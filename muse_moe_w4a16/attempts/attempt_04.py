# okc: problem=muse_moe_w4a16
"""Marlin W4A16 experts with local alignment and graph-fused post norm.

Derived from the supplied SGLang baseline (Apache-2.0). Packed NVFP4
weights, BF16 intermediates, and FP32 GEMM reductions are preserved.
"""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


@triton.jit
def _post_norm_sum(Y, Scores, Norm, Ids, Out,
                   H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    """All sizes use a fused norm, including decode's tiny token counts."""
    t = tl.program_id(0)
    h = tl.arange(0, B)
    w = tl.load(Norm + h, h < H, other=0).to(tl.float32)
    acc = tl.full((B,), 0, tl.float32)
    for k in tl.static_range(K):
        if tl.load(Ids + t * K + k) >= 0:
            y = tl.load(Y + (t * K + k) * H + h, h < H, other=0).to(tl.float32)
            y = y * w
            inv = tl.rsqrt(tl.sum(y * y, 0) / H + 1.0e-8)
            score = tl.load(Scores + t * K + k)
            acc += (y * inv) * score
    tl.store(Out + t * H + h, acc, h < H)


@triton.jit
def _align_local_routes(Ids, Local, Sorted, Experts, Total,
                        N: tl.constexpr, START: tl.constexpr,
                        E: tl.constexpr, BM: tl.constexpr,
                        NP: tl.constexpr, NE: tl.constexpr, NB: tl.constexpr):
    # Each local expert owns a disjoint padded segment. The small histogram
    # supplies identical segment boundaries to all CTAs without global atomics.
    expert = tl.program_id(0)
    p = tl.arange(0, NP)
    ids = tl.load(Ids + p, p < N, other=-1) - START
    valid = (p < N) & (ids >= 0) & (ids < E)
    if expert == 0:
        tl.store(Local + p, tl.where(valid, ids, -1), p < N)
    own = valid & (ids == expert)
    count = tl.sum(own.to(tl.int32), 0)
    if (count > 0) | (expert == 0):
        hist = tl.histogram(tl.where(valid, ids, 0), NE, mask=valid)
        padded = tl.cdiv(hist, BM) * BM
        e = tl.arange(0, NE)
        base = tl.sum(tl.where(e < expert, padded, 0), 0)
        if expert == 0:
            tl.store(Total, tl.sum(padded, 0))
        n_pad = tl.cdiv(count, BM) * BM
        # Fill and scatter are separated by a CTA barrier: different warps
        # may otherwise overwrite a real pair with its padding sentinel.
        tl.store(Sorted + base + p, N, p < n_pad)
        tl.debug_barrier()
        rank = tl.cumsum(own.to(tl.int32), 0) - 1
        tl.store(Sorted + base + rank, p, own)
        b = tl.arange(0, NB)
        tl.store(Experts + base // BM + b, expert, b < n_pad // BM)


def _align_local(ids, cfg, bm):
    n = ids.numel()
    count = cfg["local_count"]
    capacity = min(n * bm, n + count * (bm - 1))
    local = torch.empty_like(ids)
    sorted_ids = torch.empty((capacity,), dtype=torch.int32, device=ids.device)
    experts = torch.empty((triton.cdiv(capacity, bm),), dtype=torch.int32, device=ids.device)
    total = torch.empty((1,), dtype=torch.int32, device=ids.device)
    _align_local_routes[(count,)](
        ids, local, sorted_ids, experts, total,
        N=n, START=cfg["local_start"], E=count, BM=bm,
        NP=triton.next_power_of_2(n), NE=triton.next_power_of_2(count),
        NB=triton.next_power_of_2(triton.cdiv(n, bm)),
        num_warps=4,
    )
    return local, sorted_ids, experts, total

def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    # Fused mapping/alignment excludes remote routes from Marlin's schedule.
    block_size_m = 8
    local_ids, sorted_ids, block_experts, n_padded = _align_local(
        data["topk_ids"], cfg, block_size_m)
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    out = torch.empty_like(x)
    _post_norm_sum[(num_tokens,)](
        y, topk_weights, state["norm_w"], local_ids, out,
        H=size_k, K=TOP_K, B=triton.next_power_of_2(size_k),
        num_warps=8, enable_fp_fusion=False,
    )
    return out
