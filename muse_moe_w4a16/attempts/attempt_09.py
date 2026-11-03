# okc: problem=muse_moe_w4a16
"""Native NVFP4 skinny GEMMs for M<=32, Marlin W4A16 above, local alignment, fused post norm.

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
    # Native NVFP4 copy of the local experts for the skinny (M<=32) kernel:
    # the same uint8 E2M1 codes, FP8 block scales and FP32 global scales, sliced only.
    loc = slice(0, 32)
    native = {name: weights[name][loc].contiguous()
              for name in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    torch.cuda.synchronize()
    return {
        "native": native,
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

@triton.jit
def _write_segment(Own, Sorted, Experts, base, count,
                   N: tl.constexpr, BM: tl.constexpr,
                   NP: tl.constexpr, NB: tl.constexpr):
    p = tl.arange(0, NP)
    n_pad = tl.cdiv(count, BM) * BM
    tl.store(Sorted + base + p, N, p < n_pad)
    tl.debug_barrier()
    rank = tl.cumsum(Own.to(tl.int32), 0) - 1
    tl.store(Sorted + base + rank, p, Own)
    b = tl.arange(0, NB)
    tl.store(Experts + base // BM + b, tl.program_id(0), b < n_pad // BM)


@triton.jit
def _align_mixed_routes(Ids, Local, S16, E16, T16, S32, E32, T32,
                        N: tl.constexpr, START: tl.constexpr,
                        E: tl.constexpr, CUT: tl.constexpr,
                        NP: tl.constexpr, NE: tl.constexpr,
                        NB16: tl.constexpr, NB32: tl.constexpr):
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
        pad16 = tl.where(hist <= CUT, tl.cdiv(hist, 16) * 16, 0)
        pad32 = tl.where(hist > CUT, tl.cdiv(hist, 32) * 32, 0)
        e = tl.arange(0, NE)
        base16 = tl.sum(tl.where(e < expert, pad16, 0), 0)
        base32 = tl.sum(tl.where(e < expert, pad32, 0), 0)
        if expert == 0:
            tl.store(T16, tl.sum(pad16, 0))
            tl.store(T32, tl.sum(pad32, 0))
        if count > CUT:
            _write_segment(own, S32, E32, base32, count, N, 32, NP, NB32)
        else:
            _write_segment(own, S16, E16, base16, count, N, 16, NP, NB16)


def _align_mixed(ids, cfg, cut=16):
    n = ids.numel()
    count = cfg["local_count"]
    local = torch.empty_like(ids)
    groups = []
    for bm in (16, 32):
        capacity = min(n * bm, n + count * (bm - 1))
        sorted_ids = torch.empty((capacity,), dtype=torch.int32, device=ids.device)
        experts = torch.empty((triton.cdiv(capacity, bm),), dtype=torch.int32, device=ids.device)
        total = torch.empty((1,), dtype=torch.int32, device=ids.device)
        groups.append((sorted_ids, experts, total, bm))
    _align_mixed_routes[(count,)](
        ids, local, *groups[0][:3], *groups[1][:3],
        N=n, START=cfg["local_start"], E=count, CUT=cut,
        NP=triton.next_power_of_2(n), NE=triton.next_power_of_2(count),
        NB16=triton.next_power_of_2(triton.cdiv(n, 16)),
        NB32=triton.next_power_of_2(triton.cdiv(n, 32)),
        num_warps=4,
    )
    return local, groups


# Exact NVFP4 pair decode (checked in muse_prefill_layer attempt 9): PRMT maps each
# 3-bit E2M1 magnitude to its exact FP16 encoding, the sign bit is copied, both are
# widened to FP32, multiplied by the FP32 SF*GS product (as nvfp4.dequantize does)
# and rounded once to BF16. Low nibble = element 2i, high nibble = element 2i+1.
_DECODE_PAIR = tl.constexpr(r"""{
    .reg .b32 sel, tmp, vals;
    .reg .b16 lo, hi;
    .reg .f32 f0, f1;
    and.b32 sel, $2, 7;
    shl.b32 sel, sel, 4;
    and.b32 tmp, $2, 0x70;
    shl.b32 tmp, tmp, 8;
    or.b32 sel, sel, tmp;
    prmt.b32 vals, 0x3e3c3800, 0x46444240, sel;
    and.b32 tmp, $2, 8;
    shl.b32 tmp, tmp, 12;
    or.b32 vals, vals, tmp;
    and.b32 tmp, $2, 0x80;
    shl.b32 tmp, tmp, 24;
    or.b32 vals, vals, tmp;
    mov.b32 {lo, hi}, vals;
    cvt.f32.f16 f0, lo;
    cvt.f32.f16 f1, hi;
    mul.f32 f0, f0, $3;
    mul.f32 f1, f1, $3;
    cvt.rn.bf16.f32 $0, f0;
    cvt.rn.bf16.f32 $1, f1;
}""")


@triton.jit
def _nvfp4_moe_mm(X, W, SF, GS, Sorted, Experts, Total, Out,
                  N: tl.constexpr, K: tl.constexpr, ROUTES: tl.constexpr, ROW_DIV: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BKP: tl.constexpr):
    """Out[route, n] = sum_k X[route // ROW_DIV, k] * W[e, n, k] for one 16-row expert block.

    W is native NVFP4 [E, N, K/2] with FP8 scales [E, N, K/16]. Even and odd K
    elements (low/high nibbles) feed two BF16 dots; accumulation is FP32.
    """
    blk = tl.program_id(0)
    if blk * BM >= tl.load(Total):
        return
    e = tl.load(Experts + blk).to(tl.int64)
    rows = tl.load(Sorted + blk * BM + tl.arange(0, BM))
    valid = rows < ROUTES
    xrow = tl.where(valid, rows // ROW_DIV, 0).to(tl.int64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    G: tl.constexpr = BKP // 8
    g = tl.arange(0, G)
    j = tl.arange(0, 8)
    gs = tl.load(GS + e)
    w_base = W + e * (N * (K // 2)) + n[:, None, None] * (K // 2)
    s_base = SF + e * (N * (K // 16)) + n[:, None] * (K // 16)
    x_base = X + xrow[:, None] * K
    kp = tl.arange(0, BKP)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K // 2, BKP):
        packed = tl.load(w_base + k0 + g[None, :, None] * 8 + j[None, None, :])      # [BN, G, 8] u8
        scale = tl.load(s_base + k0 // 8 + g[None, :]).to(tl.float32) * gs           # [BN, G]
        w_lo, w_hi = tl.inline_asm_elementwise(
            _DECODE_PAIR, "=h,=h,r,f",
            [packed.to(tl.int32), tl.broadcast_to(scale[:, :, None], (BN, G, 8))],
            dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=1)
        w_lo = tl.reshape(w_lo, (BN, BKP))
        w_hi = tl.reshape(w_hi, (BN, BKP))
        x_even = tl.load(x_base + 2 * (k0 + kp[None, :]), mask=valid[:, None], other=0.0)
        x_odd = tl.load(x_base + 2 * (k0 + kp[None, :]) + 1, mask=valid[:, None], other=0.0)
        acc = tl.dot(x_even, tl.trans(w_lo), acc)
        acc = tl.dot(x_odd, tl.trans(w_hi), acc)
    out = Out + rows.to(tl.int64)[:, None] * N + n[None, :]
    tl.store(out, acc.to(tl.bfloat16), mask=valid[:, None])


SKINNY_MAX_TOKENS = 32
SKINNY_BN = 64
SKINNY_BKP = 64


def _skinny_mm(x, w, sf, gs, sorted_ids, block_experts, n_padded, out, n, k, routes, row_div):
    grid = (sorted_ids.numel() // 16, n // SKINNY_BN)
    _nvfp4_moe_mm[grid](
        x, w, sf, gs, sorted_ids, block_experts, n_padded, out,
        N=n, K=k, ROUTES=routes, ROW_DIV=row_div,
        BM=16, BN=SKINNY_BN, BKP=SKINNY_BKP, num_warps=4)


def _skinny_experts(x, data, cfg, state):
    """M<=32: native NVFP4 skinny GEMMs replace both Marlin calls; same intermediates."""
    nat = state["native"]
    m, hidden = x.shape
    inter = nat["w2"].shape[2] * 2
    routes = m * TOP_K
    local_ids, sorted_ids, block_experts, n_padded = _align_local(data["topk_ids"], cfg, 16)
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    cache1 = torch.empty((routes, 2 * inter), device=x.device, dtype=x.dtype)
    _skinny_mm(x, nat["w13"], nat["w13_sf"], nat["w13_gs"], sorted_ids, block_experts, n_padded,
               cache1, 2 * inter, hidden, routes, TOP_K)
    cache2 = torch.empty((routes, inter), device=x.device, dtype=x.dtype)
    silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1)
    cache3 = torch.empty((routes, hidden), device=x.device, dtype=x.dtype)
    _skinny_mm(cache2, nat["w2"], nat["w2_sf"], nat["w2_gs"], sorted_ids, block_experts, n_padded,
               cache3, hidden, inter, routes, 1)
    out = torch.empty_like(x)
    _post_norm_sum[(m,)](
        cache3.view(m, TOP_K, hidden), topk_weights, state["norm_w"], local_ids, out,
        H=hidden, K=TOP_K, B=triton.next_power_of_2(hidden),
        num_warps=8, enable_fp_fusion=False,
    )
    return out


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    # Shape-only dispatch: decode/verify-1 sizes use the native NVFP4 skinny kernel.
    if num_tokens <= SKINNY_MAX_TOKENS and cfg["local_start"] == 0 and cfg["local_count"] == 32:
        return _skinny_experts(x, data, cfg, state)
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    # Fused mapping/alignment excludes remote routes from Marlin's schedule.
    if num_tokens <= 32:
        local_ids, sorted_ids, block_experts, n_padded = _align_local(
            data["topk_ids"], cfg, 8)
        groups = [(sorted_ids, block_experts, n_padded, 8)]
    else:
        local_ids, groups = _align_mixed(data["topk_ids"], cfg)
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    for sorted_ids, block_experts, n_padded, bm in groups:
        moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, moe_block_size=bm, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    for sorted_ids, block_experts, n_padded, bm in groups:
        moe_wna16_marlin_gemm(
            cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
            size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, moe_block_size=bm, **common)
    y = cache3.view(num_tokens, TOP_K, size_k)
    out = torch.empty_like(x)
    _post_norm_sum[(num_tokens,)](
        y, topk_weights, state["norm_w"], local_ids, out,
        H=size_k, K=TOP_K, B=triton.next_power_of_2(size_k),
        num_warps=8, enable_fp_fusion=False,
    )
    return out
