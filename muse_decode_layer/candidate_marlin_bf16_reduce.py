# okc: problem=muse_decode_layer
# Attempt 16: attempt 14 + Marlin's split-K partials held in BF16 (use_fp32_reduce=False; rules 2026-11-03).
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
import triton as tr
import triton.language.extra.cuda.libdevice as libdevice
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


@triton.jit
def _dense_partials(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SPLIT: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    ki = split * (K // SPLIT) + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(K // SPLIT // BK):
        kk = ki + step * BK
        a = tl.load(X + mi[:, None] * K + kk[None, :], mi[:, None] < M, other=0)
        b = tl.load(W + ni[None, :] * K + kk[:, None], ni[None, :] < N, other=0)
        acc += tl.dot(a, b)
    tl.store(P + split * M * N + mi[:, None] * N + ni[None, :],
             acc, (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _dense_reduce(P, O, SIZE: tl.constexpr, SPLIT: tl.constexpr,
                  BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLIT)
    partial = tl.load(P + s[:, None] * SIZE + i[None, :], i[None, :] < SIZE, other=0)
    tl.store(O + i, tl.sum(partial, axis=0), i < SIZE)


@triton.jit
def _router_partials(X, HI, LO, P, M: tl.constexpr, BM: tl.constexpr):
    # The permitted BF16 high/low router representation, both accumulated FP32.
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * 32 + tl.arange(0, 32)
    split = tl.program_id(2)
    ki = split * 512 + tl.arange(0, 128)
    hi_acc = tl.full((BM, 32), 0, tl.float32)
    lo_acc = tl.full((BM, 32), 0, tl.float32)
    for step in range(4):
        kk = ki + step * 128
        x = tl.load(X + mi[:, None] * 8192 + kk[None, :], mi[:, None] < M, other=0)
        high = tl.load(HI + kk[:, None] * 256 + ni[None, :])
        low = tl.load(LO + kk[:, None] * 256 + ni[None, :])
        hi_acc += tl.dot(x, high)
        lo_acc += tl.dot(x, low)
    tl.store(P + split * M * 256 + mi[:, None] * 256 + ni[None, :],
             hi_acc + lo_acc, mi[:, None] < M)


def _router_logits(x, state):
    m = x.shape[0]
    partial = torch.empty((16, m, 256), device=x.device, dtype=torch.float32)
    out = torch.empty((m, 256), device=x.device, dtype=torch.float32)
    bm = 16 if m <= 16 else 32
    _router_partials[(triton.cdiv(m, bm), 8, 16)](
        x, state["router_hi_t"], state["router_lo_t"], partial, m, bm,
        num_warps=4, num_stages=2)
    _dense_reduce[(triton.cdiv(m * 256, 256),)](
        partial, out, m * 256, 16, 256, num_warps=4)
    return out


def _gemm(x, w):
    m, k = x.shape
    n = w.shape[0]
    if m > 8:
        return _bf16_gemm_dispatch_impl(x, w, None)
    # Static shape dispatch; all scratch tensors are fully overwritten per call.
    # The input and weight stay BF16. Both the dot products and split reduction
    # accumulate in FP32, with one final BF16 output conversion.
    split = 8 if k >= 4096 else 4
    bm = 16 if m <= 16 else 32
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _dense_partials[(triton.cdiv(m, bm), triton.cdiv(n, 64), split)](
        x, w, partial, m, n, k, bm, 64, 128, split, num_warps=4, num_stages=2)
    _dense_reduce[(triton.cdiv(m * n, 256),)](
        partial, out, m * n, split, 256, num_warps=4)
    return out


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "pre_expert_w": (weights["pre_expert_norm"] + 1).contiguous(),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


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


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = topk_ids
    topk_weights = topk_weights.to(torch.float32).contiguous()
    # Route tiles measured in attempt 3, preserving every route.
    block_size_m = 8 if num_tokens <= 16 else (16 if num_tokens <= 128 else 32)
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=False, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    # muse_moe_w4a16 attempt 6's fused post-expert norm and weighted sum (scored
    # aa4935c3095c450d): per token, each local slot's output times the effective
    # norm weight, RMS-normalized (eps 1e-8), times its routing weight, summed in FP32.
    out = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _post_norm_sum[(num_tokens,)](
        y, topk_weights, state["post_expert_w"], local_ids, out,
        H=size_k, K=TOP_K, B=triton.next_power_of_2(size_k), num_warps=8, enable_fp_fusion=False)
    return out


@triton.jit
def _norm_and_local_routes(X, W, TOP, START, Y, IDS):
    token = tl.program_id(0)
    col = tl.arange(0, 4096)
    x = tl.load(X + token * 4096 + col).to(tl.float32)
    weight = tl.load(W + col).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / 4096 + 1.0e-5)
    tl.store(Y + token * 4096 + col, x * rstd * weight)
    slot = tl.arange(0, 8)
    expert = tl.load(TOP + token * 8 + slot).to(tl.int32)
    start = tl.load(START).to(tl.int32)
    local = (expert - start) & 255
    tl.store(IDS + token * 8 + slot, tl.where(local < 32, local, -1))


def _expert_input_and_ids(x, topk_ids, local_start, state):
    y = torch.empty_like(x)
    ids = torch.empty(topk_ids.shape, device=x.device, dtype=torch.int32)
    _norm_and_local_routes[(x.shape[0],)](
        x, state["pre_expert_w"], topk_ids, local_start, y, ids,
        num_warps=4, enable_fp_fusion=False)
    return y, ids


@triton.jit
def _qkv_to_cache(QKV, Q, KC, VC, LOC, POS, CS, KS, VS,
                  T: tl.constexpr, STRIDE: tl.constexpr, ROPE: tl.constexpr,
                  SCALED: tl.constexpr, BM: tl.constexpr):
    # Adapted from SGLang FusedQKNorm. Preserve both BF16 rounding points,
    # write slot zero (live in this challenge), and support cache descales.
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.arange(0, 64)
    token, head = rows // 20, rows % 20
    valid = rows < T * 20
    raw = tl.load(QKV + token[:, None] * STRIDE + head[:, None] * 64 + col[None, :],
                  valid[:, None], other=0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(raw * raw, 1) / 64 + 1.0e-5)
    norm = (raw * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        partner = tl.load(QKV + token[:, None] * STRIDE + head[:, None] * 64
                          + (col[None, :] ^ 1), valid[:, None], other=0).to(tl.float32)
        partner = (partner * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(POS + token, valid, other=0).to(tl.int64)
        cosine = tl.load(CS + pos[:, None] * 64 + col[None, :] // 2,
                         valid[:, None], other=1)
        sine = tl.load(CS + pos[:, None] * 64 + 32 + col[None, :] // 2,
                       valid[:, None], other=0)
        norm = tl.where((col[None, :] % 2) == 0,
                        norm * cosine - partner * sine, norm * cosine + partner * sine)
        norm = norm.to(tl.bfloat16).to(tl.float32)
    tl.store(Q + token[:, None] * 1024 + head[:, None] * 64 + col[None, :],
             norm, valid[:, None] & (head[:, None] < 16))
    loc = tl.load(LOC + token, valid, other=0).to(tl.int64)
    key = norm
    value = raw
    if SCALED:
        # The separate PyTorch division also rounds scaled K and V to BF16.
        key = (key / tl.load(KS)).to(tl.bfloat16).to(tl.float32)
        value = (value / tl.load(VS)).to(tl.bfloat16).to(tl.float32)
    tl.store(KC + loc[:, None] * 128 + (head[:, None] - 16) * 64 + col[None, :],
             key, valid[:, None] & (head[:, None] >= 16) & (head[:, None] < 18))
    tl.store(VC + loc[:, None] * 128 + (head[:, None] - 18) * 64 + col[None, :],
             value, valid[:, None] & (head[:, None] >= 18))


ATTENTION_N = 128


@tr.jit
def _prepare(QKV, POS, LOC, KS, VS, KC, VC, Q,
             HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             STRIDE: tl.constexpr, THETA: tl.constexpr,
             TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = row // (HQ + HK)
    h = row % (HQ + HK)
    valid = t < TOKENS
    x = tl.arange(0, D)
    a = tl.load(QKV + t[:, None] * STRIDE + h[:, None] * D + x[None, :],
                mask=valid[:, None], other=0.).to(tl.float32)
    r = tl.rsqrt(tl.sum(a * a, 1) / D + 1.e-5)
    a = a * r[:, None]
    if THETA != 0:
        other = tl.load(QKV + t[:, None] * STRIDE + h[:, None] * D + (x[None, :] ^ 1),
                        mask=valid[:, None], other=0.).to(tl.float32) * r[:, None]
        freq = tl.div_rn(1., libdevice.pow(THETA, ((x // 2) * 2).to(tl.float32) / D))
        angle = tl.load(POS + t, mask=valid, other=0).to(tl.float32)[:, None] * freq[None, :]
        c = libdevice.cos(angle)
        s = libdevice.sin(angle)
        a = a * c + tl.where(x[None, :] % 2 == 0, -other, other) * s
    q = a.to(tl.bfloat16).to(tl.float32)
    tl.store(Q + (t[:, None] * HQ + h[:, None]) * D + x[None, :], q,
             mask=valid[:, None] & (h[:, None] < HQ))
    kh = h - HQ
    is_k = valid & (h >= HQ)
    loc = tl.load(LOC + t, mask=is_k, other=0)
    v = tl.load(QKV + t[:, None] * STRIDE + (HQ + HK + kh[:, None]) * D + x[None, :],
                mask=is_k[:, None], other=0.).to(tl.float32)
    tl.store(KC + (loc[:, None] * HK + kh[:, None]) * D + x[None, :],
             a / tl.load(KS), mask=is_k[:, None])
    tl.store(VC + (loc[:, None] * HK + kh[:, None]) * D + x[None, :],
             v / tl.load(VS), mask=is_k[:, None])
    # PDL trigger after this CTA's stores (attempt 16 triggered at the start):
    # the attention grid is staged only as _prepare drains, never alongside it.
    gdc_launch_dependents()



@tr.jit
def _gate(A, QKV, O, HQ: tl.constexpr, HK: tl.constexpr,
          D: tl.constexpr, STRIDE: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    x = tl.arange(0, D)
    idx = (t * HQ + h) * D + x
    a = tl.load(A + idx).to(tl.bfloat16).to(tl.float32)
    g = tl.load(QKV + t * STRIDE + (HQ + 2 * HK) * D + h * D + x).to(tl.float32)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / D + 1.e-5)
    tl.store(O + idx, a / (1. + tl.exp(-g)))


@tr.jit
def _attention_parts(Q, KC, VC, PT, LENS, KS, VS, PART, LSE,
                     HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                     L: tl.constexpr, PAGE: tl.constexpr, PT_STRIDE: tl.constexpr,
                     SPLITS: tl.constexpr, CHUNK: tl.constexpr, WINDOW: tl.constexpr,
                     SCALE: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    b = tl.program_id(0)
    kh = tl.program_id(1)
    split = tl.program_id(2)
    G: tl.constexpr = HQ // HK
    r = tl.arange(0, M)
    d = tl.arange(0, D)
    n = tl.arange(0, N)
    j = r // G
    # PDL: launched early (launch_pdl=True); block until _prepare has fully
    # completed and its q / KV-cache writes are visible. Nothing is written above.
    gdc_wait()
    q = tl.load(Q + ((b * L + j[:, None]) * HQ + kh * G + r[:, None] % G) * D + d[None, :],
                mask=r[:, None] < L * G, other=0.)
    seqlen = tl.load(LENS + b)
    pos = seqlen - L + j
    scale = tl.load(KS) * SCALE * 1.4426950408889634
    m = tl.full((M,), -float("inf"), tl.float32)
    z = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    first = tl.maximum(0, seqlen - L - WINDOW + 1) if WINDOW else 0
    begin = first + split * CHUNK
    end = tl.minimum(begin + CHUNK, seqlen)
    for start in range(begin, end, N):
        ts = start + n
        pages = tl.load(PT + b * PT_STRIDE + ts // PAGE, mask=ts < seqlen, other=0)
        rows = pages * PAGE + ts % PAGE
        k = tl.load(KC + (rows[None, :] * HK + kh) * D + d[:, None],
                    mask=ts[None, :] < seqlen, other=0.)
        logits = tl.dot(q, k) * scale
        logits = tl.where((ts[None, :] < seqlen) & (ts[None, :] <= pos[:, None]), logits, -float("inf"))
        if WINDOW:
            logits = tl.where(ts[None, :] > pos[:, None] - WINDOW, logits, -float("inf"))
        new_m = tl.maximum(m, tl.max(logits, 1))
        new_m = tl.where(new_m == -float("inf"), 0., new_m)
        p = tl.exp2(logits - new_m[:, None])
        alpha = tl.exp2(m - new_m)
        z = z * alpha + tl.sum(p, 1)
        v = tl.load(VC + (rows[:, None] * HK + kh) * D + d[None, :],
                    mask=ts[:, None] < seqlen, other=0.)
        acc = acc * alpha[:, None]
        if WINDOW:
            acc = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc)
        else:
            acc = tl.dot((p * 256.).to(q.dtype), v, acc)
        m = new_m
    base = ((b * HK + kh) * SPLITS + split) * (L * G) + r
    result = acc / tl.maximum(z[:, None], 1.e-20) * tl.load(VS)
    if not WINDOW:
        result = result / 256.
    tl.store(PART + base[:, None] * D + d[None, :], result, mask=r[:, None] < L * G)
    logsum = tl.where(z > 0, m + tl.log2(z), -float("inf"))
    tl.store(LSE + base, logsum, mask=r < L * G)
    # Trigger the combine grid only after this CTA's stores, so early-staged
    # combine CTAs never take SM slots from later attention waves.
    gdc_launch_dependents()


@tr.jit
def _combine_gate(PART, LSE, QKV, O, HQ: tl.constexpr, HK: tl.constexpr,
                  D: tl.constexpr, L: tl.constexpr, STRIDE: tl.constexpr,
                  SPLITS: tl.constexpr, S: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    G: tl.constexpr = HQ // HK
    b, j = t // L, t % L
    s = tl.arange(0, S)
    d = tl.arange(0, D)
    base = ((b * HK + h // G) * SPLITS + s) * (L * G) + j * G + h % G
    # The gate comes from the step's input qkv, which no kernel here writes.
    g = tl.load(QKV + t * STRIDE + (HQ + 2 * HK + h) * D + d).to(tl.float32)
    # PDL: wait until _attention_parts has completed (partials and LSE visible).
    gdc_wait()
    lse = tl.load(LSE + base, mask=s < SPLITS, other=-float("inf"))
    w = tl.exp2(lse - tl.max(lse, 0))
    a = tl.load(PART + base[:, None] * D + d[None, :], mask=s[:, None] < SPLITS, other=0)
    a = tl.sum(a * w[:, None], 0) / tl.sum(w, 0)
    a = a.to(tl.bfloat16).to(tl.float32)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / D + 1.e-5)
    tl.store(O + (t * HQ + h) * D + d, a / (1. + tl.exp(-g)))



def _attention_block(data, qkv):
    """muse_attention_decode attempt 17's kernel_fn (scored 7e9dd2b58e9746e4): q/k norm, RoPE,
    FP8 cache store, attention and the output RMSNorm x sigmoid(gate), on this layer's qkv."""
    cfg = data["config"]
    b, length = cfg["batch"], cfg["q_len"]
    hq, hk, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = b * length
    stride = qkv.stride(-2)
    window = cfg["window"]
    q = torch.empty((tokens, hq, d), device=qkv.device, dtype=torch.float8_e4m3fn)
    _prepare[(tr.cdiv(tokens * (hq + hk), 4),)](
        qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
        data["k_cache"], data["v_cache"], q, hq, hk, d, stride, cfg["rope_theta"], tokens, 4,
        num_warps=4, enable_fp_fusion=False)
    if not window and length > 1 and b * hk >= 16:
        descale = {}
        if cfg["kv_scale"] != 1.:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(b, hk),
                           v_descale=data["v_scale"].reshape(1, 1).expand(b, hk))
        window = cfg["window"]
        a = flash_attn_with_kvcache(
            q=q, k_cache=data["k_cache"], v_cache=data["v_cache"],
            page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            scheduler_metadata=data["scheduler_metadata"], **descale)
        if isinstance(a, tuple):
            a = a[0]
        out = torch.empty((b, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
        _gate[(tokens, hq)](a, qkv, out, hq, hk, d, stride, num_warps=4)
        return out.view(tokens, hq * d)
    chunk = (128 if b < 8 else 256) if window else 2048
    capacity = data["page_table"].shape[1] * cfg["page_size"]
    if not window and b < 8 and capacity <= 262144:
        chunk = 256
    span = min(capacity, window + length - 1) if window else capacity
    splits = tr.cdiv(span, chunk)
    rows = length * (hq // hk)
    partial = torch.empty((b, hk, splits, rows, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((b, hk, splits, rows), device=qkv.device, dtype=torch.float32)
    out = torch.empty((b, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _attention_parts[(b, hk, splits)](
        q, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"],
        data["k_scale"], data["v_scale"], partial, lse, hq, hk, d, length,
        cfg["page_size"], data["page_table"].stride(0), splits, chunk, window,
        cfg["sm_scale"] * cfg["qk_scale_factor"], max(16, tr.next_power_of_2(rows)), ATTENTION_N,
        num_warps=4, num_stages=2, launch_pdl=True)
    _combine_gate[(tokens, hq)](
        partial, lse, qkv, out, hq, hk, d, length, stride, splits,
        tr.next_power_of_2(splits), num_warps=8 if splits >= 128 else 4, launch_pdl=True)
    return out.view(tokens, hq * d)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    o = _attention_block(data, qkv)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > 1
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if 1 < tokens <= 32:
            logits = _router_logits(ffn_input, state)
        elif split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    expert_input, local_ids = _expert_input_and_ids(
        expert_input, topk_ids, data["local_start"], state)
    # all-reduce: identity
    expert_output = _experts(expert_input, topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
