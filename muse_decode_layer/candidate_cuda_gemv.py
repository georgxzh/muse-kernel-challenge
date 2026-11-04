# okc: problem=muse_decode_layer
# Attempt 22: attempt 20 + muse_bf16_proj attempt 17's CUDA GEMV for the T <= 8 projections.
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
from torch.utils.cpp_extension import load_inline
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


# Small-batch BF16 projections in CUDA (source below; compiled once at import with nvcc via
# load_inline): tensor-core GEMV with 128-byte-line weight reads and fixed-order split-K.
_GEMV_CU = r"""
// Small-batch BF16 projection y[M, N] = x[M, K] @ W[N, K]^T (M <= 16), FP32 accumulation,
// BF16 output, for decode/verify token counts where the weight read dominates.
//
// Tensor cores: mma.sync m16n8k16 BF16 x BF16 -> FP32; A = x (rows >= M are zero), B = W
// (row-major W is B in column-major order). Each lane reads 32 contiguous bytes of its weight
// row per 64-element chunk (4 lanes cover a 128-byte line): within a chunk, k-step s uses
// elements 16t + 4s .. 16t + 4s + 3 of lane t, a fixed permutation of K applied to x and W
// alike, so x's A fragments come from the same contiguous 32 bytes of x's rows.
// A CTA (4 warps, 32 output columns) covers one K range; split-K partials are summed in a
// fixed order by the last CTA to arrive (per-column-block arrival counters, reset to zero by
// that CTA: the self-resetting pattern of SGLang's Marlin workspace). Every product is
// accumulated in FP32.

#if defined(__CUDACC_RTC__)
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
#else
#include <stdint.h>
#endif

namespace muse_gemv {

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

__device__ __forceinline__ uint32_t w4(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint16_t f2bf(float f) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(r) : "f"(f));
  return r;
}

}  // namespace muse_gemv

// grid: (N / 32, splits); block: 128 threads. K % (64 * UNROLL) == 0 per split range.
extern "C" __global__ void __launch_bounds__(128)
muse_gemv_bf16(const uint16_t* __restrict__ X,   // [M, K] BF16
               const uint16_t* __restrict__ W,   // [N, K] BF16
               uint16_t* __restrict__ Y,         // [M, N] BF16
               float* __restrict__ P,            // [splits, M, N] FP32 partials
               int* __restrict__ CNT,            // [N / 32] arrival counters, zero at rest
               int M, int N, int K, int kc) {
  using namespace muse_gemv;
  constexpr int UNROLL = 4;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * 32 + warp * 8;
  const int split = blockIdx.y, splits = gridDim.y;
  const int kbeg = split * kc;
  const uint16_t* wrow = W + (long long)(n0 + g) * K + kbeg + 16 * t;
  const bool has_lo = g < M, has_hi = g + 8 < M;
  const uint16_t* xlo = X + (long long)(has_lo ? g : 0) * K + kbeg + 16 * t;
  const uint16_t* xhi = X + (long long)(has_hi ? g + 8 : 0) * K + kbeg + 16 * t;
  const uint4 zero = make_uint4(0, 0, 0, 0);

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < kc; k += 64 * UNROLL) {
    uint4 wv[UNROLL][2], lo[UNROLL][2], hi[UNROLL][2];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int off = k + 64 * u;
      wv[u][0] = ldg16(wrow + off);
      wv[u][1] = ldg16(wrow + off + 8);
      lo[u][0] = has_lo ? ldg16(xlo + off) : zero;
      lo[u][1] = has_lo ? ldg16(xlo + off + 8) : zero;
      hi[u][0] = has_hi ? ldg16(xhi + off) : zero;
      hi[u][1] = has_hi ? ldg16(xhi + off + 8) : zero;
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        // Words 2s, 2s+1 of the lane's 16 elements: elements 4s, 4s+1 | 4s+2, 4s+3.
        const int q = (2 * s) >> 2, r = (2 * s) & 3;
        const uint4& wa = wv[u][q];
        const uint4& la = lo[u][q];
        const uint4& ha = hi[u][q];
        mma_bf16(c, w4(la, r), w4(ha, r), w4(la, r + 1), w4(ha, r + 1), w4(wa, r), w4(wa, r + 1));
      }
    }
  }
  // c[0..1]: row g, columns n0 + 2t + {0, 1}; c[2..3]: row g + 8.
  const int col = n0 + 2 * t;
  if (splits == 1) {
    if (has_lo) {
      Y[(long long)g * N + col] = f2bf(c[0]);
      Y[(long long)g * N + col + 1] = f2bf(c[1]);
    }
    if (has_hi) {
      Y[(long long)(g + 8) * N + col] = f2bf(c[2]);
      Y[(long long)(g + 8) * N + col + 1] = f2bf(c[3]);
    }
    return;
  }
  float* ps = P + (long long)split * M * N;
  if (has_lo) { ps[(long long)g * N + col] = c[0]; ps[(long long)g * N + col + 1] = c[1]; }
  if (has_hi) { ps[(long long)(g + 8) * N + col] = c[2]; ps[(long long)(g + 8) * N + col + 1] = c[3]; }
  __threadfence();
  __syncthreads();
  __shared__ int s_last;
  if (threadIdx.x == 0) s_last = atomicAdd(CNT + blockIdx.x, 1) == splits - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  // Last CTA of this column block: sum the partials in split order, write BF16, reset.
  for (int idx = threadIdx.x; idx < M * 32; idx += blockDim.x) {
    const int m = idx / 32, n = blockIdx.x * 32 + idx % 32;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) acc += __ldcg(P + ((long long)s * M + m) * N + n);
    Y[(long long)m * N + n] = f2bf(acc);
  }
  if (threadIdx.x == 0) CNT[blockIdx.x] = 0;
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void gemv_bf16(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor p, torch::Tensor cnt,
               int64_t m, int64_t n, int64_t k, int64_t splits) {
  dim3 grid((unsigned)(n / 32), (unsigned)splits);
  muse_gemv_bf16<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint16_t*)x.data_ptr(), (const uint16_t*)w.data_ptr(), (uint16_t*)y.data_ptr(),
      p.data_ptr<float>(), cnt.data_ptr<int>(), (int)m, (int)n, (int)k, (int)(k / splits));
}
"""
_GEMV = load_inline(
    name="muse_gemv_bf16_v1",
    cpp_sources="void gemv_bf16(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor p, "
                "torch::Tensor cnt, int64_t m, int64_t n, int64_t k, int64_t splits);",
    cuda_sources=_GEMV_CU, functions=["gemv_bf16"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)


def _gemv_splits(n, k):
    # About 1024 CTAs of 32 columns, K ranges of at least 256 (shape-only).
    splits = 1
    while (n // 32) * splits * 2 <= 1024 and k // (splits * 2) >= 256 and k % (splits * 2 * 256) == 0:
        splits *= 2
    return splits


def _cuda_gemv(x, w, state):
    m, k = x.shape
    n = w.shape[0]
    splits = _gemv_splits(n, k)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    _GEMV.gemv_bf16(x.contiguous(), w, out, partial, state["gemv_counters"], m, n, k, splits)
    return out


def _gemm(x, w):
    m, k = x.shape
    n = w.shape[0]
    if m > 8:
        return _bf16_gemm_dispatch_impl(x, w, None)
    if n % 32 == 0 and k % 256 == 0:
        # muse_bf16_proj attempt 17's CUDA tensor-core GEMV (FP32 accumulation, fixed-order split-K).
        return _cuda_gemv(x, w, _GEMV_STATE)
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


_GEMV_STATE = {}


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
    # CUDA GEMV arrival counters (one per 32-column block), zero at rest.
    _GEMV_STATE["gemv_counters"] = torch.zeros((8192 // 32,), dtype=torch.int32, device=device)
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


# muse_moe_w4a16 attempt 6's adaptive route alignment (scored aa4935c3095c450d), verbatim:
# experts with <= 16 routes get 16-row Marlin tiles, busier experts 32-row tiles.
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
    if num_tokens > 32:
        # Local ids are already 0..31 or -1, so the aligner runs with START=0.
        _, groups = _align_mixed(local_ids, {"local_start": 0, "local_count": LOCAL_EXPERTS})
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
        groups = [(sorted_ids, block_experts, n_padded, block_size_m)]
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    for sorted_ids, block_experts, n_padded, bm in groups:
        moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **dict(common, moe_block_size=bm))
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    for sorted_ids, block_experts, n_padded, bm in groups:
        moe_wna16_marlin_gemm(
            cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
            size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **dict(common, moe_block_size=bm))
    y = cache3.view(num_tokens, TOP_K, size_k)
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


# Full-layer decode attention in CUDA (source below, compiled once at import with nvcc via
# load_inline). Same inputs/outputs as _attention_parts for L = 1: the FP8 query from
# _prepare, the FP8 caches, and FP32 split-KV partials + LSE for _combine_gate.
_FULL_DECODE_CU = r"""
// Full-layer decode attention (one query token per sequence, 16 query heads, 2 KV heads,
// head dim 64, FP8 E4M3 query/cache), split-KV partials in the layout _combine_gate reads.
//
// One CTA covers one (sequence, split) for BOTH KV heads, so every cached token's K row
// and V row (2 heads x 64 FP8 bytes = 128 contiguous bytes each) is gathered whole: the
// gather probes (diagnostic 01) read whole rows 1.55x faster than 64-byte half rows.
//
// Tensor cores: mma.sync m16n8k32 E4M3 x E4M3 -> FP32 for both Q.K^T and P.V.
//  * Q.K^T: A = the 16 query rows (rows 0-7: KV head 0's query heads, rows 8-15: KV head
//    1's), K dimension = the 128 bytes of a cache row. Row r is zero outside its KV head's
//    64 bytes, so the product equals the per-head dot product. The 32-wide k-steps visit
//    the row's bytes in a fixed permuted order (the same for Q and K), chosen so each
//    thread's B fragment is two 16-byte loads of its token's row.
//  * P.V: A = P (FP32 probabilities x 256 rounded to E4M3, as FA3's FP8 path and our
//    Triton kernel do), K dimension = 32 tokens in a permuted order that matches the Q.K^T
//    accumulator layout, N = the 128 row bytes in the order (n-tile nt, column g) ->
//    byte 16 * g + nt, so each thread's V bytes are one 16-byte load per token followed by
//    an in-register 4x4 byte transpose (PRMT).
//  * Softmax in FP32, base-2, per warp; the CTA's 4 warps (disjoint 32-token tiles) are
//    merged at the end like split-KV partials.
// Every key in [split * chunk, min(seqlen, (split + 1) * chunk)) is read and weighted.
//
// v3 data path: each warp stages its 32-token tiles' whole K and V rows into shared
// memory with cp.async (2 stages, page-table entries fetched one tile further ahead);
// 8 lanes copy one row, so every copy instruction moves 4 whole 128-byte rows,
// then reads MMA fragments from shared memory (16-byte chunks XOR-swizzled by row so
// the fragment reads are bank-conflict free). The 4 warps' buffers are reused for the
// final merge.

#if defined(__CUDACC_RTC__)
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
#else
#include <stdint.h>
#endif

#define MUSE_NEG_INF (__int_as_float(0xff800000))

namespace muse_fd {

constexpr int HQ = 16;      // query heads
constexpr int D = 64;       // head dim
constexpr int ROW = 128;    // bytes per cache row (2 KV heads x 64)
constexpr int WARPS = 4;
constexpr int TILE = 32;    // tokens per warp iteration

__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// Four FP32 values -> four E4M3 bytes (x0 in byte 0), round to nearest, saturating.
__device__ __forceinline__ uint32_t pack_e4m3(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(x3), "f"(x2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

__device__ __forceinline__ uint4 ldg16(const uint8_t* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

// 2^x with the GPU's approximate exp2 (accurate far beyond BF16; ex2(-inf) = 0).
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ int cache_row(const int* __restrict__ pt_b, int ts, int page) {
  return pt_b[ts / page] * page + ts % page;
}

}  // namespace muse_fd


namespace muse_fd {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const uint8_t* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ uint4 lds16(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }

// Byte offset of 16-byte chunk c of tile row r (row = 128 bytes, 8 chunks), XOR-swizzled.
__device__ __forceinline__ int sw(int r, int c) { return r * ROW + ((c ^ (r & 7)) << 4); }

}  // namespace muse_fd

extern "C" __global__ void __launch_bounds__(128, 3)
muse_attn_full_decode(const uint8_t* __restrict__ Q,     // [B, 16, 64] E4M3
                      const uint8_t* __restrict__ KC,    // [rows, 128] E4M3
                      const uint8_t* __restrict__ VC,    // [rows, 128] E4M3
                      const int* __restrict__ PT,        // [B, pt_stride]
                      const int* __restrict__ LENS,      // [B]
                      const float* __restrict__ KS,      // [] key scale
                      const float* __restrict__ VS,      // [] value scale
                      float* __restrict__ PART,          // [B, 2, splits, 8, 64]
                      float* __restrict__ LSE,           // [B, 2, splits, 8]
                      int page, int pt_stride, int splits, int chunk, float sm_scale) {
  using namespace muse_fd;
  // [warp][stage][K, V][TILE rows][128 bytes] = 64 KB; reused as the merge buffers.
  extern __shared__ __align__(128) uint8_t smem[];

  const int split = blockIdx.x;
  const int b = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;   // groupID: A/C row (and +8), B column
  const int t = lane & 3;    // thread in group
  const int seqlen = LENS[b];
  const int begin = split * chunk;
  const int end = min(begin + chunk, seqlen);
  const float scale = KS[0] * sm_scale * 1.4426950408889634f;
  const int* pt_b = PT + (long long)b * pt_stride;
  uint8_t* wbuf = smem + warp * (2 * 2 * TILE * ROW);

  // Q A-fragments. Thread (g, t) holds 32 bytes of one query head: t < 2 -> head g (KV head 0),
  // bytes t*32..t*32+31 of its 64; t >= 2 -> head 8 + g (KV head 1), bytes (t-2)*32... .
  // k-step s, logical k = t*4 + j (j < 4) <-> row byte t*32 + 8s + j; k = t*4 + 16 + j <->
  // row byte t*32 + 8s + 4 + j. Rows of the other KV head are zero.
  uint32_t qw[8];
  {
    const int head = (t < 2) ? g : 8 + g;
    const uint8_t* src = Q + ((long long)b * HQ + head) * D + (t & 1) * 32;
    uint4 w0 = *reinterpret_cast<const uint4*>(src);
    uint4 w1 = *reinterpret_cast<const uint4*>(src + 16);
    qw[0] = w0.x; qw[1] = w0.y; qw[2] = w0.z; qw[3] = w0.w;
    qw[4] = w1.x; qw[5] = w1.y; qw[6] = w1.z; qw[7] = w1.w;
  }
  const bool kv0 = t < 2;

  float o[16][4];
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
  float m_lo = MUSE_NEG_INF, m_hi = MUSE_NEG_INF;  // running max, rows g and g + 8 (quad-uniform)
  float z_lo = 0.f, z_hi = 0.f;                    // this thread's partial sums of p

  // This warp's tiles start at begin + warp * TILE, every WARPS * TILE tokens. Lane l looks
  // up the cache row of token (tile base + l); keys past `end` use the row of `begin`
  // (always valid) and their logits are masked to -inf. Staging: in copy i, lane l moves
  // chunk l % 8 of tile row l / 8 + 4i (row index shuffled from its owner lane), so each
  // instruction copies 4 whole 128-byte rows.
  const int step = WARPS * TILE;
  const int first = begin + warp * TILE;
  auto row_of = [&](int base) -> int {
    int ts = base + lane;
    ts = ts < end ? ts : begin;
    return cache_row(pt_b, ts, page);
  };
  auto stage_tile = [&](int st, int my_row) {
    uint8_t* kb = wbuf + st * (2 * TILE * ROW);
    uint8_t* vb = kb + TILE * ROW;
    const int c = lane & 7;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = (lane >> 3) + 4 * i;
      const long long row = __shfl_sync(0xffffffff, my_row, r);
      cp_async16(smem_u32(kb + sw(r, c)), KC + row * ROW + 16 * c);
      cp_async16(smem_u32(vb + sw(r, c)), VC + row * ROW + 16 * c);
    }
  };
  int row_next = 0;
  if (first < end) {
    stage_tile(0, row_of(first));
    if (first + step < end) row_next = row_of(first + step);
  }
  cp_async_commit();

  int it = 0;
  for (int base = first; base < end; base += step, ++it) {
    // Stage the next tile (its row was looked up one iteration ago), then look up the one after.
    if (base + step < end) stage_tile((it + 1) & 1, row_next);
    cp_async_commit();
    if (base + 2 * step < end) row_next = row_of(base + 2 * step);
    cp_async_wait<1>();  // this tile's group has landed
    __syncwarp();
    const uint8_t* kb = wbuf + (it & 1) * (2 * TILE * ROW);
    const uint8_t* vb = kb + TILE * ROW;
    const bool full = base + TILE <= end;

    // ---- Q.K^T: 4 n-tiles of 8 tokens; this thread's token in n-tile j is row 8j + g,
    // bytes 32t..32t+31 = chunks 2t and 2t+1.
    float c[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = 8 * j + g;
      const uint4 k0 = lds16(kb + sw(r, 2 * t)), k1 = lds16(kb + sw(r, 2 * t + 1));
      const uint32_t kw[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
      c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const uint32_t lo = qw[2 * s], hi = qw[2 * s + 1];
        mma_e4m3(c[j], kv0 ? lo : 0u, kv0 ? 0u : lo, kv0 ? hi : 0u, kv0 ? 0u : hi, kw[2 * s], kw[2 * s + 1]);
      }
    }
    // c[j][0..1]: row g, tokens base + 8j + 2t + {0, 1}; c[j][2..3]: row g + 8, same tokens.
    float mx_lo = MUSE_NEG_INF, mx_hi = MUSE_NEG_INF;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const bool ok = full || (base + 8 * j + 2 * t + e < end);
        const float a = ok ? c[j][e] * scale : MUSE_NEG_INF;
        const float bb = ok ? c[j][2 + e] * scale : MUSE_NEG_INF;
        c[j][e] = a;
        c[j][2 + e] = bb;
        mx_lo = fmaxf(mx_lo, a);
        mx_hi = fmaxf(mx_hi, bb);
      }
    }
    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, 1));
    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, 2));
    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, 1));
    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, 2));
    float nm_lo = fmaxf(m_lo, mx_lo), nm_hi = fmaxf(m_hi, mx_hi);
    nm_lo = nm_lo == MUSE_NEG_INF ? 0.f : nm_lo;
    nm_hi = nm_hi == MUSE_NEG_INF ? 0.f : nm_hi;
    const float al_lo = ex2(m_lo - nm_lo), al_hi = ex2(m_hi - nm_hi);
    m_lo = nm_lo;
    m_hi = nm_hi;
    z_lo *= al_lo;
    z_hi *= al_hi;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        c[j][e] = ex2(c[j][e] - nm_lo);
        c[j][2 + e] = ex2(c[j][2 + e] - nm_hi);
        z_lo += c[j][e];
        z_hi += c[j][2 + e];
      }
    }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      o[nt][0] *= al_lo; o[nt][1] *= al_lo;
      o[nt][2] *= al_hi; o[nt][3] *= al_hi;
    }
    // ---- P as the A operand (E4M3, x256). Logical k = t*4 + {0,1,2,3} <-> tokens 2t, 2t+1,
    // 8+2t, 9+2t of the tile; k = t*4 + 16 + {...} <-> 16+2t, 17+2t, 24+2t, 25+2t.
    const uint32_t pa0 = pack_e4m3(c[0][0] * 256.f, c[0][1] * 256.f, c[1][0] * 256.f, c[1][1] * 256.f);
    const uint32_t pa1 = pack_e4m3(c[0][2] * 256.f, c[0][3] * 256.f, c[1][2] * 256.f, c[1][3] * 256.f);
    const uint32_t pa2 = pack_e4m3(c[2][0] * 256.f, c[2][1] * 256.f, c[3][0] * 256.f, c[3][1] * 256.f);
    const uint32_t pa3 = pack_e4m3(c[2][2] * 256.f, c[2][3] * 256.f, c[3][2] * 256.f, c[3][3] * 256.f);
    // ---- V: this thread's 8 tokens (logical-k order), bytes 16g..16g+15 = chunk g.
    uint4 vr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) vr[i] = lds16(vb + sw(2 * t + (i & 1) + 8 * (i >> 1), g));
    // ---- P.V: n-tile nt, column g <-> row byte 16g + nt. B regs: bytes nt of tokens
    // vr[0..3] (b0..b3) and vr[4..7] (b4..b7): a 4x4 byte transpose per 32-bit word.
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t bt[2][4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t x0 = word(vr[4 * h + 0], w), x1 = word(vr[4 * h + 1], w);
        const uint32_t x2 = word(vr[4 * h + 2], w), x3 = word(vr[4 * h + 3], w);
        const uint32_t t01lo = prmt(x0, x1, 0x5140), t01hi = prmt(x0, x1, 0x7362);
        const uint32_t t23lo = prmt(x2, x3, 0x5140), t23hi = prmt(x2, x3, 0x7362);
        bt[h][0] = prmt(t01lo, t23lo, 0x5410);
        bt[h][1] = prmt(t01lo, t23lo, 0x7632);
        bt[h][2] = prmt(t01hi, t23hi, 0x5410);
        bt[h][3] = prmt(t01hi, t23hi, 0x7632);
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) mma_e4m3(o[4 * w + k], pa0, pa1, pa2, pa3, bt[0][k], bt[1][k]);
    }
    __syncwarp();  // this stage is restaged two iterations later only after every lane is done
  }
  cp_async_wait<0>();

  // ---- Merge the 4 warps (reusing the staging buffers). o[nt][e] (row g) / o[nt][2+e]
  // (row g+8) is row byte (2t+e)*16 + nt.
  z_lo += __shfl_xor_sync(0xffffffff, z_lo, 1);
  z_lo += __shfl_xor_sync(0xffffffff, z_lo, 2);
  z_hi += __shfl_xor_sync(0xffffffff, z_hi, 1);
  z_hi += __shfl_xor_sync(0xffffffff, z_hi, 2);
  __syncthreads();
  float (*s_o)[HQ][ROW] = reinterpret_cast<float (*)[HQ][ROW]>(smem);            // 32 KB
  float (*s_m)[HQ] = reinterpret_cast<float (*)[HQ]>(smem + WARPS * HQ * ROW * 4);
  float (*s_z)[HQ] = reinterpret_cast<float (*)[HQ]>(smem + WARPS * HQ * ROW * 4 + WARPS * HQ * 4);
  if (t == 0) {
    s_m[warp][g] = m_lo; s_m[warp][g + 8] = m_hi;
    s_z[warp][g] = z_lo; s_z[warp][g + 8] = z_hi;
  }
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      s_o[warp][g][(2 * t + e) * 16 + nt] = o[nt][e];
      s_o[warp][g + 8][(2 * t + e) * 16 + nt] = o[nt][2 + e];
    }
  }
  __syncthreads();
  const float vs = VS[0] * (1.f / 256.f);
  for (int idx = threadIdx.x; idx < HQ * D; idx += blockDim.x) {
    const int r = idx / D, d = idx % D;   // query row r: KV head r / 8, query head r % 8
    const int kh = r >> 3;
    float mm = s_m[0][r];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) mm = fmaxf(mm, s_m[w][r]);
    float zz = 0.f, oo = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      // A split with no keys leaves every warp at -inf: weight 0, LSE -inf.
      const float f = mm == MUSE_NEG_INF ? 0.f : ex2(s_m[w][r] - mm);
      zz += s_z[w][r] * f;
      oo += s_o[w][r][kh * D + d] * f;
    }
    const long long base_row = ((long long)(b * 2 + kh) * splits + split) * 8 + (r & 7);
    PART[base_row * D + d] = oo / fmaxf(zz, 1e-20f) * vs;
    if (d == 0) LSE[base_row] = zz > 0.f ? mm + log2f(zz) : MUSE_NEG_INF;
  }
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void attn_full_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                      torch::Tensor lens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part,
                      torch::Tensor lse, int64_t page, int64_t pt_stride, int64_t splits,
                      int64_t chunk, double sm_scale) {
  constexpr int kSmem = 4 * 2 * 2 * 32 * 128;  // 64 KB: per-warp double-buffered K/V tiles
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(muse_attn_full_decode, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    configured = true;
  }
  dim3 grid((unsigned)splits, (unsigned)q.size(0));
  muse_attn_full_decode<<<grid, 128, kSmem, at::cuda::getCurrentCUDAStream()>>>(
      (const uint8_t*)q.data_ptr(), (const uint8_t*)kc.data_ptr(), (const uint8_t*)vc.data_ptr(),
      pt.data_ptr<int>(), lens.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(),
      part.data_ptr<float>(), lse.data_ptr<float>(), (int)page, (int)pt_stride, (int)splits,
      (int)chunk, (float)sm_scale);
}
"""
_FULL_DECODE = load_inline(
    name="muse_attn_full_decode_v3",
    cpp_sources="void attn_full_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, "
                "torch::Tensor lens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, "
                "int64_t page, int64_t pt_stride, int64_t splits, int64_t chunk, double sm_scale);",
    cuda_sources=_FULL_DECODE_CU, functions=["attn_full_decode"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)


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
    if not window and length == 1 and hq == 16 and hk == 2 and d == 64 and b >= 8:
        _FULL_DECODE.attn_full_decode(
            q, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"],
            data["k_scale"], data["v_scale"], partial, lse, cfg["page_size"],
            data["page_table"].stride(0), splits, chunk, cfg["sm_scale"] * cfg["qk_scale_factor"])
    else:
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
