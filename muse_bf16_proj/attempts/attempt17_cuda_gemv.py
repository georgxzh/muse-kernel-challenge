# okc: problem=muse_bf16_proj
# Attempt 17: attempt 16 + a CUDA tensor-core GEMV for the BF16 layers at m <= 16.
"""Split-K tensor-core projections, with a fused hi/lo router; PDL into the router's second pass;
BF16 split-K partials for the BF16 layers (summed in FP32), FP32 partials for the router."""
from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from torch.utils.cpp_extension import load_inline


_TILE = 64  # weight tile; every tiled layer's N and K are multiples
SERIAL_ROUTER = True   # router m<=16: one kernel, in-kernel split-K reduction
SERIAL_TILED = True    # tiled qkv_gate/pooler/unpooler: one kernel as well


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


_COUNTERS = 4096  # per-tile arrival counters; every launch uses fewer tiles


@triton.jit
def _serial_reduce(P, Y, CNT, tile, offs, mask, acc,
                   MN: tl.constexpr, SPLIT: tl.constexpr):
    """Split-K epilogue without a second kernel.

    Each split CTA stores its FP32 partial, then counts in on its output tile.
    The last CTA to arrive sums all SPLIT partials in fixed order (s = 0..SPLIT-1),
    writes the output, and resets the counter to zero for the next call: the
    self-resetting lock pattern SGLang's Marlin workspace uses. Counters hold
    no data between calls.
    """
    tl.store(P + tl.program_id(2) * MN + offs, acc.to(P.dtype.element_ty), mask)
    tl.debug_barrier()          # every thread's partial is issued before the release
    ticket = tl.atomic_add(CNT + tile, 1, sem="acq_rel", scope="gpu")
    if ticket == SPLIT - 1:
        # Partials may be BF16 (rules 2026-11-03); the sum is always FP32.
        total = tl.load(P + offs, mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        for s in tl.static_range(1, SPLIT):
            total += tl.load(P + s * MN + offs, mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        tl.store(Y + offs, total.to(Y.dtype.element_ty), mask)
        tl.atomic_xchg(CNT + tile, 0, sem="relaxed", scope="gpu")


@triton.jit
def _split_mm(X, W, L, P, Y, CNT, M: tl.constexpr, N: tl.constexpr,
              K: tl.constexpr, SPLIT: tl.constexpr, ROUTER: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              SERIAL: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    si = tl.program_id(2)
    ki = si * (K // SPLIT) + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    if ROUTER:
        low = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K // SPLIT, BK)):
        kk = ki + block * BK
        x = tl.load(X + mi[:, None] * K + kk[None, :],
                    (mi[:, None] < M) & (kk[None, :] < (si + 1) * (K // SPLIT)), 0)
        w = tl.load(W + ni[None, :] * K + kk[:, None],
                    (ni[None, :] < N) & (kk[:, None] < (si + 1) * (K // SPLIT)), 0)
        acc = tl.dot(x, w, acc)
        if ROUTER:
            lo = tl.load(L + ni[None, :] * K + kk[:, None],
                         (ni[None, :] < N) & (kk[:, None] < (si + 1) * (K // SPLIT)), 0)
            low = tl.dot(x, lo, low)
    if ROUTER:
        acc += low
    offs = mi[:, None] * N + ni[None, :]
    mask = (mi[:, None] < M) & (ni[None, :] < N)
    if SERIAL:
        _serial_reduce(P, Y, CNT, tl.program_id(0) * tl.num_programs(1) + tl.program_id(1),
                       offs, mask, acc, M * N, SPLIT)
    else:
        tl.store(P + si * M * N + offs, acc, mask)
        # PDL: let _finish launch once every CTA has stored its partial.
        gdc_launch_dependents()


@triton.jit
def _tiled_mm(X, T, P, Y, CNT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              SPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, SERIAL: tl.constexpr):
    # T holds the same BF16 weights as contiguous BK x BN tiles, ordered by
    # (output tile, K tile), so one CTA reads its weights sequentially.
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    tn = tl.program_id(1)
    si = tl.program_id(2)
    kj = tl.arange(0, BK)
    nj = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(K // SPLIT // BK):
        tk = si * (K // SPLIT // BK) + block
        x = tl.load(X + mi[:, None] * K + tk * BK + kj[None, :],
                    mi[:, None] < M, 0)
        w = tl.load(T + (tn * (K // BK) + tk) * BK * BN + kj[:, None] * BN + nj[None, :])
        acc = tl.dot(x, w, acc)
    offs = mi[:, None] * N + tn * BN + nj[None, :]
    mask = (mi[:, None] < M) & (nj[None, :] < BN)
    if SERIAL:
        _serial_reduce(P, Y, CNT, tl.program_id(0) * tl.num_programs(1) + tn,
                       offs, mask, acc, M * N, SPLIT)
    else:
        tl.store(P + si * M * N + offs, acc, mask)


@triton.jit
def _finish(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr,
            BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLIT)
    # PDL-launched: block until _split_mm has completed and its partials are visible.
    gdc_wait()
    v = tl.load(P + s[:, None] * SIZE + i[None, :], i[None, :] < SIZE, 0)
    tl.store(Y + i, tl.sum(v, axis=0), i < SIZE)


@triton.jit
def _router_small(X, W, P, Y, CNT, M: tl.constexpr, N: tl.constexpr,
                  K: tl.constexpr, SPLIT: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr, SERIAL: tl.constexpr):
    row = tl.program_id(0)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    si = tl.program_id(2)
    ki = si * BK + tl.arange(0, BK)
    x = tl.load(X + row * K + ki).to(tl.float32)
    w = tl.load(W + ni[:, None] * K + ki[None, :], ni[:, None] < N, 0)
    v = tl.sum(w * x[None, :], axis=1)
    if SERIAL:
        _serial_reduce(P, Y, CNT, row * tl.num_programs(1) + tl.program_id(1),
                       row * N + ni, ni < N, v, M * N, SPLIT)
    else:
        tl.store(P + si * M * N + row * N + ni, v, ni < N)


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


def prepare(weights: Weights):
    state = dict(weights)
    hi = weights["router"].to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (weights["router"] - hi.float()).to(torch.bfloat16)
    for name in ("pooler", "unpooler"):
        state[name + "_t"] = weights[name].T.contiguous()
    for name in ("qkv_gate", "pooler", "unpooler"):
        w = weights[name]
        tiles = w.reshape(w.shape[0] // _TILE, _TILE, w.shape[1] // _TILE, _TILE)
        state[name + "_tiled"] = tiles.permute(0, 2, 3, 1).contiguous()
    # Concatenate output columns so cuBLAS computes the two allowed BF16
    # router products in one GEMM, without duplicating activations.
    state["router_pair"] = torch.cat((hi.T, state["router_lo"].T), dim=1).contiguous()
    # Split-K arrival counters: zero at rest, reset by each tile's last CTA.
    state["counters"] = torch.zeros((_COUNTERS,), dtype=torch.int32, device=hi.device)
    # CUDA GEMV arrival counters (one per 32-column block): zero at rest.
    state["gemv_counters"] = torch.zeros((8192 // 32,), dtype=torch.int32, device=hi.device)
    return state


@triton.jit
def _join_router(P, Y, SIZE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = (i // N) * (2 * N) + i % N
    # PDL-launched after cuBLAS: block until the GEMM has completed.
    gdc_wait()
    hi = tl.load(P + pos, i < SIZE, 0)
    lo = tl.load(P + pos + N, i < SIZE, 0)
    tl.store(Y + i, hi + lo, i < SIZE)

def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    m, n, k, layer = cfg["m"], cfg["n"], cfg["k"], cfg["layer"]
    router = layer == "router"
    if router and m > 32:
        products = torch.mm(x, state["router_pair"], out_dtype=torch.float32)
        out = torch.empty((m, n), device=x.device, dtype=torch.float32)
        _join_router[(triton.cdiv(m * n, 256),)](products, out, m * n, n, 256, launch_pdl=True)
        return out
    if not router and m <= 16 and n % 32 == 0 and k % 256 == 0:
        return _cuda_gemv(x, state[layer][:n], state)
    tiled = (layer in ("qkv_gate", "pooler") and m <= 16) or (layer == "unpooler" and m <= 8)
    if layer in ("pooler", "unpooler") and m > 32:
        return torch.mm(x, state[layer + "_t"][:, :n])
    if not router and not tiled:
        return torch.mm(x, state[layer][:n].T)
    split = (16 if m <= 32 else 8) if router else 8
    bm = (16 if m <= 16 else 32 if m <= 32 else 64 if m <= 128 else 128) if router else 16
    bn = 32 if router else 64
    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)
    # BF16 partials for the BF16 layers; the router (0.01% tolerance) keeps FP32.
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    # Single-kernel split-K (in-kernel last-CTA reduction) for the router;
    # the other layers keep the separate _finish pass.
    # Router m=32 lost 1.3% with the in-kernel reduction (attempt 11): keep _finish there.
    serial = (SERIAL_ROUTER and m <= 16) if router else SERIAL_TILED
    cnt = state["counters"]
    if router and m == 1:
        grid = (m, triton.cdiv(n, 4), split)
        _router_small[grid](
            x, state["router"], partial, out, cnt, m, n, k, split, 4, k // split, serial,
            num_warps=4)
    elif tiled:
        grid = (triton.cdiv(m, bm), n // _TILE, split)
        _tiled_mm[grid](
            x, state[layer + "_tiled"], partial, out, cnt, m, n, k, split, bm, _TILE, _TILE, serial,
            num_warps=4, num_stages=3)
    else:
        w = state["router_hi"] if router else state[layer]
        lo = state["router_lo"] if router else w
        grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), split)
        _split_mm[grid](
            x, w, lo, partial, out, cnt, m, n, k, split, router, bm, bn, 64, serial,
            num_warps=8 if bm == 128 else 4, num_stages=3)
    assert grid[0] * grid[1] <= _COUNTERS
    if not serial:
        _finish[(triton.cdiv(m * n, 256),)](partial, out, m * n, split, 256, launch_pdl=True)
    return out
