# okc: problem=muse_attention_decode
# Attempt 42: attempt 39 + one fused CUDA kernel for sliding-window decode (prepare + attention + combine).
"""Hybrid FP8/BF16 PV attention with FA3 for batched full verify; PDL between the Triton kernels;
full-layer decode from 8 sequences runs a CUDA kernel (load_inline) staging whole 128-byte K/V rows."""

import torch
import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch.utils.cpp_extension import load_inline

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

// ---- fused sliding-window decode kernel ----

namespace muse_fs {

constexpr int HQ = 16;      // query heads
constexpr int HK = 2;       // KV heads
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

__device__ __forceinline__ uint16_t f2bf(float f) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(r) : "f"(f));
  return r;
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ int cache_row(const int* __restrict__ pt_b, int ts, int page) {
  return pt_b[ts / page] * page + ts % page;
}

}  // namespace muse_fs


namespace muse_fs {

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

}  // namespace muse_fs

// Fused sliding-window decode (L = 1): _prepare + _attention_parts + _combine_gate in one
// launch. Each CTA (sequence b, split) computes its own query heads from qkv exactly as
// _prepare does (FP32 RMSNorm, interleaved RoPE with libdevice powf/cosf/sinf, rounded to
// BF16 then E4M3); the CTA whose key range holds the new token writes that token's K and V
// cache rows (K: normalized + rotated / k_scale, V: raw / v_scale, E4M3) and fences them
// before staging them; the attention loop is the full-decode v3 kernel's (whole 128-byte
// rows, both KV heads, FP8 mma.sync, per-warp online softmax). Each CTA writes FP32 split
// partials + LSE; the last CTA of the sequence to arrive (per-sequence counter, reset by it)
// combines them as _combine_gate does: LSE-weighted average, BF16 round, RMSNorm (eps 1e-5)
// and x sigmoid(gate), BF16 output.
extern "C" __global__ void __launch_bounds__(128, 3)
muse_attn_sliding_decode(const uint16_t* __restrict__ QKV,   // [B, STRIDE] BF16: q | k | v | gate
                         const long long* __restrict__ POS,  // [B] new token position
                         const long long* __restrict__ LOC,  // [B] new token flat cache row
                         uint8_t* __restrict__ KC,           // [rows, 128] E4M3 (new rows written)
                         uint8_t* __restrict__ VC,
                         const int* __restrict__ PT,         // [B, pt_stride]
                         const int* __restrict__ LENS,       // [B], includes the new token
                         const float* __restrict__ KS,
                         const float* __restrict__ VS,
                         float* __restrict__ PART,           // [B, 2, splits, 8, 64]
                         float* __restrict__ LSE,            // [B, 2, splits, 8]
                         uint16_t* __restrict__ OUT,         // [B, 16 * 64] BF16
                         int* __restrict__ CNT,              // [>= B], zero at rest
                         int stride, int page, int pt_stride, int splits, int chunk, int window,
                         float sm_scale, float theta) {
  using namespace muse_fs;
  // [warp][stage][K, V][TILE][128] = 64 KB staging (reused for the merge), then q (1 KB).
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* s_q = smem + WARPS * 2 * 2 * TILE * ROW;
  __shared__ int s_last;

  const int split = blockIdx.x;
  const int b = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int seqlen = LENS[b];
  const int first_key = max(0, seqlen - window);   // the window: the token and the window-1 before it
  const int begin = first_key + split * chunk;
  const int end = min(begin + chunk, seqlen);
  const float scale = KS[0] * sm_scale * 1.4426950408889634f;
  const int* pt_b = PT + (long long)b * pt_stride;
  uint8_t* wbuf = smem + warp * (2 * 2 * TILE * ROW);
  const uint16_t* qkv_b = QKV + (long long)b * stride;

  // ---- Query heads (and, in the CTA holding the new token, its K/V rows), as _prepare.
  // Warp w handles heads 4w .. 4w+3; lane l the RoPE pair (2l, 2l+1).
  {
    const float posf = (float)POS[b];
    float c = 1.f, s = 0.f;
    if (theta != 0.f) {
      const float freq = __fdiv_rn(1.f, powf(theta, __fdiv_rn((float)(2 * lane), (float)D)));
      const float ang = __fmul_rn(posf, freq);
      c = cosf(ang);
      s = sinf(ang);
    }
    auto norm_rope = [&](const uint16_t* src, float& y0, float& y1) {
      const uint32_t raw = *reinterpret_cast<const uint32_t*>(src + 2 * lane);
      const float x0 = __uint_as_float(raw << 16), x1 = __uint_as_float(raw & 0xffff0000u);
      float ss = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffff, ss, o));
      const float r = rsqrtf(__fadd_rn(__fdiv_rn(ss, (float)D), 1e-5f));
      const float a0 = __fmul_rn(x0, r), a1 = __fmul_rn(x1, r);
      if (theta != 0.f) {
        y0 = __fsub_rn(__fmul_rn(a0, c), __fmul_rn(a1, s));
        y1 = __fadd_rn(__fmul_rn(a1, c), __fmul_rn(a0, s));
      } else {
        y0 = a0;
        y1 = a1;
      }
    };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = 4 * warp + i;
      float y0, y1;
      norm_rope(qkv_b + h * D, y0, y1);
      // BF16 rounding, then E4M3 (as _prepare stores q).
      const float q0 = __uint_as_float(((uint32_t)f2bf(y0)) << 16), q1 = __uint_as_float(((uint32_t)f2bf(y1)) << 16);
      uint16_t pk;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pk) : "f"(q1), "f"(q0));
      *reinterpret_cast<uint16_t*>(s_q + h * D + 2 * lane) = pk;
    }
    const int pos = seqlen - 1;
    if (pos >= begin && pos < end) {
      const long long loc = LOC[b];
      const float ks = KS[0], vs = VS[0];
      const int kh = warp & 1;
      if (warp < 2) {
        float y0, y1;
        norm_rope(qkv_b + (HQ + kh) * D, y0, y1);
        uint16_t pk;
        const float k0 = __fdiv_rn(y0, ks), k1 = __fdiv_rn(y1, ks);
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pk) : "f"(k1), "f"(k0));
        *reinterpret_cast<uint16_t*>(KC + loc * ROW + kh * D + 2 * lane) = pk;
      } else {
        const uint32_t raw = *reinterpret_cast<const uint32_t*>(qkv_b + (HQ + 2 + kh) * D + 2 * lane);
        const float v0 = __fdiv_rn(__uint_as_float(raw << 16), vs), v1 = __fdiv_rn(__uint_as_float(raw & 0xffff0000u), vs);
        uint16_t pk;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pk) : "f"(v1), "f"(v0));
        *reinterpret_cast<uint16_t*>(VC + loc * ROW + kh * D + 2 * lane) = pk;
      }
      __threadfence();  // the new rows reach L2 before any cp.async (.cg) of this CTA reads them
    }
    __syncthreads();
  }
  uint32_t qw[8];
  {
    const int head = (t < 2) ? g : 8 + g;
    const uint8_t* src = s_q + head * D + (t & 1) * 32;
    const uint4 w0 = *reinterpret_cast<const uint4*>(src);
    const uint4 w1 = *reinterpret_cast<const uint4*>(src + 16);
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

  // ---- Last CTA of this sequence: combine the splits (as _combine_gate).
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomicAdd(CNT + b, 1) == splits - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  // Warp w: heads 4w .. 4w+3; lane: dims 2l, 2l+1.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int h = 4 * warp + i, kh = h >> 3, qh = h & 7;
    const long long row0 = (long long)(b * 2 + kh) * splits * 8 + qh;   // + split * 8
    float mx = MUSE_NEG_INF;
    for (int sp = 0; sp < splits; ++sp) mx = fmaxf(mx, __ldcg(LSE + row0 + sp * 8));
    float wsum = 0.f, a0 = 0.f, a1 = 0.f;
    for (int sp = 0; sp < splits; ++sp) {
      const float w = ex2(__ldcg(LSE + row0 + sp * 8) - mx);
      const float2 pv = __ldcg(reinterpret_cast<const float2*>(PART + (row0 + sp * 8) * D + 2 * lane));
      wsum += w;
      a0 += pv.x * w;
      a1 += pv.y * w;
    }
    a0 = __uint_as_float(((uint32_t)f2bf(a0 / wsum)) << 16);
    a1 = __uint_as_float(((uint32_t)f2bf(a1 / wsum)) << 16);
    float ss = a0 * a0 + a1 * a1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    const float r = rsqrtf(ss / (float)D + 1e-5f);
    const uint32_t graw = *reinterpret_cast<const uint32_t*>(qkv_b + (HQ + 2 * HK + h) * D + 2 * lane);
    const float g0 = __uint_as_float(graw << 16), g1 = __uint_as_float(graw & 0xffff0000u);
    const float o0 = a0 * r / (1.f + __expf(-g0)), o1 = a1 * r / (1.f + __expf(-g1));
    const uint32_t packed = (uint32_t)f2bf(o0) | ((uint32_t)f2bf(o1) << 16);
    *reinterpret_cast<uint32_t*>(OUT + (long long)b * HQ * D + h * D + 2 * lane) = packed;
  }
  if (threadIdx.x == 0) CNT[b] = 0;
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

void attn_sliding_decode(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc,
                         torch::Tensor vc, torch::Tensor pt, torch::Tensor lens, torch::Tensor ks,
                         torch::Tensor vs, torch::Tensor part, torch::Tensor lse, torch::Tensor out,
                         torch::Tensor cnt, int64_t stride, int64_t page, int64_t pt_stride,
                         int64_t splits, int64_t chunk, int64_t window, double sm_scale, double theta) {
  constexpr int kSmem = 4 * 2 * 2 * 32 * 128 + 16 * 64;  // staging (64 KB) + queries (1 KB)
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(muse_attn_sliding_decode, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    configured = true;
  }
  dim3 grid((unsigned)splits, (unsigned)lens.size(0));
  muse_attn_sliding_decode<<<grid, 128, kSmem, at::cuda::getCurrentCUDAStream()>>>(
      (const uint16_t*)qkv.data_ptr(), pos.data_ptr<int64_t>(), loc.data_ptr<int64_t>(),
      (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), pt.data_ptr<int>(), lens.data_ptr<int>(),
      ks.data_ptr<float>(), vs.data_ptr<float>(), part.data_ptr<float>(), lse.data_ptr<float>(),
      (uint16_t*)out.data_ptr(), cnt.data_ptr<int>(), (int)stride, (int)page, (int)pt_stride,
      (int)splits, (int)chunk, (int)window, (float)sm_scale, (float)theta);
}
"""
_FULL_DECODE = load_inline(
    name="muse_attn_cuda_v6",
    cpp_sources="void attn_full_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, "
                "torch::Tensor lens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, "
                "int64_t page, int64_t pt_stride, int64_t splits, int64_t chunk, double sm_scale);\n"
                "void attn_sliding_decode(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, "
                "torch::Tensor vc, torch::Tensor pt, torch::Tensor lens, torch::Tensor ks, torch::Tensor vs, "
                "torch::Tensor part, torch::Tensor lse, torch::Tensor out, torch::Tensor cnt, int64_t stride, "
                "int64_t page, int64_t pt_stride, int64_t splits, int64_t chunk, int64_t window, double sm_scale, "
                "double theta);",
    cuda_sources=_FULL_DECODE_CU, functions=["attn_full_decode", "attn_sliding_decode"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)

# Per-sequence arrival counters for the fused sliding kernel's final combine (zero at rest;
# the last CTA of each sequence resets its counter).
_SLIDING_CNT = torch.zeros((1 << 16,), dtype=torch.int32, device="cuda")


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



def kernel_fn(data):
    cfg = data["config"]
    b, length = cfg["batch"], cfg["q_len"]
    hq, hk, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"]
    tokens = b * length
    stride = qkv.stride(-2)
    window = cfg["window"]
    if window and length == 1 and hq == 16 and hk == 2 and d == 64:
        # One fused CUDA kernel: q/k norm + RoPE + cache store, whole-row attention, combine + gate.
        chunk = 128 if b < 8 else 256
        capacity = data["page_table"].shape[1] * cfg["page_size"]
        splits = tr.cdiv(min(capacity, window), chunk)
        partial = torch.empty((b, hk, splits, hq // hk, d), device=qkv.device, dtype=torch.float32)
        lse = torch.empty((b, hk, splits, hq // hk), device=qkv.device, dtype=torch.float32)
        out = torch.empty((b, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
        _FULL_DECODE.attn_sliding_decode(
            qkv, data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"],
            data["page_table"], data["cache_seqlens"], data["k_scale"], data["v_scale"],
            partial, lse, out, _SLIDING_CNT, stride, cfg["page_size"], data["page_table"].stride(0),
            splits, chunk, window, cfg["sm_scale"] * cfg["qk_scale_factor"], float(cfg["rope_theta"]))
        return out
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
        return out
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
    # CUDA kernel from 8 sequences up (2048-token splits); one sequence keeps the Triton
    # kernel, whose 256-token splits suit it better (attempt 36: 20.1 vs 18.8 us).
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
    return out
