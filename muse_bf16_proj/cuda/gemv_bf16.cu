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
