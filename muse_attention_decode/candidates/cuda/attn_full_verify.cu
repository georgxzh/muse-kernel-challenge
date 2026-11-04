// Full-layer verify attention (an 8-token causal block per sequence, 16 query heads,
// 2 KV heads, head dim 64, FP8 E4M3 query/cache), split-KV partials in the layout
// _combine_gate reads: PART [B, 2, splits, 64 rows, 64], LSE [B, 2, splits, 64], where a
// KV head's row is (block token j) * 8 + (query head within the group).
//
// One CTA covers one (sequence, split) for both KV heads; its 128 threads stage each
// 32-token tile's whole K and V rows (128 contiguous bytes each) into shared memory with
// coalesced cp.async (2 stages; 8 threads per row), so the cache is gathered in whole rows.
// Warp w owns 32 query rows: KV head w / 2, block tokens 4 (w % 2) .. 4 (w % 2) + 3, all 8
// query heads; it never needs another warp's rows, so there is no cross-warp merge.
//
// Tensor cores: mma.sync m16n8k32 E4M3 x E4M3 -> FP32.
//  * Q.K^T per m-tile (16 rows), K dimension = the KV head's 64 row bytes in natural order
//    (2 k-steps); B fragments are 4-byte shared-memory reads.
//  * P.V: A = P (FP32 probabilities x 256 rounded to E4M3, as FA3's FP8 path and our Triton
//    kernel), K = 32 tokens in the permuted order matching the Q.K^T accumulator layout,
//    N = the head's 64 bytes in the order (n-tile nt, column g) -> byte 8g + nt, so each
//    thread reads 8 contiguous V bytes per token and transposes them in registers (PRMT).
//  * Softmax in FP32, base 2. Causal: key position ts is visible to block token j iff
//    ts <= seqlen - 8 + j. Every key in the split is read and weighted.

#if defined(__CUDACC_RTC__)
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
#else
#include <stdint.h>
#endif

#define MUSE_NEG_INF (__int_as_float(0xff800000))

namespace muse_fv {

constexpr int HQ = 16;
constexpr int D = 64;
constexpr int ROW = 128;
constexpr int L = 8;        // block tokens
constexpr int TILE = 32;    // tokens per tile
constexpr int THREADS = 128;

__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t pack_e4m3(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(x3), "f"(x2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const uint8_t* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// Byte offset of byte `x` of tile row r (16-byte chunks XOR-swizzled by row).
__device__ __forceinline__ int sw(int r, int x) { return r * ROW + ((((x >> 4) ^ (r & 7)) << 4) | (x & 15)); }

__device__ __forceinline__ int cache_row(const int* __restrict__ pt_b, int ts, int page) {
  return pt_b[ts / page] * page + ts % page;
}

}  // namespace muse_fv

extern "C" __global__ void __launch_bounds__(128, 3)
muse_attn_full_verify(const uint8_t* __restrict__ Q,     // [B * 8, 16, 64] E4M3
                      const uint8_t* __restrict__ KC,    // [rows, 128] E4M3
                      const uint8_t* __restrict__ VC,    // [rows, 128] E4M3
                      const int* __restrict__ PT,        // [B, pt_stride]
                      const int* __restrict__ LENS,      // [B], includes the 8 new tokens
                      const float* __restrict__ KS,
                      const float* __restrict__ VS,
                      float* __restrict__ PART,          // [B, 2, splits, 64, 64]
                      float* __restrict__ LSE,           // [B, 2, splits, 64]
                      int page, int pt_stride, int splits, int chunk, float sm_scale) {
  using namespace muse_fv;
  __shared__ __align__(128) uint8_t smem[2][2][TILE * ROW];  // [stage][K, V] = 16 KB

  const int split = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int kh = warp >> 1;             // this warp's KV head
  const int j0 = (warp & 1) * 4;        // its block tokens j0 .. j0 + 3
  const int seqlen = LENS[b];
  const int begin = split * chunk;
  const int end = min(begin + chunk, seqlen);
  const float scale = KS[0] * sm_scale * 1.4426950408889634f;
  const int* pt_b = PT + (long long)b * pt_stride;

  // m-tile mt holds rows rr = j0*8 + mt*16 + {g, g+8} of this KV head: block token
  // j = rr / 8 = j0 + 2 mt + (0 or 1), query head kh*8 + g. Row lo: token j0 + 2mt, row hi:
  // token j0 + 2mt + 1 (both query head kh*8 + g).
  // Q A-fragments, natural byte order: k-step s covers head bytes 32s .. 32s + 31.
  uint32_t qa[2][2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int j = j0 + 2 * mt + h;
      const uint8_t* qrow = Q + (((long long)b * L + j) * HQ + kh * 8 + g) * D;
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        qa[mt][s][h] = *reinterpret_cast<const uint32_t*>(qrow + 32 * s + 4 * t);           // a0..a3 / a4..a7
        qa[mt][s][2 + h] = *reinterpret_cast<const uint32_t*>(qrow + 32 * s + 16 + 4 * t);  // a8..a11 / a12..a15
      }
    }
  }
  int pos[2][2];  // last visible key position for (mt, lo/hi)
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    pos[mt][0] = seqlen - L + j0 + 2 * mt;
    pos[mt][1] = seqlen - L + j0 + 2 * mt + 1;
  }

  float o[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) o[mt][nt][0] = o[mt][nt][1] = o[mt][nt][2] = o[mt][nt][3] = 0.f;
  float m[2][2], z[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) { m[mt][0] = m[mt][1] = MUSE_NEG_INF; z[mt][0] = z[mt][1] = 0.f; }

  // Staging: pass p (0, 1) of a tile: thread copies chunk tid % 8 of tile row p * 16 + tid / 8,
  // K and V. Its row lookups are for those two tokens; keys past `end` use `begin`'s row.
  const int c8 = tid & 7;
  auto rows_of = [&](int base, int& r0, int& r1) {
    int ts0 = base + (tid >> 3), ts1 = base + 16 + (tid >> 3);
    ts0 = ts0 < end ? ts0 : begin;
    ts1 = ts1 < end ? ts1 : begin;
    r0 = cache_row(pt_b, ts0, page);
    r1 = cache_row(pt_b, ts1, page);
  };
  auto stage_tile = [&](int st, int r0, int r1) {
    const int rr0 = tid >> 3, rr1 = 16 + (tid >> 3);
    cp_async16(smem_u32(&smem[st][0][sw(rr0, 16 * c8)]), KC + (long long)r0 * ROW + 16 * c8);
    cp_async16(smem_u32(&smem[st][1][sw(rr0, 16 * c8)]), VC + (long long)r0 * ROW + 16 * c8);
    cp_async16(smem_u32(&smem[st][0][sw(rr1, 16 * c8)]), KC + (long long)r1 * ROW + 16 * c8);
    cp_async16(smem_u32(&smem[st][1][sw(rr1, 16 * c8)]), VC + (long long)r1 * ROW + 16 * c8);
  };
  int n0 = 0, n1 = 0;
  if (begin < end) {
    int r0, r1;
    rows_of(begin, r0, r1);
    stage_tile(0, r0, r1);
    if (begin + TILE < end) rows_of(begin + TILE, n0, n1);
  }
  cp_async_commit();

  int it = 0;
  for (int base = begin; base < end; base += TILE, ++it) {
    if (base + TILE < end) stage_tile((it + 1) & 1, n0, n1);
    cp_async_commit();
    if (base + 2 * TILE < end) rows_of(base + 2 * TILE, n0, n1);
    cp_async_wait<1>();
    __syncthreads();  // tile `it` visible to every warp
    const uint8_t* kb = smem[it & 1][0];
    const uint8_t* vb = smem[it & 1][1];

    // ---- Q.K^T: 2 m-tiles x 4 n-tiles (8 tokens each); thread's token in n-tile nj is
    // tile row 8 nj + g; B bytes kh*64 + 32 s + 4 t .. (b0..b3) and + 16 (b4..b7).
    float c[2][4][4];
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int r = 8 * nj + g;
      uint32_t kw[2][2];
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        kw[s][0] = *reinterpret_cast<const uint32_t*>(kb + sw(r, kh * 64 + 32 * s + 4 * t));
        kw[s][1] = *reinterpret_cast<const uint32_t*>(kb + sw(r, kh * 64 + 32 * s + 16 + 4 * t));
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        c[mt][nj][0] = c[mt][nj][1] = c[mt][nj][2] = c[mt][nj][3] = 0.f;
#pragma unroll
        for (int s = 0; s < 2; ++s)
          mma_e4m3(c[mt][nj], qa[mt][s][0], qa[mt][s][1], qa[mt][s][2], qa[mt][s][3], kw[s][0], kw[s][1]);
      }
    }
    // c[mt][nj][e] (e < 2): row lo of m-tile mt, token base + 8 nj + 2t + e; [2 + e]: row hi.
    const bool full = base + TILE <= end && base + TILE - 1 <= seqlen - L;  // no mask needed
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float mx[2] = {MUSE_NEG_INF, MUSE_NEG_INF};
#pragma unroll
      for (int nj = 0; nj < 4; ++nj) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int ts = base + 8 * nj + 2 * t + e;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const bool ok = full || (ts < end && ts <= pos[mt][h]);
            const float v = ok ? c[mt][nj][2 * h + e] * scale : MUSE_NEG_INF;
            c[mt][nj][2 * h + e] = v;
            mx[h] = fmaxf(mx[h], v);
          }
        }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffff, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffff, mx[h], 2));
        float nm = fmaxf(m[mt][h], mx[h]);
        nm = nm == MUSE_NEG_INF ? 0.f : nm;
        const float al = ex2(m[mt][h] - nm);
        m[mt][h] = nm;
        z[mt][h] *= al;
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float p = ex2(c[mt][nj][2 * h + e] - nm);
            c[mt][nj][2 * h + e] = p;
            z[mt][h] += p;
          }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) { o[mt][nt][2 * h] *= al; o[mt][nt][2 * h + 1] *= al; }
      }
    }
    // ---- V: this thread's 8 tokens in logical-k order (2t, 2t+1, 8+2t, 9+2t, 16+2t, ...),
    // head bytes 8g .. 8g + 7 (row bytes kh*64 + 8g ..).
    uint32_t vw[8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = 2 * t + (i & 1) + 8 * (i >> 1);
      const uint2 v2 = *reinterpret_cast<const uint2*>(vb + sw(r, kh * 64 + 8 * g));
      vw[i][0] = v2.x;
      vw[i][1] = v2.y;
    }
    // B regs for n-tile nt (column g <-> head byte 8g + nt): byte nt of tokens vw[0..3]
    // (b0..b3) and vw[4..7] (b4..b7).
    uint32_t bt[8][2];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t x0 = vw[4 * h + 0][w], x1 = vw[4 * h + 1][w], x2 = vw[4 * h + 2][w], x3 = vw[4 * h + 3][w];
        const uint32_t t01lo = prmt(x0, x1, 0x5140), t01hi = prmt(x0, x1, 0x7362);
        const uint32_t t23lo = prmt(x2, x3, 0x5140), t23hi = prmt(x2, x3, 0x7362);
        bt[4 * w + 0][h] = prmt(t01lo, t23lo, 0x5410);
        bt[4 * w + 1][h] = prmt(t01lo, t23lo, 0x7632);
        bt[4 * w + 2][h] = prmt(t01hi, t23hi, 0x5410);
        bt[4 * w + 3][h] = prmt(t01hi, t23hi, 0x7632);
      }
    }
    // ---- P.V per m-tile. P logical k = t*4 + {0..3} <-> tokens 2t, 2t+1, 8+2t, 9+2t;
    // k = t*4 + 16 + {...} <-> 16+2t, 17+2t, 24+2t, 25+2t.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const float (*cc)[4] = c[mt];
      const uint32_t pa0 = pack_e4m3(cc[0][0] * 256.f, cc[0][1] * 256.f, cc[1][0] * 256.f, cc[1][1] * 256.f);
      const uint32_t pa1 = pack_e4m3(cc[0][2] * 256.f, cc[0][3] * 256.f, cc[1][2] * 256.f, cc[1][3] * 256.f);
      const uint32_t pa2 = pack_e4m3(cc[2][0] * 256.f, cc[2][1] * 256.f, cc[3][0] * 256.f, cc[3][1] * 256.f);
      const uint32_t pa3 = pack_e4m3(cc[2][2] * 256.f, cc[2][3] * 256.f, cc[3][2] * 256.f, cc[3][3] * 256.f);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) mma_e4m3(o[mt][nt], pa0, pa1, pa2, pa3, bt[nt][0], bt[nt][1]);
    }
    __syncthreads();  // every warp is done with this stage before it is restaged
  }
  cp_async_wait<0>();

  // ---- Write this warp's rows. o[mt][nt][e] / [2 + e]: row lo / hi, head byte 8 (2t + e) + nt.
  const float vs = VS[0] * (1.f / 256.f);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float zz = z[mt][h];
      zz += __shfl_xor_sync(0xffffffff, zz, 1);
      zz += __shfl_xor_sync(0xffffffff, zz, 2);
      const int j = j0 + 2 * mt + h;
      const long long row = ((long long)(b * 2 + kh) * splits + split) * 64 + j * 8 + g;
      const float inv = vs / fmaxf(zz, 1e-20f);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) PART[row * D + 8 * (2 * t + e) + nt] = o[mt][nt][2 * h + e] * inv;
      }
      if (t == 0) LSE[row] = zz > 0.f ? m[mt][h] + log2f(zz) : MUSE_NEG_INF;
    }
  }
}
