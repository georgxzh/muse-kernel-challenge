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
// v2 data path: each warp stages its 32-token tiles' whole K and V rows into shared
// memory with cp.async (2 stages, page-table entries fetched one tile further ahead),
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

__device__ __forceinline__ long long cache_row(const int* __restrict__ pt_b, int ts, int page) {
  return (long long)pt_b[ts / page] * page + ts % page;
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

  // This warp's tiles start at begin + warp * TILE, every WARPS * TILE tokens. Lane l stages
  // token (tile base + l): its whole K row and V row (8 chunks each). Keys past `end` stage
  // the row of `begin` (always valid); their logits are masked to -inf.
  const int step = WARPS * TILE;
  const int first = begin + warp * TILE;
  auto row_of = [&](int base) -> long long {
    int ts = base + lane;
    ts = ts < end ? ts : begin;
    return cache_row(pt_b, ts, page);
  };
  auto stage_tile = [&](int st, long long row) {
    const uint8_t* kg = KC + row * ROW;
    const uint8_t* vg = VC + row * ROW;
    uint8_t* kb = wbuf + st * (2 * TILE * ROW);
    uint8_t* vb = kb + TILE * ROW;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      cp_async16(smem_u32(kb + sw(lane, c)), kg + 16 * c);
      cp_async16(smem_u32(vb + sw(lane, c)), vg + 16 * c);
    }
  };
  long long row_next = 0;
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
