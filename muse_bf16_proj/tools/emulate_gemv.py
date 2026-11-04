"""Lane-level numpy emulation of cuda/gemv_bf16.cu's fragment mapping (mma.m16n8k16 BF16
layouts from the PTX ISA), float64, vs x @ W^T. Checks indexing only."""
import numpy as np


def mma16816(c, a, b):
    """a[lane] = 4 regs x 2 values (a0..a7), b[lane] = 2 regs x 2 values; c[lane][4]."""
    A = np.zeros((16, 16)); B = np.zeros((16, 8))
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(4):
            for e in range(2):
                row = g + 8 * (i % 2); col = 2 * t + e + 8 * (i // 2)
                A[row, col] = a[lane][i][e]
        for i in range(2):
            for e in range(2):
                B[2 * t + e + 8 * i, g] = b[lane][i][e]
    C = A @ B
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        c[lane][0] += C[g, 2 * t]; c[lane][1] += C[g, 2 * t + 1]
        c[lane][2] += C[g + 8, 2 * t]; c[lane][3] += C[g + 8, 2 * t + 1]


def warp_tile(x, w, n0, kbeg, kc, M):
    """One warp: columns n0..n0+7 over K range [kbeg, kbeg + kc)."""
    c = [[0.0] * 4 for _ in range(32)]
    for k in range(0, kc, 64):
        for s in range(4):
            a = []; b = []
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                base = kbeg + k + 16 * t + 4 * s
                xr = lambda row: (x[row, base:base + 4] if row < M else np.zeros(4))
                lo, hi = xr(g), xr(g + 8)
                wv = w[n0 + g, base:base + 4]
                a.append([lo[0:2], hi[0:2], lo[2:4], hi[2:4]])
                b.append([wv[0:2], wv[2:4]])
            mma16816(c, a, b)
    out = np.zeros((16, 8))
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        out[g, 2 * t] = c[lane][0]; out[g, 2 * t + 1] = c[lane][1]
        out[g + 8, 2 * t] = c[lane][2]; out[g + 8, 2 * t + 1] = c[lane][3]
    return out[:M]


rng = np.random.default_rng(0)
worst = 0
for M, N, K, splits in ((1, 64, 512, 2), (8, 32, 1024, 4), (16, 96, 256, 1), (5, 32, 768, 3)):
    x = rng.normal(size=(M, K)); w = rng.normal(size=(N, K))
    kc = K // splits
    y = np.zeros((M, N))
    for nb in range(N // 32):
        for warp in range(4):
            n0 = nb * 32 + warp * 8
            for sp in range(splits):
                y[:, n0:n0 + 8] += warp_tile(x, w, n0, sp * kc, kc, M)
    err = np.abs(y - x @ w.T).max(); worst = max(worst, err)
    print(f"M={M} N={N} K={K} splits={splits}: max err {err:.2e}")
print("WORST", worst)
