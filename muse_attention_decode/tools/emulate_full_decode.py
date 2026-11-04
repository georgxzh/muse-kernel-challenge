"""Lane-level numpy emulation of candidates/cuda/attn_full_decode.cu's index math.

Emulates every lane's registers exactly as the kernel builds them (Q/K/V/P byte
fragments, PRMT transposes, mma.m16n8k32 fragment layouts from the PTX ISA, the
per-warp online softmax and the 4-warp merge) in float64 without FP8 rounding, and
compares PART/LSE with a direct attention computation. Checks the layouts and
permutations, not FP8 numerics or performance.
"""
import numpy as np

HQ, D, ROW, WARPS, TILE = 16, 64, 128, 4, 32


def mma(c, a_regs, b_regs):
    """mma.m16n8k32 row.col E4M3: a_regs[lane][4][4 bytes], b_regs[lane][2][4 bytes]; c[lane][4]."""
    A = np.zeros((16, 32)); B = np.zeros((32, 8))
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(4):
            for j in range(4):
                row = g + 8 * (i % 2)
                col = t * 4 + j + 16 * (i // 2)
                A[row, col] = a_regs[lane][i][j]
        for i in range(2):
            for j in range(4):
                B[t * 4 + j + 16 * i, g] = b_regs[lane][i][j]
    C = A @ B
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        c[lane][0] += C[g, 2 * t]; c[lane][1] += C[g, 2 * t + 1]
        c[lane][2] += C[g + 8, 2 * t]; c[lane][3] += C[g + 8, 2 * t + 1]


def prmt(a, b, sel):
    src = list(a) + list(b)
    return [src[(sel >> (4 * k)) & 0xF] for k in range(4)]


def run(q, kc, vc, pt, seqlen, page, chunk, split, ks=1.0, vs=1.0, sm_scale=0.125):
    """One CTA: (sequence, split). q [16, 64]; kc/vc [rows, 128]; pt [pages]."""
    scale = ks * sm_scale * 1.4426950408889634
    begin = split * chunk
    end = min(begin + chunk, seqlen)
    row_of = lambda ts: pt[ts // page] * page + ts % page
    zero4 = [0.0] * 4
    s_m = np.zeros((WARPS, HQ)); s_z = np.zeros((WARPS, HQ)); s_o = np.zeros((WARPS, HQ, ROW))
    for warp in range(WARPS):
        qw = []
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            head = g if t < 2 else 8 + g
            by = q[head, (t & 1) * 32:(t & 1) * 32 + 32]
            qw.append([list(by[4 * w:4 * w + 4]) for w in range(8)])
        o = [[[0.0] * 4 for _ in range(16)] for _ in range(32)]
        m_lo = [-np.inf] * 32; m_hi = [-np.inf] * 32; z_lo = [0.0] * 32; z_hi = [0.0] * 32
        base = begin + warp * TILE
        while base < end:
            full = base + TILE <= end
            kr = []; vr = []
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                kl = []
                for j in range(4):
                    ts = base + 8 * j + g
                    ts = ts if ts < end else begin
                    r = row_of(ts)
                    by = kc[r, t * 32:t * 32 + 32]
                    kl.append([list(by[4 * w:4 * w + 4]) for w in range(8)])
                kr.append(kl)
                vl = []
                for i in range(8):
                    ts = base + 2 * t + (i & 1) + 8 * (i >> 1)
                    ts = ts if ts < end else begin
                    r = row_of(ts)
                    by = vc[r, 16 * g:16 * g + 16]
                    vl.append([list(by[4 * w:4 * w + 4]) for w in range(4)])
                vr.append(vl)
            c = [[[0.0] * 4 for _ in range(4)] for _ in range(32)]
            for j in range(4):
                for s in range(4):
                    a_regs = []; b_regs = []
                    for lane in range(32):
                        t = lane & 3
                        lo, hi = qw[lane][2 * s], qw[lane][2 * s + 1]
                        kv0 = t < 2
                        a_regs.append([lo if kv0 else zero4, zero4 if kv0 else lo, hi if kv0 else zero4, zero4 if kv0 else hi])
                        b_regs.append([kr[lane][j][2 * s], kr[lane][j][2 * s + 1]])
                    cj = [c[lane][j] for lane in range(32)]
                    mma(cj, a_regs, b_regs)
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for j in range(4):
                    for e in range(2):
                        ok = full or (base + 8 * j + 2 * t + e < end)
                        c[lane][j][e] = c[lane][j][e] * scale if ok else -np.inf
                        c[lane][j][2 + e] = c[lane][j][2 + e] * scale if ok else -np.inf
            for quad in range(8):
                lanes = range(4 * quad, 4 * quad + 4)
                mx_lo = max(c[l][j][e] for l in lanes for j in range(4) for e in range(2))
                mx_hi = max(c[l][j][2 + e] for l in lanes for j in range(4) for e in range(2))
                for l in lanes:
                    nm_lo = max(m_lo[l], mx_lo); nm_hi = max(m_hi[l], mx_hi)
                    nm_lo = 0.0 if nm_lo == -np.inf else nm_lo
                    nm_hi = 0.0 if nm_hi == -np.inf else nm_hi
                    al_lo = 2.0 ** (m_lo[l] - nm_lo); al_hi = 2.0 ** (m_hi[l] - nm_hi)
                    m_lo[l], m_hi[l] = nm_lo, nm_hi
                    z_lo[l] *= al_lo; z_hi[l] *= al_hi
                    for j in range(4):
                        for e in range(2):
                            c[l][j][e] = 2.0 ** (c[l][j][e] - nm_lo)
                            c[l][j][2 + e] = 2.0 ** (c[l][j][2 + e] - nm_hi)
                            z_lo[l] += c[l][j][e]; z_hi[l] += c[l][j][2 + e]
                    for nt in range(16):
                        o[l][nt][0] *= al_lo; o[l][nt][1] *= al_lo
                        o[l][nt][2] *= al_hi; o[l][nt][3] *= al_hi
            pa = []
            for lane in range(32):
                cc = c[lane]
                pa.append([[cc[0][0] * 256, cc[0][1] * 256, cc[1][0] * 256, cc[1][1] * 256],
                           [cc[0][2] * 256, cc[0][3] * 256, cc[1][2] * 256, cc[1][3] * 256],
                           [cc[2][0] * 256, cc[2][1] * 256, cc[3][0] * 256, cc[3][1] * 256],
                           [cc[2][2] * 256, cc[2][3] * 256, cc[3][2] * 256, cc[3][3] * 256]])
            for w in range(4):
                bt = []
                for lane in range(32):
                    btl = [[None] * 4, [None] * 4]
                    for h in range(2):
                        x0, x1, x2, x3 = (vr[lane][4 * h + i][w] for i in range(4))
                        t01lo, t01hi = prmt(x0, x1, 0x5140), prmt(x0, x1, 0x7362)
                        t23lo, t23hi = prmt(x2, x3, 0x5140), prmt(x2, x3, 0x7362)
                        btl[h] = [prmt(t01lo, t23lo, 0x5410), prmt(t01lo, t23lo, 0x7632),
                                  prmt(t01hi, t23hi, 0x5410), prmt(t01hi, t23hi, 0x7632)]
                    bt.append(btl)
                for k in range(4):
                    ok_ = [o[lane][4 * w + k] for lane in range(32)]
                    mma(ok_, pa, [[bt[lane][0][k], bt[lane][1][k]] for lane in range(32)])
            base += WARPS * TILE
        for quad in range(8):
            lanes = list(range(4 * quad, 4 * quad + 4))
            zl = sum(z_lo[l] for l in lanes); zh = sum(z_hi[l] for l in lanes)
            g = quad
            s_m[warp, g] = m_lo[lanes[0]]; s_m[warp, g + 8] = m_hi[lanes[0]]
            s_z[warp, g] = zl; s_z[warp, g + 8] = zh
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for nt in range(16):
                for e in range(2):
                    s_o[warp, g, (2 * t + e) * 16 + nt] = o[lane][nt][e]
                    s_o[warp, g + 8, (2 * t + e) * 16 + nt] = o[lane][nt][2 + e]
    part = np.zeros((2, 8, D)); lse = np.zeros((2, 8))
    for r in range(HQ):
        kh = r >> 3
        mm = s_m[:, r].max()
        f = np.zeros(WARPS) if mm == -np.inf else 2.0 ** (s_m[:, r] - mm)
        zz = (s_z[:, r] * f).sum()
        for d in range(D):
            oo = (s_o[:, r, kh * D + d] * f).sum()
            part[kh, r & 7, d] = oo / max(zz, 1e-20) * vs / 256
        lse[kh, r & 7] = mm + np.log2(zz) if zz > 0 else -np.inf
    return part, lse


def reference(q, kc, vc, pt, seqlen, page, chunk, split, ks=1.0, vs=1.0, sm_scale=0.125):
    begin = split * chunk
    end = min(begin + chunk, seqlen)
    part = np.zeros((2, 8, D)); lse = np.full((2, 8), -np.inf)
    if end <= begin:
        return part, lse
    rows = [pt[ts // page] * page + ts % page for ts in range(begin, end)]
    for kh in range(2):
        K = kc[rows, kh * D:(kh + 1) * D]; V = vc[rows, kh * D:(kh + 1) * D]
        for qh in range(8):
            logit = (K @ q[kh * 8 + qh]) * ks * sm_scale * 1.4426950408889634
            m = logit.max(); p = 2.0 ** (logit - m)
            part[kh, qh] = (p @ V) / p.sum() * vs
            lse[kh, qh] = m + np.log2(p.sum())
    return part, lse


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst = 0.0
    for case, (seqlen, page, chunk, split) in enumerate([(300, 1, 256, 0), (300, 1, 256, 1), (1000, 64, 256, 3),
                                                         (130, 1, 256, 0), (300, 1, 256, 2), (77, 64, 2048, 0)]):
        pages = (seqlen + page - 1) // page + 2
        n_rows = pages * page
        pt = rng.permutation(pages).astype(np.int64)
        q = rng.normal(size=(HQ, D)); kc = rng.normal(size=(n_rows, ROW)); vc = rng.normal(size=(n_rows, ROW))
        got_p, got_l = run(q, kc, vc, pt, seqlen, page, chunk, split, ks=0.5, vs=2.0)
        ref_p, ref_l = reference(q, kc, vc, pt, seqlen, page, chunk, split, ks=0.5, vs=2.0)
        ep = np.abs(got_p - ref_p).max()
        el = np.nanmax(np.abs(np.where(np.isinf(ref_l), 0, got_l - ref_l))) if np.isfinite(ref_l).any() else 0.0
        inf_ok = np.array_equal(np.isinf(got_l), np.isinf(ref_l))
        worst = max(worst, ep, el)
        print(f"case {case}: seqlen={seqlen} page={page} chunk={chunk} split={split}  max|dPART|={ep:.2e}  max|dLSE|={el:.2e}  inf-match={inf_ok}")
    print("WORST", worst)
