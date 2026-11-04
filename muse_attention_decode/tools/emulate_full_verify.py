"""Lane-level numpy emulation of candidates/cuda/attn_full_verify.cu (8-token causal verify).

Mirrors each warp's register fragments (Q/K natural byte order, P/V permuted orders, PRMT
transposes) with the PTX mma.m16n8k32 layouts from emulate_full_decode.py, in float64
without FP8 rounding, and compares PART/LSE with direct causal attention.
"""
import numpy as np
from emulate_full_decode import mma, prmt

HQ, D, ROW, L, TILE = 16, 64, 128, 8, 32


def run(q, kc, vc, pt, seqlen, page, chunk, split, ks=1.0, vs=1.0, sm_scale=0.125):
    """q [8, 16, 64] (block tokens, heads, dims); kc/vc [rows, 128]."""
    scale = ks * sm_scale * 1.4426950408889634
    begin = split * chunk
    end = min(begin + chunk, seqlen)
    row_of = lambda ts: pt[ts // page] * page + ts % page
    part = np.zeros((2, 64, D)); lse = np.zeros((2, 64))
    for warp in range(4):
        kh, j0 = warp >> 1, (warp & 1) * 4
        qa = [[[None] * 4 for _ in range(2)] for _ in range(32)]
        qa = [[[[None] * 4 for _ in range(2)] for _ in range(2)] for _ in range(32)]
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for mt in range(2):
                for h in range(2):
                    j = j0 + 2 * mt + h
                    qrow = q[j, kh * 8 + g]
                    for s in range(2):
                        qa[lane][mt][s][h] = list(qrow[32 * s + 4 * t:32 * s + 4 * t + 4])
                        qa[lane][mt][s][2 + h] = list(qrow[32 * s + 16 + 4 * t:32 * s + 16 + 4 * t + 4])
        pos = [[seqlen - L + j0 + 2 * mt + h for h in range(2)] for mt in range(2)]
        o = [[[[0.0] * 4 for _ in range(8)] for _ in range(2)] for _ in range(32)]
        m = [[[-np.inf, -np.inf] for _ in range(2)] for _ in range(32)]
        z = [[[0.0, 0.0] for _ in range(2)] for _ in range(32)]
        base = begin
        while base < end:
            tile_tok = [(base + r) if base + r < end else begin for r in range(TILE)]
            K = np.stack([kc[row_of(ts)] for ts in tile_tok]); V = np.stack([vc[row_of(ts)] for ts in tile_tok])
            c = [[[[0.0] * 4 for _ in range(4)] for _ in range(2)] for _ in range(32)]
            for nj in range(4):
                kw = []
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    r = 8 * nj + g
                    kw.append([[list(K[r, kh * 64 + 32 * s + 4 * t:kh * 64 + 32 * s + 4 * t + 4]),
                                list(K[r, kh * 64 + 32 * s + 16 + 4 * t:kh * 64 + 32 * s + 16 + 4 * t + 4])] for s in range(2)])
                for mt in range(2):
                    for s in range(2):
                        cs = [c[lane][mt][nj] for lane in range(32)]
                        mma(cs, [qa[lane][mt][s] for lane in range(32)], [kw[lane][s] for lane in range(32)])
            full = base + TILE <= end and base + TILE - 1 <= seqlen - L
            for mt in range(2):
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    for nj in range(4):
                        for e in range(2):
                            ts = base + 8 * nj + 2 * t + e
                            for h in range(2):
                                ok = full or (ts < end and ts <= pos[mt][h])
                                c[lane][mt][nj][2 * h + e] = c[lane][mt][nj][2 * h + e] * scale if ok else -np.inf
                for quad in range(8):
                    lanes = range(4 * quad, 4 * quad + 4)
                    for h in range(2):
                        mx = max(c[l][mt][nj][2 * h + e] for l in lanes for nj in range(4) for e in range(2))
                        for l in lanes:
                            nm = max(m[l][mt][h], mx)
                            nm = 0.0 if nm == -np.inf else nm
                            al = 2.0 ** (m[l][mt][h] - nm)
                            m[l][mt][h] = nm
                            z[l][mt][h] *= al
                            for nj in range(4):
                                for e in range(2):
                                    p = 2.0 ** (c[l][mt][nj][2 * h + e] - nm)
                                    c[l][mt][nj][2 * h + e] = p
                                    z[l][mt][h] += p
                            for nt in range(8):
                                o[l][mt][nt][2 * h] *= al; o[l][mt][nt][2 * h + 1] *= al
            bt = []
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                vw = []
                for i in range(8):
                    r = 2 * t + (i & 1) + 8 * (i >> 1)
                    by = V[r, kh * 64 + 8 * g:kh * 64 + 8 * g + 8]
                    vw.append([list(by[0:4]), list(by[4:8])])
                b8 = [[None, None] for _ in range(8)]
                for w in range(2):
                    for h in range(2):
                        x0, x1, x2, x3 = (vw[4 * h + i][w] for i in range(4))
                        t01lo, t01hi = prmt(x0, x1, 0x5140), prmt(x0, x1, 0x7362)
                        t23lo, t23hi = prmt(x2, x3, 0x5140), prmt(x2, x3, 0x7362)
                        b8[4 * w + 0][h] = prmt(t01lo, t23lo, 0x5410)
                        b8[4 * w + 1][h] = prmt(t01lo, t23lo, 0x7632)
                        b8[4 * w + 2][h] = prmt(t01hi, t23hi, 0x5410)
                        b8[4 * w + 3][h] = prmt(t01hi, t23hi, 0x7632)
                bt.append(b8)
            for mt in range(2):
                pa = []
                for lane in range(32):
                    cc = c[lane][mt]
                    pa.append([[cc[0][0] * 256, cc[0][1] * 256, cc[1][0] * 256, cc[1][1] * 256],
                               [cc[0][2] * 256, cc[0][3] * 256, cc[1][2] * 256, cc[1][3] * 256],
                               [cc[2][0] * 256, cc[2][1] * 256, cc[3][0] * 256, cc[3][1] * 256],
                               [cc[2][2] * 256, cc[2][3] * 256, cc[3][2] * 256, cc[3][3] * 256]])
                for nt in range(8):
                    os_ = [o[lane][mt][nt] for lane in range(32)]
                    mma(os_, pa, [[bt[lane][nt][0], bt[lane][nt][1]] for lane in range(32)])
            base += TILE
        for mt in range(2):
            for h in range(2):
                for quad in range(8):
                    lanes = list(range(4 * quad, 4 * quad + 4))
                    zz = sum(z[l][mt][h] for l in lanes)
                    g = quad
                    j = j0 + 2 * mt + h
                    row = j * 8 + g
                    for l in lanes:
                        t = l & 3
                        for nt in range(8):
                            for e in range(2):
                                part[kh, row, 8 * (2 * t + e) + nt] = o[l][mt][nt][2 * h + e] * vs / 256 / max(zz, 1e-20)
                    lse[kh, row] = m[lanes[0]][mt][h] + np.log2(zz) if zz > 0 else -np.inf
    return part, lse


def reference(q, kc, vc, pt, seqlen, page, chunk, split, ks=1.0, vs=1.0, sm_scale=0.125):
    begin = split * chunk
    end = min(begin + chunk, seqlen)
    part = np.zeros((2, 64, D)); lse = np.full((2, 64), -np.inf)
    rows = [pt[ts // page] * page + ts % page for ts in range(begin, end)]
    if not rows:
        return part, lse
    for kh in range(2):
        K = kc[rows, kh * 64:(kh + 1) * 64]; V = vc[rows, kh * 64:(kh + 1) * 64]
        for j in range(L):
            vis = np.array([ts <= seqlen - L + j for ts in range(begin, end)])
            for qh in range(8):
                if not vis.any():
                    continue
                logit = (K @ q[j, kh * 8 + qh]) * ks * sm_scale * 1.4426950408889634
                logit = np.where(vis, logit, -np.inf)
                mx = logit.max(); p = 2.0 ** (logit - mx)
                part[kh, j * 8 + qh] = (p @ V) / p.sum() * vs
                lse[kh, j * 8 + qh] = mx + np.log2(p.sum())
    return part, lse


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    worst = 0.0
    for case, (seqlen, page, chunk, split) in enumerate([(300, 1, 256, 1), (300, 1, 256, 0), (1000, 64, 256, 3),
                                                         (40, 1, 2048, 0), (300, 1, 256, 2), (2056, 64, 2048, 1)]):
        pages = (seqlen + page - 1) // page + 2
        pt = rng.permutation(pages).astype(np.int64)
        q = rng.normal(size=(L, HQ, D)); kc = rng.normal(size=(pages * page, ROW)); vc = rng.normal(size=(pages * page, ROW))
        got_p, got_l = run(q, kc, vc, pt, seqlen, page, chunk, split, ks=0.5, vs=2.0)
        ref_p, ref_l = reference(q, kc, vc, pt, seqlen, page, chunk, split, ks=0.5, vs=2.0)
        ep = np.abs(got_p - ref_p).max()
        fin = np.isfinite(ref_l)
        el = np.abs(got_l[fin] - ref_l[fin]).max() if fin.any() else 0.0
        inf_ok = np.array_equal(np.isinf(got_l), np.isinf(ref_l))
        worst = max(worst, ep, el)
        print(f"case {case}: seqlen={seqlen} page={page} chunk={chunk} split={split}  max|dPART|={ep:.2e}  max|dLSE|={el:.2e}  inf-match={inf_ok}")
    print("WORST", worst)
