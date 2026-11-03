# okc: problem=muse_bf16_proj
"""Split-K tensor-core projections, with a fused hi/lo router."""
from typing import TypedDict

import torch
import triton
import triton.language as tl


_TILE = 64  # weight tile; every tiled layer's N and K are multiples


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


@triton.jit
def _split_mm(X, W, L, P, M: tl.constexpr, N: tl.constexpr,
              K: tl.constexpr, SPLIT: tl.constexpr, ROUTER: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    tl.store(P + si * M * N + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _tiled_mm(X, T, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              SPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr):
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
    tl.store(P + si * M * N + mi[:, None] * N + tn * BN + nj[None, :], acc,
             mi[:, None] < M)


@triton.jit
def _packed_mm(X, W, C, E, P, BASE: tl.constexpr, M: tl.constexpr,
               N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    si = tl.program_id(2)
    kj = tl.arange(0, BK)
    ej = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(K // SPLIT // BK):
        start = si * (K // SPLIT) + block * BK
        kk = start + kj
        # Unique, coalesced exponent-byte loads. Expand adjacent nibbles
        # locally instead of loading each byte twice with a repeated index.
        exps = tl.load(E + ni[:, None] * (K // 2) + start // 2 + ej[None, :])
        codes = tl.interleave(exps & 15, exps >> 4).to(tl.uint32)
        payload = tl.load(C + ni[:, None] * K + kk[None, :]).to(tl.uint32)
        bits = ((payload & 128) << 8) | ((BASE + codes - 1) << 7) | (payload & 127)
        original = tl.load(W + ni[:, None] * K + kk[None, :], codes == 0, 0)
        original_bits = original.to(tl.uint16, bitcast=True).to(tl.uint32)
        bits = tl.where(codes == 0, original_bits, bits).to(tl.uint16)
        weight = bits.to(tl.bfloat16, bitcast=True)
        x = tl.load(X + mi[:, None] * K + kk[None, :], mi[:, None] < M, 0)
        acc = tl.dot(x, tl.trans(weight), acc)
    tl.store(P + si * M * N + mi[:, None] * N + ni[None, :], acc,
             mi[:, None] < M)

@triton.jit
def _finish(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr,
            BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLIT)
    v = tl.load(P + s[:, None] * SIZE + i[None, :], i[None, :] < SIZE, 0)
    tl.store(Y + i, tl.sum(v, axis=0), i < SIZE)


@triton.jit
def _router_small(X, W, P, M: tl.constexpr, N: tl.constexpr,
                  K: tl.constexpr, SPLIT: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    si = tl.program_id(2)
    ki = si * BK + tl.arange(0, BK)
    x = tl.load(X + row * K + ki).to(tl.float32)
    w = tl.load(W + ni[:, None] * K + ki[None, :], ni[:, None] < N, 0)
    v = tl.sum(w * x[None, :], axis=1)
    tl.store(P + si * M * N + row * N + ni, v, ni < N)


def prepare(weights: Weights):
    state = dict(weights)
    hi = weights["router"].to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (weights["router"] - hi.float()).to(torch.bfloat16)
    for name in ("pooler", "unpooler"):
        state[name + "_t"] = weights[name].T.contiguous()
    for name in ("qkv_gate", "pooler", "unpooler"):
        w = weights[name]
        bits = w.view(torch.int16).to(torch.int32) & 65535
        exponent = (bits >> 7) & 255
        base = max(0, int(exponent.max().item()) - 14)
        code = torch.where((exponent >= base) & (exponent < base + 15), exponent - base + 1, 0)
        payload = ((bits & 127) | ((bits >> 8) & 128)).to(torch.uint8)
        packed_exp = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        state[name + "_packed"] = (payload, packed_exp, base)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    m, n, k, layer = cfg["m"], cfg["n"], cfg["k"], cfg["layer"]
    router = layer == "router"
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
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    if router and m == 1:
        _router_small[(m, triton.cdiv(n, 4), split)](
            x, state["router"], partial, m, n, k, split, 4, k // split,
            num_warps=4)
    elif tiled:
        payload, exps, base = state[layer + "_packed"]
        _packed_mm[(triton.cdiv(m, bm), n // _TILE, split)](
            x, state[layer], payload, exps, partial, base, m, n, k, split, bm, _TILE, _TILE,
            num_warps=4, num_stages=2)
    else:
        w = state["router_hi"] if router else state[layer]
        lo = state["router_lo"] if router else w
        _split_mm[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
            x, w, lo, partial, m, n, k, split, router, bm, bn, 64,
            num_warps=8 if bm == 128 else 4, num_stages=3)
    _finish[(triton.cdiv(m * n, 256),)](partial, out, m * n, split, 256)
    return out
