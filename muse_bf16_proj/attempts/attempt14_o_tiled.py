# okc: problem=muse_bf16_proj
"""Split-K tensor-core projections, with a fused hi/lo router."""
from typing import TypedDict

import torch
import triton
import triton.language as tl


_TILE = 64  # weight tile; every tiled layer's N and K are multiples
SERIAL_ROUTER = True   # router m<=16: one kernel, in-kernel split-K reduction
SERIAL_TILED = True    # tiled qkv_gate/pooler/unpooler: one kernel as well
O_TILED_MAX_M = 16     # o (n=8192, k=1024): tiled single-kernel path at m<=16 (was cuBLAS)
TILED_SPLITS = {"qkv_gate": 8, "pooler": 8, "unpooler": 16, "o": 4}


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
    tl.store(P + tl.program_id(2) * MN + offs, acc, mask)
    tl.debug_barrier()          # every thread's partial is issued before the release
    ticket = tl.atomic_add(CNT + tile, 1, sem="acq_rel", scope="gpu")
    if ticket == SPLIT - 1:
        total = tl.load(P + offs, mask, other=0.0, cache_modifier=".cg")
        for s in tl.static_range(1, SPLIT):
            total += tl.load(P + s * MN + offs, mask, other=0.0, cache_modifier=".cg")
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


def prepare(weights: Weights):
    state = dict(weights)
    hi = weights["router"].to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (weights["router"] - hi.float()).to(torch.bfloat16)
    for name in ("pooler", "unpooler"):
        state[name + "_t"] = weights[name].T.contiguous()
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        w = weights[name]
        tiles = w.reshape(w.shape[0] // _TILE, _TILE, w.shape[1] // _TILE, _TILE)
        state[name + "_tiled"] = tiles.permute(0, 2, 3, 1).contiguous()
    # Concatenate output columns so cuBLAS computes the two allowed BF16
    # router products in one GEMM, without duplicating activations.
    state["router_pair"] = torch.cat((hi.T, state["router_lo"].T), dim=1).contiguous()
    # Split-K arrival counters: zero at rest, reset by each tile's last CTA.
    state["counters"] = torch.zeros((_COUNTERS,), dtype=torch.int32, device=hi.device)
    return state


@triton.jit
def _join_router(P, Y, SIZE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = (i // N) * (2 * N) + i % N
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
        _join_router[(triton.cdiv(m * n, 256),)](products, out, m * n, n, 256)
        return out
    tiled = ((layer in ("qkv_gate", "pooler") and m <= 16) or (layer == "unpooler" and m <= 8)
             or (layer == "o" and m <= O_TILED_MAX_M))
    if layer in ("pooler", "unpooler") and m > 32:
        return torch.mm(x, state[layer + "_t"][:, :n])
    if not router and not tiled:
        return torch.mm(x, state[layer][:n].T)
    # Tiled K-splits per layer (attempt 13: more splits help only the layers with
    # few output tiles; qkv_gate/pooler regressed at 16).
    split = (16 if m <= 32 else 8) if router else (TILED_SPLITS[layer] if tiled else 8)
    bm = (16 if m <= 16 else 32 if m <= 32 else 64 if m <= 128 else 128) if router else 16
    bn = 32 if router else 64
    out = torch.empty((m, n), device=x.device,
                      dtype=torch.float32 if router else torch.bfloat16)
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
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
        _finish[(triton.cdiv(m * n, 256),)](partial, out, m * n, split, 256)
    return out
