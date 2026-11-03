# okc: problem=muse_attention_decode
"""Hybrid FP8/BF16 PV attention, FA3 for batched full verify, PDL; full-layer loop with 5 stages
(double-buffered K/V gathers, page indices prefetched ahead)."""

import torch
import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from sgl_kernel.flash_attn import flash_attn_with_kvcache

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
    _attention_parts[(b, hk, splits)](
        q, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"],
        data["k_scale"], data["v_scale"], partial, lse, hq, hk, d, length,
        cfg["page_size"], data["page_table"].stride(0), splits, chunk, window,
        cfg["sm_scale"] * cfg["qk_scale_factor"], max(16, tr.next_power_of_2(rows)), ATTENTION_N,
        num_warps=4, num_stages=5 if not window else 2, launch_pdl=True)
    _combine_gate[(tokens, hq)](
        partial, lse, qkv, out, hq, hk, d, length, stride, splits,
        tr.next_power_of_2(splits), num_warps=8 if splits >= 128 else 4, launch_pdl=True)
    return out
