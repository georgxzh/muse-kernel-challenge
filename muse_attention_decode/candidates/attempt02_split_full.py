# okc: problem=muse_attention_decode
"""Fused attention preparation and output gate around SGLang's FP8 FA3."""

import torch
import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from sgl_kernel.flash_attn import flash_attn_with_kvcache


@tr.jit
def _prepare(QKV, POS, LOC, KS, VS, KC, VC, Q,
             HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             STRIDE: tl.constexpr, THETA: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    x = tl.arange(0, D)
    a = tl.load(QKV + t * STRIDE + h * D + x).to(tl.float32)
    r = tl.rsqrt(tl.sum(a * a, 0) / D + 1.e-5)
    a = a * r
    if THETA != 0:
        # Interleaved RoPE, evaluated at the actual GPU token position.
        other = tl.load(QKV + t * STRIDE + h * D + (x ^ 1)).to(tl.float32) * r
        freq = tl.div_rn(1., libdevice.pow(THETA, ((x // 2) * 2).to(tl.float32) / D))
        angle = tl.load(POS + t).to(tl.float32) * freq
        c = libdevice.cos(angle)
        s = libdevice.sin(angle)
        a = a * c + tl.where(x % 2 == 0, -other, other) * s
    if h < HQ:
        # SGLang stores normalized BF16 queries before converting to E4M3.
        a = a.to(tl.bfloat16).to(tl.float32)
        tl.store(Q + (t * HQ + h) * D + x, a)
    else:
        kh = h - HQ
        loc = tl.load(LOC + t)
        kscale = tl.load(KS)
        vscale = tl.load(VS)
        v = tl.load(QKV + t * STRIDE + (HQ + HK + kh) * D + x).to(tl.float32)
        tl.store(KC + (loc * HK + kh) * D + x, a / kscale)
        tl.store(VC + (loc * HK + kh) * D + x, v / vscale)


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
                     SPLITS: tl.constexpr, CHUNK: tl.constexpr,
                     SCALE: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    b = tl.program_id(0)
    kh = tl.program_id(1)
    split = tl.program_id(2)
    G: tl.constexpr = HQ // HK
    r = tl.arange(0, M)
    d = tl.arange(0, D)
    n = tl.arange(0, N)
    j = r // G
    q = tl.load(Q + ((b * L + j[:, None]) * HQ + kh * G + r[:, None] % G) * D + d[None, :],
                mask=r[:, None] < L * G, other=0.).to(tl.bfloat16)
    seqlen = tl.load(LENS + b)
    pos = seqlen - L + j
    scale = tl.load(KS) * SCALE * 1.4426950408889634
    m = tl.full((M,), -float("inf"), tl.float32)
    z = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    begin = split * CHUNK
    end = tl.minimum(begin + CHUNK, seqlen)
    for start in range(begin, end, N):
        ts = start + n
        pages = tl.load(PT + b * PT_STRIDE + ts // PAGE, mask=ts < seqlen, other=0)
        rows = pages * PAGE + ts % PAGE
        k = tl.load(KC + (rows[None, :] * HK + kh) * D + d[:, None],
                    mask=ts[None, :] < seqlen, other=0.).to(tl.bfloat16)
        logits = tl.dot(q, k) * scale
        logits = tl.where((ts[None, :] < seqlen) & (ts[None, :] <= pos[:, None]), logits, -float("inf"))
        new_m = tl.maximum(m, tl.max(logits, 1))
        new_m = tl.where(new_m == -float("inf"), 0., new_m)
        p = tl.exp2(logits - new_m[:, None])
        alpha = tl.exp2(m - new_m)
        z = z * alpha + tl.sum(p, 1)
        v = tl.load(VC + (rows[:, None] * HK + kh) * D + d[None, :],
                    mask=ts[:, None] < seqlen, other=0.).to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m = new_m
    base = ((b * HK + kh) * SPLITS + split) * (L * G) + r
    result = acc / tl.maximum(z[:, None], 1.e-20) * tl.load(VS)
    tl.store(PART + base[:, None] * D + d[None, :], result, mask=r[:, None] < L * G)
    logsum = tl.where(z > 0, m + tl.log2(z), -float("inf"))
    tl.store(LSE + base, logsum, mask=r < L * G)


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
    lse = tl.load(LSE + base, mask=s < SPLITS, other=-float("inf"))
    w = tl.exp2(lse - tl.max(lse, 0))
    a = tl.load(PART + base[:, None] * D + d[None, :], mask=s[:, None] < SPLITS, other=0)
    a = tl.sum(a * w[:, None], 0) / tl.sum(w, 0)
    a = a.to(tl.bfloat16).to(tl.float32)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / D + 1.e-5)
    g = tl.load(QKV + t * STRIDE + (HQ + 2 * HK + h) * D + d).to(tl.float32)
    tl.store(O + (t * HQ + h) * D + d, a / (1. + tl.exp(-g)))



def kernel_fn(data):
    cfg = data["config"]
    b, length = cfg["batch"], cfg["q_len"]
    hq, hk, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"]
    tokens = b * length
    stride = qkv.stride(-2)
    q = torch.empty((tokens, hq, d), device=qkv.device, dtype=torch.float8_e4m3fn)
    _prepare[(tokens, hq + hk)](
        qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
        data["k_cache"], data["v_cache"], q, hq, hk, d, stride, cfg["rope_theta"],
        num_warps=4, enable_fp_fusion=False)
    if cfg["window"] == 0:
        chunk = 2048
        splits = tr.cdiv(data["page_table"].shape[1] * cfg["page_size"], chunk)
        rows = length * (hq // hk)
        partial = torch.empty((b, hk, splits, rows, d), device=qkv.device, dtype=torch.float32)
        lse = torch.empty((b, hk, splits, rows), device=qkv.device, dtype=torch.float32)
        out = torch.empty((b, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
        _attention_parts[(b, hk, splits)](
            q, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"],
            data["k_scale"], data["v_scale"], partial, lse, hq, hk, d, length,
            cfg["page_size"], data["page_table"].stride(0), splits, chunk,
            cfg["sm_scale"] * cfg["qk_scale_factor"], max(16, tr.next_power_of_2(rows)), 64,
            num_warps=4, num_stages=2)
        _combine_gate[(tokens, hq)](
            partial, lse, qkv, out, hq, hk, d, length, stride, splits,
            tr.next_power_of_2(splits), num_warps=8 if splits >= 128 else 4)
        return out
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
