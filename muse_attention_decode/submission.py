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
