"""Fuse Muse BF16 Q/K norm, interleaved RoPE and page-16 FP8 cache writes.

The BF16 round trips before query scaling and before RoPE/cache conversion
are explicit. One Q-head program owns each corresponding KV head's writes.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _norm_rope(
    x,
    COS,
    POS,
    eps,
    scale,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    Q_SCALE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    d = tl.arange(0, D)
    f = x.to(tl.float32)
    inv = tl.rsqrt(tl.sum(f * f, axis=0) / D + eps)
    value = (f * inv).to(tl.bfloat16).to(tl.float32)
    if Q_SCALE:
        value = (value * scale).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        pos = tl.load(POS + tl.program_id(0))
        # Day0 pos_enc.cu stores the FP32 cache loads in scalar_t (BF16),
        # and its BF16 operators round both products before the final add.
        cosine = tl.load(COS + pos * COS_STRIDE + d // 2).to(tl.bfloat16).to(tl.float32)
        sine = (
            tl.load(COS + pos * COS_STRIDE + D // 2 + d // 2)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        other = tl.reshape(tl.flip(tl.reshape(value, (D // 2, 2)), 1), (D,))
        rotated = tl.where(d % 2 == 0, -other, other)
        first = (value * cosine).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sine).to(tl.bfloat16).to(tl.float32)
        value = first + second
    return value.to(tl.bfloat16)


@tr.jit
def _muse_qk_rope_cache(
    Q,
    K,
    V,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    QO,
    KO,
    KS,
    VS,
    QS: tl.constexpr,
    KSTR: tl.constexpr,
    VSTR: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    q = tl.load(Q + token * QS + head * D + d)
    q = _norm_rope(q, COS, POS, EPS, SCALE, D, ROPE, True, COS_STRIDE)
    tl.store(QO + token * HQ * D + head * D + d, q)
    if head % (HQ // HK) == 0:
        kh = head // (HQ // HK)
        k = tl.load(K + token * KSTR + kh * D + d)
        k = _norm_rope(k, COS, POS, EPS, 1.0, D, ROPE, False, COS_STRIDE)
        tl.store(KO + token * HK * D + kh * D + d, k)
        slot = tl.load(LOC + token).to(tl.int64)
        if slot >= 0:
            if HAS_MAP:
                slot = tl.load(MAP + slot).to(tl.int64)
            if slot >= 0:
                v = tl.load(V + token * VSTR + kh * D + d)
                if HAS_SCALE:
                    k_value = k.to(tl.float32) / tl.load(KS)
                    v_value = v.to(tl.float32) / tl.load(VS)
                else:
                    k_value = k.to(tl.float32)
                    v_value = v.to(tl.float32)
                offset = slot * HK * D + kh * D + d
                tl.store(KC + offset, k_value)
                tl.store(VC + offset, v_value)


def fused_qk_rope_cache(
    q,
    k,
    v,
    positions,
    cos_sin,
    cache_k,
    cache_v,
    locations,
    swa_mapping,
    eps,
    q_scale,
    k_scale,
    v_scale,
    warps=1,
    fp_fusion=True,
):
    m = q.shape[0]
    d = 64
    hq = q.shape[1] // d
    hk = k.shape[1] // d
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
    assert cache_k.is_contiguous() and cache_v.is_contiguous()
    assert q.shape == (m, hq * d) and k.shape == v.shape == (m, hk * d)
    qo = torch.empty_like(q, memory_format=torch.contiguous_format)
    ko = torch.empty_like(k, memory_format=torch.contiguous_format)
    if m:
        _muse_qk_rope_cache[(m, hq)](
            q,
            k,
            v,
            positions,
            cos_sin,
            locations,
            swa_mapping,
            cache_k,
            cache_v,
            qo,
            ko,
            k_scale,
            v_scale,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            hq,
            hk,
            d,
            float(eps),
            float(q_scale),
            cos_sin is not None,
            swa_mapping is not None,
            k_scale is not None,
            cos_sin.stride(0) if cos_sin is not None else d,
            num_warps=warps,
            enable_fp_fusion=fp_fusion,
        )
    return qo, ko
