"""Per-head QK RMSNorm + interleaved RoPE + paged FP8 KV-cache store (Muse Spark, head_dim 64, page 16).

* ``qk_norm_rope_cache``: from projected Q/K/V. One program per (token, Q head) normalizes, scales and rotates
  Q; the first Q head of each GQA group also normalizes and rotates its K and stores K/V into the FP8 cache
  (optionally through the full -> SWA slot mapping).
* ``qkv_proj_qk_norm_rope_cache``: the same epilogue fed straight from the BF16 split-K partials of the fused
  QKV + output-gate projection (``bf16_splitk_gemm``), which also emits V and the gate; there each K/V head has
  its own program instead of riding on the first Q head of its group.
* ``qk_norm_rope_cache_extend``: extend, one program per token over all its heads; the norm is
  ``fused_qk_norm``'s body verbatim (same [heads, D] tile, reduction and warps), so it lowers to the same
  reduction and matches the unfused extend (``fused_qk_norm``, RoPE, cache write) bit for bit.

The BF16 round trips of the unfused sequence (norm output, query scaling, RoPE products, projection store)
are explicit, so the results match the separate norm / RoPE / cache-write kernels.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.bf16_splitk_gemm import bf16_splitk_gemm_partials


@triton.jit
def _rope_table(COS, pos, D: tl.constexpr, COS_STRIDE: tl.constexpr):
    d = tl.arange(0, D)
    # Day0 pos_enc.cu stores the FP32 cache loads in scalar_t (BF16),
    # and its BF16 operators round both products before the final add.
    cosine = tl.load(COS + pos * COS_STRIDE + d // 2).to(tl.bfloat16).to(tl.float32)
    sine = (
        tl.load(COS + pos * COS_STRIDE + D // 2 + d // 2).to(tl.bfloat16).to(tl.float32)
    )
    return cosine, sine


@triton.jit
def _rms_scale(x, eps, scale, D: tl.constexpr, Q_SCALE: tl.constexpr):
    f = x.to(tl.float32)
    inv = tl.rsqrt(tl.sum(f * f, axis=0) / D + eps)
    value = (f * inv).to(tl.bfloat16).to(tl.float32)
    if Q_SCALE:
        value = (value * scale).to(tl.bfloat16).to(tl.float32)
    return value


@triton.jit
def _rotate(value, cosine, sine, D: tl.constexpr):
    d = tl.arange(0, D)
    other = tl.reshape(tl.flip(tl.reshape(value, (D // 2, 2)), 1), (D,))
    rotated = tl.where(d % 2 == 0, -other, other)
    first = (value * cosine).to(tl.bfloat16).to(tl.float32)
    second = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    return first + second


@triton.jit
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
    value = _rms_scale(x, eps, scale, D, Q_SCALE)
    if ROPE:
        cosine, sine = _rope_table(COS, tl.load(POS + tl.program_id(0)), D, COS_STRIDE)
        value = _rotate(value, cosine, sine, D)
    return value.to(tl.bfloat16)


@triton.jit
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


@triton.jit
def _muse_qk_norm_rope_cache_tok(
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
    q_stride,
    k_stride,
    v_stride,
    eps,
    q_scale,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    t = tl.program_id(0)
    offs_q = tl.arange(0, HQ * D)
    q = tl.load(Q + t * q_stride + offs_q).to(tl.float32)
    q2 = tl.reshape(q, (HQ, D))
    rstd_q = tl.rsqrt(tl.sum(q2 * q2, axis=1) / D + eps)
    qn = (q2 * rstd_q[:, None]).to(tl.bfloat16).to(tl.float32) * q_scale
    qn = qn.to(tl.bfloat16).to(tl.float32)
    offs_k = tl.arange(0, HK * D)
    k = tl.load(K + t * k_stride + offs_k).to(tl.float32)
    k2 = tl.reshape(k, (HK, D))
    rstd_k = tl.rsqrt(tl.sum(k2 * k2, axis=1) / D + eps)
    kn = (k2 * rstd_k[:, None]).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        d = tl.arange(0, D)
        pos = tl.load(POS + t)
        cosine = tl.load(COS + pos * COS_STRIDE + d // 2).to(tl.bfloat16).to(tl.float32)
        sine = (
            tl.load(COS + pos * COS_STRIDE + D // 2 + d // 2)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        even = (d % 2 == 0)[None, :]
        oq = tl.reshape(tl.flip(tl.reshape(qn, (HQ, D // 2, 2)), 2), (HQ, D))
        rq = tl.where(even, -oq, oq)
        qn = (qn * cosine[None, :]).to(tl.bfloat16).to(tl.float32) + (
            rq * sine[None, :]
        ).to(tl.bfloat16).to(tl.float32)
        ok = tl.reshape(tl.flip(tl.reshape(kn, (HK, D // 2, 2)), 2), (HK, D))
        rk = tl.where(even, -ok, ok)
        kn = (kn * cosine[None, :]).to(tl.bfloat16).to(tl.float32) + (
            rk * sine[None, :]
        ).to(tl.bfloat16).to(tl.float32)
    qb = tl.reshape(qn, (HQ * D,)).to(tl.bfloat16)
    kb = tl.reshape(kn, (HK * D,)).to(tl.bfloat16)
    tl.store(QO + t * (HQ * D) + offs_q, qb)
    tl.store(KO + t * (HK * D) + offs_k, kb)
    slot = tl.load(LOC + t).to(tl.int64)
    if slot >= 0:
        if HAS_MAP:
            slot = tl.load(MAP + slot).to(tl.int64)
        if slot >= 0:
            v = tl.load(V + t * v_stride + offs_k)
            if HAS_SCALE:
                k_value = kb.to(tl.float32) / tl.load(KS)
                v_value = v.to(tl.float32) / tl.load(VS)
            else:
                k_value = kb.to(tl.float32)
                v_value = v.to(tl.float32)
            tl.store(KC + slot * (HK * D) + offs_k, k_value)
            tl.store(VC + slot * (HK * D) + offs_k, v_value)


@triton.jit
def _project_slice(
    P, TOKEN, FIRST, M: tl.constexpr, N: tl.constexpr, SK: tl.constexpr, D: tl.constexpr
):
    split = tl.arange(0, SK)
    col = FIRST + tl.arange(0, D)
    return tl.load(P + split[:, None] * M * N + TOKEN * N + col[None, :])


@triton.jit
def _sum_slice(partial):
    return tl.sum(partial, axis=0).to(tl.bfloat16)


# Programs [0, HQ) of a token own one Q head and its gate; programs [HQ, HQ + HK) own one K/V head and its cache
# store. Each program issues all of its global loads before any math: the kernel is latency bound at the decode /
# verify widths it serves.
@triton.jit
def _qkv_epilogue(
    P,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    QO,
    KO,
    VO,
    GO,
    KS,
    VS,
    M: tl.constexpr,
    N: tl.constexpr,
    SK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    if head < HQ:
        gate = _project_slice(P, token, (HQ + 2 * HK) * D + head * D, M, N, SK, D)
        q = _project_slice(P, token, head * D, M, N, SK, D)
        if ROPE:
            cosine, sine = _rope_table(COS, tl.load(POS + token), D, COS_STRIDE)
        tl.store(GO + token * HQ * D + head * D + d, _sum_slice(gate))
        q = _rms_scale(_sum_slice(q), EPS, SCALE, D, True)
        if ROPE:
            q = _rotate(q, cosine, sine, D)
        tl.store(QO + token * HQ * D + head * D + d, q.to(tl.bfloat16))
    else:
        kh = head - HQ
        k = _project_slice(P, token, HQ * D + kh * D, M, N, SK, D)
        v = _project_slice(P, token, (HQ + HK) * D + kh * D, M, N, SK, D)
        k_scale = tl.load(KS)
        v_scale = tl.load(VS)
        slot = tl.load(LOC + token).to(tl.int64)
        if HAS_MAP:
            slot = tl.where(
                slot >= 0, tl.load(MAP + tl.maximum(slot, 0)).to(tl.int64), slot
            )
        if ROPE:
            cosine, sine = _rope_table(COS, tl.load(POS + token), D, COS_STRIDE)
        k = _rms_scale(_sum_slice(k), EPS, 1.0, D, False)
        if ROPE:
            k = _rotate(k, cosine, sine, D)
        k = k.to(tl.bfloat16)
        v = _sum_slice(v)
        tl.store(KO + token * HK * D + kh * D + d, k)
        tl.store(VO + token * HK * D + kh * D + d, v)
        if slot >= 0:
            offset = slot * HK * D + kh * D + d
            tl.store(KC + offset, k.to(tl.float32) / k_scale)
            tl.store(VC + offset, v.to(tl.float32) / v_scale)


def qk_norm_rope_cache(
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
    """Return (normalized + rotated Q [M, HQ*64], K [M, HK*64]); K/V of each token go to the FP8 cache."""
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


def qk_norm_rope_cache_extend(
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
):
    """Extend form of ``qk_norm_rope_cache`` (one program per token, any row count): the same outputs and cache
    store, with Q / K bit-identical to ``fused_qk_norm`` followed by RoPE."""
    m = q.shape[0]
    d = 64
    hq = q.shape[1] // d
    hk = k.shape[1] // d
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
    assert cache_k.is_contiguous() and cache_v.is_contiguous()
    assert q.shape == (m, hq * d) and k.shape == v.shape == (m, hk * d)
    assert q.stride(1) == k.stride(1) == v.stride(1) == 1
    qo = torch.empty((m, hq * d), device=q.device, dtype=torch.bfloat16)
    ko = torch.empty((m, hk * d), device=q.device, dtype=torch.bfloat16)
    if m:
        _muse_qk_norm_rope_cache_tok[(m,)](
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
            float(eps),
            float(q_scale),
            hq,
            hk,
            d,
            cos_sin is not None,
            swa_mapping is not None,
            k_scale is not None,
            cos_sin.stride(0) if cos_sin is not None else d,
            num_warps=4,
        )
    return qo, ko


def qkv_proj_qk_norm_rope_cache(
    x,
    weight,
    positions,
    cos_sin,
    cache_k,
    cache_v,
    locations,
    mapping,
    eps,
    q_scale,
    k_scale,
    v_scale,
    *,
    num_heads,
    num_kv_heads,
):
    """Fused QKV + gate projection of ``x`` [M <= 32, 8192] with the QK-norm / RoPE / FP8-cache epilogue.

    The weight rows are [Q, K, V, gate] for this rank's heads (head_dim 64), e.g. N 2304 at TP8, 4608 at TP4.
    Returns (Q [M, HQ*64], K [M, HK*64], V [M, HK*64], gate [M, HQ*64]), BF16.
    """
    m, k = x.shape
    hq, hk, d = num_heads, num_kv_heads, 64
    n = (2 * hq + 2 * hk) * d
    assert 0 < m <= 32 and k == 8192 and weight.shape == (n, 8192)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
    assert cache_k.is_contiguous() and cache_v.is_contiguous()
    # Split-K 8 / block_n 32 were tuned at N 2304 (TP8); not re-tuned for N 4608.
    # Above 16 rows one 32-row tile streams the weight once instead of twice.
    partials = torch.empty((8, m, n), device=x.device, dtype=torch.float32)
    bf16_splitk_gemm_partials(
        x,
        weight,
        partials,
        block_n=32,
        block_k=128,
        split_k=8,
        block_m=16 if m <= 16 else 32,
    )
    qo = torch.empty((m, hq * d), device=x.device, dtype=torch.bfloat16)
    ko = torch.empty((m, hk * d), device=x.device, dtype=torch.bfloat16)
    vo = torch.empty_like(ko)
    go = torch.empty_like(qo)
    _qkv_epilogue[(m, hq + hk)](
        partials,
        positions,
        cos_sin,
        locations,
        mapping,
        cache_k,
        cache_v,
        qo,
        ko,
        vo,
        go,
        k_scale,
        v_scale,
        m,
        n,
        8,
        hq,
        hk,
        d,
        float(eps),
        float(q_scale),
        cos_sin is not None,
        mapping is not None,
        cos_sin.stride(0) if cos_sin is not None else 64,
        num_warps=1,
    )
    return qo, ko, vo, go
