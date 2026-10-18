"""BF16 Muse QKV/gate projection with the existing QK/RoPE/cache epilogue.

Two ordered launches: BF16 split-K partials, then reduction + normalization +
legacy interleaved RoPE + FP8 writes. Preserves the projection BF16 round point.
"""

import torch
import triton as tr
import triton.language as tl

from sglang.srt.rocm_muse_spark.dense_kernels import _dense_partial
from sglang.srt.rocm_muse_spark.muse_qk_rope_cache import _norm_rope


@tr.jit
def _project_slice(
    P, TOKEN, FIRST, M: tl.constexpr, N: tl.constexpr, SK: tl.constexpr, D: tl.constexpr
):
    split = tl.arange(0, SK)
    col = FIRST + tl.arange(0, D)
    partial = tl.load(P + split[:, None] * M * N + TOKEN * N + col[None, :])
    return tl.sum(partial, axis=0).to(tl.bfloat16)


@tr.jit
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
    gate = _project_slice(P, token, (HQ + 2 * HK) * D + head * D, M, N, SK, D)
    tl.store(GO + token * HQ * D + head * D + d, gate)
    q = _project_slice(P, token, head * D, M, N, SK, D)
    q = _norm_rope(q, COS, POS, EPS, SCALE, D, ROPE, True, COS_STRIDE)
    tl.store(QO + token * HQ * D + head * D + d, q)
    if head % (HQ // HK) == 0:
        kh = head // (HQ // HK)
        k = _project_slice(P, token, HQ * D + kh * D, M, N, SK, D)
        v = _project_slice(P, token, (HQ + HK) * D + kh * D, M, N, SK, D)
        k = _norm_rope(k, COS, POS, EPS, 1.0, D, ROPE, False, COS_STRIDE)
        tl.store(KO + token * HK * D + kh * D + d, k)
        tl.store(VO + token * HK * D + kh * D + d, v)
        slot = tl.load(LOC + token).to(tl.int64)
        if slot >= 0:
            if HAS_MAP:
                slot = tl.load(MAP + slot).to(tl.int64)
            if slot >= 0:
                offset = slot * HK * D + kh * D + d
                tl.store(KC + offset, k.to(tl.float32) / tl.load(KS))
                tl.store(VC + offset, v.to(tl.float32) / tl.load(VS))


def project_qkv_cache(
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
):
    m, k = x.shape
    assert 0 < m <= 32 and k == 8192 and weight.shape == (2304, 8192)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
    assert cache_k.is_contiguous() and cache_v.is_contiguous()
    partials = torch.empty((8, m, 2304), device=x.device, dtype=torch.float32)
    _dense_partial[(72, tr.cdiv(m, 16), 8)](
        x,
        weight,
        partials,
        m,
        2304,
        8192,
        16,
        32,
        128,
        8,
        ".cg",
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    qo = torch.empty((m, 1024), device=x.device, dtype=torch.bfloat16)
    ko = torch.empty((m, 128), device=x.device, dtype=torch.bfloat16)
    vo = torch.empty_like(ko)
    go = torch.empty_like(qo)
    _qkv_epilogue[(m, 16)](
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
        2304,
        8,
        16,
        2,
        64,
        float(eps),
        float(q_scale),
        cos_sin is not None,
        mapping is not None,
        cos_sin.stride(0) if cos_sin is not None else 64,
        num_warps=1,
    )
    return qo, ko, vo, go
