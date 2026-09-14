"""Small-batch MLA for 16 TP-local heads and the 584-byte paged KV layout."""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _load_kv(
    CACHE,
    ids,
    valid,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    FORMAT: tl.constexpr,
):
    d = tl.arange(0, D)
    page = ids // PAGE
    offset = ids % PAGE
    base = page[:, None].to(tl.int64) * STRIDE
    tl.static_assert(FORMAT == 584)
    live = valid[:, None] & (d[None, :] < 448)
    v = (
        tl.load(CACHE + base + offset[:, None] * 576 + d[None, :], live, 0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    sf = tl.load(
        CACHE + base + PAGE * 576 + offset[:, None] * 8 + d[None, :] // 64, live, 0
    ).to(tl.int32)
    scale = tl.where(sf == 0, 0x00400000, sf << 23).to(tl.float32, bitcast=True)
    rope_ptr = (CACHE + base + offset[:, None] * 576 + 448 + (d[None, :] - 448) * 2).to(
        tl.pointer_type(tl.bfloat16)
    )
    rope = tl.load(rope_ptr, valid[:, None] & (d[None, :] >= 448), 0.0).to(tl.float32)
    value = tl.where(d[None, :] < 448, v * scale, rope)
    return value.to(tl.bfloat16)


@triton.jit
def _small_paged_attention_partial(
    Q,
    K,
    E,
    I,
    EI,
    L,
    EL,
    P,
    MAX,
    SUM,
    QS: tl.constexpr,
    QH: tl.constexpr,
    IS: tl.constexpr,
    EIS: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    EP: tl.constexpr,
    ES: tl.constexpr,
    NK: tl.constexpr,
    NE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    KT: tl.constexpr,
    BT: tl.constexpr,
    SCALE: tl.constexpr,
    KFMT: tl.constexpr,
    EFMT: tl.constexpr,
    KTOKENS: tl.constexpr,
    ETOKENS: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    b, t = tl.program_id(0), tl.program_id(1)
    n = tl.arange(0, BT)
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    q = tl.load(Q + b * QS + h[:, None] * QH + d[None, :])
    if t < KT:
        at = t * BT + n
        ln = tl.load(L + b)
        ids = tl.load(I + b * IS + at, at < NK, -1)
        valid = (at < NK) & (at < ln) & (ids >= 0) & (ids < KTOKENS)
        kv = _load_kv(K, tl.maximum(ids, 0), valid, D, KP, KS, KFMT)
    else:
        at = (t - KT) * BT + n
        ln = tl.load(EL + b)
        ids = tl.load(EI + b * EIS + at, at < NE, -1)
        valid = (at < NE) & (at < ln) & (ids >= 0) & (ids < ETOKENS)
        kv = _load_kv(E, tl.maximum(ids, 0), valid, D, EP, ES, EFMT)
    score = tl.dot(q, tl.trans(kv)) * SCALE
    score = tl.where(valid[None, :], score, -float("inf"))
    maximum = tl.max(score, 1)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    prob = tl.exp(score - maximum[:, None])
    denom = tl.sum(prob, 1)
    accum = tl.dot(prob.to(tl.bfloat16), kv)
    tl.store(P + ((b * NT + t) * H + h[:, None]) * D + d[None, :], accum)
    tl.store(MAX + (b * NT + t) * H + h, maximum)
    tl.store(SUM + (b * NT + t) * H + h, denom)
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _small_paged_attention_combine(
    P,
    MAX,
    SUM,
    SINK,
    O,
    F,
    POS,
    INVERSE_ROPE: tl.constexpr,
    NT: tl.constexpr,
    ST: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    b, h = tl.program_id(0), tl.program_id(1)
    t = tl.arange(0, ST)
    d = tl.program_id(2) * BD + tl.arange(0, BD)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    den = tl.load(SUM + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.load(MAX + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.where(den > 0, mx, -float("inf"))
    sink = tl.load(SINK + h)
    global_max = tl.maximum(tl.max(mx, 0), sink)
    # A +inf sink defines a zero output; avoid inf-inf while evaluating.
    finite_max = tl.where(tl.abs(global_max) == float("inf"), 0.0, global_max)
    factor = tl.exp(mx - finite_max)
    total = tl.sum(den * factor, 0) + tl.exp(sink - finite_max)
    val = tl.load(
        P + ((b * NT + t[:, None]) * H + h) * D + d[None, :], t[:, None] < NT, 0
    )
    result = tl.sum(val * factor[:, None], 0) / total
    result = tl.where((total > 0) & (sink != float("inf")), result, 0.0)
    # Preserve the BF16 attention-output rounding before inverse RoPE.
    rounded = result.to(tl.bfloat16).to(tl.float32)
    if INVERSE_ROPE:
        partner = tl.gather(rounded, tl.arange(0, BD) ^ 1, 0)
        position = tl.load(POS + b).to(tl.int64)
        freq = position * 64 + (d - 448) // 2 * 2
        cos = tl.load(F + freq, d >= 448, 0)
        sin = tl.load(F + freq + 1, d >= 448, 0)
        even = tl.fma(rounded, cos, partner * sin)
        odd = tl.fma(-partner, sin, rounded * cos)
        rotated = tl.where((d & 1) == 0, even, odd)
        rounded = tl.where(d >= 448, rotated, rounded)
    tl.store(O + (b * H + h) * D + d, rounded.to(tl.bfloat16))
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def small_paged_attention(
    q,
    kv,
    indices,
    lengths,
    sink,
    extra_kv=None,
    extra_indices=None,
    extra_lengths=None,
    *,
    inverse_rope=None,
):
    # The model only consumes the attention output, so no LSE is materialized.
    block, warps, combine_warps = 32, 4, 4
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    b = q.shape[0]
    h = 16
    d = 512
    assert 0 < b <= 8 and q.shape[-2] >= h and q.shape[-1] == d
    assert q.ndim == 4 and q.shape[1] == 1 and q.stride(-1) == 1
    assert q.dtype == torch.bfloat16
    assert kv.dtype in (torch.uint8, torch.float8_e4m3fn)
    # The pool exposes the packed byte storage through its configured FP8
    # dtype. Reinterpret the bytes; do not numerically cast the packed data.
    kv = kv.view(torch.uint8)
    assert kv.shape[-1] == 584
    assert indices.stride(-1) == 1 and lengths.is_contiguous()
    assert sink.stride(0) == 1
    k = indices.shape[-1]
    ek = extra_indices.shape[-1] if extra_indices is not None else 0
    # For mixed SWA + compressed attention, smaller output tiles avoid the
    # register pressure of reducing all 512 channels in one CTA.
    combine_block = 64 if ek else 512
    assert 0 < k <= 192 and 0 <= ek <= 1024
    kt = triton.cdiv(k, block)
    nt = kt + triton.cdiv(ek, block)
    partial = torch.empty((b, nt, h, d), device=q.device, dtype=torch.float32)
    maximum = torch.empty((b, nt, h), device=q.device, dtype=torch.float32)
    sums = torch.empty_like(maximum)
    out = torch.empty((b, h, d), device=q.device, dtype=torch.bfloat16)
    if extra_kv is None:
        extra_kv = kv
        extra_indices = indices
        extra_lengths = lengths
    else:
        assert extra_kv.shape[-1] == 584
        assert extra_kv.dtype in (torch.uint8, torch.float8_e4m3fn)
        extra_kv = extra_kv.view(torch.uint8)
        assert extra_indices.stride(-1) == 1 and extra_lengths.is_contiguous()
    _small_paged_attention_partial[(b, nt)](
        q,
        kv,
        extra_kv,
        indices,
        extra_indices,
        lengths,
        extra_lengths,
        partial,
        maximum,
        sums,
        QS=q.stride(0),
        QH=q.stride(-2),
        IS=indices.stride(0),
        EIS=extra_indices.stride(0),
        KP=kv.shape[1],
        KS=kv.stride(0),
        EP=extra_kv.shape[1],
        ES=extra_kv.stride(0),
        NK=k,
        NE=ek,
        H=h,
        D=d,
        NT=nt,
        KT=kt,
        BT=block,
        SCALE=512**-0.5,
        KFMT=kv.shape[-1],
        EFMT=extra_kv.shape[-1],
        KTOKENS=kv.shape[0] * kv.shape[1],
        ETOKENS=extra_kv.shape[0] * extra_kv.shape[1],
        num_warps=warps,
        **pdl_kwargs,
    )
    freq, pos = sink, lengths
    if inverse_rope is not None:
        freqs_cis, pos = inverse_rope
        assert freqs_cis.dtype == torch.complex64 and freqs_cis.is_contiguous()
        assert freqs_cis.shape[1] == 32 and pos.shape == (b,)
        assert pos.is_contiguous() and pos.dtype in (torch.int32, torch.int64)
        freq = torch.view_as_real(freqs_cis)
    _small_paged_attention_combine[(b, h, triton.cdiv(d, combine_block))](
        partial,
        maximum,
        sums,
        sink,
        out,
        freq,
        pos,
        INVERSE_ROPE=inverse_rope is not None,
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=h,
        D=d,
        BD=combine_block,
        num_warps=combine_warps,
        **pdl_kwargs,
    )
    return out
