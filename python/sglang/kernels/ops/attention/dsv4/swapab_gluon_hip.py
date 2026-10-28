"""Native 16-head MFMA layouts for gfx950 paged attention."""

import functools

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .decode_attention_sm100 import swapab_attention as _swapab_attention


@gluon.jit
def _load_v4_hip(
    CACHE,
    ids,
    valid,
    PAGE: gl.constexpr,
    STRIDE: gl.constexpr,
    KV_LAYOUT: gl.constexpr,
    DATA_BYTES: gl.constexpr,
    SCALE_BYTES: gl.constexpr,
    TILE: gl.constexpr,
):
    d = gl.arange(0, 512, gl.SliceLayout(0, KV_LAYOUT))
    base = (ids // PAGE).to(gl.int64)[:, None] * STRIDE
    slot = (ids % PAGE)[:, None]
    mask = valid[:, None] & (d[None, :] < 448)
    bits = gl.load(CACHE + base + slot * DATA_BYTES + d[None, :], mask, 0)
    exponent = gl.load(
        CACHE + base + PAGE * DATA_BYTES + slot * SCALE_BYTES + d[None, :] // TILE,
        mask,
        0,
    )
    nope = gl.amd.cdna4.scaled_upcast(
        bits.to(gl.float8e4nv, bitcast=True), exponent, gl.bfloat16
    )
    # a uint8 pointer splits the rope into 2-byte masked loads; bf16-typed it
    # stays one 16-byte load per thread
    rope_ptr = (CACHE + base + slot * DATA_BYTES + 448).to(
        gl.pointer_type(gl.bfloat16)
    ) + (d[None, :] - 448)
    rope = gl.load(rope_ptr, valid[:, None] & (d[None, :] >= 448), 0)
    return gl.where(d[None, :] < 448, nope, rope)


@gluon.jit
def partial_gluon(
    Q,
    K,
    E,
    IDX,
    EI,
    L,
    EL,
    PART,
    MAX,
    SUM,
    QS: gl.constexpr,
    QH: gl.constexpr,
    IS: gl.constexpr,
    EIS: gl.constexpr,
    KP: gl.constexpr,
    KS: gl.constexpr,
    EP: gl.constexpr,
    ES: gl.constexpr,
    NK: gl.constexpr,
    NE: gl.constexpr,
    NT: gl.constexpr,
    KT: gl.constexpr,
    BT: gl.constexpr,
    H: gl.constexpr,
    SCALE: gl.constexpr,
    KTOKENS: gl.constexpr,
    ETOKENS: gl.constexpr,
    COMPENSATE: gl.constexpr,
    SWAP_AB: gl.constexpr,
    DATA_BYTES: gl.constexpr,
    SCALE_BYTES: gl.constexpr,
    TILE: gl.constexpr,
):
    gl.static_assert(SWAP_AB and H == 16 and (BT == 32 or BT == 64 or BT == 128))
    b, t = gl.program_id(0), gl.program_id(1)
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
    n = gl.arange(0, BT, gl.SliceLayout(1, kv_layout))
    if t < KT:
        at = t * BT + n
        length = gl.load(L + b)
        ids = gl.load(IDX + b * IS + at, at < NK, -1)
        valid = (at < NK) & (at < length) & (ids >= 0) & (ids < KTOKENS)
        kv = _load_v4_hip(
            K,
            gl.maximum(ids, 0),
            valid,
            KP,
            KS,
            kv_layout,
            DATA_BYTES,
            SCALE_BYTES,
            TILE,
        )
    else:
        at = (t - KT) * BT + n
        length = gl.load(EL + b)
        ids = gl.load(EI + b * EIS + at, at < NE, -1)
        valid = (at < NE) & (at < length) & (ids >= 0) & (ids < ETOKENS)
        kv = _load_v4_hip(
            E,
            gl.maximum(ids, 0),
            valid,
            EP,
            ES,
            kv_layout,
            DATA_BYTES,
            SCALE_BYTES,
            TILE,
        )
    qh = gl.arange(0, H, gl.SliceLayout(1, kv_layout))
    qd = gl.arange(0, 512, gl.SliceLayout(0, kv_layout))
    q = gl.load(Q + b * QS + qh[:, None] * QH + qd[None, :])
    # transposed=False: a lane holds 4 consecutive d of one head in the PV result,
    # so `value` stores to PART [h, d] without an LDS relayout
    mf: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[4, 1]
    )
    # every warp holds all BT tokens, and k_width 4 is the accumulator's token
    # order, so P feeds the PV MFMA from registers
    mp: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, 4]
    )
    kv_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [BT, 512],
        gl.PaddedSharedLayout.with_identity_for([[128, 8]], [BT, 512], [1, 0]),
        kv,
    )
    scores = (
        gl.amd.cdna4.mfma(
            kv_smem.load(gl.DotOperandLayout(0, mf, 8)),
            gl.convert_layout(q.permute((1, 0)), gl.DotOperandLayout(1, mf, 8)),
            gl.zeros((BT, H), gl.float32, mf),
        )
        * SCALE
    )
    valid = gl.convert_layout(valid, gl.SliceLayout(1, mf))
    scores = gl.convert_layout(gl.where(valid[:, None], scores, -float("inf")), mp)
    mx = gl.max(scores, 0)
    mx = gl.where(mx == -float("inf"), 0.0, mx)
    prob = gl.exp(scores - mx[None, :])
    denom = gl.sum(prob, 0)
    hi = prob.to(gl.bfloat16)
    kv_t = kv_smem.permute((1, 0))
    value = gl.amd.cdna4.mfma(
        kv_t.load(gl.DotOperandLayout(0, mf, 4)),
        gl.convert_layout(hi, gl.DotOperandLayout(1, mf, 4), assert_trivial=True),
        gl.zeros((512, H), gl.float32, mf),
    )
    if COMPENSATE:
        lo = (prob - hi.to(gl.float32)).to(gl.bfloat16)
        value = gl.amd.cdna4.mfma(
            kv_t.load(gl.DotOperandLayout(0, mf, 4)),
            gl.convert_layout(lo, gl.DotOperandLayout(1, mf, 4), assert_trivial=True),
            value,
        )
    d = gl.arange(0, 512, gl.SliceLayout(1, mf))
    h = gl.arange(0, H, gl.SliceLayout(0, mf))
    gl.store(PART + ((b * NT + t) * H + h[None, :]) * 512 + d[:, None], value)
    hs = gl.arange(0, H, gl.SliceLayout(0, mp))
    gl.store(MAX + (b * NT + t) * H + hs, mx)
    gl.store(SUM + (b * NT + t) * H + hs, denom)


@triton.jit
def rope_tables(
    d, FREQS, pos, freq_stride, HEAD_DIM: tl.constexpr, ROPE_DIM: tl.constexpr
):
    is_rope = d >= HEAD_DIM - ROPE_DIM
    freq_index = ((d - (HEAD_DIM - ROPE_DIM)) // 2) * 2
    cos = tl.load(FREQS + pos * freq_stride + freq_index, is_rope, 0)
    sin = tl.load(FREQS + pos * freq_stride + freq_index + 1, is_rope, 0)
    return cos, sin


@triton.jit
def inverse_rope_apply(
    out, d, cos, sin, HEAD_DIM: tl.constexpr, ROPE_DIM: tl.constexpr
):
    """Rotates the rounded values, as the model's standalone RoPE kernel reads them."""
    x = out.to(tl.float32)
    is_rope = d >= HEAD_DIM - ROPE_DIM
    signed = tl.where(d % 2 == 0, -x * sin, x * sin)
    swapped = tl.reshape(tl.flip(tl.reshape(signed, (d.shape[0] // 2, 2)), 1), d.shape)
    rotated = tl.fma(x, cos, swapped)
    return tl.where(is_rope, rotated.to(out.dtype), out)


@triton.jit
def combine_gfx950(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    FREQS,
    POSITIONS,
    FREQ_STRIDE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NT: tl.constexpr,
    ST: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t, d = tl.arange(0, ST), tile * BD + tl.arange(0, BD)
    # issued before the tl.max reduce: Triton does not hoist loads across it
    sink = tl.load(SINK + h)
    if ROPE_DIM > 0:
        pos = tl.load(POSITIONS + b)
        cos, sin = rope_tables(d, FREQS, pos, FREQ_STRIDE, 512, ROPE_DIM)
    den = tl.load(SUM + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.load(MAX + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.where(den > 0, mx, -float("inf"))
    m = tl.maximum(tl.max(mx, 0), sink)
    m = tl.where(tl.abs(m) == float("inf"), 0.0, m)
    factor = tl.exp(mx - m)
    denominator = tl.sum(den * factor, 0) + tl.exp(sink - m)
    vals = tl.load(
        PART + ((b * NT + t[:, None]) * H + h) * 512 + d[None, :],
        t[:, None] < NT,
        0,
    )
    out = tl.sum(vals * factor[:, None], 0) / denominator
    out = tl.where((denominator > 0) & (sink != float("inf")), out, 0.0)
    out = out.to(OUT.dtype.element_ty)
    if ROPE_DIM > 0:
        out = inverse_rope_apply(out, d, cos, sin, 512, ROPE_DIM)
    tl.store(OUT + (b * H + h) * 512 + d, out)


swapab_attention = functools.partial(
    _swapab_attention,
    partial_gluon=partial_gluon,
    combine=combine_gfx950,
    block=32,
)
