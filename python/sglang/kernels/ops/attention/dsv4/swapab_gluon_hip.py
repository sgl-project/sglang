"""Native 16-head MFMA layouts for gfx950 paged attention."""

import functools

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .decode_attention_sm100 import swapab_attention as _swapab_attention

# router row cap
MAX_BATCH = 16


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


# a buffer descriptor's num_records (Triton sets 2**31 - 2 bytes); a cache past it
# keeps the global-load path, whose offsets are 64-bit
BUFFER_BYTES = gl.constexpr(2**31 - 2)


@gluon.jit
def _load_v4_buffer(
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
    # an invalid row's offsets start at 2**31, past num_records, so the hardware reads
    # 0. Columns >= 448 of the fp8 and scale loads read the row's own rope and pad
    # bytes, which the final select drops.
    d = gl.arange(0, 512, gl.SliceLayout(0, KV_LAYOUT))
    page = (ids // PAGE) * STRIDE
    slot = ids % PAGE
    row = gl.where(valid, page + slot * DATA_BYTES, -(2**31))[:, None]
    bits = gl.amd.cdna4.buffer_load(CACHE, row + d[None, :])
    scales = gl.where(valid, page + PAGE * DATA_BYTES + slot * SCALE_BYTES, -(2**31))
    exponent = gl.amd.cdna4.buffer_load(CACHE, scales[:, None] + d[None, :] // TILE)
    nope = gl.amd.cdna4.scaled_upcast(
        bits.to(gl.float8e4nv, bitcast=True), exponent, gl.bfloat16
    )
    # a masked-off vector also reads 0: with `other` unset Triton adds no select
    rope = gl.amd.cdna4.buffer_load(
        CACHE.to(gl.pointer_type(gl.bfloat16)),
        (row >> 1) + (d[None, :] - 224),
        (d >= 448)[None, :],
    )
    return gl.where(d[None, :] < 448, nope, rope)


@gluon.jit
def _load_kv(
    CACHE,
    ids,
    valid,
    TOKENS: gl.constexpr,
    PAGE: gl.constexpr,
    STRIDE: gl.constexpr,
    KV_LAYOUT: gl.constexpr,
    DATA_BYTES: gl.constexpr,
    SCALE_BYTES: gl.constexpr,
    TILE: gl.constexpr,
):
    # the rope offsets count bf16 elements, so a row must start on an even byte
    if TOKENS // PAGE * STRIDE <= BUFFER_BYTES and STRIDE % 2 == 0:
        return _load_v4_buffer(
            CACHE, ids, valid, PAGE, STRIDE, KV_LAYOUT, DATA_BYTES, SCALE_BYTES, TILE
        )
    else:
        return _load_v4_hip(
            CACHE, ids, valid, PAGE, STRIDE, KV_LAYOUT, DATA_BYTES, SCALE_BYTES, TILE
        )


@gluon.jit
def _load_q(
    Q, QS: gl.constexpr, QH: gl.constexpr, H: gl.constexpr, Q_LAYOUT: gl.constexpr
):
    qh = gl.arange(0, H, gl.SliceLayout(1, Q_LAYOUT))
    qd = gl.arange(0, 512, gl.SliceLayout(0, Q_LAYOUT))
    return gl.load(Q + gl.program_id(0) * QS + qh[:, None] * QH + qd[None, :])


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
    # warps split the rows, so each register's columns, and with them the fp8 or
    # rope side of 448, are known at compile time and the gather has no dead loads
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    q_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
    n = gl.arange(0, BT, gl.SliceLayout(1, kv_layout))
    side_k = t < KT
    at = gl.where(side_k, t, t - KT) * BT + n
    row = gl.where(side_k, IDX, EI) + b * gl.where(side_k, IS, EIS)
    if NK % BT == 0 and NE % BT == 0:
        ids = gl.amd.cdna4.buffer_load(row, at)
    else:
        ids = gl.amd.cdna4.buffer_load(row, at, at < gl.where(side_k, NK, NE))
    length = gl.load(gl.where(side_k, L, EL) + b)
    # opaque copy of t: stops the scheduler from hoisting the K/E side constants
    # above the ids and length loads
    tk = gl.inline_asm_elementwise(
        "s_mov_b32 $0, $1", "=s,s", [t], dtype=gl.int32, is_pure=False, pack=1
    )
    q = _load_q(Q, QS, QH, H, q_layout)
    if tk < KT:
        valid = (at < NK) & (at < length) & (ids >= 0) & (ids < KTOKENS)
        kv = _load_kv(
            K,
            gl.maximum(ids, 0),
            valid,
            KTOKENS,
            KP,
            KS,
            kv_layout,
            DATA_BYTES,
            SCALE_BYTES,
            TILE,
        )
    else:
        valid = (at < NE) & (at < length) & (ids >= 0) & (ids < ETOKENS)
        kv = _load_kv(
            E,
            gl.maximum(ids, 0),
            valid,
            ETOKENS,
            EP,
            ES,
            kv_layout,
            DATA_BYTES,
            SCALE_BYTES,
            TILE,
        )
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
    q_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [H, 512],
        gl.PaddedSharedLayout.with_identity_for([[128, 8]], [H, 512], [1, 0]),
        q,
    )
    scores = (
        gl.amd.cdna4.mfma(
            kv_smem.load(gl.DotOperandLayout(0, mf, 8)),
            q_smem.permute((1, 0)).load(gl.DotOperandLayout(1, mf, 8)),
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
    ROPE_TILES: tl.constexpr = (512 - ROPE_DIM) % BD == 0
    # issued before the tl.max reduce: Triton does not hoist loads across it
    sink = tl.load(SINK + h)
    if ROPE_DIM > 0:
        pos = tl.load(POSITIONS + b)
        cos, sin = rope_tables(d, FREQS, pos, FREQ_STRIDE, 512, ROPE_DIM)
        if ROPE_TILES:
            signed_sin = tl.where(d % 2 == 0, -sin, sin)
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
        if ROPE_TILES:
            if tile * BD >= 512 - ROPE_DIM:
                x = out.to(tl.float32)
                swapped = tl.reshape(
                    tl.flip(tl.reshape(x * signed_sin, (BD // 2, 2)), 1), d.shape
                )
                out = tl.fma(x, cos, swapped).to(out.dtype)
        else:
            out = inverse_rope_apply(out, d, cos, sin, 512, ROPE_DIM)
    tl.store(OUT + (b * H + h) * 512 + d, out)


swapab_attention = functools.partial(
    _swapab_attention,
    partial_gluon=partial_gluon,
    combine=combine_gfx950,
    block=32,
)
