"""Native 16-head MFMA layouts for gfx950 paged attention."""

import functools

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
    mf: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[4, 1]
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
    scores = gl.where(valid[:, None], scores, -float("inf"))
    mx = gl.max(scores, 0)
    mx = gl.where(mx == -float("inf"), 0.0, mx)
    prob = gl.exp(scores - mx[None, :])
    denom = gl.sum(prob, 0)
    hi = prob.to(gl.bfloat16)
    kv_t = kv_smem.permute((1, 0))
    value = gl.amd.cdna4.mfma(
        kv_t.load(gl.DotOperandLayout(0, mf, 8)),
        gl.convert_layout(hi, gl.DotOperandLayout(1, mf, 8)),
        gl.zeros((512, H), gl.float32, mf),
    )
    if COMPENSATE:
        lo = (prob - hi.to(gl.float32)).to(gl.bfloat16)
        value = gl.amd.cdna4.mfma(
            kv_t.load(gl.DotOperandLayout(0, mf, 8)),
            gl.convert_layout(lo, gl.DotOperandLayout(1, mf, 8)),
            value,
        )
    hd: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
    value = gl.convert_layout(value.permute((1, 0)), hd)
    h = gl.arange(0, H, gl.SliceLayout(1, hd))
    d = gl.arange(0, 512, gl.SliceLayout(0, hd))
    gl.store(PART + ((b * NT + t) * H + h[:, None]) * 512 + d[None, :], value)
    stat: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    hs = gl.arange(0, H, stat)
    gl.store(MAX + (b * NT + t) * H + hs, gl.convert_layout(mx, stat))
    gl.store(SUM + (b * NT + t) * H + hs, gl.convert_layout(denom, stat))


# the SM100 host path (sink, combine, inverse RoPE) on the gfx950 split-KV kernel above
swapab_attention = functools.partial(
    _swapab_attention, partial_gluon=partial_gluon, block=32
)
