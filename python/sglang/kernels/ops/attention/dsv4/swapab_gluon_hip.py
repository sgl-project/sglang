"""Native 16-head MFMA layouts for gfx950 paged attention."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .swapab_common import load_v4


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
    gl.static_assert(SWAP_AB and H == 16 and (BT == 64 or BT == 128))
    b, t = gl.program_id(0), gl.program_id(1)
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
    n = gl.arange(0, BT, gl.SliceLayout(1, kv_layout))
    if t < KT:
        at = t * BT + n
        length = gl.load(L + b)
        ids = gl.load(IDX + b * IS + at, at < NK, -1)
        valid = (at < NK) & (at < length) & (ids >= 0) & (ids < KTOKENS)
        kv = load_v4(
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
        kv = load_v4(
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
    scores = (
        gl.amd.cdna4.mfma(
            gl.convert_layout(kv, gl.DotOperandLayout(0, mf, 8)),
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
    value = gl.amd.cdna4.mfma(
        gl.convert_layout(kv.permute((1, 0)), gl.DotOperandLayout(0, mf, 8)),
        gl.convert_layout(hi, gl.DotOperandLayout(1, mf, 8)),
        gl.zeros((512, H), gl.float32, mf),
    )
    if COMPENSATE:
        lo = (prob - hi.to(gl.float32)).to(gl.bfloat16)
        value = gl.amd.cdna4.mfma(
            gl.convert_layout(kv.permute((1, 0)), gl.DotOperandLayout(0, mf, 8)),
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
