"""Native 16-head Blackwell (tcgen05) MMA layouts for paged V4-layout attention."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    get_tmem_reg_layout,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
)


@gluon.jit
def _load_v4(
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
    # V4 row: 448 fp8 nope + 64 bf16 rope = DATA_BYTES, plus one ue8m0 scale per
    # TILE values in the page's scale rows.
    d = gl.arange(0, 512, gl.SliceLayout(0, KV_LAYOUT))
    base = (ids // PAGE).to(gl.int64)[:, None] * STRIDE
    slot = (ids % PAGE)[:, None]
    mask = valid[:, None] & (d[None, :] < 448)
    bits = gl.load(CACHE + base + slot * DATA_BYTES + d[None, :], mask, 0)
    fp8 = bits.to(gl.float8e4nv, bitcast=True).to(gl.float32)
    exponent = gl.load(
        CACHE + base + PAGE * DATA_BYTES + slot * SCALE_BYTES + d[None, :] // TILE,
        mask,
        0,
    ).to(gl.int32)
    scale = gl.where(exponent == 0, 0x00400000, exponent << 23).to(
        gl.float32, bitcast=True
    )
    rope_ptr = (CACHE + base + slot * DATA_BYTES + 448 + (d[None, :] - 448) * 2).to(
        gl.pointer_type(gl.bfloat16)
    )
    rope = gl.load(rope_ptr, valid[:, None] & (d[None, :] >= 448), 0)
    return gl.where(d[None, :] < 448, fp8 * scale, rope.to(gl.float32)).to(gl.bfloat16)


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
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    n = gl.arange(0, BT, gl.SliceLayout(1, kv_layout))
    if t < KT:
        at = t * BT + n
        length = gl.load(L + b)
        ids = gl.load(IDX + b * IS + at, at < NK, -1)
        valid = (at < NK) & (at < length) & (ids >= 0) & (ids < KTOKENS)
        kv = _load_v4(
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
        kv = _load_v4(
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
    q_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [H, 512],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16),
        value=q,
    )
    kv_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [BT, 512],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16),
        value=kv,
    )
    score_tmem = allocate_tensor_memory(
        gl.float32, [BT, H], TensorMemoryLayout(block=(BT, H), col_stride=1)
    )
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    tcgen05_mma(kv_smem, q_smem.permute((1, 0)), score_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    score_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BT, H), TensorMemoryLayout(block=(BT, H), col_stride=1), 4
    )
    scores = score_tmem.load(score_layout) * SCALE
    valid = gl.convert_layout(valid, gl.SliceLayout(1, score_layout))
    scores = gl.where(valid[:, None], scores, -float("inf"))
    mx = gl.max(scores, 0)
    mx = gl.where(mx == -float("inf"), 0.0, mx)
    prob = gl.exp(scores - mx[None, :])
    denom = gl.sum(prob, 0)
    p_hi = prob.to(gl.bfloat16)
    p_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [BT, H],
        gl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16),
        value=p_hi,
    )
    out_tmem = allocate_tensor_memory(
        gl.float32, [512, H], TensorMemoryLayout(block=(128, H), col_stride=1)
    )
    fence_async_shared()
    tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=1)
    if COMPENSATE:
        p_lo = (prob - p_hi.to(gl.float32)).to(gl.bfloat16)
        p_smem.store(p_lo)
        fence_async_shared()
        tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    out_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (512, H), TensorMemoryLayout(block=(128, H), col_stride=1), 4
    )
    value = out_tmem.load(out_layout)
    hd_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    value_hd = gl.convert_layout(value.permute((1, 0)), hd_layout)
    h = gl.arange(0, H, gl.SliceLayout(1, hd_layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, hd_layout))
    gl.store(PART + ((b * NT + t) * H + h[:, None]) * 512 + d[None, :], value_hd)
    stat_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    hs = gl.arange(0, H, stat_layout)
    gl.store(MAX + (b * NT + t) * H + hs, gl.convert_layout(mx, stat_layout))
    gl.store(SUM + (b * NT + t) * H + hs, gl.convert_layout(denom, stat_layout))
