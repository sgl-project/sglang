# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
)

try:
    from triton.experimental.gluon.language.nvidia.blackwell import get_tmem_reg_layout
except ImportError:
    from triton.experimental.gluon.language.nvidia.blackwell import (
        tensor_memory_descriptor_type,
    )
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def get_tmem_reg_layout(element_ty, shape, layout, num_warps):
        ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
        return ty.get_reg_layout(num_warps=num_warps)


TMEM_LANES = gl.constexpr(128)


@gluon.jit
def _pdl_wait():
    gl.inline_asm_elementwise(
        "griddepcontrol.wait; // $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def _pdl_launch_dependents():
    gl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // $0",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_kv(
    CACHE,
    ids,
    valid,
    PAGE: gl.constexpr,
    STRIDE: gl.constexpr,
    D: gl.constexpr,
    ROPE: gl.constexpr,
    KV_ROW: gl.constexpr,
    KV_SROW: gl.constexpr,
    KV_TILE: gl.constexpr,
    KV_LAYOUT: gl.constexpr,
):
    NOPE: gl.constexpr = D - ROPE
    d = gl.arange(0, D, gl.SliceLayout(0, KV_LAYOUT))
    base = (ids // PAGE).to(gl.int64)[:, None] * STRIDE
    slot = (ids % PAGE)[:, None]
    mask = valid[:, None] & (d[None, :] < NOPE)
    bits = gl.load(CACHE + base + slot * KV_ROW + d[None, :], mask, 0)
    fp8 = bits.to(gl.float8e4nv, bitcast=True).to(gl.float32)
    exponent = gl.load(
        CACHE + base + PAGE * KV_ROW + slot * KV_SROW + d[None, :] // KV_TILE, mask, 0
    ).to(gl.int32)
    scale = gl.where(exponent == 0, 0x00400000, exponent << 23).to(
        gl.float32, bitcast=True
    )
    rope_ptr = (CACHE + base + slot * KV_ROW + NOPE + (d[None, :] - NOPE) * 2).to(
        gl.pointer_type(gl.bfloat16)
    )
    rope = gl.load(rope_ptr, valid[:, None] & (d[None, :] >= NOPE), 0)
    return gl.where(d[None, :] < NOPE, fp8 * scale, rope.to(gl.float32)).to(gl.bfloat16)


@gluon.jit
def _load_mla(
    CACHE,
    ids,
    valid,
    KS: gl.constexpr,
    D: gl.constexpr,
    ROPE: gl.constexpr,
    KV_TILE: gl.constexpr,
    KV_LAYOUT: gl.constexpr,
):
    d = gl.arange(0, D, gl.SliceLayout(0, KV_LAYOUT))
    r = gl.arange(0, ROPE, gl.SliceLayout(0, KV_LAYOUT))
    base = ids.to(gl.int64)[:, None] * KS
    mask = valid[:, None] & (d[None, :] < D)
    bits = gl.load(CACHE + base + d[None, :], mask, 0)
    fp8 = bits.to(gl.float8e4nv, bitcast=True).to(gl.float32)
    scale_ptr = (CACHE + base + D + (d[None, :] // KV_TILE) * 4).to(
        gl.pointer_type(gl.float32)
    )
    scale = gl.load(scale_ptr, mask, 0)
    rope_ptr = (CACHE + base + D + (D // KV_TILE) * 4 + r[None, :] * 2).to(
        gl.pointer_type(gl.bfloat16)
    )
    rope = gl.load(rope_ptr, valid[:, None] & (r[None, :] < ROPE), 0)
    return (fp8 * scale).to(gl.bfloat16), rope


@gluon.jit
def _tile_ids(
    IDX, L, b, at, STRIDE: gl.constexpr, N: gl.constexpr, TOKENS: gl.constexpr
):
    length = gl.load(L + b)
    ids = gl.load(IDX + b * STRIDE + at, at < N, -1)
    valid = (at < N) & (at < length) & (ids >= 0) & (ids < TOKENS)
    return gl.maximum(ids, 0), valid


@gluon.jit
def _is_late(EL, POS, b, C_RATIO: gl.constexpr, TOKENS_PER_REQ: gl.constexpr):
    el = gl.load(EL + b)
    return gl.load(POS + b).to(gl.int32) + 1 - el * C_RATIO <= b % TOKENS_PER_REQ


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
    FREQS,
    POS,
    SYNC,
    QR,
    QS: gl.constexpr,
    QH: gl.constexpr,
    IS: gl.constexpr,
    EIS: gl.constexpr,
    KS: gl.constexpr,
    NK: gl.constexpr,
    NE: gl.constexpr,
    NT: gl.constexpr,
    KT: gl.constexpr,
    BT: gl.constexpr,
    H: gl.constexpr,
    D: gl.constexpr,
    SCALE: gl.constexpr,
    KTOKENS: gl.constexpr,
    ROPE: gl.constexpr,
    KV_TILE: gl.constexpr,
    KP: gl.constexpr = 0,
    EP: gl.constexpr = 0,
    ES: gl.constexpr = 0,
    ETOKENS: gl.constexpr = 0,
    QEPS: gl.constexpr = 0.0,
    KV_ROW: gl.constexpr = 0,
    KV_SROW: gl.constexpr = 0,
    TILE_MODE: gl.constexpr = 0,
    C_RATIO: gl.constexpr = 0,
    TOKENS_PER_REQ: gl.constexpr = 1,
    DECOUPLE: gl.constexpr = False,
    QRS: gl.constexpr = 0,
    QRH: gl.constexpr = 0,
    MLA: gl.constexpr = False,
):
    gl.static_assert(BT == 64 or BT == 128)
    gl.static_assert(TILE_MODE == 0 or C_RATIO > 0)
    if TILE_MODE != 2:
        _pdl_launch_dependents()
    b, t = gl.program_id(0), gl.program_id(1)
    if TILE_MODE != 0:
        late = _is_late(EL, POS, b, C_RATIO, TOKENS_PER_REQ)
        last = KT + (gl.maximum(gl.load(EL + b), 1) - 1) // BT
        if TILE_MODE == 1:
            if late and t == last:
                return
        else:
            if not late:
                return
            t = last
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    n = gl.arange(0, BT, gl.SliceLayout(1, kv_layout))
    load_after_wait: gl.constexpr = TILE_MODE == 2 or MLA
    if not load_after_wait:
        if t < KT:
            valid = n < 0
            kv = gl.zeros([BT, D], gl.bfloat16, kv_layout)
        else:
            ids, valid = _tile_ids(EI, EL, b, (t - KT) * BT + n, EIS, NE, ETOKENS)
            kv = _load_kv(
                E, ids, valid, EP, ES, D, ROPE, KV_ROW, KV_SROW, KV_TILE, kv_layout
            )

    qh = gl.arange(0, H, gl.SliceLayout(1, kv_layout))
    qd = gl.arange(0, D, gl.SliceLayout(0, kv_layout))
    nvmma: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16
    )
    p_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=32, element_bitwidth=16
    )
    if not MLA:
        pos = gl.load(POS + b).to(gl.int64)
        j = qd - (D - ROPE)
        is_rope = j >= 0
        pair = j // 2
        cos_off = pos * ROPE + (2 * pair).to(gl.int64)
        cosf = gl.load(FREQS + cos_off, is_rope, 1.0)
        sinf = gl.load(FREQS + cos_off + 1, is_rope, 0.0)
        sign = gl.where(is_rope & ((j & 1) == 0), -1.0, 1.0)
    if load_after_wait:
        kv_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, D], nvmma)
    else:
        kv_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, D], nvmma, value=kv)
    if MLA:
        kr_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, ROPE], nvmma)
    score_tmem = allocate_tensor_memory(
        gl.float32, [BT, H], TensorMemoryLayout(block=(BT, H), col_stride=1)
    )
    out_tmem = allocate_tensor_memory(
        gl.float32, [D, H], TensorMemoryLayout(block=(TMEM_LANES, H), col_stride=1)
    )
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    _pdl_wait()
    if MLA:
        if t < KT:
            ids, valid = _tile_ids(IDX, L, b, t * BT + n, IS, NK, KTOKENS)
        else:
            ids, valid = _tile_ids(EI, EL, b, (t - KT) * BT + n, EIS, NE, KTOKENS)
        kv, kr = _load_mla(K, ids, valid, KS, D, ROPE, KV_TILE, kv_layout)
        kv_smem.store(kv)
        kr_smem.store(kr)
    elif TILE_MODE == 2:
        ids, valid = _tile_ids(EI, EL, b, (t - KT) * BT + n, EIS, NE, ETOKENS)
        kv = _load_kv(
            E, ids, valid, EP, ES, D, ROPE, KV_ROW, KV_SROW, KV_TILE, kv_layout
        )
        kv_smem.store(kv)
    elif t < KT:
        ids, valid = _tile_ids(IDX, L, b, t * BT + n, IS, NK, KTOKENS)
        kv = _load_kv(
            K, ids, valid, KP, KS, D, ROPE, KV_ROW, KV_SROW, KV_TILE, kv_layout
        )
        kv_smem.store(kv)

    q = gl.load(Q + b * QS + qh[:, None] * QH + qd[None, :])
    if not MLA:
        qf = q.to(gl.float32)
        ss = gl.sum(qf * qf, axis=1)
        rinv = (1.0 / gl.sqrt(ss * (1.0 / D) + QEPS))[:, None]
        even, odd = gl.split(gl.reshape(qf, [H, D // 2, 2]))
        swapped = gl.convert_layout(
            gl.reshape(gl.join(odd, even), [H, D]), qf.type.layout
        )
        qp = gl.where(is_rope[None, :], swapped, qf)
        q = (
            (qf * rinv) * cosf[None, :] + sign[None, :] * (qp * rinv) * sinf[None, :]
        ).to(gl.bfloat16)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [H, D], nvmma, value=q)
    if MLA:
        qr_d = gl.arange(0, ROPE, gl.SliceLayout(0, kv_layout))
        qr = gl.load(QR + b * QRS + qh[:, None] * QRH + qr_d[None, :])
        qr_smem = gl.allocate_shared_memory(gl.bfloat16, [H, ROPE], nvmma, value=qr)
    fence_async_shared()
    tcgen05_mma(kv_smem, q_smem.permute((1, 0)), score_tmem, use_acc=False)
    if MLA:
        tcgen05_mma(kr_smem, qr_smem.permute((1, 0)), score_tmem, use_acc=True)
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
    # Feed P as bf16 p_hi + p_lo so the PV MMA compensates P's bf16 rounding.
    p_hi = prob.to(gl.bfloat16)
    p_lo = (prob - p_hi.to(gl.float32)).to(gl.bfloat16)
    p_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, H], p_layout, value=p_hi)
    p_lo_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, H], p_layout, value=p_lo)
    fence_async_shared()
    tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=False)
    tcgen05_mma(kv_smem.permute((1, 0)), p_lo_smem, out_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=1)
    mbarrier.invalidate(bar)
    out_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32,
        (D, H),
        TensorMemoryLayout(block=(TMEM_LANES, H), col_stride=1),
        4,
    )
    value = out_tmem.load(out_layout)
    hd_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    value_hd = gl.convert_layout(value.permute((1, 0)), hd_layout)
    h = gl.arange(0, H, gl.SliceLayout(1, hd_layout))
    d = gl.arange(0, D, gl.SliceLayout(0, hd_layout))
    gl.store(PART + ((b * NT + t) * H + h[:, None]) * D + d[None, :], value_hd)
    stat_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    hs = gl.arange(0, H, stat_layout)
    gl.store(MAX + (b * NT + t) * H + hs, gl.convert_layout(mx, stat_layout))
    gl.store(SUM + (b * NT + t) * H + hs, gl.convert_layout(denom, stat_layout))
    if TILE_MODE == 2:
        _pdl_launch_dependents()
        if DECOUPLE:
            # Only late tokens get here: tell combine_push their last tile is written.
            gl.barrier()
            done = gl.load((SYNC + 2).to(gl.pointer_type(gl.int64)), volatile=True)
            seq = (done >> 20).to(gl.int32)
            gl.atomic_xchg(SYNC + 8 + b, seq + 1, sem="release", scope="gpu")
