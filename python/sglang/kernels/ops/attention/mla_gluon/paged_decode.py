"""Generated Kimi-K3 MLA paged decode attention kernels for gfx950.

Source: OpenAI-Partners/artemis-kernel-integrations PR 17,
commit 35b249f7a551278946a81b7da1d58c286c41fb8f.
"""

# ruff: noqa
# fmt: off

"""Selected paged attention decode schedules and shared helpers."""

import torch
import triton
from triton.language.core import range as loop_range
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _m1_softmax_exp(x):
    return gl.exp2(x * 1.4426950408889634)


@gluon.jit
def _m1_compensated_pv(high, low, values, acc):

    correction = gl.amd.cdna4.mfma(low, values, gl.full(acc.shape, 0.0, gl.float32, acc.type.layout))
    acc = gl.amd.cdna4.mfma(high, values, acc)
    return acc + correction * (1.0 / 65536.0)


@gluon.jit
def _m1_consume_tile(Q, K, V, Indices, scale, row, value_block, block_start, begin, length, maximum, denom, acc, HEADS: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, BLOCK: gl.constexpr, VD: gl.constexpr, FIRST: gl.constexpr):
    mat: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4 if BLOCK <= 64 else 8])
    if VD <= 256:
        softmax_layout: gl.constexpr = gl.BlockedLayout([1, 4 if VD == 256 else 2], [4, 16], [4, 1], [1, 0])
    elif BLOCK == 64:
        softmax_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    else:
        softmax_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [8, 1], [1, 0])
    DIRECT: gl.constexpr = BLOCK == 64
    qa: gl.constexpr = gl.DotOperandLayout(0, mat, 16)
    kb: gl.constexpr = gl.DotOperandLayout(1, mat, 16)
    PV: gl.constexpr = 4 if VD == 256 else 8
    pa: gl.constexpr = gl.DotOperandLayout(0, mat, PV)
    vb: gl.constexpr = gl.DotOperandLayout(1, mat, PV)
    qlayout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1 if BLOCK <= 64 else 2, 4], [1, 0])
    if BLOCK == 64:
        klayout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
    else:
        klayout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [8, 1], [1, 0])
    TWO_PANELS: gl.constexpr = VD == 128 or (VD == 512 and BLOCK == 128)
    FOUR_PANELS: gl.constexpr = VD == 256 or (VD == 512 and BLOCK == 64)
    STAGE: gl.constexpr = VD // (4 if FOUR_PANELS else 2 if TWO_PANELS else 1)
    PHASES: gl.constexpr = 1 if VD == 128 else 2 if VD == 256 else 8
    vlayout: gl.constexpr = gl.BlockedLayout([1, 16], [1024 // STAGE, STAGE // 16], [4 if BLOCK <= 64 else 8, 1], [1, 0])
    tk = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, klayout))
    tm = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, softmax_layout))
    dv = value_block * VD + gl.arange(0, STAGE, layout=gl.SliceLayout(0, vlayout))
    pos = block_start + tk
    slot = gl.load(Indices + begin + pos, pos < length, 0).to(gl.int32)
    vpos = block_start + gl.arange(0, BLOCK, layout=gl.SliceLayout(1, vlayout))
    if VD >= 128:
        vslot = gl.load(Indices + begin + vpos, vpos < length, 0).to(gl.int32)
    else:
        vslot = gl.convert_layout(slot, gl.SliceLayout(1, vlayout))
    hq = gl.arange(0, 16, layout=gl.SliceLayout(1, qlayout))
    dq = gl.arange(0, 512, layout=gl.SliceLayout(0, qlayout))
    if VD == 64:
        q = gl.amd.cdna4.buffer_load(Q + row * QS0, hq[:, None] * QS1 + dq[None, :], hq[:, None] < HEADS, 0)
    else:
        q = gl.load(Q + row * QS0 + hq[:, None] * QS1 + dq[None, :], hq[:, None] < HEADS, 0)
    rope_heads = gl.arange(0, 16, layout=gl.SliceLayout(1, qa))
    rope_dims = gl.arange(0, 64, layout=gl.SliceLayout(0, qa))
    qr = gl.amd.cdna4.buffer_load(Q + row * QS0, rope_heads[:, None] * QS1 + 512 + rope_dims[None, :], rope_heads[:, None] < HEADS, 0)
    q = gl.convert_layout(q, qa)
    if DIRECT:
        key_slot = gl.convert_layout(slot, gl.SliceLayout(0, kb))
        dk_direct = gl.arange(0, 512, layout=gl.SliceLayout(1, kb))
        k_operand = gl.load(K + dk_direct[:, None] + key_slot[None, :] * KS0)
        rk_direct = gl.arange(0, 64, layout=gl.SliceLayout(1, kb))
        kr_operand = gl.load(K + 512 + rk_direct[:, None] + key_slot[None, :] * KS0)
    else:
        dk = gl.arange(0, 512, layout=gl.SliceLayout(0, klayout))
        k = gl.amd.cdna4.buffer_load(K, slot[:, None] * KS0 + dk[None, :])
        rk = gl.arange(0, 64, layout=gl.SliceLayout(0, klayout))
        kr = gl.amd.cdna4.buffer_load(K, slot[:, None] * KS0 + 512 + rk[None, :])
    v_smem = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK, STAGE), gl.SwizzledSharedLayout(16, 1, PHASES, [1, 0]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem, V, vslot[:, None] * VS0 + dv[None, :], mask=vpos[:, None] < length, cache_modifier='' if BLOCK <= 64 else '.cg')
    if TWO_PANELS or FOUR_PANELS:
        v_smem_right = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK, STAGE), gl.SwizzledSharedLayout(16, 1, PHASES, [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem_right, V, vslot[:, None] * VS0 + dv[None, :] + STAGE, mask=vpos[:, None] < length, cache_modifier='' if BLOCK <= 64 else '.cg')
    if FOUR_PANELS:
        v_smem_2 = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK, STAGE), gl.SwizzledSharedLayout(16, 1, PHASES, [1, 0]))
        v_smem_3 = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK, STAGE), gl.SwizzledSharedLayout(16, 1, PHASES, [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem_2, V, vslot[:, None] * VS0 + dv[None, :] + 2 * STAGE, mask=vpos[:, None] < length, cache_modifier='' if BLOCK <= 64 else '.cg')
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem_3, V, vslot[:, None] * VS0 + dv[None, :] + 3 * STAGE, mask=vpos[:, None] < length, cache_modifier='' if BLOCK <= 64 else '.cg')
    gl.amd.cdna4.async_copy.commit_group()
    if DIRECT:
        scores = gl.amd.cdna4.mfma(qr, kr_operand.to(gl.bfloat16), gl.full((16, BLOCK), 0.0, gl.float32, mat))
    else:
        scores = gl.amd.cdna4.mfma(qr, gl.convert_layout(kr.T, kb).to(gl.bfloat16), gl.full((16, BLOCK), 0.0, gl.float32, mat))
    if DIRECT:
        scores = gl.amd.cdna4.mfma(q, k_operand.to(gl.bfloat16), scores) * scale
    else:
        scores = gl.amd.cdna4.mfma(q, gl.convert_layout(k.T, kb).to(gl.bfloat16), scores) * scale
    scores = gl.convert_layout(scores, softmax_layout)
    scores = gl.where(block_start + tm[None, :] < length, scores, -float('inf'))
    if FIRST:
        new_max = gl.max(scores, 1)
    else:
        new_max = gl.maximum(maximum, gl.max(scores, 1))
        alpha = _m1_softmax_exp(maximum - new_max)
    p = gl.where(block_start + tm[None, :] < length, _m1_softmax_exp(scores - new_max[:, None]), 0.0)
    if FIRST:
        denom = gl.sum(p, 1)
        acc = gl.full((16, VD), 0.0, gl.float32, mat)
    else:
        denom = denom * alpha + gl.sum(p, 1)
        alpha_mat = gl.convert_layout(alpha, gl.SliceLayout(1, mat))
        acc = acc * alpha_mat[:, None]
    p_high = p.to(gl.float16)
    p_low = ((p - p_high.to(gl.float32)) * 65536.0).to(gl.float16)
    prob_vec: gl.constexpr = 16 if BLOCK == 128 else 4 if VD == 256 else 8
    prob_phases: gl.constexpr = 16 if VD == 64 or VD == 256 else 8
    p_shared_high = gl.allocate_shared_memory(gl.float16, (16, BLOCK), gl.SwizzledSharedLayout(prob_vec, 1, prob_phases, [1, 0]))
    p_shared_low = gl.allocate_shared_memory(gl.float16, (16, BLOCK), gl.SwizzledSharedLayout(prob_vec, 1, prob_phases, [1, 0]))
    p_shared_high.store(p_high)
    p_shared_low.store(p_low)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    if FOUR_PANELS:
        high = gl.amd.cdna4.async_copy.load_shared_relaxed(p_shared_high, pa)
        low = gl.amd.cdna4.async_copy.load_shared_relaxed(p_shared_low, pa)
        acc_0 = gl.amd.slice(acc, [16, STAGE], [0, 0])
        v_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_smem, vb).to(gl.float16)
        acc_0 = _m1_compensated_pv(high, low, v_0, acc_0)
        acc_1 = gl.amd.slice(acc, [16, STAGE], [0, STAGE])
        v_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_smem_right, vb).to(gl.float16)
        acc_1 = _m1_compensated_pv(high, low, v_1, acc_1)
        acc_2 = gl.amd.slice(acc, [16, STAGE], [0, 2 * STAGE])
        v_2 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_smem_2, vb).to(gl.float16)
        acc_2 = _m1_compensated_pv(high, low, v_2, acc_2)
        acc_3 = gl.amd.slice(acc, [16, STAGE], [0, 3 * STAGE])
        v_3 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_smem_3, vb).to(gl.float16)
        acc_3 = _m1_compensated_pv(high, low, v_3, acc_3)
        acc_left = gl.join(acc_0, acc_1).permute((0, 2, 1)).reshape((16, 2 * STAGE))
        acc_right = gl.join(acc_2, acc_3).permute((0, 2, 1)).reshape((16, 2 * STAGE))
        acc = gl.convert_layout(gl.join(acc_left, acc_right).permute((0, 2, 1)).reshape((16, VD)), mat)
    elif TWO_PANELS:
        high = gl.amd.cdna4.async_copy.load_shared_relaxed(p_shared_high, pa)
        low = gl.amd.cdna4.async_copy.load_shared_relaxed(p_shared_low, pa)
        acc_left = gl.amd.slice(acc, [16, VD // 2], [0, 0])
        acc_right = gl.amd.slice(acc, [16, VD // 2], [0, VD // 2])
        v_left = gl.amd.cdna4.async_copy.load_shared_relaxed(v_smem, vb).to(gl.float16)
        acc_left = _m1_compensated_pv(high, low, v_left, acc_left)
        v_right = gl.amd.cdna4.async_copy.load_shared_relaxed(v_smem_right, vb).to(gl.float16)
        acc_right = _m1_compensated_pv(high, low, v_right, acc_right)
        acc = gl.convert_layout(gl.join(acc_left, acc_right).permute((0, 2, 1)).reshape((16, VD)), mat)
    else:
        v_half = gl.amd.cdna4.async_copy.load_shared_relaxed(v_smem, vb).to(gl.float16)
        high = gl.amd.cdna4.async_copy.load_shared_relaxed(p_shared_high, pa)
        low = gl.amd.cdna4.async_copy.load_shared_relaxed(p_shared_low, pa)
        acc = _m1_compensated_pv(high, low, v_half, acc)
    return (new_max, denom, acc)


@gluon.jit
def _m1_attention_partials(Q, K, V, Indptr, Indices, Partials, Stats, scale, HEADS: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, SPLITS: gl.constexpr, BLOCK: gl.constexpr, VD: gl.constexpr, RECORD: gl.constexpr):
    row = gl.program_id(0)
    part = gl.program_id(1)
    value_block = 0 if VD == 512 else gl.program_id(2)
    if VD == 64:
        linear = part + SPLITS * value_block
        part = linear // 32 * 4 + linear % 4
        value_block = linear % 32 // 4
    mat: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4 if BLOCK <= 64 else 8])
    if VD <= 256:
        softmax_layout: gl.constexpr = gl.BlockedLayout([1, 4 if VD == 256 else 2], [4, 16], [4, 1], [1, 0])
    elif BLOCK == 64:
        softmax_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    else:
        softmax_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [8, 1], [1, 0])
    h = gl.arange(0, 16, layout=gl.SliceLayout(1, mat))
    d = value_block * VD + gl.arange(0, VD, layout=gl.SliceLayout(0, mat))
    if VD == 128 or VD == 256:
        d = d.to(gl.uint32)
    begin = gl.load(Indptr + row)
    end = gl.load(Indptr + row + 1)
    length = end - begin
    maximum = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, softmax_layout))
    denom = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, softmax_layout))
    acc = gl.full((16, VD), 0.0, gl.float32, mat)
    if VD == 64 or part * BLOCK < length:
        maximum, denom, acc = _m1_consume_tile(Q, K, V, Indices, scale, row, value_block, part * BLOCK, begin, length, maximum, denom, acc, HEADS, QS0, QS1, KS0, VS0, BLOCK, VD, True)
    for block_start in loop_range((part + SPLITS) * BLOCK, length, SPLITS * BLOCK, disable_licm=True):
        maximum, denom, acc = _m1_consume_tile(Q, K, V, Indices, scale, row, value_block, block_start, begin, length, maximum, denom, acc, HEADS, QS0, QS1, KS0, VS0, BLOCK, VD, False)
    if RECORD < 0:
        width: gl.constexpr = -RECORD
        offset = (row * HEADS + h[:, None] // 2 * 2) * SPLITS * 512 + (d[None, :] // width * SPLITS + part) * (2 * width) + h[:, None] % 2 * width + d[None, :] % width
        gl.store(Partials + offset, acc, h[:, None] < HEADS)
    elif RECORD > 0:
        offset = row * HEADS * SPLITS * 512 + (d[None, :] // RECORD * SPLITS + part) * (HEADS * RECORD) + h[:, None] * RECORD + d[None, :] % RECORD
        gl.store(Partials + offset, acc, h[:, None] < HEADS)
    else:
        gl.store(Partials + (row * HEADS + h[:, None]) * (SPLITS * 512) + part * 512 + d[None, :], acc, h[:, None] < HEADS)
    if value_block == 0:
        stat_h = gl.arange(0, 16, layout=gl.SliceLayout(1, softmax_layout))
        gl.store(Stats + (row * HEADS + stat_h) * (SPLITS * 2) + part * 2, maximum, stat_h < HEADS)
        gl.store(Stats + (row * HEADS + stat_h) * (SPLITS * 2) + part * 2 + 1, denom, stat_h < HEADS)


@gluon.jit
def _m1_merge_small(Partials, Stats, Out, SPLITS: gl.constexpr):
    row_head = gl.program_id(0)
    dblock = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [1, 1], [1, 0])
    s = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, layout))
    d = dblock * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    base = Partials + row_head * (SPLITS * 512)
    stats_base = Stats + row_head * (SPLITS * 2)
    maximum = gl.amd.cdna4.buffer_load(stats_base, s * 2)
    denom = gl.amd.cdna4.buffer_load(stats_base + 1, s * 2)
    global_max = gl.max(maximum, 0)
    weights = _m1_softmax_exp(maximum - global_max)
    numerator = gl.amd.cdna4.buffer_load(base, s[:, None] * 512 + d[None, :])
    total = gl.sum(denom * weights, 0)
    out = gl.sum(numerator * weights[:, None], 0) * gl.div_rn(1.0, total)
    gl.store(Out + row_head * 512 + d, out)


def _m1_launch_config(tokens):

    if tokens == 1:
        return (64, 64, 64, 0)
    if tokens == 2:
        return (32, 64, 128, 16)
    if tokens == 4:
        return (32, 64, 256, -16)
    if tokens == 8:
        return (32, 64, 512, 16)
    return (16, 128, 512, 8)


def _m1_allocate_buffers(query, splits):

    tokens, heads, _ = query.shape
    numerator_size = tokens * heads * splits * 512
    stats_size = tokens * heads * splits * 2
    storage = torch.empty((numerator_size + stats_size + tokens * heads * 256,), device=query.device, dtype=torch.float32)
    output = storage[numerator_size + stats_size:].view(query.dtype).view(tokens, heads, 512)
    partials = storage[:numerator_size]
    stats = storage[numerator_size:numerator_size + stats_size]
    return (partials, stats, output)


def paged_attention_decode_m1(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:
    tokens, heads, _ = query.shape
    splits, block, value_tile, record = _m1_launch_config(tokens)
    partials, stats, output = _m1_allocate_buffers(query, splits)
    if output_tensor is not None:
        output = output_tensor
    _m1_attention_partials[tokens, splits, 512 // value_tile](query, key_cache, value_cache, kv_indptr, kv_indices, partials, stats, scale, heads, query.stride(0), query.stride(1), key_cache.stride(0), value_cache.stride(0), splits, block, value_tile, record, num_warps=4, allow_flush_denorm=True)
    _m1_merge_small[tokens * heads, 16](partials, stats, output, splits, num_warps=1, allow_flush_denorm=True)
    return output


@gluon.jit
def _m128_stage_operand(x, DST: gl.constexpr, TRANSPOSE: gl.constexpr, PAD: gl.constexpr):
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[x.shape[1], PAD]], [x.shape[0], x.shape[1]], [1, 0])
    shared = gl.allocate_shared_memory(x.dtype, [x.shape[0], x.shape[1]], shared_layout, x)
    if TRANSPOSE:
        result = shared.permute((1, 0)).load(DST)
    else:
        result = shared.load(DST)
    return result


@gluon.jit
def _m128_probability_planes(p, DST: gl.constexpr):
    hi = p.to(gl.bfloat16)
    residual = p - hi.to(gl.float32)
    mid = residual.to(gl.bfloat16)
    lo = (residual - mid.to(gl.float32)).to(gl.bfloat16)
    plane_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[p.shape[1], 8]], [4 * p.shape[0], p.shape[1]], [1, 0])
    packed = gl.join(gl.join(hi, mid), gl.join(lo, lo))
    packed = packed.permute((3, 2, 0, 1)).reshape((4 * p.shape[0], p.shape[1]))
    planes = gl.allocate_shared_memory(gl.bfloat16, [4 * p.shape[0], p.shape[1]], plane_layout, packed)
    return (planes.slice(0, p.shape[0], dim=0).load(DST), planes.slice(p.shape[0], p.shape[0], dim=0).load(DST), planes.slice(2 * p.shape[0], p.shape[0], dim=0).load(DST))


@gluon.jit
def _m128_merge_checkpoint(numerator, maximum, denominator, Partial, Stats, records, heads, dims, H: gl.constexpr, STAT_STRIDE: gl.constexpr):
    old_maximum = gl.amd.cdna4.buffer_load(Stats, records * STAT_STRIDE, heads < H, other=-float('inf'))
    old_denominator = gl.amd.cdna4.buffer_load(Stats, records * STAT_STRIDE + 1, heads < H, other=0.0)
    old_numerator = gl.amd.cdna4.buffer_load(Partial, records[:, None] * 512 + dims[None, :], heads[:, None] < H, other=0.0)
    next_maximum = gl.maximum(old_maximum, maximum)
    old_factor = gl.exp2((old_maximum - next_maximum) * 1.4426950408889634)
    factor = gl.exp2((maximum - next_maximum) * 1.4426950408889634)
    numerator = old_numerator * old_factor[:, None] + numerator * factor[:, None]
    denominator = old_denominator * old_factor + denominator * factor
    return (numerator, next_maximum, denominator)


@gluon.jit
def _m128_mla_partials(Q, K, V, Indptr, Indices, Partial, Stats, scale, H: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, S: gl.constexpr, BT: gl.constexpr, HEAD_PITCH: gl.constexpr):
    FULL_TILE_LOOP: gl.constexpr = S == 2
    row = gl.program_id(0)
    split = gl.program_id(1)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16] if S == 2 else [8, 8], [4, 1], [1, 0])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4])
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    tail_layout: gl.constexpr = gl.BlockedLayout([1, 4] if S == 2 else [1, 8], [4, 16] if S == 2 else [8, 8], [4, 1], [1, 0])
    dims = gl.arange(0, 512, layout=gl.SliceLayout(0, load_layout))
    tail = gl.arange(0, 64, layout=gl.SliceLayout(0, tail_layout))
    query_load: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    heads = gl.arange(0, 16, layout=gl.SliceLayout(1, query_load))
    query_dims = gl.arange(0, 512, layout=gl.SliceLayout(0, query_load))
    qr_heads = gl.arange(0, 16, layout=gl.SliceLayout(1, a_layout))
    qr_dims = gl.arange(0, 64, layout=gl.SliceLayout(0, a_layout))
    q = gl.amd.cdna4.buffer_load(Q + row * QS0, heads[:, None] * QS1 + query_dims[None, :], heads[:, None] < H)
    qr = gl.amd.cdna4.buffer_load(Q + row * QS0, qr_heads[:, None] * QS1 + 512 + qr_dims[None, :], qr_heads[:, None] < H)
    if S == 4:
        q_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [16, 512], [1, 0])
        q_shared = gl.allocate_shared_memory(q.dtype, [16, 512], q_shared_layout, q)
    else:
        q = _m128_stage_operand(q, a_layout, False, 8)
    token_offsets = gl.arange(0, BT, layout=gl.SliceLayout(1, load_layout))
    start = gl.load(Indptr + row)
    length = gl.load(Indptr + row + 1) - start
    tiles = gl.cdiv(length, BT)
    base_tiles = tiles // S
    extra_tiles = tiles % S
    first_tile = split * base_tiles + gl.minimum(split, extra_tiles)
    split_tiles = base_tiles + (split < extra_tiles).to(gl.int32)
    lo = first_tile * BT
    hi = gl.minimum((first_tile + split_tiles) * BT, length)
    ACCUMULATION_WINDOW: gl.constexpr = 2048 if S == 2 else 4096
    STAT_STRIDE: gl.constexpr = 4
    record_heads = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
    record_dims = gl.arange(0, 512, layout=gl.SliceLayout(0, mma))
    record_ids = (row * HEAD_PITCH + record_heads) * S + split
    fp32_partials = Partial + gl.num_programs(0) * H * S * 256
    maximum = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, mma))
    denominator = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma))
    numerator = gl.full((16, 512), 0.0, gl.float32, mma)
    denominator_by_position = gl.full((16, BT), 0.0, gl.float32, mma)
    score_tokens = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    for phase in gl.static_range(2 if FULL_TILE_LOOP else 1):
        if FULL_TILE_LOOP:
            full_end = hi // BT * BT
            if FULL_TILE_LOOP and phase == 0:
                begin, end = (lo, full_end)
            else:
                begin, end = (gl.maximum(lo, full_end), hi)
        else:
            begin, end = (lo, hi)
        for pos in range(begin, end, BT):
            if FULL_TILE_LOOP and phase == 0:
                valid = gl.full((BT,), True, gl.int1, gl.SliceLayout(1, load_layout))
            else:
                valid = pos + token_offsets < hi
            slots = gl.load(Indices + start + pos + token_offsets, valid, 0).to(gl.int32)
            values = gl.amd.cdna4.buffer_load(V, slots[:, None] * VS0 + dims[None, :], valid[:, None], cache='.cg')
            keys = gl.amd.cdna4.buffer_load(K, slots[:, None] * KS0 + dims[None, :])
            tail_slots = gl.convert_layout(slots, gl.SliceLayout(1, tail_layout))
            kr = gl.amd.cdna4.buffer_load(K, tail_slots[:, None] * KS0 + 512 + tail[None, :], cache='.cg' if S == 2 else '')
            gl.inline_asm_elementwise('s_setprio 1', constraints='=s', args=[], dtype=gl.int32, is_pure=False, pack=1)
            if S == 4:
                k_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [BT, 512], [1, 0])
                k_shared = gl.allocate_shared_memory(keys.dtype, [BT, 512], k_shared_layout, keys)
            else:
                key_operand = _m128_stage_operand(keys, b_layout, True, 16).to(gl.bfloat16)
            tail_operand = _m128_stage_operand(kr, b_layout, True, 16).to(gl.bfloat16)
            scores = gl.full((16, BT), 0.0, gl.float32, mma)
            if S == 4:
                for block in gl.static_range(2):
                    query_operand = q_shared.slice(block * 256, 256, dim=1).load(a_layout)
                    key_operand = k_shared.slice(block * 256, 256, dim=1).permute((1, 0)).load(b_layout).to(gl.bfloat16)
                    scores = gl.amd.cdna4.mfma(query_operand, key_operand, scores)
            else:
                scores = gl.amd.cdna4.mfma(q, key_operand, scores)
            scores = gl.amd.cdna4.mfma(qr, tail_operand, scores) * scale
            if S == 4:
                v_operand = _m128_stage_operand(values, b_layout, False, 16).to(gl.bfloat16)
            if not (FULL_TILE_LOOP and phase == 0):
                scores = gl.where((pos + score_tokens < hi)[None, :], scores, -float('inf'))
            next_max = gl.maximum(maximum, gl.max(scores, 1))
            alpha = gl.exp2((maximum - next_max) * 1.4426950408889634)
            p = gl.exp2((scores - next_max[:, None]) * 1.4426950408889634)
            if not (FULL_TILE_LOOP and phase == 0):
                p = gl.where((pos + score_tokens < hi)[None, :], p, 0.0)
            if S == 2:
                if pos == lo:
                    numerator = gl.full((16, 512), 0.0, gl.float32, mma)
                else:
                    numerator *= alpha[:, None]
            else:
                numerator *= alpha[:, None]
            if S == 2:
                denominator_by_position = denominator_by_position * alpha[:, None] + p
            else:
                denominator = denominator * alpha + gl.sum(p, 1)
            probability_hi, probability_mid, probability_lo = _m128_probability_planes(p, a_layout)
            if S == 2:
                v_operand = _m128_stage_operand(values, b_layout, False, 16).to(gl.bfloat16)
            numerator = gl.amd.cdna4.mfma(probability_lo, v_operand, numerator)
            numerator = gl.amd.cdna4.mfma(probability_mid, v_operand, numerator)
            numerator = gl.amd.cdna4.mfma(probability_hi, v_operand, numerator)
            gl.inline_asm_elementwise('s_setprio 0', constraints='=s', args=[], dtype=gl.int32, is_pure=False, pack=1)
            maximum = next_max
            if not FULL_TILE_LOOP or phase == 0:
                if (pos + BT - lo) % ACCUMULATION_WINDOW == 0 and pos + BT < hi:
                    if S == 2:
                        denominator = gl.sum(denominator_by_position, 1)
                    if pos - lo >= ACCUMULATION_WINDOW:
                        numerator, maximum, denominator = _m128_merge_checkpoint(numerator, maximum, denominator, fp32_partials, Stats, record_ids, record_heads, record_dims, H, STAT_STRIDE)
                    gl.amd.cdna4.buffer_store(numerator, fp32_partials, record_ids[:, None] * 512 + record_dims[None, :], record_heads[:, None] < H)
                    gl.amd.cdna4.buffer_store(maximum, Stats, record_ids * STAT_STRIDE, record_heads < H)
                    gl.amd.cdna4.buffer_store(denominator, Stats, record_ids * STAT_STRIDE + 1, record_heads < H)
                    gl.barrier()
                    numerator = gl.full((16, 512), 0.0, gl.float32, mma)
                    maximum = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, mma))
                    denominator = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma))
                    denominator_by_position = gl.full((16, BT), 0.0, gl.float32, mma)
    if S == 2:
        denominator = gl.sum(denominator_by_position, 1)
    if hi - lo > ACCUMULATION_WINDOW:
        numerator, maximum, denominator = _m128_merge_checkpoint(numerator, maximum, denominator, fp32_partials, Stats, record_ids, record_heads, record_dims, H, STAT_STRIDE)
    out_heads = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
    out_dims = gl.arange(0, 512, layout=gl.SliceLayout(0, mma))
    records = (row * HEAD_PITCH + out_heads) * S + split
    store_layout: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 128], [0, 256]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 64], [0, 8]], warp_bases=[[0, 16], [0, 32]], block_bases=[], shape=[16, 512])
    store_values = gl.convert_layout(numerator.to(gl.float16), store_layout)
    store_heads = gl.arange(0, 16, layout=gl.SliceLayout(1, store_layout))
    store_dims = gl.arange(0, 512, layout=gl.SliceLayout(0, store_layout))
    store_records = (row * H + store_heads) * S + split
    gl.amd.cdna4.buffer_store(store_values, Partial.to(gl.pointer_type(gl.float16)), store_records[:, None] * 512 + store_dims[None, :], (store_heads < H)[:, None])
    magnitude = gl.max(gl.abs(numerator), 1)
    compact = (magnitude <= denominator * 0.25) & (magnitude <= 65504.0) & (denominator > 0.0)
    if gl.sum(((out_heads < H) & ~compact).to(gl.int32), 0) != 0:
        gl.amd.cdna4.buffer_store(numerator, fp32_partials, records[:, None] * 512 + out_dims[None, :], (out_heads < H)[:, None] & ~compact[:, None])
    tag_bits = compact.to(gl.int32).to(gl.float32, bitcast=True)
    padding = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma))
    fields = gl.join(gl.join(maximum, tag_bits), gl.join(denominator, padding)).reshape((16, 4))
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
    fields = gl.convert_layout(fields, stats_layout)
    stats_heads = gl.arange(0, 16, layout=gl.SliceLayout(1, stats_layout))
    stats_fields = gl.arange(0, 4, layout=gl.SliceLayout(0, stats_layout))
    stats_records = (row * HEAD_PITCH + stats_heads) * S + split
    gl.amd.cdna4.buffer_store(fields, Stats, stats_records[:, None] * STAT_STRIDE + stats_fields[None, :], stats_heads[:, None] < H)


@gluon.jit
def _m128_merge_mla(Partial, Stats, O, S: gl.constexpr, H: gl.constexpr, HEAD_PITCH: gl.constexpr):
    row_head = gl.program_id(0)
    record_head = row_head // H * HEAD_PITCH + row_head % H
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
    dims = gl.arange(0, 512, layout=layout)
    maximum = gl.load(Stats + record_head * S * 4)
    compact_all = gl.load(Stats.to(gl.pointer_type(gl.int32)) + record_head * S * 4 + 2) != 0
    for split in gl.static_range(1, S):
        maximum = gl.maximum(maximum, gl.load(Stats + (record_head * S + split) * 4))
        compact_all &= gl.load(Stats.to(gl.pointer_type(gl.int32)) + (record_head * S + split) * 4 + 2) != 0
    numerator = gl.full((512,), 0.0, gl.float32, layout)
    denominator = 0.0
    if compact_all:
        for split in gl.static_range(S):
            split_max = gl.load(Stats + (record_head * S + split) * 4)
            split_sum = gl.load(Stats + (record_head * S + split) * 4 + 1)
            factor = gl.exp2((split_max - maximum) * 1.4426950408889634)
            values = gl.load(Partial.to(gl.pointer_type(gl.float16)) + (row_head * S + split) * 512 + dims, cache_modifier='.cg').to(gl.float32)
            numerator += values * factor
            denominator += split_sum * factor
    else:
        for split in gl.static_range(S):
            split_max = gl.load(Stats + (record_head * S + split) * 4)
            split_sum = gl.load(Stats + (record_head * S + split) * 4 + 1)
            factor = gl.exp2((split_max - maximum) * 1.4426950408889634)
            compact = gl.load(Stats.to(gl.pointer_type(gl.int32)) + (record_head * S + split) * 4 + 2) != 0
            values_half = gl.load(Partial.to(gl.pointer_type(gl.float16)) + (row_head * S + split) * 512 + dims, compact, 0.0, cache_modifier='.cg').to(gl.float32)
            values_full = gl.load(Partial + gl.num_programs(0) * S * 256 + (record_head * S + split) * 512 + dims, ~compact, 0.0, cache_modifier='.cg')
            values = gl.where(compact, values_half, values_full)
            numerator += values * factor
            denominator += split_sum * factor
    gl.store(O + row_head * 512 + dims, (numerator * (1.0 / denominator)).to(O.dtype.element_ty))


def paged_attention_decode_m128(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:

    tokens, heads, _ = query.shape
    splits = 4
    output = torch.empty((tokens, heads, 512), dtype=query.dtype, device=query.device)
    record_heads = heads
    record_width = 768
    stat_width = 4
    partial = torch.empty((tokens, record_heads, splits, record_width), dtype=torch.float32, device=query.device)
    stats = torch.empty((tokens, record_heads, splits, stat_width), dtype=torch.float32, device=query.device)
    if output_tensor is not None:
        output = output_tensor
    _m128_mla_partials[tokens, splits](query, key_cache, value_cache, kv_indptr, kv_indices, partial, stats, scale, heads, query.stride(0), query.stride(1), key_cache.stride(0), value_cache.stride(0), splits, 64, record_heads, num_warps=4, num_stages=1, enable_fp_fusion=False)
    _m128_merge_mla[tokens * heads,](partial, stats, output, splits, heads, record_heads, num_warps=1)
    return output


@gluon.jit
def _m12_16_generic_decode(Q, K, V, Indptr, Indices, Out, KScale, VScale, scale, H: gl.constexpr, HK: gl.constexpr, D: gl.constexpr, DV: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, KS1: gl.constexpr, VS0: gl.constexpr, VS1: gl.constexpr, BD: gl.constexpr, BV: gl.constexpr, SCALED: gl.constexpr):
    row, head = (gl.program_id(0), gl.program_id(1))
    kv_head = head // (H // HK)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    d = gl.arange(0, BD, gl.SliceLayout(0, layout))
    v = gl.arange(0, BV, gl.SliceLayout(0, layout))
    t = gl.arange(0, 16, gl.SliceLayout(1, layout))
    q = gl.load(Q + row * QS0 + head * QS1 + d, d < D, 0).to(gl.float32)
    start = gl.load(Indptr + row)
    end = gl.load(Indptr + row + 1)
    if SCALED:
        ks = gl.load(KScale)
        vs = gl.load(VScale)
    else:
        ks = 1.0
        vs = 1.0
    maximum = gl.full((), -float('inf'), gl.float32, gl.BlockedLayout([1], [64], [4], [0]))
    denom = gl.full((), 0.0, gl.float32, gl.BlockedLayout([1], [64], [4], [0]))
    acc = gl.full((BV,), 0.0, gl.float32, gl.SliceLayout(0, layout))
    for base in range(start, end, 16):
        valid = base + t < end
        slots = gl.load(Indices + base + t, valid, 0).to(gl.int64)
        k = gl.load(K + slots[:, None] * KS0 + kv_head * KS1 + d[None, :], valid[:, None] & (d[None, :] < D), 0.0).to(gl.float32)
        values = gl.load(V + slots[:, None] * VS0 + kv_head * VS1 + v[None, :], valid[:, None] & (v[None, :] < DV), 0.0).to(gl.float32)
        scores = gl.sum(k * ks * q[None, :], 1) * scale
        scores = gl.where(valid, scores, -float('inf'))
        new_max = gl.maximum(maximum, gl.max(scores, 0))
        alpha = gl.exp(maximum - new_max)
        p = gl.where(valid, gl.exp(scores - new_max), 0.0)
        acc = acc * alpha + gl.sum(p[:, None] * values * vs, 0)
        denom = denom * alpha + gl.sum(p, 0)
        maximum = new_max
    gl.store(Out + (row * H + head) * DV + v, (acc / denom).to(gl.bfloat16), v < DV)


@gluon.jit
def _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, base, length, shard, maximum, denom, acc, scale, KS0: gl.constexpr, VS0: gl.constexpr, BT: gl.constexpr, BV: gl.constexpr, BK: gl.constexpr, WARPS: gl.constexpr, FIRST: gl.constexpr, FULL: gl.constexpr=False):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WARPS])
    qa: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    kb: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    qka: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    qkb: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    kload: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, WARPS], [1, 0])
    value_lanes: gl.constexpr = 32 if BV == 512 and VS0 % 16 == 0 else min(16, BV // 16)
    vl: gl.constexpr = gl.BlockedLayout([1, 16], [64 // value_lanes, value_lanes], [WARPS, 1], [1, 0])
    kt = gl.arange(0, BT, gl.SliceLayout(0, kload))
    kd = gl.arange(0, BK, gl.SliceLayout(1, kload))
    kr = gl.arange(0, 64, gl.SliceLayout(1, kload))
    vt = gl.arange(0, BT, gl.SliceLayout(1, vl))
    vd = gl.arange(0, BV, gl.SliceLayout(0, vl))
    mt = gl.arange(0, BT, gl.SliceLayout(0, mma))
    slots = gl.load(Indices + start + base + kt, (base + kt < length) | FULL, 0).to(gl.int32)
    if BT == 128 and BV == 256:
        vslots = gl.convert_layout(slots, gl.SliceLayout(1, vl))
        values = gl.amd.cdna4.buffer_load(V, vslots[:, None] * VS0 + shard * BV + vd[None, :], (base + vt[:, None] < length) | FULL, other=0.0)
        vmem.store(values)
    if BT == 128 and BV == 256 or (BT == 64 and BV == 512):
        scores0 = gl.full((16, BT), 0.0, gl.float32, mma)
        scores1 = gl.full((16, BT), 0.0, gl.float32, mma)
        scores2 = gl.full((16, BT), 0.0, gl.float32, mma)
        scores3 = gl.full((16, BT), 0.0, gl.float32, mma)
        for channel in gl.static_range(8):
            keys = gl.amd.cdna4.buffer_load(K, slots[None, :] * KS0 + channel * BK + kd[:, None], (base + kt[None, :] < length) | FULL, other=0.0)
            kval = gl.convert_layout(keys, qkb).to(gl.bfloat16)
            qval = qmem.slice(channel * BK, BK, 1).load(qka)
            if channel % 4 == 0:
                scores0 = gl.amd.cdna4.mfma(qval, kval, scores0)
            elif channel % 4 == 1:
                scores1 = gl.amd.cdna4.mfma(qval, kval, scores1)
            elif channel % 4 == 2:
                scores2 = gl.amd.cdna4.mfma(qval, kval, scores2)
            else:
                scores3 = gl.amd.cdna4.mfma(qval, kval, scores3)
        keys_rope = gl.amd.cdna4.buffer_load(K, slots[None, :] * KS0 + 512 + kr[:, None], (base + kt[None, :] < length) | FULL, other=0.0)
        scores3 = gl.amd.cdna4.mfma(rope, gl.convert_layout(keys_rope, qkb).to(gl.bfloat16), scores3)
        scores = scores0 + scores1 + (scores2 + scores3)
    else:
        scores = gl.full((16, BT), 0.0, gl.float32, mma)
        for channel in gl.static_range(512 // BK):
            keys = gl.amd.cdna4.buffer_load(K, slots[None, :] * KS0 + channel * BK + kd[:, None], (base + kt[None, :] < length) | FULL, other=0.0)
            kval = gl.convert_layout(keys, qkb).to(gl.bfloat16)
            qval = qmem.slice(channel * BK, BK, 1).load(qka)
            scores = gl.amd.cdna4.mfma(qval, kval, scores)
        keys_rope = gl.amd.cdna4.buffer_load(K, slots[None, :] * KS0 + 512 + kr[:, None], (base + kt[None, :] < length) | FULL, other=0.0)
        scores = gl.amd.cdna4.mfma(rope, gl.convert_layout(keys_rope, qkb).to(gl.bfloat16), scores)
    if not (BT == 128 and BV == 256):
        vslots = gl.convert_layout(slots, gl.SliceLayout(1, vl))
        if BV == 512 and VS0 % 16 == 0:
            if WARPS == 8:
                sub_t = gl.arange(0, 32, gl.SliceLayout(1, vl))
                columns = (vd[None, :] // 16 ^ sub_t[:, None] % 8) * 16 + vd[None, :] % 16
                for panel in gl.static_range(4):
                    sub_slots = gl.amd.slice(vslots, [32], [panel * 32])
                    voffsets = sub_slots[:, None] * VS0 + shard * BV + columns
                    copy_mem = vmem._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0])).slice(panel * 32, 32, 0)
                    if panel == 0:
                        gl.amd.cdna4.async_copy.buffer_load_to_shared(copy_mem, V.to(gl.pointer_type(gl.uint8)), voffsets, (base + panel * 32 + sub_t[:, None] < length) | FULL)
                    else:
                        gl.amd.cdna4.async_copy.buffer_load_to_shared(copy_mem, V.to(gl.pointer_type(gl.uint8)), voffsets, (base + panel * 32 + sub_t[:, None] < length) | FULL, cache_modifier='.cg')
            else:
                columns = (vd[None, :] // 16 ^ vt[:, None] % 8) * 16 + vd[None, :] % 16
                voffsets = vslots[:, None] * VS0 + shard * BV + columns
                copy_mem = vmem._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
                gl.amd.cdna4.async_copy.buffer_load_to_shared(copy_mem, V.to(gl.pointer_type(gl.uint8)), voffsets, (base + vt[:, None] < length) | FULL)
            gl.amd.cdna4.async_copy.commit_group()
        else:
            values = gl.amd.cdna4.buffer_load(V, vslots[:, None] * VS0 + shard * BV + vd[None, :], (base + vt[:, None] < length) | FULL, other=0.0)
            vmem.store(values)
    if FULL:
        scores = gl.fma(scores, scale, 0.0)
    else:
        scores = scores * scale
    scores = gl.where((base + mt[None, :] < length) | FULL, scores, -float('inf'))
    if BV == 64:
        prob_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [WARPS, 1], [0, 1])
        scores = gl.convert_layout(scores, prob_layout)
        pt = gl.arange(0, BT, gl.SliceLayout(0, prob_layout))
        if FIRST:
            new_max = gl.max(scores, 1)
            alpha = 0.0
        else:
            old_max = gl.convert_layout(maximum, gl.SliceLayout(1, prob_layout))
            new_max = gl.maximum(old_max, gl.max(scores, 1))
            alpha = gl.convert_layout(gl.exp(old_max - new_max), gl.SliceLayout(1, mma))
        p = gl.where((base + pt[None, :] < length) | FULL, gl.exp2((scores - new_max[:, None]) * 1.4426950408889634), 0.0)
        new_max = gl.convert_layout(new_max, gl.SliceLayout(1, mma))
        tile_denom = gl.convert_layout(gl.sum(p, 1), gl.SliceLayout(1, mma))
    else:
        if FIRST:
            new_max = gl.max(scores, 1)
            alpha = 0.0
        else:
            new_max = gl.maximum(maximum, gl.max(scores, 1))
            alpha = gl.exp(maximum - new_max)
        p = gl.where((base + mt[None, :] < length) | FULL, gl.exp2((scores - new_max[:, None]) * 1.4426950408889634), 0.0)
        tile_denom = gl.sum(p, 1)
    if FIRST:
        denom = tile_denom
        acc = gl.full((16, BV), 0.0, gl.float32, mma)
    else:
        denom = denom * alpha + tile_denom
        acc = acc * alpha[:, None]
    high = p.to(gl.bfloat16)
    low = (p - high.to(gl.float32)).to(gl.bfloat16)
    if BV == 128 or BV == 256:
        packed = high.to(gl.uint16, bitcast=True).to(gl.uint32) | low.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        packed = gl.convert_layout(packed, qa)
        high = (packed & 65535).to(gl.uint16).to(gl.bfloat16, bitcast=True)
        low = (packed >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    else:
        low = gl.convert_layout(low, qa)
        high = gl.convert_layout(high, qa)
    if BV == 512:
        if VS0 % 16 == 0:
            gl.amd.cdna4.async_copy.wait_group(0)
        if BT == 64:
            acc0 = gl.amd.slice(acc, [16, 128], [0, 0])
            acc1 = gl.amd.slice(acc, [16, 128], [0, 128])
            acc2 = gl.amd.slice(acc, [16, 128], [0, 256])
            acc3 = gl.amd.slice(acc, [16, 128], [0, 384])
            vb0 = vmem.slice(0, 128, 1).load(kb).to(V.dtype.element_ty, bitcast=True).to(gl.bfloat16)
            acc0 = gl.amd.cdna4.mfma(low, vb0, acc0)
            acc0 = gl.amd.cdna4.mfma(high, vb0, acc0)
            vb1 = vmem.slice(128, 128, 1).load(kb).to(V.dtype.element_ty, bitcast=True).to(gl.bfloat16)
            acc1 = gl.amd.cdna4.mfma(low, vb1, acc1)
            acc1 = gl.amd.cdna4.mfma(high, vb1, acc1)
            vb2 = vmem.slice(256, 128, 1).load(kb).to(V.dtype.element_ty, bitcast=True).to(gl.bfloat16)
            acc2 = gl.amd.cdna4.mfma(low, vb2, acc2)
            acc2 = gl.amd.cdna4.mfma(high, vb2, acc2)
            vb3 = vmem.slice(384, 128, 1).load(kb).to(V.dtype.element_ty, bitcast=True).to(gl.bfloat16)
            acc3 = gl.amd.cdna4.mfma(low, vb3, acc3)
            acc3 = gl.amd.cdna4.mfma(high, vb3, acc3)
            acc01 = gl.join(acc0, acc1).permute((0, 2, 1)).reshape((16, 256))
            acc23 = gl.join(acc2, acc3).permute((0, 2, 1)).reshape((16, 256))
            acc = gl.convert_layout(gl.join(acc01, acc23).permute((0, 2, 1)).reshape((16, BV)), mma)
        else:
            acc0 = gl.amd.slice(acc, [16, BV // 2], [0, 0])
            acc1 = gl.amd.slice(acc, [16, BV // 2], [0, BV // 2])
            vb0 = vmem.slice(0, BV // 2, 1).load(kb).to(V.dtype.element_ty, bitcast=True).to(gl.bfloat16)
            acc0 = gl.amd.cdna4.mfma(low, vb0, acc0)
            acc0 = gl.amd.cdna4.mfma(high, vb0, acc0)
            vb1 = vmem.slice(BV // 2, BV // 2, 1).load(kb).to(V.dtype.element_ty, bitcast=True).to(gl.bfloat16)
            acc1 = gl.amd.cdna4.mfma(low, vb1, acc1)
            acc1 = gl.amd.cdna4.mfma(high, vb1, acc1)
            acc = gl.convert_layout(gl.join(acc0, acc1).permute((0, 2, 1)).reshape((16, BV)), mma)
    else:
        vb = vmem.load(kb).to(V.dtype.element_ty, bitcast=True).to(gl.bfloat16)
        acc = gl.amd.cdna4.mfma(low, vb, acc)
        acc = gl.amd.cdna4.mfma(high, vb, acc)
    maximum = new_max
    return (maximum, denom, acc)


@gluon.jit
def _m12_16_mla_partials(Q, K, V, Indptr, Indices, Numerator, Stats, scale, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, PARTS: gl.constexpr, BT: gl.constexpr, BV: gl.constexpr, BK: gl.constexpr, WARPS: gl.constexpr):
    row = gl.program_id(0)
    if BT == 64 and BV == 128:
        shard = gl.program_id(1) // 2
        part = gl.program_id(1) % 2 + 2 * gl.program_id(2)
    else:
        part = gl.program_id(1)
        shard = gl.program_id(2)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WARPS])
    qka: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    qload: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WARPS, 1], [1, 0])
    qh = gl.arange(0, 16, gl.SliceLayout(1, qload))
    qd = gl.arange(0, 512, gl.SliceLayout(0, qload))
    rope_load: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [WARPS, 1], [0, 1])
    rh = gl.arange(0, 16, gl.SliceLayout(1, rope_load))
    qr = gl.arange(0, 64, gl.SliceLayout(0, rope_load))
    q_latent = gl.load(Q + row * QS0 + gl.minimum(qh[:, None], 11) * QS1 + qd[None, :])
    q_rope = gl.load(Q + row * QS0 + gl.minimum(rh[:, None], 11) * QS1 + 512 + qr[None, :])
    if BT == 64 and BV == 256 or BV == 512:
        query_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 8]], [16, 512], [1, 0])
    else:
        query_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 2 if BV == 256 else 1, 8, [1, 0])
    qmem = gl.allocate_shared_memory(gl.bfloat16, [16, 512], query_shared, q_latent)
    rope = gl.convert_layout(q_rope, qka, assert_trivial=True)
    if BV == 512 and VS0 % 16 == 0:
        vmem = gl.allocate_shared_memory(gl.uint8, [BT, BV], gl.SwizzledSharedLayout(16, 1, 8, [1, 0]))
    else:
        vmem = gl.allocate_shared_memory(V.dtype.element_ty, [BT, BV], gl.SwizzledSharedLayout(16, 1, 8, [1, 0]))
    start = gl.load(Indptr + row)
    length = gl.load(Indptr + row + 1) - start
    STEP: gl.constexpr = BT
    if BV == 64:
        if length <= PARTS * BT:
            partition_tiles = 1
            begin = part * BT
            finish = gl.minimum(length, begin + BT)
        else:
            partition_tiles = gl.cdiv(length, PARTS * BT)
            begin = part * partition_tiles * BT
            finish = gl.minimum(length, begin + partition_tiles * BT)
    else:
        partition_tiles = gl.cdiv(length, PARTS * BT)
        begin = part * partition_tiles * BT
        finish = gl.minimum(length, begin + partition_tiles * BT)
    mh = gl.arange(0, 16, gl.SliceLayout(1, mma))
    mv = shard * BV + gl.arange(0, BV, gl.SliceLayout(0, mma))
    maximum = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, mma))
    denom = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma))
    acc = gl.full((16, BV), 0.0, gl.float32, mma)
    if BV == 64 or (BT == 128 and BV == 256) or (BT == 64 and BV == 512):
        if (partition_tiles == 1) & (begin + BT <= length):
            maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, begin, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, True, True)
        elif begin < finish:
            maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, begin, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, True)
            for base in range(begin + STEP, finish, STEP):
                maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, base, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, False)
    elif BT == 64 and (BV == 128 or BV == 256):
        maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, begin, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, True)
        for base in range(begin + BT, finish, BT):
            maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, base, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, False)
    elif begin < finish:
        base = begin
        if BV == 512:
            if begin + BT <= length:
                maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, base, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, True, True)
            else:
                maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, base, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, True)
        else:
            maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, base, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, True)
        for base in range(begin + STEP, finish, STEP):
            maximum, denom, acc = _m12_16_attention_tile(qmem, rope, vmem, K, V, Indices, start, base, length, shard, maximum, denom, acc, scale, KS0, VS0, BT, BV, BK, WARPS, False)
    record = (((row * 3 + mh[:, None] // 4) * 32 + mv[None, :] // 16) * PARTS + part) * 64 + mh[:, None] % 4 * 16 + mv[None, :] % 16
    if BV == 512:
        gl.amd.cdna4.buffer_store(acc, Numerator, record, mh[:, None] < 12, cache='.cs')
    else:
        gl.store(Numerator + record, acc, mh[:, None] < 12)
    if shard == 0:
        stat = (row * PARTS + part) * 12 + mh
        gl.store(Stats + stat * 2, maximum, mh < 12)
        gl.store(Stats + stat * 2 + 1, denom, mh < 12)


@gluon.jit
def _m12_16_merge_partials(Numerator, Stats, Out, PARTS: gl.constexpr, BC: gl.constexpr, BH: gl.constexpr, STREAM: gl.constexpr):
    row = gl.program_id(0)
    head = gl.program_id(1) * BH
    col = gl.program_id(2) * BC
    PL: gl.constexpr = min(16, 256 // (BH * BC))
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [BH, PL, 64 // (BH * PL)], [1, 1, 1], [2, 0, 1])
    sl: gl.constexpr = gl.SliceLayout(2, layout)
    h = head + gl.arange(0, BH, gl.SliceLayout(1, sl))
    p = gl.arange(0, PARTS, gl.SliceLayout(0, sl))
    c = col + gl.arange(0, BC, gl.SliceLayout(0, gl.SliceLayout(0, layout)))
    stat = (row * PARTS + p[None, :]) * 12 + h[:, None]
    maximum = gl.load(Stats + stat * 2)
    denom = gl.load(Stats + stat * 2 + 1)
    weight = gl.exp(maximum - gl.max(maximum, 1)[:, None])
    denom_sum = gl.sum(denom * weight, 1)
    hs = gl.expand_dims(gl.expand_dims(h, 1), 2)
    ps = gl.expand_dims(gl.expand_dims(p, 0), 2)
    cs = gl.expand_dims(gl.expand_dims(c, 0), 0)
    record = (((row * 3 + hs // 4) * 32 + cs // 16) * PARTS + ps) * 64 + hs % 4 * 16 + cs % 16
    if STREAM:
        nums = gl.amd.cdna4.buffer_load(Numerator, record, cache='.cs')
    else:
        nums = gl.load(Numerator + record)
    summed = gl.sum(nums * weight[:, :, None], 1)
    out_layout: gl.constexpr = gl.SliceLayout(1, layout)
    reciprocal = gl.convert_layout(1.0 / denom_sum, gl.SliceLayout(1, out_layout))
    result = summed * reciprocal[:, None]
    oh = head + gl.arange(0, BH, gl.SliceLayout(1, out_layout))
    oc = col + gl.arange(0, BC, gl.SliceLayout(0, out_layout))
    gl.store(Out + (row * 12 + oh[:, None]) * 512 + oc[None, :], result.to(gl.bfloat16))


def paged_attention_decode_m12_16(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:
    tokens, heads, head_dim = query.shape
    kv_heads = key_cache.shape[1]
    value_dim = value_cache.shape[2]
    output = torch.empty((tokens, heads, value_dim), dtype=query.dtype, device=query.device) if output_tensor is None else output_tensor
    fast = heads == 12 and kv_heads == 1 and (head_dim == 576) and (value_dim == 512) and (query.dtype == torch.bfloat16) and (key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)) and (value_cache.dtype == key_cache.dtype) and (k_scale is None) and (v_scale is None) and (key_cache.shape[0] * key_cache.stride(0) < 2 ** 31) and (value_cache.shape[0] * value_cache.stride(0) < 2 ** 31)
    if fast:
        parts = 16
        bt = 128
        bv = min(512, 32 * tokens)
        bv = triton.next_power_of_2(bv)
        scratch = torch.empty((tokens * parts * 12 * 514,), device=query.device, dtype=torch.float32)
        numerators = scratch[:tokens * parts * 12 * 512]
        stats = scratch[tokens * parts * 12 * 512:]
        grid = (tokens, parts, 512 // bv)
        warps = 8
        bk = 256
        _m12_16_mla_partials[grid](query, key_cache, value_cache, kv_indptr, kv_indices, numerators, stats, scale, query.stride(0), query.stride(1), key_cache.stride(0), value_cache.stride(0), parts, bt, bv, bk, warps, num_warps=warps, waves_per_eu=0)
        bh = 2
        bc = 32
        _m12_16_merge_partials[tokens, 12 // bh, 512 // bc](numerators, stats, output, parts, bc, bh, tokens >= 8, num_warps=1)
    else:
        _m12_16_generic_decode[tokens, heads](query, key_cache, value_cache, kv_indptr, kv_indices, output, query if k_scale is None else k_scale, query if v_scale is None else v_scale, scale, heads, kv_heads, head_dim, value_dim, query.stride(0), query.stride(1), key_cache.stride(0), key_cache.stride(1), value_cache.stride(0), value_cache.stride(1), triton.next_power_of_2(head_dim), triton.next_power_of_2(value_dim), k_scale is not None, num_warps=4)
    return output


@gluon.jit
def _m2_pv_leaf(p_hi, p_lo, v_smem, acc, mat: gl.constexpr, kb: gl.constexpr):
    v_half = gl.amd.cdna4.async_copy.load_shared_relaxed(v_smem, kb).to(gl.float16)
    correction = gl.amd.cdna4.mfma(p_lo, v_half, gl.full(acc.shape, 0.0, gl.float32, mat))
    acc = gl.amd.cdna4.mfma(p_hi, v_half, acc)
    return acc + correction * (1.0 / 65536.0)


@gluon.jit
def _m2_pv_two_panels(p_hi, p_lo, panel0, panel1, acc, mat: gl.constexpr, kb: gl.constexpr):
    width: gl.constexpr = acc.shape[1] // 2
    acc0, acc1 = gl.split(acc.reshape((16, 2, width)).permute((0, 2, 1)))
    acc0 = _m2_pv_leaf(p_hi, p_lo, panel0, gl.convert_layout(acc0, mat), mat, kb)
    acc1 = _m2_pv_leaf(p_hi, p_lo, panel1, gl.convert_layout(acc1, mat), mat, kb)
    joined = gl.join(acc0, acc1).permute((0, 2, 1)).reshape((16, 2 * width))
    return gl.convert_layout(joined, mat)


@gluon.jit
def _m2_pv_four_panels(p_hi, p_lo, panel0, panel1, panel2, panel3, acc, mat: gl.constexpr, kb: gl.constexpr):
    acc0, acc1 = gl.split(acc.reshape((16, 2, 128)).permute((0, 2, 1)))
    acc0 = _m2_pv_two_panels(p_hi, p_lo, panel0, panel1, gl.convert_layout(acc0, mat), mat, kb)
    acc1 = _m2_pv_two_panels(p_hi, p_lo, panel2, panel3, gl.convert_layout(acc1, mat), mat, kb)
    joined = gl.join(acc0, acc1).permute((0, 2, 1)).reshape((16, 256))
    return gl.convert_layout(joined, mat)


@gluon.jit
def _m2_consume_tile(Q, K, V, Indices, scale, row, value_block, block_start, begin, length, maximum, denom, acc, HEADS: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, BLOCK: gl.constexpr, VD: gl.constexpr, FIRST: gl.constexpr):
    mat: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4 if BLOCK <= 64 else 8])
    pv_width: gl.constexpr = 8 if VD == 128 else 16 if BLOCK == 64 and VD == 512 else 4 if VD < 512 else 8
    qa: gl.constexpr = gl.DotOperandLayout(0, mat, pv_width)
    kb: gl.constexpr = gl.DotOperandLayout(1, mat, pv_width)
    direct_qk: gl.constexpr = BLOCK == 64
    qk_a: gl.constexpr = gl.DotOperandLayout(0, mat, 16)
    qk_b: gl.constexpr = gl.DotOperandLayout(1, mat, 16)
    qlayout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1 if BLOCK <= 64 else 2, 4], [1, 0])
    klayout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [8, 1], [1, 0])
    COPY_D: gl.constexpr = 256 if VD == 512 else 64
    PHASES: gl.constexpr = 16 if VD == 512 else 1 if VD == 128 else 8
    vlayout: gl.constexpr = gl.BlockedLayout([1, 16], [1024 // COPY_D, COPY_D // 16], [4 if BLOCK <= 64 else 8, 1], [1, 0])
    if direct_qk:
        tk = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, qk_b))
    else:
        tk = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, klayout))
    soft_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16] if BLOCK == 64 else [2, 32], [4, 1] if BLOCK == 64 else [8, 1], [1, 0])
    tm = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, soft_layout))
    dv = value_block * VD + gl.arange(0, COPY_D, layout=gl.SliceLayout(0, vlayout))
    pos = block_start + tk
    slot = gl.load(Indices + begin + pos, pos < length, 0).to(gl.int32)
    if VD > 64:
        vpos = block_start + gl.arange(0, BLOCK, layout=gl.SliceLayout(1, vlayout))
        vslot = gl.load(Indices + begin + vpos, vpos < length, 0).to(gl.int32)
    else:
        vslot = gl.convert_layout(slot, gl.SliceLayout(1, vlayout))
        vpos = block_start + gl.arange(0, BLOCK, layout=gl.SliceLayout(1, vlayout))
    hq = gl.arange(0, 16, layout=gl.SliceLayout(1, qlayout))
    dq = gl.arange(0, 512, layout=gl.SliceLayout(0, qlayout))
    if direct_qk:
        q = gl.amd.cdna4.buffer_load(Q + row * QS0, hq[:, None] * QS1 + dq[None, :], hq[:, None] < HEADS, 0)
    else:
        q = gl.load(Q + row * QS0 + hq[:, None] * QS1 + dq[None, :], hq[:, None] < HEADS, 0)
    if direct_qk:
        rope_layout: gl.constexpr = qk_a
    else:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4 if BLOCK <= 64 else 8, 1], [1, 0])
    rope_heads = gl.arange(0, 16, layout=gl.SliceLayout(1, rope_layout))
    rope_dims = gl.arange(0, 64, layout=gl.SliceLayout(0, rope_layout))
    if direct_qk:
        qr = gl.amd.cdna4.buffer_load(Q + row * QS0, rope_heads[:, None] * QS1 + 512 + rope_dims[None, :], rope_heads[:, None] < HEADS, 0)
    else:
        qr = gl.load(Q + row * QS0 + rope_heads[:, None] * QS1 + 512 + rope_dims[None, :], rope_heads[:, None] < HEADS, 0)
    q = gl.convert_layout(q, qk_a)
    qr = gl.convert_layout(qr, qk_a)
    if direct_qk:
        dk = gl.arange(0, 512, layout=gl.SliceLayout(1, qk_b))
        rk = gl.arange(0, 64, layout=gl.SliceLayout(1, qk_b))
        k_dot = gl.amd.cdna4.buffer_load(K, dk[:, None] + slot[None, :] * KS0)
        kr_dot = gl.amd.cdna4.buffer_load(K, 512 + rk[:, None] + slot[None, :] * KS0)
    else:
        dk = gl.arange(0, 512, layout=gl.SliceLayout(0, klayout))
        rk = gl.arange(0, 64, layout=gl.SliceLayout(0, klayout))
        k = gl.amd.cdna4.buffer_load(K, slot[:, None] * KS0 + dk[None, :])
        kr = gl.amd.cdna4.buffer_load(K, slot[:, None] * KS0 + 512 + rk[None, :])
    v_smem = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK, COPY_D), gl.SwizzledSharedLayout(16, 1, PHASES, [1, 0]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem, V, vslot[:, None] * VS0 + dv[None, :], mask=vpos[:, None] < length, cache_modifier='' if BLOCK <= 64 else '.cg')
    if VD >= 128:
        v_smem1 = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK, COPY_D), gl.SwizzledSharedLayout(16, 1, PHASES, [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem1, V, vslot[:, None] * VS0 + COPY_D + dv[None, :], mask=vpos[:, None] < length, cache_modifier='' if BLOCK <= 64 else '.cg')
    if VD == 256:
        v_smem2 = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK, COPY_D), gl.SwizzledSharedLayout(16, 1, PHASES, [1, 0]))
        v_smem3 = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK, COPY_D), gl.SwizzledSharedLayout(16, 1, PHASES, [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem2, V, vslot[:, None] * VS0 + 128 + dv[None, :], mask=vpos[:, None] < length)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem3, V, vslot[:, None] * VS0 + 192 + dv[None, :], mask=vpos[:, None] < length)
    gl.amd.cdna4.async_copy.commit_group()
    if direct_qk:
        scores = gl.amd.cdna4.mfma(qr, kr_dot.to(gl.bfloat16), gl.full((16, BLOCK), 0.0, gl.float32, mat))
        scores = gl.amd.cdna4.mfma(q, k_dot.to(gl.bfloat16), scores) * scale
    else:
        scores = gl.amd.cdna4.mfma(qr, gl.convert_layout(kr.T, qk_b).to(gl.bfloat16), gl.full((16, BLOCK), 0.0, gl.float32, mat))
        scores = gl.amd.cdna4.mfma(q, gl.convert_layout(k.T, qk_b).to(gl.bfloat16), scores) * scale
    scores = gl.convert_layout(scores, soft_layout)
    scores = gl.where(block_start + tm[None, :] < length, scores, -float('inf'))
    if FIRST:
        new_max = gl.max(scores, 1)
    else:
        new_max = gl.maximum(maximum, gl.max(scores, 1))
        alpha = _m1_softmax_exp(maximum - new_max)
    p = gl.where(block_start + tm[None, :] < length, _m1_softmax_exp(scores - new_max[:, None]), 0.0)
    if FIRST:
        denom = gl.sum(p, 1)
        acc = gl.full((16, VD), 0.0, gl.float32, mat)
    else:
        denom = denom * alpha + gl.sum(p, 1)
        acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, mat))[:, None]
    if VD != 256:
        high = p.to(gl.float16)
        low = ((p - high.to(gl.float32)) * 65536.0).to(gl.float16)
        probability_layout: gl.constexpr = gl.SwizzledSharedLayout(pv_width, 1, BLOCK // pv_width, [1, 0])
        high_smem = gl.allocate_shared_memory(gl.float16, (16, BLOCK), probability_layout, high)
        low_smem = gl.allocate_shared_memory(gl.float16, (16, BLOCK), probability_layout, low)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    if VD == 256:
        high = p.to(gl.float16)
        low = ((p - high.to(gl.float32)) * 65536.0).to(gl.float16)
        packed = high.to(gl.uint16, bitcast=True).to(gl.uint32) | low.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        packed = gl.convert_layout(packed, qa)
        p_hi = packed.to(gl.uint16).to(gl.float16, bitcast=True)
        p_lo = (packed >> 16).to(gl.uint16).to(gl.float16, bitcast=True)
    else:
        p_hi = gl.amd.cdna4.async_copy.load_shared_relaxed(high_smem, qa)
        p_lo = gl.amd.cdna4.async_copy.load_shared_relaxed(low_smem, qa)
    if VD == 64:
        acc = _m2_pv_leaf(p_hi, p_lo, v_smem, acc, mat, kb)
    elif VD == 256:
        acc = _m2_pv_four_panels(p_hi, p_lo, v_smem, v_smem1, v_smem2, v_smem3, acc, mat, kb)
    else:
        acc = _m2_pv_two_panels(p_hi, p_lo, v_smem, v_smem1, acc, mat, kb)
    return (new_max, denom, acc)


@gluon.jit
def _m2_attention_partials(Q, K, V, Indptr, Indices, Partials, Stats, scale, HEADS: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, SPLITS: gl.constexpr, BLOCK: gl.constexpr, VD: gl.constexpr):
    row = gl.program_id(0)
    part = gl.program_id(1)
    value_block = 0 if VD == 512 else gl.program_id(2)
    if VD == 64:
        linear = part + SPLITS * value_block
        part = linear // 32 * 4 + linear % 4
        value_block = linear // 4 % 8
    mat: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4 if BLOCK <= 64 else 8])
    h = gl.arange(0, 16, layout=gl.SliceLayout(1, mat))
    d = value_block * VD + gl.arange(0, VD, layout=gl.SliceLayout(0, mat))
    begin = gl.load(Indptr + row)
    end = gl.load(Indptr + row + 1)
    length = end - begin
    soft_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16] if BLOCK == 64 else [2, 32], [4, 1] if BLOCK == 64 else [8, 1], [1, 0])
    maximum = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, soft_layout))
    denom = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, soft_layout))
    acc = gl.full((16, VD), 0.0, gl.float32, mat)
    if BLOCK == 64 and VD == 256 or part * BLOCK < length:
        maximum, denom, acc = _m2_consume_tile(Q, K, V, Indices, scale, row, value_block, part * BLOCK, begin, length, maximum, denom, acc, HEADS, QS0, QS1, KS0, VS0, BLOCK, VD, True)
    for block_start in loop_range((part + SPLITS) * BLOCK, length, SPLITS * BLOCK, disable_licm=True):
        maximum, denom, acc = _m2_consume_tile(Q, K, V, Indices, scale, row, value_block, block_start, begin, length, maximum, denom, acc, HEADS, QS0, QS1, KS0, VS0, BLOCK, VD, False)
    if VD == 128 or VD == 256:
        d = d.to(gl.uint32)
    if VD >= 128:
        offset = row * HEADS * SPLITS * 512 + (d[None, :] // 16 * SPLITS + part) * (HEADS * 16) + h[:, None] * 16 + d[None, :] % 16
        gl.store(Partials + offset, acc, h[:, None] < HEADS)
    else:
        gl.store(Partials + (row * HEADS + h[:, None]) * (SPLITS * 512) + part * 512 + d[None, :], acc, h[:, None] < HEADS)
    if value_block == 0:
        stat_h = gl.arange(0, 16, layout=gl.SliceLayout(1, soft_layout))
        gl.store(Stats + (row * HEADS + stat_h) * (SPLITS * 2) + part * 2, maximum, stat_h < HEADS)
        gl.store(Stats + (row * HEADS + stat_h) * (SPLITS * 2) + part * 2 + 1, denom, stat_h < HEADS)


@gluon.jit
def _m2_grouped_merge(Partials, Stats, Out, HEADS: gl.constexpr, SPLITS: gl.constexpr, BLOCK_D: gl.constexpr, GROUP: gl.constexpr, THREAD_D: gl.constexpr, BUFFER_OUTPUT: gl.constexpr=False):
    row_head = gl.program_id(0) * HEADS + gl.program_id(1) * GROUP
    dblock = gl.program_id(2)
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [GROUP, 64 // GROUP // THREAD_D, THREAD_D], [1, 1, 1], [2, 1, 0])
    h = gl.arange(0, GROUP, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    split = gl.arange(0, SPLITS, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    d = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    gl.static_assert(HEADS % GROUP == 0)
    stat = h[:, None] * (SPLITS * 2) + split[None, :] * 2
    maximum = gl.amd.cdna4.buffer_load(Stats + row_head * (SPLITS * 2), stat)
    denom = gl.amd.cdna4.buffer_load(Stats + row_head * (SPLITS * 2) + 1, stat)
    global_max = gl.max(maximum, 1)
    weights = _m1_softmax_exp(maximum - global_max[:, None])
    channels = (dblock * BLOCK_D + d).to(gl.uint32)
    offset = (channels[None, None, :] // 16 * SPLITS + split[None, :, None]) * (HEADS * 16) + (gl.program_id(1) * GROUP + h[:, None, None]) * 16 + channels[None, None, :] % 16
    numerator = gl.amd.cdna4.buffer_load(Partials + gl.program_id(0) * HEADS * SPLITS * 512, offset)
    inverse = gl.div_rn(1.0, gl.sum(denom * weights, 1))
    inverse = gl.convert_layout(inverse, gl.SliceLayout(1, gl.SliceLayout(1, layout)))
    h = gl.convert_layout(h, gl.SliceLayout(1, gl.SliceLayout(1, layout)))
    out = gl.sum(numerator * weights[:, :, None], 1) * inverse[:, None]
    if BUFFER_OUTPUT:
        gl.amd.cdna4.buffer_store(out.to(Out.dtype.element_ty), Out, (row_head + h[:, None]) * 512 + dblock * BLOCK_D + d[None, :])
    else:
        gl.store(Out + (row_head + h[:, None]) * 512 + dblock * BLOCK_D + d[None, :], out)


def paged_attention_decode_m2(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:
    tokens, heads, _ = query.shape
    splits, block, value_tile = (32, 64, 128)
    numerator_size = tokens * heads * splits * 512
    output_floats = tokens * heads * 256
    storage = torch.empty((output_floats + tokens * heads * splits * 514,), device=query.device, dtype=torch.float32)
    output = storage[:output_floats].view(query.dtype).view(tokens, heads, 512)
    partials = storage[output_floats:output_floats + numerator_size]
    stats = storage[output_floats + numerator_size:]
    if output_tensor is not None:
        output = output_tensor
    _m2_attention_partials[tokens, splits, 512 // value_tile](query, key_cache, value_cache, kv_indptr, kv_indices, partials, stats, scale, heads, query.stride(0), query.stride(1), key_cache.stride(0), value_cache.stride(0), splits, block, value_tile, num_warps=4, allow_flush_denorm=True)
    group = 2
    merge_tile, thread_d = (32, 8)
    _m2_grouped_merge[tokens, triton.cdiv(heads, group), 512 // merge_tile](partials, stats, output, heads, splits, merge_tile, group, thread_d, BUFFER_OUTPUT=tokens == 4, num_warps=1, allow_flush_denorm=True)
    return output


@gluon.jit
def _m24_32_mla_split(Q, K, V, Indptr, Indices, Arena, scale, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, HEADS: gl.constexpr, S: gl.constexpr, B: gl.constexpr, RECORDS: gl.constexpr):
    Stats = Arena.to(gl.pointer_type(gl.float32))
    Partial = Arena + RECORDS * 4
    row = gl.program_id(0)
    split = gl.program_id(1)
    key_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4])
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 4], [1, 0])
    qh = gl.arange(0, 16, gl.SliceLayout(1, query_layout))
    qd = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
    q = gl.load(Q + row * QS0 + qh[:, None] * QS1 + qd[None, :], qh[:, None] < HEADS, other=0.0)
    tail_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
    th = gl.arange(0, 16, gl.SliceLayout(1, tail_layout))
    td = gl.arange(0, 64, gl.SliceLayout(0, tail_layout))
    qr = gl.load(Q + row * QS0 + th[:, None] * QS1 + 512 + td[None, :], th[:, None] < HEADS, other=0.0)
    q_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 8]], [32, 512], [1, 0])
    v_shared: gl.constexpr = gl.PaddedSharedLayout([[1024, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0], [1, 0], [2, 0], [4, 0], [8, 0], [32, 0]], [], [B, 512])
    v_smem = gl.allocate_shared_memory(V.dtype.element_ty, (B, 512), v_shared)
    q_smem = v_smem._reinterpret(Q.dtype.element_ty, (32, 512), q_shared).slice(0, 16, 0)
    qr = gl.convert_layout(qr, a_layout)
    kd = gl.arange(0, 256, gl.SliceLayout(0, key_layout))
    krd = gl.arange(0, 64, gl.SliceLayout(0, key_layout))
    kt = gl.arange(0, B, gl.SliceLayout(1, key_layout))
    copy_layout: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [32, 0]], lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]], warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[B, 512])
    vt = gl.arange(0, B, gl.SliceLayout(1, copy_layout))
    vd = gl.arange(0, 512, gl.SliceLayout(0, copy_layout))
    begin = gl.load(Indptr + row)
    length = gl.load(Indptr + row + 1) - begin
    span = gl.cdiv(length, S * B) * B
    first = split * span
    end = gl.minimum(first + span, length)
    m = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, mma))
    denom = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma))
    acc = gl.full((16, 512), 0.0, gl.float32, mma)
    for start in range(first, end, B):
        if S == 16:
            gl.inline_asm_elementwise('s_setprio 1', constraints='=s', args=[], dtype=gl.int32, is_pure=False, pack=1)
        pos = start + kt
        slots = gl.load(Indices + begin + pos, pos < end, other=0).to(gl.int32)
        kr = gl.load(K + slots[:, None] * KS0 + 512 + krd[None, :])
        kr = gl.convert_layout(kr.permute((1, 0)), b_layout).to(gl.bfloat16)
        scores = gl.full((16, B), 0.0, gl.float32, mma)
        if S == 16:
            next_key = gl.load(K + slots[:, None] * KS0 + 256 + kd[None, :])
        for panel in gl.static_range(2):
            if S == 16 and panel == 1:
                k = next_key
            else:
                k = gl.load(K + slots[:, None] * KS0 + panel * 256 + kd[None, :])
            k = gl.convert_layout(k.permute((1, 0)), b_layout).to(gl.bfloat16)
            if panel == 0:
                q_smem.store(q)
            qp = q_smem.slice(panel * 256, 256, 1).load(a_layout)
            scores = gl.amd.cdna4.mfma(qp, k, scores)
        scores = gl.amd.cdna4.mfma(qr, kr, scores) * scale
        copy_slots = gl.load(Indices + begin + start + vt, start + vt < end, other=0).to(gl.int32)
        if start + B <= end:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem, V, copy_slots[:, None] * VS0 + vd[None, :])
        else:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem, V, copy_slots[:, None] * VS0 + vd[None, :], mask=start + vt[:, None] < end)
        gl.amd.cdna4.async_copy.commit_group()
        if S == 16:
            gl.inline_asm_elementwise('s_setprio 0', constraints='=s', args=[], dtype=gl.int32, is_pure=False, pack=1)
        score_pos = start + gl.arange(0, B, gl.SliceLayout(0, mma))
        scores = gl.where(score_pos[None, :] < end, scores, -float('inf'))
        new_m = gl.maximum(m, gl.max(scores, 1))
        p = gl.where(score_pos[None, :] < end, gl.exp2((scores - new_m[:, None]) * 1.4426950408889634), 0.0)
        if S == 16:
            alpha = gl.exp(m - new_m)
        else:
            alpha = gl.exp2((m - new_m) * 1.4426950408889634)
        denom = denom * alpha + gl.sum(p, 1)
        if start != first:
            acc = acc * alpha[:, None]
        gl.amd.cdna4.async_copy.wait_group(0)
        v = v_smem.load(b_layout).to(gl.float16)
        pp = gl.convert_layout(p.to(gl.float16), a_layout)
        acc = gl.amd.cdna4.mfma(pp, v, acc)
        m = new_m
    oh = gl.arange(0, 16, gl.SliceLayout(1, mma))
    if S == 16:
        record = (row * S + split) * HEADS + oh
    else:
        record = (row * HEADS + oh) * S + split
    store_dims = gl.arange(0, 512, gl.SliceLayout(0, mma))
    GH: gl.constexpr = 4
    offsets = (((row * (HEADS // GH) + oh[:, None] // GH) * S + split) * 128 + store_dims[None, :] // 4) * GH * 4 + oh[:, None] % GH * 4 + store_dims[None, :] % 4
    gl.amd.cdna4.buffer_store(ptr=Partial, offsets=offsets, stored_value=acc.to(gl.bfloat16), mask=oh[:, None] < HEADS, cache='.wt')
    packed_stats = m.to(gl.uint32, bitcast=True).to(gl.uint64) | denom.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
    gl.amd.cdna4.buffer_store(ptr=Stats.to(gl.pointer_type(gl.uint64)), offsets=record, stored_value=packed_stats, mask=oh < HEADS, cache='.wt')


@gluon.jit
def _m24_32_merge_attention(Arena, Out, HEADS: gl.constexpr, S: gl.constexpr, BV: gl.constexpr, RECORDS: gl.constexpr):
    Stats = Arena.to(gl.pointer_type(gl.float32))
    Partial = Arena + RECORDS * 4
    if S == 16:
        row_group: gl.constexpr = 16 if RECORDS // (HEADS * S) % 16 == 0 else 1
        pid = gl.program_id(0)
        row = pid // (row_group * (512 // BV)) * row_group + pid % row_group
        tile = pid // row_group % (512 // BV)
    else:
        row = gl.program_id(0)
        tile = gl.program_id(2)
    group = gl.program_id(1)
    GH: gl.constexpr = 4
    SP: gl.constexpr = 2 if S == 16 else 1
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [SP, GH, 64 // (SP * GH)], [1, 1, 1], [2, 1, 0])
    stat_layout: gl.constexpr = gl.SliceLayout(2, layout)
    ss = gl.arange(0, S, gl.SliceLayout(1, stat_layout))
    hh = gl.arange(0, GH, gl.SliceLayout(0, stat_layout))
    heads = group * GH + hh
    if S == 16:
        records = (row * S + ss[:, None]) * HEADS + heads[None, :]
    else:
        records = (row * HEADS + heads[None, :]) * S + ss[:, None]
    if S == 16:
        m = gl.load(Stats + records * 2)
        d = gl.load(Stats + records * 2 + 1)
    else:
        packed = gl.amd.cdna4.buffer_load(Stats.to(gl.pointer_type(gl.uint64)), records)
        m = packed.to(gl.uint32).to(gl.float32, bitcast=True)
        d = (packed >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
    final_m = gl.max(m, 0)
    weights = gl.exp2((m - final_m[None, :]) * 1.4426950408889634)
    reciprocal = 1.0 / gl.sum(d * weights, 0)
    sd_layout: gl.constexpr = gl.SliceLayout(1, layout)
    dims = tile * BV + gl.arange(0, BV, gl.SliceLayout(0, sd_layout))
    offsets = (((row * (HEADS // GH) + group) * S + ss[:, None, None]) * 128 + dims[None, None, :] // 4) * GH * 4 + hh[None, :, None] * 4 + dims[None, None, :] % 4
    if S == 16:
        numerator = gl.amd.cdna4.buffer_load(Partial, offsets, cache='.cg').to(gl.float32)
    else:
        base = Partial + (row * (HEADS // GH) + group) * S * 2048
        local_offsets = ss[:, None, None] * 2048 + dims[None, None, :] // 4 * 16 + hh[None, :, None] * 4 + dims[None, None, :] % 4
        numerator = gl.amd.cdna4.buffer_load(base, local_offsets).to(gl.float32)
    reciprocal = gl.convert_layout(reciprocal, gl.SliceLayout(1, gl.SliceLayout(0, layout)))
    result = gl.sum(numerator * weights[:, :, None], 0) * reciprocal[:, None]
    out_layout: gl.constexpr = gl.SliceLayout(0, layout)
    oh = group * GH + gl.arange(0, GH, gl.SliceLayout(1, out_layout))
    od = tile * BV + gl.arange(0, BV, gl.SliceLayout(0, out_layout))
    gl.store(Out + (row * HEADS + oh[:, None]) * 512 + od[None, :], result)


def paged_attention_decode_m24_32(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:
    tokens, heads, _ = query.shape
    splits = 16
    output = query.new_empty((tokens, heads, 512))
    records = tokens * heads * splits
    arena = query.new_empty((records * (512 + 4),), dtype=torch.bfloat16)
    if output_tensor is not None:
        output = output_tensor
    _m24_32_mla_split[tokens, splits](query, key_cache, value_cache, kv_indptr, kv_indices, arena, scale, *query.stride()[:2], key_cache.stride(0), value_cache.stride(0), heads, splits, 64, records, num_warps=4)
    merge_width = 32
    merge_grid = (tokens * (512 // merge_width), heads // 4)
    _m24_32_merge_attention[merge_grid](arena, output, heads, splits, merge_width, records, num_warps=1)
    return output


@gluon.jit
def _m256_set_wave_priority(HIGH: gl.constexpr):
    if HIGH:
        gl.inline_asm_elementwise('s_setprio 1\ns_mov_b32 $0, 0', constraints='=s,~{memory}', args=[], dtype=gl.int32, is_pure=False, pack=1)
    else:
        gl.inline_asm_elementwise('s_setprio 0\ns_mov_b32 $0, 0', constraints='=s,~{memory}', args=[], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _m256_copy_cache_tile(Smem, Cache, offsets, valid, full):
    if full:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(Smem, Cache, offsets, cache_modifier='.cg')
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(Smem, Cache, offsets, valid, 0.0, cache_modifier='.cg')
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m256_attention_tile(Key, Value, Indices, value_smem, query, query_rope, sequence_start, base, end, maximum, denominator, numerator, scale, KEY_ROW_STRIDE: gl.constexpr, VALUE_ROW_STRIDE: gl.constexpr, SPLITS: gl.constexpr, BLOCK_TOKENS: gl.constexpr, FULL: gl.constexpr, FIRST: gl.constexpr):
    load_layout: gl.constexpr = gl.BlockedLayout([2, 16], [8, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4])
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=SPLITS == 2, warps_per_cta=[1, 4])
    key_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, pv_layout, 8)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, pv_layout, 8)
    rope_load_layout: gl.constexpr = gl.BlockedLayout([2, 8], [8, 8], [4, 1], [1, 0])
    value_load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [2, 32], [4, 1], [1, 0])
    dims = gl.arange(0, 512, gl.SliceLayout(0, load_layout))
    rope_dims = gl.arange(0, 64, gl.SliceLayout(0, rope_load_layout))
    tokens = gl.arange(0, BLOCK_TOKENS, gl.SliceLayout(1, load_layout))
    value_dims = gl.arange(0, 512, gl.SliceLayout(0, value_load_layout))
    value_tokens = gl.arange(0, BLOCK_TOKENS, gl.SliceLayout(1, value_load_layout))
    positions = base + tokens
    valid = gl.full((BLOCK_TOKENS,), True, gl.int1, gl.SliceLayout(1, load_layout)) if FULL and SPLITS == 2 else positions < end
    _m256_set_wave_priority(True)
    if SPLITS == 4:
        slots = gl.amd.cdna4.buffer_load(Indices + sequence_start, positions, valid, 0).to(gl.int32)
    else:
        slots = gl.load(Indices + sequence_start + positions, valid, 0).to(gl.int32)
    key = gl.amd.cdna4.buffer_load(Key, slots[:, None] * KEY_ROW_STRIDE + dims[None, :], valid[:, None])
    rope_slots = gl.convert_layout(slots, gl.SliceLayout(1, rope_load_layout))
    rope_valid = gl.convert_layout(valid, gl.SliceLayout(1, rope_load_layout))
    key_rope = gl.amd.cdna4.buffer_load(Key, rope_slots[:, None] * KEY_ROW_STRIDE + 512 + rope_dims[None, :], rope_valid[:, None], cache='.cg' if SPLITS == 2 else '')
    value_slots = gl.convert_layout(slots, gl.SliceLayout(1, value_load_layout))
    _m256_copy_cache_tile(value_smem, Value, value_slots[:, None] * VALUE_ROW_STRIDE + value_dims[None, :], base + value_tokens[:, None] < end, True if FULL else base + BLOCK_TOKENS <= end)
    if SPLITS == 2:
        _m256_set_wave_priority(False)
    key = gl.convert_layout(gl.permute(key, (1, 0)), key_layout).to(gl.bfloat16)
    key_rope = gl.convert_layout(gl.permute(key_rope, (1, 0)), key_layout).to(gl.bfloat16)
    score0 = gl.amd.cdna4.mfma(gl.amd.slice(query, (16, 128), (0, 0)), gl.amd.slice(key, (128, BLOCK_TOKENS), (0, 0)), gl.zeros((16, BLOCK_TOKENS), gl.float32, mma_layout))
    score1 = gl.amd.cdna4.mfma(gl.amd.slice(query, (16, 128), (0, 128)), gl.amd.slice(key, (128, BLOCK_TOKENS), (128, 0)), gl.zeros((16, BLOCK_TOKENS), gl.float32, mma_layout))
    score2 = gl.amd.cdna4.mfma(gl.amd.slice(query, (16, 128), (0, 256)), gl.amd.slice(key, (128, BLOCK_TOKENS), (256, 0)), gl.zeros((16, BLOCK_TOKENS), gl.float32, mma_layout))
    score3 = gl.amd.cdna4.mfma(gl.amd.slice(query, (16, 128), (0, 384)), gl.amd.slice(key, (128, BLOCK_TOKENS), (384, 0)), gl.zeros((16, BLOCK_TOKENS), gl.float32, mma_layout))
    scores = score0 + score1 + (score2 + score3)
    scores = gl.amd.cdna4.mfma(query_rope, key_rope, scores) * scale
    score_positions = base + gl.arange(0, BLOCK_TOKENS, gl.SliceLayout(0, mma_layout))
    score_valid = score_positions[None, :] < end
    if not FULL:
        scores = gl.where(score_valid, scores, -float('inf'))
    if FIRST:
        next_maximum = gl.max(scores, 1)
    else:
        next_maximum = gl.maximum(maximum, gl.max(scores, 1))
        alpha = gl.exp(maximum - next_maximum)
    if FULL:
        probabilities = gl.exp(scores - next_maximum[:, None])
    else:
        probabilities = gl.where(score_valid, gl.exp(scores - next_maximum[:, None]), 0.0)
    if FIRST:
        denominator = gl.sum(probabilities, 1)
    else:
        denominator = denominator * alpha + gl.sum(probabilities, 1)
        pv_alpha = gl.convert_layout(alpha, gl.SliceLayout(1, pv_layout))
        numerator = numerator * pv_alpha[:, None]
    if SPLITS == 2:
        p_hi_local = probabilities.to(gl.bfloat16)
        residual = probabilities - p_hi_local.to(gl.float32)
        p_mid_local = residual.to(gl.bfloat16)
        p_low_local = (residual - p_mid_local.to(gl.float32)).to(gl.bfloat16)
        p_hi = gl.convert_layout(p_hi_local, probability_layout)
        p_mid = gl.convert_layout(p_mid_local, probability_layout)
        p_low = gl.convert_layout(p_low_local, probability_layout)
    if SPLITS == 4:
        _m256_set_wave_priority(False)
    gl.amd.cdna4.async_copy.wait_group(0)
    value = gl.amd.cdna4.async_copy.load_shared_relaxed(value_smem, value_layout).to(gl.bfloat16)
    if SPLITS == 4:
        probabilities = gl.convert_layout(probabilities, probability_layout)
        p_hi = probabilities.to(gl.bfloat16)
        residual = probabilities - p_hi.to(gl.float32)
        p_mid = residual.to(gl.bfloat16)
        p_low = (residual - p_mid.to(gl.float32)).to(gl.bfloat16)
    numerator = gl.amd.cdna4.mfma(p_low, value, numerator)
    numerator = gl.amd.cdna4.mfma(p_mid, value, numerator)
    numerator = gl.amd.cdna4.mfma(p_hi, value, numerator)
    return (next_maximum, denominator, numerator)


@gluon.jit
def _m256_attention_partials(Query, Key, Value, Indptr, Indices, Partials, Overflow, Flags, Stats, scale, HEADS: gl.constexpr, QUERY_ROW_STRIDE: gl.constexpr, QUERY_HEAD_STRIDE: gl.constexpr, KEY_ROW_STRIDE: gl.constexpr, VALUE_ROW_STRIDE: gl.constexpr, SPLITS: gl.constexpr, BLOCK_TOKENS: gl.constexpr, ROW_GROUP: gl.constexpr):
    pid = gl.program_id(0)
    if SPLITS == 4:
        pid = pid.to(gl.uint32)
    row = pid // (SPLITS * ROW_GROUP) * ROW_GROUP + pid % ROW_GROUP
    split = pid // ROW_GROUP % SPLITS
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4])
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=SPLITS == 2, warps_per_cta=[1, 4])
    query_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    query_load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    heads = gl.arange(0, 16, gl.SliceLayout(1, query_load_layout))
    query_dims = gl.arange(0, 512, gl.SliceLayout(0, query_load_layout))
    query_rope_load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [0, 1])
    rope_heads = gl.arange(0, 16, gl.SliceLayout(1, query_rope_load_layout))
    query_rope_dims = gl.arange(0, 64, gl.SliceLayout(0, query_rope_load_layout))
    query_base = Query + row * QUERY_ROW_STRIDE
    query_offsets = heads[:, None] * QUERY_HEAD_STRIDE
    query = gl.amd.cdna4.buffer_load(query_base, query_offsets + query_dims[None, :], heads[:, None] < HEADS, 0.0)
    query_rope = gl.amd.cdna4.buffer_load(query_base, rope_heads[:, None] * QUERY_HEAD_STRIDE + 512 + query_rope_dims[None, :], rope_heads[:, None] < HEADS, 0.0)
    query = gl.convert_layout(query, query_layout)
    query_rope = gl.convert_layout(query_rope, query_layout)
    sequence_start = gl.load(Indptr + row)
    length = gl.load(Indptr + row + 1) - sequence_start
    tiles_per_split = gl.cdiv(gl.cdiv(length, BLOCK_TOKENS), SPLITS)
    begin = split * tiles_per_split * BLOCK_TOKENS
    end = gl.minimum(begin + tiles_per_split * BLOCK_TOKENS, length)
    maximum = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, mma_layout))
    denominator = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma_layout))
    numerator = gl.full((16, 512), 0.0, gl.float32, pv_layout)
    value_load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [2, 32], [4, 1], [1, 0])
    if SPLITS == 4:
        value_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    else:
        value_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[1024, 32]], [BLOCK_TOKENS, 512], [1, 0])
    value_smem = gl.allocate_shared_memory(Value.dtype.element_ty, [BLOCK_TOKENS, 512], value_shared_layout)
    needs_clear = end % BLOCK_TOKENS != 0
    if needs_clear:
        value_smem.store(gl.zeros((BLOCK_TOKENS, 512), Value.dtype.element_ty, value_load_layout))
    full_end = end // BLOCK_TOKENS * BLOCK_TOKENS
    loop_begin = begin
    if end > gl.maximum(begin, full_end):
        maximum, denominator, numerator = _m256_attention_tile(Key, Value, Indices, value_smem, query, query_rope, sequence_start, full_end, end, maximum, denominator, numerator, scale, KEY_ROW_STRIDE, VALUE_ROW_STRIDE, SPLITS, BLOCK_TOKENS, False, True)
    elif SPLITS == 2 and begin < full_end:
        maximum, denominator, numerator = _m256_attention_tile(Key, Value, Indices, value_smem, query, query_rope, sequence_start, begin, end, maximum, denominator, numerator, scale, KEY_ROW_STRIDE, VALUE_ROW_STRIDE, SPLITS, BLOCK_TOKENS, True, True)
        loop_begin += BLOCK_TOKENS
    for base in range(loop_begin, full_end, BLOCK_TOKENS):
        maximum, denominator, numerator = _m256_attention_tile(Key, Value, Indices, value_smem, query, query_rope, sequence_start, base, end, maximum, denominator, numerator, scale, KEY_ROW_STRIDE, VALUE_ROW_STRIDE, SPLITS, BLOCK_TOKENS, True, False)
    out_heads = gl.arange(0, 16, gl.SliceLayout(1, pv_layout))
    out_dims = gl.arange(0, 512, gl.SliceLayout(0, pv_layout))
    record = (row * SPLITS + split) * HEADS
    partial_offsets = (record + out_heads[:, None]) * 512 + out_dims[None, :]
    if SPLITS == 2:
        grouped_numerator = gl.reshape(numerator, (16, 128, 4))
        group_max = gl.max(gl.abs(grouped_numerator), 2)
        limit = gl.minimum(65000.0, denominator * 0.125)
        limit = gl.convert_layout(limit, gl.SliceLayout(1, group_max.type.layout))
        group_safe = group_max <= limit[:, None]
        grouped_safe = group_safe[:, :, None] & gl.full((16, 128, 4), True, gl.int1, grouped_numerator.type.layout)
        safe = gl.convert_layout(gl.reshape(grouped_safe, (16, 512)), pv_layout)
        gl.store(Partials + partial_offsets, numerator.to(gl.float16), out_heads[:, None] < HEADS)
        flag_components = gl.reshape(~group_safe, (16, 8, 16))
        bit_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, flag_components.type.layout))
        bits = gl.arange(0, 8, bit_layout)
        packed_flags = gl.reduce_or(flag_components.to(gl.int32) << bits[None, :, None], 1)
        flag_heads = gl.arange(0, 16, gl.SliceLayout(1, packed_flags.type.layout))
        flag_columns = gl.arange(0, 16, gl.SliceLayout(0, packed_flags.type.layout))
        gl.store(Flags + (record + flag_heads[:, None]) * 16 + flag_columns[None, :], packed_flags, flag_heads[:, None] < HEADS)
        gl.store(Overflow + partial_offsets, numerator, (out_heads[:, None] < HEADS) & ~safe)
    else:
        gl.store(Partials + partial_offsets, numerator, out_heads[:, None] < HEADS)
    stats_heads = gl.arange(0, 16, gl.SliceLayout(1, mma_layout))
    if SPLITS == 4:
        stats_offsets = (row * HEADS + stats_heads) * (2 * SPLITS) + 2 * split
        gl.store(Stats + gl.join(stats_offsets, stats_offsets + 1), gl.join(maximum, denominator), gl.join(stats_heads < HEADS, stats_heads < HEADS))
    else:
        stats_base = Stats + (row * SPLITS + split) * 2 * HEADS
        gl.store(stats_base + stats_heads, maximum, stats_heads < HEADS)
        gl.store(stats_base + HEADS + stats_heads, denominator, stats_heads < HEADS)


@gluon.jit
def _m256_merge_compact_partials(Partials, Overflow, Flags, Stats, Output, HEADS: gl.constexpr, HEAD_GROUP: gl.constexpr):
    row = gl.program_id(0)
    head_base = gl.program_id(1) * HEAD_GROUP
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [HEAD_GROUP, 1], [1, 0])
    heads = head_base + gl.arange(0, HEAD_GROUP, gl.SliceLayout(1, layout))
    dims = gl.arange(0, 512, gl.SliceLayout(0, layout))
    max0 = gl.load(Stats + row * 4 * HEADS + heads)
    max1 = gl.load(Stats + (row * 4 + 2) * HEADS + heads)
    maximum = gl.maximum(max0, max1)
    numerator = gl.full((HEAD_GROUP, 512), 0.0, gl.float32, layout)
    denominator = gl.full((HEAD_GROUP,), 0.0, gl.float32, gl.SliceLayout(1, layout))
    partial_layout: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 1, 64], [HEAD_GROUP, 1, 1], [2, 1, 0])
    partial_heads = head_base + gl.arange(0, HEAD_GROUP, gl.SliceLayout(1, gl.SliceLayout(2, partial_layout)))
    splits = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, partial_layout)))
    partial_dims = gl.arange(0, 512, gl.SliceLayout(0, gl.SliceLayout(1, partial_layout)))
    offsets = ((row * 2 + splits[None, :, None]) * HEADS + partial_heads[:, None, None]) * 512 + partial_dims[None, None, :]
    partials = gl.load(Partials + offsets, cache_modifier='.cg')
    grouped = gl.reshape(partials, (HEAD_GROUP, 2, 128, 4))
    group_layout: gl.constexpr = gl.SliceLayout(3, grouped.type.layout)
    flag_heads = head_base + gl.arange(0, HEAD_GROUP, gl.SliceLayout(1, gl.SliceLayout(2, group_layout)))
    flag_splits = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, group_layout)))
    groups = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, group_layout)))
    flags = gl.load(Flags + ((row * 2 + flag_splits[None, :, None]) * HEADS + flag_heads[:, None, None]) * 16 + groups[None, None, :] % 16).to(gl.int32)
    group_overflow = flags >> groups[None, None, :] // 16 & 1 != 0
    expanded_mask = group_overflow[:, :, :, None] & gl.full((HEAD_GROUP, 2, 128, 4), True, gl.int1, grouped.type.layout)
    overflow_mask = gl.convert_layout(gl.reshape(expanded_mask, (HEAD_GROUP, 2, 512)), partial_layout)
    partials = gl.load(Overflow + offsets, overflow_mask, partials.to(gl.float32), cache_modifier='.cg')
    for split in gl.static_range(2):
        local_maximum = max0 if split == 0 else max1
        local_denominator = gl.load(Stats + (row * 4 + split * 2 + 1) * HEADS + heads)
        alpha = gl.exp(local_maximum - maximum)
        partial = gl.convert_layout(gl.reshape(gl.amd.slice(partials, (HEAD_GROUP, 1, 512), (0, split, 0)), (HEAD_GROUP, 512)), layout)
        numerator += partial * alpha[:, None]
        denominator += local_denominator * alpha
    gl.store(Output + (row * HEADS + heads[:, None]) * 512 + dims[None, :], (numerator * (1.0 / denominator[:, None])).to(gl.bfloat16), cache_modifier='.cs')


def paged_attention_decode_m256(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:

    rows, heads, dim = query.shape
    assert (heads, dim) == (12, 576)
    assert key_cache.shape[1:] == (1, 576)
    assert value_cache.shape[1:] == (1, 512)
    assert k_scale is None and v_scale is None
    splits = 2
    output = torch.empty((rows, heads, 512), device=query.device, dtype=query.dtype)
    partials = torch.empty((rows, splits, heads, 512), device=query.device, dtype=torch.float16)
    overflow = torch.empty((rows, splits, heads, 512), device=query.device, dtype=torch.float32)
    flags = torch.empty((rows, splits, heads, 16), device=query.device, dtype=torch.uint8)
    stats_shape = (rows, splits, 2, heads)
    stats = torch.empty(stats_shape, device=query.device, dtype=torch.float32)
    row_group = 64
    if output_tensor is not None:
        output = output_tensor
    _m256_attention_partials[rows * splits,](query, key_cache, value_cache, kv_indptr, kv_indices, partials, overflow, flags, stats, scale, heads, query.stride(0), query.stride(1), key_cache.stride(0), value_cache.stride(0), splits, 64, row_group, num_warps=4)
    _m256_merge_compact_partials[rows, heads](partials, overflow, flags, stats, output, heads, 1, num_warps=1)
    return output


@gluon.jit
def _m4_consume_async(Q, K, V, Indices, scale, row, block_start, begin, length, maximum, denom, acc, HEADS: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, FIRST: gl.constexpr, BLOCK: gl.constexpr, VD: gl.constexpr, WARPS: gl.constexpr, QLAYOUT: gl.constexpr, KLAYOUT: gl.constexpr, V_CACHE: gl.constexpr, SWIZZLE: gl.constexpr, PV_PACK: gl.constexpr, VALUE_AXIS: gl.constexpr):
    mat: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WARPS])
    if WARPS == 4:
        score_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    else:
        score_layout: gl.constexpr = mat
    qa: gl.constexpr = gl.DotOperandLayout(0, mat, 16)
    kb: gl.constexpr = gl.DotOperandLayout(1, mat, 16)
    pa: gl.constexpr = gl.DotOperandLayout(0, mat, PV_PACK)
    vb: gl.constexpr = gl.DotOperandLayout(1, mat, PV_PACK)
    klayout: gl.constexpr = gl.BlockedLayout([1, KLAYOUT[0]], [KLAYOUT[1], 64 // KLAYOUT[1]], [KLAYOUT[2], WARPS // KLAYOUT[2]], [1, 0])
    PANEL_COUNT: gl.constexpr = 4 if VD == 256 or (VD == 512 and WARPS == 4) else 2 if VD == 128 else 1
    PANEL: gl.constexpr = VD // PANEL_COUNT
    vlayout: gl.constexpr = gl.BlockedLayout([1, 16], [1024 // PANEL, PANEL // 16], [WARPS, 1], [1, 0])
    tk = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, klayout))
    tv = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, vlayout))
    tm = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, score_layout))
    if VD == 512:
        value_block = 0
    else:
        value_block = gl.program_id(VALUE_AXIS)
    dv = value_block * VD + gl.arange(0, PANEL, layout=gl.SliceLayout(0, vlayout))
    pos = block_start + tk
    slot = gl.load(Indices + begin + pos, pos < length, 0).to(gl.int32)
    if VD >= 128:
        vpos = block_start + tv
        vslot = gl.load(Indices + begin + vpos, vpos < length, 0).to(gl.int32)
    else:
        vslot = gl.convert_layout(slot, gl.SliceLayout(1, vlayout))
    qlayout: gl.constexpr = gl.BlockedLayout([1, QLAYOUT[0]], [QLAYOUT[1], 64 // QLAYOUT[1]], [QLAYOUT[2], WARPS // QLAYOUT[2]], [1, 0])
    hq = gl.arange(0, 16, layout=gl.SliceLayout(1, qlayout))
    dq = gl.arange(0, 512, layout=gl.SliceLayout(0, qlayout))
    rq = gl.arange(0, 64, layout=gl.SliceLayout(0, qa))
    hr = gl.arange(0, 16, layout=gl.SliceLayout(1, qa))
    q = gl.amd.cdna4.buffer_load(Q + row * QS0, hq[:, None] * QS1 + dq[None, :], hq[:, None] < HEADS, 0)
    qr = gl.amd.cdna4.buffer_load(Q + row * QS0, hr[:, None] * QS1 + 512 + rq[None, :], hr[:, None] < HEADS, 0)
    qr = gl.convert_layout(qr, qa)
    q = gl.convert_layout(q, qa)
    direct_slot = gl.convert_layout(slot, gl.SliceLayout(0, kb))
    direct_rk = gl.arange(0, 64, layout=gl.SliceLayout(1, kb))
    if WARPS == 8:
        dk = gl.arange(0, 512, layout=gl.SliceLayout(0, klayout))
        k = gl.amd.cdna4.buffer_load(K, slot[:, None] * KS0 + dk[None, :], cache='')
    else:
        direct_dk = gl.arange(0, 512, layout=gl.SliceLayout(1, kb))
        k_dot = gl.amd.cdna4.buffer_load(K, direct_slot[None, :] * KS0 + direct_dk[:, None], cache='')
    kr_dot = gl.amd.cdna4.buffer_load(K, direct_slot[None, :] * KS0 + 512 + direct_rk[:, None], cache='')
    v_shared = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK, PANEL], gl.SwizzledSharedLayout(SWIZZLE[0], SWIZZLE[1], SWIZZLE[2], [1, 0]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(v_shared, V, vslot[:, None] * VS0 + dv[None, :], mask=block_start + tv[:, None] < length, cache_modifier=V_CACHE)
    if PANEL_COUNT >= 2:
        v_shared_1 = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK, PANEL], gl.SwizzledSharedLayout(SWIZZLE[0], SWIZZLE[1], SWIZZLE[2], [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_shared_1, V, vslot[:, None] * VS0 + PANEL + dv[None, :], mask=block_start + tv[:, None] < length, cache_modifier=V_CACHE)
    if PANEL_COUNT == 4:
        v_shared_2 = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK, PANEL], gl.SwizzledSharedLayout(SWIZZLE[0], SWIZZLE[1], SWIZZLE[2], [1, 0]))
        v_shared_3 = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK, PANEL], gl.SwizzledSharedLayout(SWIZZLE[0], SWIZZLE[1], SWIZZLE[2], [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_shared_2, V, vslot[:, None] * VS0 + 2 * PANEL + dv[None, :], mask=block_start + tv[:, None] < length, cache_modifier=V_CACHE)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_shared_3, V, vslot[:, None] * VS0 + 3 * PANEL + dv[None, :], mask=block_start + tv[:, None] < length, cache_modifier=V_CACHE)
    gl.amd.cdna4.async_copy.commit_group()
    if WARPS == 8:
        k_dot = gl.convert_layout(k.T, kb)
    scores = gl.amd.cdna4.mfma(q, k_dot.to(gl.bfloat16), gl.full((16, BLOCK), 0.0, gl.float32, mat))
    scores = gl.amd.cdna4.mfma(qr, kr_dot.to(gl.bfloat16), scores) * scale
    scores = gl.convert_layout(scores, score_layout)
    scores = gl.where(block_start + tm[None, :] < length, scores, -float('inf'))
    if FIRST:
        new_max = gl.max(scores, 1)
    else:
        new_max = gl.maximum(maximum, gl.max(scores, 1))
        alpha = _m1_softmax_exp(maximum - new_max)
    p = gl.where(block_start + tm[None, :] < length, _m1_softmax_exp(scores - new_max[:, None]), 0.0)
    if FIRST:
        denom = gl.sum(p, 1)
    else:
        denom = denom * alpha + gl.sum(p, 1)
    if FIRST:
        acc = gl.full((16, VD), 0.0, gl.float32, mat)
    else:
        acc_alpha = gl.convert_layout(alpha, gl.SliceLayout(1, mat))
        acc = acc * acc_alpha[:, None]
    p_hi_stage = p.to(gl.float16)
    p_lo_stage = ((p - p_hi_stage.to(gl.float32)) * 65536.0).to(gl.float16)
    if WARPS == 8:
        packed_stage = p_hi_stage.to(gl.uint16, bitcast=True).to(gl.uint32)
        packed_stage |= p_lo_stage.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        probability_shared = gl.allocate_shared_memory(gl.uint32, [16, BLOCK], gl.SwizzledSharedLayout(4, 1, 8, [1, 0]), packed_stage)
    else:
        P_VEC: gl.constexpr = 4 if VD == 256 else 8
        P_PHASE: gl.constexpr = 16 if VD == 64 or VD == 256 else 8
        p_hi_shared = gl.allocate_shared_memory(gl.float16, [16, BLOCK], gl.SwizzledSharedLayout(P_VEC, 1, P_PHASE, [1, 0]), p_hi_stage)
        p_lo_shared = gl.allocate_shared_memory(gl.float16, [16, BLOCK], gl.SwizzledSharedLayout(P_VEC, 1, P_PHASE, [1, 0]), p_lo_stage)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    if PANEL_COUNT == 4:
        p_hi_dot = p_hi_shared.load(pa)
        p_lo_dot = p_lo_shared.load(pa)
        acc_0 = gl.amd.slice(acc, [16, PANEL], [0, 0])
        vf_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared, vb).to(gl.float16)
        correction_0 = gl.amd.cdna4.mfma(p_lo_dot, vf_0, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_0 = gl.amd.cdna4.mfma(p_hi_dot, vf_0, acc_0)
        acc_0 = acc_0 + correction_0 * (1.0 / 65536.0)
        acc_1 = gl.amd.slice(acc, [16, PANEL], [0, PANEL])
        vf_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_1, vb).to(gl.float16)
        correction_1 = gl.amd.cdna4.mfma(p_lo_dot, vf_1, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_1 = gl.amd.cdna4.mfma(p_hi_dot, vf_1, acc_1)
        acc_1 = acc_1 + correction_1 * (1.0 / 65536.0)
        acc_2 = gl.amd.slice(acc, [16, PANEL], [0, 2 * PANEL])
        vf_2 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_2, vb).to(gl.float16)
        correction_2 = gl.amd.cdna4.mfma(p_lo_dot, vf_2, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_2 = gl.amd.cdna4.mfma(p_hi_dot, vf_2, acc_2)
        acc_2 = acc_2 + correction_2 * (1.0 / 65536.0)
        acc_3 = gl.amd.slice(acc, [16, PANEL], [0, 3 * PANEL])
        vf_3 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_3, vb).to(gl.float16)
        correction_3 = gl.amd.cdna4.mfma(p_lo_dot, vf_3, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_3 = gl.amd.cdna4.mfma(p_hi_dot, vf_3, acc_3)
        acc_3 = acc_3 + correction_3 * (1.0 / 65536.0)
        acc_01 = gl.join(acc_0, acc_1).permute(0, 2, 1).reshape((16, 2 * PANEL))
        acc_23 = gl.join(acc_2, acc_3).permute(0, 2, 1).reshape((16, 2 * PANEL))
        acc = gl.join(acc_01, acc_23).permute(0, 2, 1).reshape((16, VD))
        acc = gl.convert_layout(acc, mat, assert_trivial=True)
    elif PANEL_COUNT == 2:
        p_hi_dot = p_hi_shared.load(pa)
        p_lo_dot = p_lo_shared.load(pa)
        acc_0 = gl.amd.slice(acc, [16, PANEL], [0, 0])
        acc_1 = gl.amd.slice(acc, [16, PANEL], [0, PANEL])
        vf_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared, vb).to(gl.float16)
        correction_0 = gl.amd.cdna4.mfma(p_lo_dot, vf_0, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_0 = gl.amd.cdna4.mfma(p_hi_dot, vf_0, acc_0)
        acc_0 = acc_0 + correction_0 * (1.0 / 65536.0)
        vf_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_1, vb).to(gl.float16)
        correction_1 = gl.amd.cdna4.mfma(p_lo_dot, vf_1, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_1 = gl.amd.cdna4.mfma(p_hi_dot, vf_1, acc_1)
        acc_1 = acc_1 + correction_1 * (1.0 / 65536.0)
        acc = gl.join(acc_0, acc_1).permute(0, 2, 1).reshape((16, VD))
        acc = gl.convert_layout(acc, mat, assert_trivial=True)
    else:
        vf = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared, vb).to(gl.float16)
        if WARPS == 8:
            packed = probability_shared.load(pa)
            p_hi_dot = packed.to(gl.uint16).to(gl.float16, bitcast=True)
            p_lo_dot = (packed >> 16).to(gl.uint16).to(gl.float16, bitcast=True)
        else:
            p_hi_dot = p_hi_shared.load(pa)
            p_lo_dot = p_lo_shared.load(pa)
        correction = gl.amd.cdna4.mfma(p_lo_dot, vf, gl.full((16, VD), 0.0, gl.float32, mat))
        acc = gl.amd.cdna4.mfma(p_hi_dot, vf, acc)
        acc = acc + correction * (1.0 / 65536.0)
    return (new_max, denom, acc)


@gluon.jit
def _m4_mla_partials_async(Q, K, V, Indptr, Indices, Partials, Stats, scale, HEADS: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, SPLITS: gl.constexpr, BLOCK: gl.constexpr, VD: gl.constexpr, WARPS: gl.constexpr, QLAYOUT: gl.constexpr, KLAYOUT: gl.constexpr, V_CACHE: gl.constexpr, SWIZZLE: gl.constexpr, PV_PACK: gl.constexpr, VALUE_AXIS: gl.constexpr, SINGLE_ROW: gl.constexpr, INTERLEAVE: gl.constexpr):
    if SINGLE_ROW:
        row = 0
        part = gl.program_id(2)
    else:
        row = gl.program_id(0)
        part = gl.program_id(1)
    mat: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WARPS])
    if WARPS == 4:
        score_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    else:
        score_layout: gl.constexpr = mat
    h = gl.arange(0, 16, layout=gl.SliceLayout(1, mat))
    if VD == 512:
        value_block = 0
    else:
        value_block = gl.program_id(VALUE_AXIS)
    d = value_block * VD + gl.arange(0, VD, layout=gl.SliceLayout(0, mat))
    if VD <= 256:
        d = d.to(gl.uint32)
    begin = gl.load(Indptr + row)
    end = gl.load(Indptr + row + 1)
    length = end - begin
    maximum = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, score_layout))
    denom = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, score_layout))
    acc = gl.full((16, VD), 0.0, gl.float32, mat)
    if part * BLOCK < length:
        maximum, denom, acc = _m4_consume_async(Q, K, V, Indices, scale, row, part * BLOCK, begin, length, maximum, denom, acc, HEADS, QS0, QS1, KS0, VS0, True, BLOCK, VD, WARPS, QLAYOUT, KLAYOUT, V_CACHE, SWIZZLE, PV_PACK, VALUE_AXIS)
    for block_start in loop_range((part + SPLITS) * BLOCK, length, SPLITS * BLOCK, disable_licm=True):
        maximum, denom, acc = _m4_consume_async(Q, K, V, Indices, scale, row, block_start, begin, length, maximum, denom, acc, HEADS, QS0, QS1, KS0, VS0, False, BLOCK, VD, WARPS, QLAYOUT, KLAYOUT, V_CACHE, SWIZZLE, PV_PACK, VALUE_AXIS)
    offset = ((row * SPLITS + part) * (512 // INTERLEAVE) + d[None, :] // INTERLEAVE) * (HEADS * INTERLEAVE)
    offset += h[:, None] * INTERLEAVE + d[None, :] % INTERLEAVE
    gl.store(Partials + offset, acc, h[:, None] < HEADS)
    if value_block == 0:
        hs = gl.arange(0, 16, layout=gl.SliceLayout(1, score_layout))
        record = (row * HEADS + hs) * SPLITS + part
        gl.store(Stats + record * 2, maximum, hs < HEADS)
        gl.store(Stats + record * 2 + 1, denom, hs < HEADS)


@gluon.jit
def _m4_grouped_merge(Partials, Stats, Out, HEADS: gl.constexpr, SPLITS: gl.constexpr, GROUP: gl.constexpr, INTERLEAVE: gl.constexpr, D_TILE: gl.constexpr, UNSIGNED_CHANNELS: gl.constexpr, M_RPT: gl.constexpr, M_LANES: gl.constexpr, M_WARPS: gl.constexpr):
    row = gl.program_id(0)
    group = gl.program_id(1)
    dblock = gl.program_id(2)
    layout: gl.constexpr = gl.BlockedLayout(M_RPT, M_LANES, M_WARPS, [2, 1, 0])
    h = group * GROUP + gl.arange(0, GROUP, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    s = gl.arange(0, SPLITS, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    d = dblock * D_TILE + gl.arange(0, D_TILE, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    if UNSIGNED_CHANNELS:
        d = d.to(gl.uint32)
    record = (row * HEADS + h[:, None]) * SPLITS + s[None, :]
    full_heads: gl.constexpr = HEADS % GROUP == 0
    maximum = gl.amd.cdna4.buffer_load(Stats, record * 2, (h[:, None] < HEADS) | full_heads, -float('inf'))
    denom = gl.amd.cdna4.buffer_load(Stats, record * 2 + 1, (h[:, None] < HEADS) | full_heads, 0.0)
    global_max = gl.max(maximum, 1)
    weights = _m1_softmax_exp(maximum - global_max[:, None])
    if GROUP == 4 and SPLITS == 32:
        Partials += row * SPLITS * HEADS * 512 + dblock * D_TILE * HEADS + group * GROUP * INTERLEAVE
        offset = s[None, :, None] * HEADS * 512
        offset += (h[:, None, None] - group * GROUP) * INTERLEAVE
        offset += d[None, None, :] - dblock * D_TILE
    else:
        offset = ((row * SPLITS + s[None, :, None]) * (512 // INTERLEAVE) + d[None, None, :] // INTERLEAVE) * (HEADS * INTERLEAVE)
        offset += h[:, None, None] * INTERLEAVE + d[None, None, :] % INTERLEAVE
    numerator = gl.amd.cdna4.buffer_load(Partials, offset, (h[:, None, None] < HEADS) | full_heads, 0.0)
    total = gl.sum(denom * weights, 1)
    h = gl.convert_layout(h, gl.SliceLayout(1, gl.SliceLayout(1, layout)))
    inverse = gl.div_rn(1.0, total)
    inverse = gl.convert_layout(inverse, gl.SliceLayout(1, gl.SliceLayout(1, layout)))
    out = gl.sum(numerator * weights[:, :, None], 1) * inverse[:, None]
    gl.store(Out + (row * HEADS + h[:, None]) * 512 + d[None, :], out, (h[:, None] < HEADS) | full_heads)


def paged_attention_decode_m4(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:
    tokens, heads, _ = query.shape
    splits, block, value_tile = (32, 64, 256)
    warps, qlayout, klayout = (4, (8, 4, 1), (8, 16, 4))
    swizzle, pv_pack = ((16, 1, 2), 4)
    interleave = 16
    group, d_tile = (2, 16)
    merge_lanes, merge_warps = ((2, 8, 4), (1, 1, 1))
    records = tokens * heads * splits
    scratch = torch.empty(records * 514 + tokens * heads * 256, device=query.device, dtype=torch.float32)
    output = scratch[records * 514:].view(torch.bfloat16).view(tokens, heads, 512)
    partials = scratch[:records * 512]
    stats = scratch[records * 512:records * 514]
    grid = (tokens, splits, 512 // value_tile)
    if output_tensor is not None:
        output = output_tensor
    _m4_mla_partials_async[grid](query, key_cache, value_cache, kv_indptr, kv_indices, partials, stats, scale, heads, query.stride(0), query.stride(1), key_cache.stride(0), value_cache.stride(0), splits, block, value_tile, warps, qlayout, klayout, '', swizzle, pv_pack, 2, tokens == 1, interleave, num_warps=warps, allow_flush_denorm=True)
    _m4_grouped_merge[tokens, triton.cdiv(heads, group), 512 // d_tile](partials, stats, output, heads, splits, group, interleave, d_tile, tokens <= 4, (1, 1, 4), merge_lanes, merge_warps, num_warps=merge_warps[0], allow_flush_denorm=True)
    return output


@gluon.jit
def _m64_stat_record(row, split, head, HEADS: gl.constexpr, SPLITS: gl.constexpr):
    if SPLITS == 16:
        return (row * SPLITS + split) * HEADS + head
    else:
        return (row * HEADS + head) * SPLITS + split


@gluon.jit
def _m64_partial_offset(row, group, split, head_in_group, channel, HEADS: gl.constexpr, SPLITS: gl.constexpr):
    return (((row * (HEADS // 4) + group) * SPLITS + split) * 128 + channel // 4) * 4 * 4 + head_in_group * 4 + channel % 4


@gluon.jit
def _m64_softmax_tile(scores, valid, running_max, SPLITS: gl.constexpr):

    scores = gl.where(valid, scores, -float('inf'))
    next_max = gl.maximum(running_max, gl.max(scores, 1))
    probabilities = gl.where(valid, gl.exp2((scores - next_max[:, None]) * 1.4426950408889634), 0.0)
    if SPLITS == 16:
        rescale = gl.exp(running_max - next_max)
    else:
        rescale = gl.exp2((running_max - next_max) * 1.4426950408889634)
    return (next_max, probabilities, rescale)


@gluon.jit
def _m64_mla_split(Q, K, V, Indptr, Indices, Partial, Stats, scale, Q_ROW_STRIDE: gl.constexpr, Q_HEAD_STRIDE: gl.constexpr, K_ROW_STRIDE: gl.constexpr, V_ROW_STRIDE: gl.constexpr, HEADS: gl.constexpr, SPLITS: gl.constexpr, BLOCK_TOKENS: gl.constexpr):
    row = gl.program_id(0)
    split = gl.program_id(1)
    key_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4])
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 4], [1, 0])
    qh = gl.arange(0, 16, gl.SliceLayout(1, query_layout))
    qd = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
    q = gl.load(Q + row * Q_ROW_STRIDE + qh[:, None] * Q_HEAD_STRIDE + qd[None, :], qh[:, None] < HEADS, other=0.0)
    tail_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
    th = gl.arange(0, 16, gl.SliceLayout(1, tail_layout))
    td = gl.arange(0, 64, gl.SliceLayout(0, tail_layout))
    q_rope = gl.load(Q + row * Q_ROW_STRIDE + th[:, None] * Q_HEAD_STRIDE + 512 + td[None, :], th[:, None] < HEADS, other=0.0)
    q_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 8]], [32, 512], [1, 0])
    v_shared: gl.constexpr = gl.PaddedSharedLayout([[1024, 16]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0], [1, 0], [2, 0], [4, 0], [8, 0], [32, 0]], [], [BLOCK_TOKENS, 512])
    v_smem = gl.allocate_shared_memory(V.dtype.element_ty, (BLOCK_TOKENS, 512), v_shared)
    q_smem = v_smem._reinterpret(Q.dtype.element_ty, (32, 512), q_shared).slice(0, 16, 0)
    q_rope = gl.convert_layout(q_rope, a_layout)
    kd = gl.arange(0, 256, gl.SliceLayout(0, key_layout))
    rope_dims = gl.arange(0, 64, gl.SliceLayout(0, key_layout))
    kt = gl.arange(0, BLOCK_TOKENS, gl.SliceLayout(1, key_layout))
    copy_layout: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [32, 0]], lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]], warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_TOKENS, 512])
    vt = gl.arange(0, BLOCK_TOKENS, gl.SliceLayout(1, copy_layout))
    vd = gl.arange(0, 512, gl.SliceLayout(0, copy_layout))
    begin = gl.load(Indptr + row)
    length = gl.load(Indptr + row + 1) - begin
    span = gl.cdiv(length, SPLITS * BLOCK_TOKENS) * BLOCK_TOKENS
    first = split * span
    end = gl.minimum(first + span, length)
    running_max = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, mma))
    running_sum = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma))
    accumulator = gl.full((16, 512), 0.0, gl.float32, mma)
    for start in range(first, end, BLOCK_TOKENS):
        gl.inline_asm_elementwise('s_setprio 1', constraints='=s', args=[], dtype=gl.int32, is_pure=False, pack=1)
        pos = start + kt
        slots = gl.load(Indices + begin + pos, pos < end, other=0).to(gl.int32)
        k_rope = gl.load(K + slots[:, None] * K_ROW_STRIDE + 512 + rope_dims[None, :])
        if SPLITS == 16:
            k_rope = gl.convert_layout(k_rope.permute((1, 0)), b_layout).to(gl.bfloat16)
        k_later = gl.load(K + slots[:, None] * K_ROW_STRIDE + 256 + kd[None, :])
        scores = gl.full((16, BLOCK_TOKENS), 0.0, gl.float32, mma)
        for panel in gl.static_range(2):
            if panel == 1:
                k = k_later
            else:
                k = gl.load(K + slots[:, None] * K_ROW_STRIDE + panel * 256 + kd[None, :])
            k = gl.convert_layout(k.permute((1, 0)), b_layout).to(gl.bfloat16)
            if panel == 0:
                q_smem.store(q)
            qp = q_smem.slice(panel * 256, 256, 1).load(a_layout)
            scores = gl.amd.cdna4.mfma(qp, k, scores)
        if SPLITS == 8:
            k_rope = gl.convert_layout(k_rope.permute((1, 0)), b_layout).to(gl.bfloat16)
        scores = gl.amd.cdna4.mfma(q_rope, k_rope, scores) * scale
        copy_slots = gl.load(Indices + begin + start + vt, start + vt < end, other=0).to(gl.int32)
        if start + BLOCK_TOKENS <= end:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem, V, copy_slots[:, None] * V_ROW_STRIDE + vd[None, :])
        else:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(v_smem, V, copy_slots[:, None] * V_ROW_STRIDE + vd[None, :], mask=start + vt[:, None] < end)
        gl.amd.cdna4.async_copy.commit_group()
        gl.inline_asm_elementwise('s_setprio 0', constraints='=s', args=[], dtype=gl.int32, is_pure=False, pack=1)
        score_pos = start + gl.arange(0, BLOCK_TOKENS, gl.SliceLayout(0, mma))
        next_max, p, alpha = _m64_softmax_tile(scores, score_pos[None, :] < end, running_max, SPLITS)
        running_sum = running_sum * alpha + gl.sum(p, 1)
        if start != first:
            accumulator = accumulator * alpha[:, None]
        gl.amd.cdna4.async_copy.wait_group(0)
        v = v_smem.load(b_layout).to(gl.float16)
        pp = gl.convert_layout(p.to(gl.float16), a_layout)
        accumulator = gl.amd.cdna4.mfma(pp, v, accumulator)
        running_max = next_max
    oh = gl.arange(0, 16, gl.SliceLayout(1, mma))
    record = _m64_stat_record(row, split, oh, HEADS, SPLITS)
    store_dims = gl.arange(0, 512, gl.SliceLayout(0, mma))
    offsets = _m64_partial_offset(row, oh[:, None] // 4, split, oh[:, None] % 4, store_dims[None, :], HEADS, SPLITS)
    gl.amd.cdna4.buffer_store(ptr=Partial, offsets=offsets, stored_value=accumulator.to(gl.bfloat16), mask=oh[:, None] < HEADS, cache='.wt')
    packed = running_max.to(gl.uint32, bitcast=True).to(gl.uint64)
    packed = packed | running_sum.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
    gl.amd.cdna4.buffer_store(ptr=Stats.to(gl.pointer_type(gl.uint64)), offsets=record, stored_value=packed, mask=oh < HEADS, cache='.wt')


@gluon.jit
def _m64_merge_attention(Partial, Stats, Out, HEADS: gl.constexpr, SPLITS: gl.constexpr, BLOCK_VALUES: gl.constexpr, ROW_GROUP: gl.constexpr=1):
    if SPLITS == 16:
        row = gl.program_id(0) % ROW_GROUP + gl.program_id(2) * ROW_GROUP
        tile = gl.program_id(0) // ROW_GROUP
    else:
        row = gl.program_id(0)
        tile = gl.program_id(2)
    group = gl.program_id(1)
    GROUP_HEADS: gl.constexpr = 4
    SPLIT_LANES: gl.constexpr = 2 if SPLITS == 16 else 1
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [SPLIT_LANES, GROUP_HEADS, 64 // (SPLIT_LANES * GROUP_HEADS)], [1, 1, 1], [2, 1, 0])
    stat_layout: gl.constexpr = gl.SliceLayout(2, layout)
    split_ids = gl.arange(0, SPLITS, gl.SliceLayout(1, stat_layout))
    group_heads = gl.arange(0, GROUP_HEADS, gl.SliceLayout(0, stat_layout))
    heads = group * GROUP_HEADS + group_heads
    records = _m64_stat_record(row, split_ids[:, None], heads[None, :], HEADS, SPLITS)
    if SPLITS == 16:
        split_max = gl.load(Stats + records * 2)
        split_sum = gl.load(Stats + records * 2 + 1)
    else:
        split_max = gl.amd.cdna4.buffer_load(Stats, records * 2)
        split_sum = gl.amd.cdna4.buffer_load(Stats, records * 2 + 1)
    final_max = gl.max(split_max, 0)
    weights = gl.exp2((split_max - final_max[None, :]) * 1.4426950408889634)
    reciprocal = 1.0 / gl.sum(split_sum * weights, 0)
    sd_layout: gl.constexpr = gl.SliceLayout(1, layout)
    dims = tile * BLOCK_VALUES + gl.arange(0, BLOCK_VALUES, gl.SliceLayout(0, sd_layout))
    if SPLITS == 16:
        offsets = _m64_partial_offset(row, group, split_ids[:, None, None], group_heads[None, :, None], dims[None, None, :], HEADS, SPLITS)
        numerator = gl.amd.cdna4.buffer_load(Partial, offsets, cache='.cg').to(gl.float32)
    else:
        base_partial = Partial + (row * (HEADS // 4) + group) * SPLITS * 2048
        local_offsets = split_ids[:, None, None] * 2048 + dims[None, None, :] // 4 * 16 + group_heads[None, :, None] * 4 + dims[None, None, :] % 4
        numerator = gl.amd.cdna4.buffer_load(base_partial, local_offsets).to(gl.float32)
    reciprocal = gl.convert_layout(reciprocal, gl.SliceLayout(1, gl.SliceLayout(0, layout)))
    result = gl.sum(numerator * weights[:, :, None], 0) * reciprocal[:, None]
    out_layout: gl.constexpr = gl.SliceLayout(0, layout)
    oh = group * GROUP_HEADS + gl.arange(0, GROUP_HEADS, gl.SliceLayout(1, out_layout))
    od = tile * BLOCK_VALUES + gl.arange(0, BLOCK_VALUES, gl.SliceLayout(0, out_layout))
    gl.store(Out + (row * HEADS + oh[:, None]) * 512 + od[None, :], result)


def paged_attention_decode_m64(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:
    tokens, heads, _ = query.shape
    splits, merge_width = (8, 64)
    records = tokens * heads * splits
    stat_words = records * 2
    partial_words = records * 512 // 2
    arena = query.new_empty((stat_words + partial_words,), dtype=torch.float32)
    stats = arena[:stat_words]
    partial = arena[stat_words:].view(torch.bfloat16)
    output = query.new_empty((tokens, heads, 512))
    if output_tensor is not None:
        output = output_tensor
    _m64_mla_split[tokens, splits](query, key_cache, value_cache, kv_indptr, kv_indices, partial, stats, scale, *query.stride()[:2], key_cache.stride(0), value_cache.stride(0), heads, splits, 64, num_warps=4)
    row_group = 1
    merge_grid = (tokens, heads // 4, 512 // merge_width)
    _m64_merge_attention[merge_grid](partial, stats, output, heads, splits, merge_width, row_group, num_warps=1)
    return output


@gluon.jit
def _m8_consume_async(Q, K, V, Indices, scale, row, block_start, begin, length, maximum, denom, acc, HEADS: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, FIRST: gl.constexpr, BLOCK: gl.constexpr, VD: gl.constexpr, WARPS: gl.constexpr, QLAYOUT: gl.constexpr, KLAYOUT: gl.constexpr, V_CACHE: gl.constexpr, SWIZZLE: gl.constexpr, PV_PACK: gl.constexpr, VALUE_AXIS: gl.constexpr):
    mat: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WARPS])
    if VD <= 128 or (VD == 512 and WARPS == 4):
        score_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    else:
        score_layout: gl.constexpr = mat
    qk_pack: gl.constexpr = 16
    pv_pack: gl.constexpr = PV_PACK
    qa: gl.constexpr = gl.DotOperandLayout(0, mat, qk_pack)
    kb: gl.constexpr = gl.DotOperandLayout(1, mat, qk_pack)
    pa: gl.constexpr = gl.DotOperandLayout(0, mat, pv_pack)
    vb: gl.constexpr = gl.DotOperandLayout(1, mat, pv_pack)
    klayout: gl.constexpr = gl.BlockedLayout([1, KLAYOUT[0]], [KLAYOUT[1], 64 // KLAYOUT[1]], [KLAYOUT[2], WARPS // KLAYOUT[2]], [1, 0])
    PANEL_COUNT: gl.constexpr = 4 if VD == 256 or (VD == 512 and WARPS == 4) else 2 if VD == 128 else 1
    PANEL: gl.constexpr = VD // PANEL_COUNT
    vlayout: gl.constexpr = gl.BlockedLayout([1, 16], [1024 // PANEL, PANEL // 16], [WARPS, 1], [1, 0])
    tk = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, klayout))
    tv = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, vlayout))
    tm = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, score_layout))
    if VD == 512:
        value_block = 0
    else:
        value_block = gl.program_id(VALUE_AXIS)
    dv = value_block * VD + gl.arange(0, PANEL, layout=gl.SliceLayout(0, vlayout))
    pos = block_start + tk
    slot = gl.load(Indices + begin + pos, pos < length, 0).to(gl.int32)
    if VD >= 128:
        vpos = block_start + tv
        vslot = gl.load(Indices + begin + vpos, vpos < length, 0).to(gl.int32)
    else:
        vslot = gl.convert_layout(slot, gl.SliceLayout(1, vlayout))
    qlayout: gl.constexpr = gl.BlockedLayout([1, QLAYOUT[0]], [QLAYOUT[1], 64 // QLAYOUT[1]], [QLAYOUT[2], WARPS // QLAYOUT[2]], [1, 0])
    hq = gl.arange(0, 16, layout=gl.SliceLayout(1, qlayout))
    dq = gl.arange(0, 512, layout=gl.SliceLayout(0, qlayout))
    rq = gl.arange(0, 64, layout=gl.SliceLayout(0, qa))
    hr = gl.arange(0, 16, layout=gl.SliceLayout(1, qa))
    q = gl.amd.cdna4.buffer_load(Q + row * QS0, hq[:, None] * QS1 + dq[None, :], hq[:, None] < HEADS, 0)
    qr = gl.amd.cdna4.buffer_load(Q + row * QS0, hr[:, None] * QS1 + 512 + rq[None, :], hr[:, None] < HEADS, 0)
    qr = gl.convert_layout(qr, qa)
    q = gl.convert_layout(q, qa)
    direct_slot = gl.convert_layout(slot, gl.SliceLayout(0, kb))
    direct_rk = gl.arange(0, 64, layout=gl.SliceLayout(1, kb))
    if WARPS == 8:
        dk = gl.arange(0, 512, layout=gl.SliceLayout(0, klayout))
        k = gl.amd.cdna4.buffer_load(K, slot[:, None] * KS0 + dk[None, :], cache='')
    else:
        direct_dk = gl.arange(0, 512, layout=gl.SliceLayout(1, kb))
        k_dot = gl.amd.cdna4.buffer_load(K, direct_slot[None, :] * KS0 + direct_dk[:, None], cache='')
    kr_dot = gl.amd.cdna4.buffer_load(K, direct_slot[None, :] * KS0 + 512 + direct_rk[:, None], cache='')
    v_shared = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK, PANEL], gl.SwizzledSharedLayout(SWIZZLE[0], SWIZZLE[1], SWIZZLE[2], [1, 0]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(v_shared, V, vslot[:, None] * VS0 + dv[None, :], mask=block_start + tv[:, None] < length, cache_modifier=V_CACHE)
    if PANEL_COUNT >= 2:
        v_shared_1 = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK, PANEL], gl.SwizzledSharedLayout(SWIZZLE[0], SWIZZLE[1], SWIZZLE[2], [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_shared_1, V, vslot[:, None] * VS0 + PANEL + dv[None, :], mask=block_start + tv[:, None] < length, cache_modifier=V_CACHE)
    if PANEL_COUNT == 4:
        v_shared_2 = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK, PANEL], gl.SwizzledSharedLayout(SWIZZLE[0], SWIZZLE[1], SWIZZLE[2], [1, 0]))
        v_shared_3 = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK, PANEL], gl.SwizzledSharedLayout(SWIZZLE[0], SWIZZLE[1], SWIZZLE[2], [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_shared_2, V, vslot[:, None] * VS0 + 2 * PANEL + dv[None, :], mask=block_start + tv[:, None] < length, cache_modifier=V_CACHE)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(v_shared_3, V, vslot[:, None] * VS0 + 3 * PANEL + dv[None, :], mask=block_start + tv[:, None] < length, cache_modifier=V_CACHE)
    gl.amd.cdna4.async_copy.commit_group()
    if WARPS == 8:
        k_dot = gl.convert_layout(k.T, kb)
    scores = gl.amd.cdna4.mfma(q, k_dot.to(gl.bfloat16), gl.full((16, BLOCK), 0.0, gl.float32, mat))
    scores = gl.amd.cdna4.mfma(qr, kr_dot.to(gl.bfloat16), scores) * scale
    scores = gl.convert_layout(scores, score_layout)
    scores = gl.where(block_start + tm[None, :] < length, scores, -float('inf'))
    if FIRST:
        new_max = gl.max(scores, 1)
    else:
        new_max = gl.maximum(maximum, gl.max(scores, 1))
        alpha = _m1_softmax_exp(maximum - new_max)
    p = gl.where(block_start + tm[None, :] < length, _m1_softmax_exp(scores - new_max[:, None]), 0.0)
    if FIRST:
        denom = gl.sum(p, 1)
    else:
        denom = denom * alpha + gl.sum(p, 1)
    if FIRST:
        acc = gl.full((16, VD), 0.0, gl.float32, mat)
    else:
        acc_alpha = gl.convert_layout(alpha, gl.SliceLayout(1, mat))
        acc = acc * acc_alpha[:, None]
    if VD == 128 or (VD == 512 and WARPS == 4):
        p_hi_stage = p.to(gl.float16)
        p_lo_stage = ((p - p_hi_stage.to(gl.float32)) * 65536.0).to(gl.float16)
        probability_hi_shared = gl.allocate_shared_memory(gl.float16, [16, BLOCK], gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), p_hi_stage)
        probability_lo_shared = gl.allocate_shared_memory(gl.float16, [16, BLOCK], gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), p_lo_stage)
    if WARPS == 8 or VD == 64:
        p_hi_stage = p.to(gl.float16)
        p_lo_stage = ((p - p_hi_stage.to(gl.float32)) * 65536.0).to(gl.float16)
        packed_stage = p_hi_stage.to(gl.uint16, bitcast=True).to(gl.uint32)
        packed_stage |= p_lo_stage.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        probability_shared = gl.allocate_shared_memory(gl.uint32, [16, BLOCK], gl.SwizzledSharedLayout(8 if VD == 64 else 4, 1, 8, [1, 0]), packed_stage)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    if PANEL_COUNT == 4:
        if VD == 512:
            p_hi_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(probability_hi_shared, pa)
            p_lo_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(probability_lo_shared, pa)
        else:
            p_hi = p.to(gl.float16)
            p_lo = ((p - p_hi.to(gl.float32)) * 65536.0).to(gl.float16)
            p_hi_dot = gl.convert_layout(p_hi, pa)
            p_lo_dot = gl.convert_layout(p_lo, pa)
        acc_0 = gl.amd.slice(acc, [16, PANEL], [0, 0])
        vf_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared, vb).to(gl.float16)
        correction_0 = gl.amd.cdna4.mfma(p_lo_dot, vf_0, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_0 = gl.amd.cdna4.mfma(p_hi_dot, vf_0, acc_0)
        acc_0 = acc_0 + correction_0 * (1.0 / 65536.0)
        acc_1 = gl.amd.slice(acc, [16, PANEL], [0, PANEL])
        vf_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_1, vb).to(gl.float16)
        correction_1 = gl.amd.cdna4.mfma(p_lo_dot, vf_1, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_1 = gl.amd.cdna4.mfma(p_hi_dot, vf_1, acc_1)
        acc_1 = acc_1 + correction_1 * (1.0 / 65536.0)
        acc_2 = gl.amd.slice(acc, [16, PANEL], [0, 2 * PANEL])
        vf_2 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_2, vb).to(gl.float16)
        correction_2 = gl.amd.cdna4.mfma(p_lo_dot, vf_2, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_2 = gl.amd.cdna4.mfma(p_hi_dot, vf_2, acc_2)
        acc_2 = acc_2 + correction_2 * (1.0 / 65536.0)
        acc_3 = gl.amd.slice(acc, [16, PANEL], [0, 3 * PANEL])
        vf_3 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_3, vb).to(gl.float16)
        correction_3 = gl.amd.cdna4.mfma(p_lo_dot, vf_3, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_3 = gl.amd.cdna4.mfma(p_hi_dot, vf_3, acc_3)
        acc_3 = acc_3 + correction_3 * (1.0 / 65536.0)
        acc_01 = gl.join(acc_0, acc_1).permute(0, 2, 1).reshape((16, 2 * PANEL))
        acc_23 = gl.join(acc_2, acc_3).permute(0, 2, 1).reshape((16, 2 * PANEL))
        acc = gl.join(acc_01, acc_23).permute(0, 2, 1).reshape((16, VD))
        acc = gl.convert_layout(acc, mat, assert_trivial=True)
    elif PANEL_COUNT == 2:
        if VD == 128:
            p_hi_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(probability_hi_shared, pa)
            p_lo_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(probability_lo_shared, pa)
        else:
            p_hi = p.to(gl.float16)
            p_lo = ((p - p_hi.to(gl.float32)) * 65536.0).to(gl.float16)
            p_hi_dot = gl.convert_layout(p_hi, pa)
            p_lo_dot = gl.convert_layout(p_lo, pa)
        acc_0 = gl.amd.slice(acc, [16, PANEL], [0, 0])
        acc_1 = gl.amd.slice(acc, [16, PANEL], [0, PANEL])
        vf_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared, vb).to(gl.float16)
        if VD != 128:
            vf_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_1, vb).to(gl.float16)
        correction_0 = gl.amd.cdna4.mfma(p_lo_dot, vf_0, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_0 = gl.amd.cdna4.mfma(p_hi_dot, vf_0, acc_0)
        acc_0 = acc_0 + correction_0 * (1.0 / 65536.0)
        if VD == 128:
            vf_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared_1, vb).to(gl.float16)
        correction_1 = gl.amd.cdna4.mfma(p_lo_dot, vf_1, gl.full((16, PANEL), 0.0, gl.float32, mat))
        acc_1 = gl.amd.cdna4.mfma(p_hi_dot, vf_1, acc_1)
        acc_1 = acc_1 + correction_1 * (1.0 / 65536.0)
        acc = gl.join(acc_0, acc_1).permute(0, 2, 1).reshape((16, VD))
        acc = gl.convert_layout(acc, mat, assert_trivial=True)
    else:
        vf = gl.amd.cdna4.async_copy.load_shared_relaxed(v_shared, vb).to(gl.float16)
        packed = probability_shared.load(pa)
        p_hi_dot = packed.to(gl.uint16).to(gl.float16, bitcast=True)
        p_lo_dot = (packed >> 16).to(gl.uint16).to(gl.float16, bitcast=True)
        correction = gl.amd.cdna4.mfma(p_lo_dot, vf, gl.full((16, VD), 0.0, gl.float32, mat))
        acc = gl.amd.cdna4.mfma(p_hi_dot, vf, acc)
        acc = acc + correction * (1.0 / 65536.0)
    return (new_max, denom, acc)


@gluon.jit
def _m8_mla_partials_async(Q, K, V, Indptr, Indices, Partials, Stats, scale, HEADS: gl.constexpr, QS0: gl.constexpr, QS1: gl.constexpr, KS0: gl.constexpr, VS0: gl.constexpr, SPLITS: gl.constexpr, BLOCK: gl.constexpr, VD: gl.constexpr, WARPS: gl.constexpr, QLAYOUT: gl.constexpr, KLAYOUT: gl.constexpr, V_CACHE: gl.constexpr, SWIZZLE: gl.constexpr, PV_PACK: gl.constexpr, VALUE_AXIS: gl.constexpr, SINGLE_ROW: gl.constexpr, INTERLEAVE: gl.constexpr):
    if SINGLE_ROW:
        row = 0
        part = gl.program_id(2)
    else:
        row = gl.program_id(0)
        part = gl.program_id(1)
    mat: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WARPS])
    if VD <= 128 or (VD == 512 and WARPS == 4):
        score_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    else:
        score_layout: gl.constexpr = mat
    h = gl.arange(0, 16, layout=gl.SliceLayout(1, mat))
    if VD == 512:
        value_block = 0
    else:
        value_block = gl.program_id(VALUE_AXIS)
    d = value_block * VD + gl.arange(0, VD, layout=gl.SliceLayout(0, mat))
    if VD == 128 or VD == 256:
        d = d.to(gl.uint32)
    begin = gl.load(Indptr + row)
    end = gl.load(Indptr + row + 1)
    length = end - begin
    maximum = gl.full((16,), -float('inf'), gl.float32, gl.SliceLayout(1, score_layout))
    denom = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, score_layout))
    acc = gl.full((16, VD), 0.0, gl.float32, mat)
    if part * BLOCK < length:
        maximum, denom, acc = _m8_consume_async(Q, K, V, Indices, scale, row, part * BLOCK, begin, length, maximum, denom, acc, HEADS, QS0, QS1, KS0, VS0, True, BLOCK, VD, WARPS, QLAYOUT, KLAYOUT, V_CACHE, SWIZZLE, PV_PACK, VALUE_AXIS)
    for block_start in loop_range((part + SPLITS) * BLOCK, length, SPLITS * BLOCK, disable_licm=True):
        maximum, denom, acc = _m8_consume_async(Q, K, V, Indices, scale, row, block_start, begin, length, maximum, denom, acc, HEADS, QS0, QS1, KS0, VS0, False, BLOCK, VD, WARPS, QLAYOUT, KLAYOUT, V_CACHE, SWIZZLE, PV_PACK, VALUE_AXIS)
    offset = ((row * SPLITS + part) * (512 // INTERLEAVE) + d[None, :] // INTERLEAVE) * (HEADS * INTERLEAVE)
    offset += h[:, None] * INTERLEAVE + d[None, :] % INTERLEAVE
    gl.store(Partials + offset, acc, h[:, None] < HEADS)
    if value_block == 0:
        hs = gl.arange(0, 16, layout=gl.SliceLayout(1, score_layout))
        record = (row * HEADS + hs) * SPLITS + part
        gl.store(Stats + record * 2, maximum, hs < HEADS)
        gl.store(Stats + record * 2 + 1, denom, hs < HEADS)


def paged_attention_decode_m8(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_context: int, k_scale: torch.Tensor | None=None, v_scale: torch.Tensor | None=None, output_tensor=None) -> torch.Tensor:
    tokens, heads, _ = query.shape
    splits, block, value_tile = (32, 64, 512)
    warps, qlayout, klayout = (4, (8, 4, 1), (8, 16, 4))
    swizzle, pv_pack = ((16, 1, 8), 8)
    interleave = 16
    group, d_tile = (4, 16)
    merge_lanes, merge_warps = ((4, 4, 4), (1, 1, 1))
    records = tokens * heads * splits
    output = torch.empty((tokens, heads, 512), device=query.device, dtype=query.dtype)
    scratch = torch.empty(records * 514, device=query.device, dtype=torch.float32)
    partials = scratch[:records * 512]
    stats = scratch[records * 512:records * 514]
    grid = (tokens, splits, 512 // value_tile)
    if output_tensor is not None:
        output = output_tensor
    _m8_mla_partials_async[grid](query, key_cache, value_cache, kv_indptr, kv_indices, partials, stats, scale, heads, query.stride(0), query.stride(1), key_cache.stride(0), value_cache.stride(0), splits, block, value_tile, warps, qlayout, klayout, '', swizzle, pv_pack, 2, tokens == 1, interleave, num_warps=warps, allow_flush_denorm=True)
    _m4_grouped_merge[tokens, triton.cdiv(heads, group), 512 // d_tile](partials, stats, output, heads, splits, group, interleave, d_tile, tokens in (2, 4), (1, 1, 4), merge_lanes, merge_warps, num_warps=merge_warps[0], allow_flush_denorm=True)
    return output
