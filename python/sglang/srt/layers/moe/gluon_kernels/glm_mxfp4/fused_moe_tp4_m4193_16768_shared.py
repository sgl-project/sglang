# fmt: off
from math import prod

"""Shared TP4 fused-MoE specialization for active batches M=4193 through M=16768."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _add(a, b):
    return a + b

@gluon.jit
def _maximum(a, b):
    return gl.maximum(a, b)

@gluon.jit
def _sigmoid(x):
    return 1.0 / (1.0 + gl.exp(-x))

@gluon.jit
def _router_projection(X, W, Y, M: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BK, BK // 8], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [BK // 8, 512 // BK], [1, 4], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    rows = gl.program_id(0) * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    cols = gl.program_id(1) * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    a = gl.load(X + rows[:, None] * SX + ak[None, :], rows[:, None] < M, 0)
    b = gl.load(W + cols[None, :] * K + bk[:, None])
    for k in range(K // BK - 1):
        aa = gl.convert_layout(a, ad)
        bb = gl.convert_layout(b, bd)
        next_k = k + 1
        a = gl.load(X + rows[:, None] * SX + next_k * BK + ak[None, :], rows[:, None] < M, 0)
        b = gl.load(W + cols[None, :] * K + next_k * BK + bk[:, None])
        acc = gl.amd.cdna4.mfma(aa, bb, acc)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(a, ad), gl.convert_layout(b, bd), acc)
    rm = gl.program_id(0) * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    cn = gl.program_id(1) * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(Y + rm[:, None] * 256 + cn[None, :], acc, rm[:, None] < M)

@gluon.jit
def _router(Logits, Bias, Ids, Weights, Inverse, expert_start, LOCAL_EXPERTS: gl.constexpr, SCALE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    m = gl.program_id(0)
    e = gl.arange(0, 256, layout)
    probability = _sigmoid(gl.load(Logits + m * 256 + e).to(gl.float32))
    score = probability + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    selected = gl.full((256,), 0.0, gl.float32, layout)
    selected_ids = gl.full((256,), 0, gl.int32, layout)
    total = 0.0
    for j in range(8):
        maximum = gl.max(score, 0)
        idx = gl.min(gl.where(available & (score == maximum), e, 256), 0)
        if idx >= 256:
            idx = gl.min(gl.where(available, e, 256), 0)
        index = gl.full((1,), idx, gl.int32, layout)
        prob = gl.sum(gl.gather(probability, index, 0), 0)
        local = (idx >= expert_start) & (idx < expert_start + LOCAL_EXPERTS)
        gl.store(Ids + m * 9 + j, gl.where(local, idx, 257))
        selected = gl.where(e == j, prob, selected)
        selected_ids = gl.where(e == j, idx, selected_ids)
        total += prob
        available = available & (e != idx)
        score = gl.where(e == idx, -float('inf'), score)
    owned = (selected_ids >= expert_start) & (selected_ids < expert_start + LOCAL_EXPERTS)
    gl.store(Weights + m * 8 + e, gl.where(owned, selected / total * SCALE, 0.0), e < 8)
    gl.store(Inverse + m * 8 + e, 0, e < 8)
    gl.store(Ids + m * 9 + 8, 256)

@gluon.jit
def _chunk_counts(Ids, Counts, Sorted, Experts, DownExperts, ROUTES: gl.constexpr, CHUNKS: gl.constexpr, CAPACITY: gl.constexpr, BM: gl.constexpr, DOWN_TILES: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    chunk = gl.program_id(0)
    r = chunk * 256 + gl.arange(0, 256, layout)
    e = gl.load(Ids + r, r < ROUTES, 257)
    histogram = gl.histogram(e, 512, layout=layout)
    expert = gl.arange(0, 512, layout)
    gl.store(Counts + expert * CHUNKS + chunk, histogram, (expert < 257) & (chunk < CHUNKS))
    r = chunk * 1024 + gl.arange(0, 1024, layout)
    gl.store(Sorted + r, -1, r < CAPACITY)
    gl.store(Experts + r, -1, r < CAPACITY // BM)
    gl.store(DownExperts + r, -1, r < DOWN_TILES)

@gluon.jit
def _chunk_prefix(Counts, Prefix, Totals, CHUNKS: gl.constexpr, BLOCK: gl.constexpr):
    e = gl.program_id(0)
    c = gl.arange(0, BLOCK, gl.BlockedLayout([1], [64], [4], [0]))
    counts = gl.load(Counts + e * CHUNKS + c, c < CHUNKS, 0)
    prefix = gl.associative_scan(counts, 0, _add) - counts
    gl.store(Prefix + e * CHUNKS + c, prefix, c < CHUNKS)
    gl.store(Totals + e, gl.sum(counts, 0))

@gluon.jit
def _build_expert_blocks(Counts, Offsets, Experts, DownExperts, BM: gl.constexpr, BLOCK: gl.constexpr, DOWN_BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    e = gl.program_id(0)
    count = gl.load(Counts + e)
    all_e = gl.arange(0, 512, layout)
    all_counts = gl.load(Counts + all_e, all_e < 257, 0)
    offset = gl.sum(gl.where(all_e < e, gl.cdiv(all_counts, BM), 0), 0)
    gl.store(Offsets + e, offset * BM)
    compact = gl.sum(gl.where(all_e < e, all_counts, 0), 0)
    gl.store(Offsets + 257 + e, compact - offset * BM)
    i = gl.arange(0, BLOCK, layout)
    rows = gl.minimum(count - i * BM, BM)
    descriptor = e + rows * 512
    gl.store(Experts + offset + i, descriptor, i < gl.cdiv(count, BM))
    if e < 256:
        down_offset = gl.sum(gl.where(all_e < e, gl.cdiv(all_counts, 64), 0), 0)
        j = gl.arange(0, DOWN_BLOCK, layout)
        down_rows = gl.minimum(count - j * 64, 64)
        activation_tile = offset * (BM // 64) + j
        down_descriptor = (compact + j * 64).to(gl.int64) << 32 | activation_tile.to(gl.int64) << 16 | down_rows * 512 + e
        gl.store(DownExperts + down_offset + j, down_descriptor, j < gl.cdiv(count, 64))

@gluon.jit
def _scatter_chunk(Ids, Offsets, Prefix, Sorted, Inverse, chunk, ROUTES: gl.constexpr, CHUNKS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    lane = gl.arange(0, 256, layout)
    r = chunk * 256 + lane
    e = gl.load(Ids + r, r < ROUTES, 257)
    keys = e * 256 + lane
    for stage in gl.static_range(1, 9):
        for step in gl.static_range(stage):
            distance = 1 << stage - 1 - step
            other = gl.gather(keys, lane ^ distance, 0)
            take_min = (lane & 1 << stage == 0) == (lane & distance == 0)
            keys = gl.where(take_min, gl.minimum(keys, other), gl.maximum(keys, other))
    expert = keys // 256
    previous = gl.gather(expert, gl.maximum(lane - 1, 0), 0)
    starts = gl.where((lane == 0) | (expert != previous), lane, 0)
    starts = gl.associative_scan(starts, 0, _maximum)
    route = chunk * 256 + keys % 256
    valid = (expert < 257) & (route < ROUTES)
    offset = gl.load(Offsets + expert, valid, 0)
    prefix = gl.load(Prefix + expert * CHUNKS + chunk, valid, 0)
    sorted_row = offset + prefix + lane - starts
    gl.store(Sorted + sorted_row, route, valid)
    delta = gl.load(Offsets + 257 + expert, valid, 0)
    compact_route = route - route // 9
    gl.store(Inverse + compact_route, sorted_row + delta, valid & (expert < 256))

@gluon.jit
def _quantize_packed(x, SHARED: gl.constexpr):
    peak = gl.max(gl.abs(x), 1)
    if SHARED:
        peak_bits = peak.to(gl.uint32, bitcast=True)
        floor_exp = (peak_bits >> 23 & 255).to(gl.int32) - 127
        threshold = gl.exp2(floor_exp.to(gl.float32)) * 1.75
        exponent = floor_exp - 2 + (peak >= threshold).to(gl.int32)
    else:
        divided = gl.div_rn(peak, 6.0)
        bits = divided.to(gl.uint32, bitcast=True)
        exponent = (bits >> 23 & 255).to(gl.int32) - 127 + (bits & 8388607 != 0)
    exponent = gl.maximum(-127, gl.minimum(127, exponent))
    scale = gl.exp2(exponent.to(gl.float32))
    inverse_scale = gl.div_rn(1.0, scale)
    a = gl.abs(x * inverse_scale[:, None])
    a = gl.where(a == a, a, 6.0)
    low, high = gl.split(a.reshape((x.shape[0], 16, 2)))
    packed = gl.inline_asm_elementwise('v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0', constraints='=v,v,v', args=[low, high], dtype=gl.uint32, is_pure=True, pack=1).to(gl.uint8)
    sign = gl.where(x < 0, 8, 0).to(gl.uint8)
    low_sign, high_sign = gl.split(sign.reshape((x.shape[0], 16, 2)))
    return (packed | low_sign | high_sign << 4, (exponent + 127).to(gl.uint8))

@gluon.jit
def _quantize_input_tile(X, Q, QS, tile, M: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, VALUES: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, VALUES], [2 * VALUES, 32 // VALUES], [4, 1], [1, 0])
    group = tile * GROUPS + gl.arange(0, GROUPS, gl.SliceLayout(1, layout))
    k = gl.arange(0, 32, gl.SliceLayout(0, layout))
    row = group // (K // 32)
    col = (group % (K // 32))[:, None] * 32 + k[None, :]
    x = gl.load(X + row[:, None] * SX + col, row[:, None] < M, 0).to(gl.float32)
    routed, routed_scale = _quantize_packed(x, False)
    shared, shared_scale = _quantize_packed(x, True)
    out_layout: gl.constexpr = gl.BlockedLayout([1, VALUES // 2], [2 * VALUES, 32 // VALUES], [4, 1], [1, 0])
    packed_group = tile * GROUPS + gl.arange(0, GROUPS, gl.SliceLayout(1, out_layout))
    packed_k = gl.arange(0, 16, gl.SliceLayout(0, out_layout))
    packed_row = packed_group // (K // 32)
    offset = packed_group[:, None] * 16 + packed_k[None, :]
    gl.store(Q + offset, gl.convert_layout(routed, out_layout), packed_row[:, None] < M)
    gl.store(Q + M * (K // 2) + offset, gl.convert_layout(shared, out_layout), packed_row[:, None] < M)
    scale_offset = (group & ~7) + ((group & 1) << 2) + ((group & 7) >> 1)
    gl.store(QS + scale_offset, routed_scale, row < M)
    gl.store(QS + M * (K // 32) + scale_offset, shared_scale, row < M)

@gluon.jit
def _weight_minimum(Scales, Minimum, tile):
    lane = gl.arange(0, 4096, gl.BlockedLayout([16], [64], [4], [0]))
    scale = gl.load(Scales + tile * 4096 + lane)
    gl.store(Minimum + tile, gl.min(scale.to(gl.int32), 0).to(gl.uint8))

@gluon.jit
def _scatter_quantize(Ids, Offsets, Prefix, Sorted, Inverse, X, Q, QS, ROUTES: gl.constexpr, CHUNKS: gl.constexpr, M: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, VALUES: gl.constexpr, W2Scales, W2Minimum, SCALE_CODEC: gl.constexpr):
    tile = gl.program_id(0)
    if tile < CHUNKS:
        _scatter_chunk(Ids, Offsets, Prefix, Sorted, Inverse, tile, ROUTES, CHUNKS)
    elif SCALE_CODEC and tile >= CHUNKS + gl.cdiv(M * K // 32, GROUPS):
        _weight_minimum(W2Scales, W2Minimum, tile - CHUNKS - gl.cdiv(M * K // 32, GROUPS))
    else:
        _quantize_input_tile(X, Q, QS, tile - CHUNKS, M, K, SX, GROUPS, VALUES)

@gluon.jit
def _store_w13_activation(acc, Y, YS, expert, start_row, column, N: gl.constexpr):
    BM: gl.constexpr = acc.shape[0]
    BN: gl.constexpr = acc.shape[1]
    ep: gl.constexpr = gl.BlockedLayout([1, 32], [16, 4], [4, 1], [1, 0])
    paired = acc.reshape((BM, 2, BN // 2))
    gate, up = gl.split(paired.permute((0, 2, 1)))
    if expert == 256:
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
    activated = (gate * _sigmoid(gate) * up).to(gl.bfloat16)
    activated = gl.convert_layout(activated, ep)
    activated = activated.reshape((BM * BN // 64, 32)).to(gl.float32)
    if expert == 256:
        quantized, scales = _quantize_packed(activated, True)
    else:
        quantized, scales = _quantize_packed(activated, False)
    packed_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    result = gl.convert_layout(quantized.reshape((BM, BN // 4)), packed_layout)
    rm = start_row + gl.arange(0, BM, gl.SliceLayout(1, packed_layout))
    cn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, packed_layout))
    gl.store(Y + rm[:, None] * (N // 4) + cn[None, :], result)
    scale_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 2], [4, 1], [0, 1])
    scales = gl.convert_layout(scales.reshape((BM, BN // 64)), scale_layout)
    sr = start_row + gl.arange(0, BM, gl.SliceLayout(1, scale_layout))
    sg = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, scale_layout))
    gl.store(YS + sr[:, None] * (N // 64) + sg[None, :], scales)

@gluon.jit
def _w2_row_max(value):
    rows: gl.constexpr = value.shape[0]
    native_rows: gl.constexpr = gl.SliceLayout(1, value.type.layout)
    partial = gl.max(value.reshape((rows, 2, 4, 32)), 1)
    partial = gl.max(partial, 2)
    exchange: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [1, 4], [0, 1])
    shared = gl.allocate_shared_memory(value.dtype, (rows, 4), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), partial)
    result = gl.max(shared.load(exchange), 1)
    return gl.convert_layout(result, native_rows, assert_trivial=True)

@gluon.jit
def _store_w2_panel(acc, raw_base, code_base, header_base, column, valid_rows, N: gl.constexpr, BN: gl.constexpr, PITCH: gl.constexpr, exponent, SCALE_CODEC: gl.constexpr):
    native_layout: gl.constexpr = acc.type.layout
    panel_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    if not SCALE_CODEC:
        peak = _w2_row_max(gl.abs(acc))
        exponent = (peak.to(gl.uint32, bitcast=True) >> 23 & 255).to(gl.int32) - 141
    exponent = gl.maximum(-126, gl.minimum(112, exponent))
    quantum = (exponent + 127 << 23).to(gl.float32, bitcast=True)
    inverse = (127 - exponent << 23).to(gl.float32, bitcast=True)
    code = (acc * inverse[:, None]).to(gl.int32).to(gl.int16)
    reconstructed = code.to(gl.float32) * quantum[:, None]
    mismatch = reconstructed.to(gl.uint32, bitcast=True) ^ acc.to(gl.uint32, bitcast=True)
    escaped = _w2_row_max(mismatch) != 0
    rows = gl.arange(0, acc.shape[0], gl.SliceLayout(1, native_layout))
    gl.store(header_base + rows * (N // BN) + column, gl.where(escaped, 0, exponent + 127).to(gl.uint8), rows < valid_rows)
    code = gl.convert_layout(code, panel_layout)
    code_rows = gl.arange(0, acc.shape[0], gl.SliceLayout(1, panel_layout))
    code_columns = column * BN + gl.arange(0, BN, gl.SliceLayout(0, panel_layout))
    gl.amd.cdna4.buffer_store(stored_value=code, ptr=code_base, offsets=code_rows[:, None] * PITCH + code_columns[None, :], mask=code_rows[:, None] < valid_rows, cache='.cs')
    if gl.sum((escaped & (rows < valid_rows)).to(gl.int32), 0) != 0:
        columns = column * BN + gl.arange(0, BN, gl.SliceLayout(0, native_layout))
        offsets = rows[:, None] * PITCH + columns[None, :]
        gl.amd.cdna4.buffer_store(stored_value=acc, ptr=raw_base, offsets=offsets, mask=(rows[:, None] < valid_rows) & escaped[:, None], cache='.cs')

@gluon.jit
def _route_metadata(dense_rows, weights, headers, route, ROW_LAYOUT: gl.constexpr):
    BM: gl.constexpr = dense_rows.shape[0]
    index = gl.full((BM, 1), route, gl.int32, dense_rows.type.layout)
    dense = gl.convert_layout(gl.gather(dense_rows, index, 1).reshape((BM,)), ROW_LAYOUT)
    weight = gl.convert_layout(gl.gather(weights, index, 1).reshape((BM,)), ROW_LAYOUT)
    bits = gl.convert_layout(gl.gather(headers, index, 1).reshape((BM,)), ROW_LAYOUT)
    quantum = (bits.to(gl.uint32) << 23).to(gl.float32, bitcast=True)
    return (dense, weight, quantum)

@gluon.jit
def _decode_w2_panel(P, code, quantum, offset, valid):
    part = code.to(gl.float32) * quantum[:, None]
    return gl.load(P + offset, valid & (quantum[:, None] == 0), part, cache_modifier='.cg')

@gluon.jit
def _weight_scale_word(n, group, K: gl.constexpr):
    group_tiles: gl.constexpr = gl.cdiv(K // 32, 8)
    tile = (n // 32)[:, None] * group_tiles + (group // 8)[None, :]
    offset = (tile * 4 + (group % 4)[None, :]) * 16 + (n % 16)[:, None]
    shift = (n[:, None] >> 4 & 1) * 8 + (group[None, :] >> 2 & 1) * 16
    return (offset, shift)

@gluon.jit
def _projection_load(ptr, offset, BUFFERED: gl.constexpr):
    if BUFFERED:
        return gl.amd.cdna4.buffer_load(ptr, offset.to(gl.uint32))
    return gl.load(ptr + offset)

@gluon.jit
def _native_down_projection(X, XS, W, WS, rows, expert, column, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, ROUTED: gl.constexpr=True, SCALE_CODEC: gl.constexpr=False, min_weight_scale=0):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 64] if ROUTED and BM >= 32 else [16, 16, 128], transposed=True, warps_per_cta=[1, 4])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([16, 1], [2, 32] if ROUTED and BM >= 32 else [4, 16], [1, 4], [1, 0])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [BM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    row = gl.convert_layout(rows, gl.SliceLayout(1, al))
    scale_row = gl.convert_layout(rows, gl.SliceLayout(1, asl))
    ak = gl.arange(0, BK // 2, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK // 2, gl.SliceLayout(1, bl))
    n = gl.arange(0, BN, gl.SliceLayout(0, bl))
    sn = gl.arange(0, BN, gl.SliceLayout(1, bsl))
    wn = column * BN + n
    scale_n = column * BN + sn
    ask = gl.arange(0, BK // 32, gl.SliceLayout(0, asl))
    bsk = gl.arange(0, BK // 32, gl.SliceLayout(0, bsl))
    wbase = W + expert.to(gl.int64) * (N * K // 2)
    wsbase = WS + expert * (N * (gl.cdiv(K // 32, 8) * 8))
    bo = wn[None, :] // 16 * (K * 8) + bk[:, None] // 16 * 256 + wn[None, :] % 16 * 16 + bk[:, None] % 16
    acc = gl.zeros((BM, BN), gl.float32, mma)
    if ROUTED and SCALE_CODEC:
        gl.static_assert(K == BK)
        quantum_exponent = gl.full((BM,), 0, gl.int32, gl.SliceLayout(1, mma))
    for step in range(K // BK):
        av = _projection_load(X, row[:, None] * (K // 2) + step * (BK // 2) + ak[None, :], ROUTED)
        bv = _projection_load(wbase, bo + step * (BK * 8), ROUTED)
        gl.static_assert(BN == 256)
        if ROUTED:
            a_scale_offset = scale_row[:, None] * (K // 32) + step * (BK // 32) + (ask & ~3)[None, :]
            a_scale_word = _projection_load(XS.to(gl.pointer_type(gl.uint32)), a_scale_offset // 4, True)
            ascale = (a_scale_word >> (ask & 3)[None, :] * 8).to(gl.uint8)
        else:
            a_scale_offset = scale_row[:, None] * (K // 32) + step * (BK // 32) + (ask & ~7)[None, :]
            a_scale_word = _projection_load(XS.to(gl.pointer_type(gl.uint64)), a_scale_offset // 8, False)
            ascale = (a_scale_word >> (ask & 7)[None, :] * 8).to(gl.uint8)
        scale_groups = step * (BK // 32) + bsk
        word_offset, byte_shift = _weight_scale_word(scale_n, scale_groups, K)
        scale_word = _projection_load(wsbase.to(gl.pointer_type(gl.uint32)), word_offset, ROUTED)
        bscale = (scale_word >> byte_shift).to(gl.uint8)
        if ROUTED and SCALE_CODEC:
            min_a = gl.convert_layout(gl.min(ascale.to(gl.int32), 1), gl.SliceLayout(1, mma))
            quantum_exponent = min_a + min_weight_scale - 256
        b_dot = gl.convert_layout(bv, bd, assert_trivial=True)
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(av, ad), ascale, 'e2m1', b_dot, bscale, 'e2m1', acc)
    if ROUTED and SCALE_CODEC:
        return (acc, quantum_exponent)
    else:
        return acc

@gluon.jit
def _native_up_async(X, XS, W, WS, rows, expert, column, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    gl.static_assert(BK == 256)
    gl.static_assert(K % BK == 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 64] if BM == 128 else [16, 16, 128], transposed=True, warps_per_cta=[1, 4] if BM == 16 else [2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([1, 16], [1, 64], [4, 1], [1, 0])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [BM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    row = gl.convert_layout(rows, gl.SliceLayout(1, al))
    scale_row = gl.convert_layout(rows, gl.SliceLayout(1, asl))
    ak = gl.arange(0, BK // 2, gl.SliceLayout(0, al))
    ng = gl.arange(0, BN // 16, gl.SliceLayout(1, bl))
    byte = gl.arange(0, BK * 8, gl.SliceLayout(0, bl))
    group = column * (BN // 32) + ng % (BN // 32) + ng // (BN // 32) * (N // 32)
    ao = row[:, None] * (K // 2) + ak[None, :]
    bo = group[:, None] * (K * 8) + byte[None, :]
    ask = gl.arange(0, BK // 32, gl.SliceLayout(0, asl))
    asoff = (scale_row[:, None] * (K // 32) + (ask & ~7)[None, :]) // 8
    ashift = (((ask & 1) << 2) + ((ask & 7) >> 1))[None, :] * 8
    wbase = W + expert.to(gl.int64) * (N * K // 2)
    wsbase = WS + expert * (N * (K // 32))
    asptr = XS.to(gl.pointer_type(gl.uint64))
    scale_copy: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
    scale_group = gl.arange(0, BN // 32, gl.SliceLayout(1, scale_copy))
    scale_byte = gl.arange(0, BK, gl.SliceLayout(0, scale_copy))
    scale_group = column * (BN // 64) + scale_group % (BN // 64) + scale_group // (BN // 64) * (N // 64)
    bsoff = scale_group[:, None] * K + scale_byte[None, :]
    a_smem = gl.allocate_shared_memory(gl.uint8, (BM, BK // 2), gl.SwizzledSharedLayout(16, 1, 8, [1, 0]))
    b_smem = gl.allocate_shared_memory(gl.uint8, (BN // 16, BK * 8), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    b_view = b_smem.reshape((BN // 16, BK // 32, 16, 16)).permute((1, 3, 0, 2)).reshape((BK // 2, BN))
    bs_smem = gl.allocate_shared_memory(gl.uint8, (BN // 32, BK), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    bs_view = bs_smem.reshape((BN // 32, 4, 16, 2, 2)).permute((0, 4, 2, 3, 1)).reshape((BN, BK // 32))
    if BM >= 32:
        activation_scale_copy: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [4, 1], [1, 0])
        activation_scale_rows = gl.convert_layout(rows, gl.SliceLayout(1, activation_scale_copy))
        activation_scale_bytes = gl.arange(0, BK // 32, gl.SliceLayout(0, activation_scale_copy))
        activation_scale_offsets = activation_scale_rows[:, None] * (K // 32) + activation_scale_bytes[None, :]
        as_smem = gl.allocate_shared_memory(gl.uint8, (BM, BK // 32), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        as_view = as_smem.reshape((BM, 2, 4)).permute((0, 2, 1)).reshape((BM, BK // 32))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(as_smem, XS, activation_scale_offsets.to(gl.uint32))
    else:
        a_word = gl.amd.cdna4.buffer_load(asptr, asoff.to(gl.uint32))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_smem, X, ao.to(gl.uint32))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_smem, wbase, bo.to(gl.uint32))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bs_smem, wsbase, bsoff.to(gl.uint32))
    gl.amd.cdna4.async_copy.commit_group()
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for step in range(K // BK - 1):
        if BM < 32:
            ascale = (a_word >> ashift).to(gl.uint8)
            a_word = gl.amd.cdna4.buffer_load(asptr, (asoff + (step + 1) * (BK // 256)).to(gl.uint32))
        gl.amd.cdna4.async_copy.wait_group(0)
        a = a_smem.load(ad)
        b = b_view.load(bd)
        bscale = bs_view.load(bsl)
        if BM >= 32:
            ascale = as_view.load(asl)
        gl.barrier()
        if BM >= 32:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(as_smem, XS, (activation_scale_offsets + (step + 1) * (BK // 32)).to(gl.uint32))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_smem, X, (ao + (step + 1) * (BK // 2)).to(gl.uint32))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_smem, wbase, (bo + (step + 1) * (BK * 8)).to(gl.uint32))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bs_smem, wsbase, (bsoff + (step + 1) * BK).to(gl.uint32))
        gl.amd.cdna4.async_copy.commit_group()
        acc = gl.amd.cdna4.mfma_scaled(a, ascale, 'e2m1', b, bscale, 'e2m1', acc)
    gl.amd.cdna4.async_copy.wait_group(0)
    a = a_smem.load(ad)
    b = b_view.load(bd)
    if BM >= 32:
        ascale = as_view.load(asl)
    else:
        ascale = (a_word >> ashift).to(gl.uint8)
    bscale = bs_view.load(bsl)
    acc = gl.amd.cdna4.mfma_scaled(a, ascale, 'e2m1', b, bscale, 'e2m1', acc)
    return acc

@gluon.jit
def _project_w13_tile(X, XS, W, Scales, Sorted, Y, YS, block, expert, weight_expert, column, N: gl.constexpr, K: gl.constexpr, M: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, TILE_M: gl.constexpr):
    row_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    row = block * BM + gl.arange(0, TILE_M, row_layout)
    route = gl.load(Sorted + row)
    row = gl.maximum(route // 9 + gl.where(expert == 256, M, 0), 0)
    acc = _native_up_async(X, XS, W, Scales, row, weight_expert, column, N, K, TILE_M, BN, BK)
    _store_w13_activation(acc, Y, YS, expert, block * BM, column, N)

@gluon.jit
def _w13_projection(X, XS, W, Scales, Sorted, Experts, Y, YS, expert_start, N: gl.constexpr, K: gl.constexpr, M: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, LOCAL_EXPERTS: gl.constexpr):
    physical_pid = gl.program_id(0)
    jobs: gl.constexpr = gl.cdiv(9 * M + 257 * (BM - 1), BM) * (N // BN)
    group_base = physical_pid // 256 * 256
    group_jobs = gl.minimum(256, jobs - group_base)
    partition = physical_pid % 8
    pid = group_base + partition * (group_jobs // 8) + gl.minimum(partition, group_jobs % 8) + physical_pid % 256 // 8
    block = pid // (N // BN)
    column = pid % (N // BN)
    descriptor = gl.load(Experts + block)
    if descriptor >= 0:
        expert = (descriptor & 511).to(gl.int32)
        owned = (expert >= expert_start) & (expert < expert_start + LOCAL_EXPERTS)
        weight_expert = gl.where(owned, expert - expert_start, LOCAL_EXPERTS)
        valid_rows = (descriptor >> 9 & 255).to(gl.int32)
        if M < 8192 and valid_rows <= 16:
            _project_w13_tile(X, XS, W, Scales, Sorted, Y, YS, block, expert, weight_expert, column, N, K, M, BM, BN, BK, 16)
        elif valid_rows <= 32:
            _project_w13_tile(X, XS, W, Scales, Sorted, Y, YS, block, expert, weight_expert, column, N, K, M, BM, BN, BK, 32)
        elif valid_rows <= 64:
            _project_w13_tile(X, XS, W, Scales, Sorted, Y, YS, block, expert, weight_expert, column, N, K, M, BM, BN, BK, 64)
        else:
            _project_w13_tile(X, XS, W, Scales, Sorted, Y, YS, block, expert, weight_expert, column, N, K, M, BM, BN, BK, BM)

@gluon.jit
def _project_w2_tile(X, XS, W, Scales, Minimum, P, Codes, Headers, block, expert, column, valid_rows, dense_base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, TILE_M: gl.constexpr, PITCH: gl.constexpr, SCALE_CODEC: gl.constexpr):
    row_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    row = block * 64 + gl.arange(0, TILE_M, row_layout)
    if SCALE_CODEC:
        min_weight_scale = gl.load(Minimum + expert * (N // BN) + column).to(gl.int32)
        acc, exponent = _native_down_projection(X, XS, W, Scales, row, expert, column, N, K, TILE_M, BN, BK, True, True, min_weight_scale)
    else:
        acc = _native_down_projection(X, XS, W, Scales, row, expert, column, N, K, TILE_M, BN, BK)
        exponent = 0
    _store_w2_panel(acc, P + dense_base.to(gl.int64) * PITCH, Codes + dense_base.to(gl.int64) * PITCH, Headers + dense_base * (N // BN), column, valid_rows, N, BN, PITCH, exponent, SCALE_CODEC)

@gluon.jit
def _w2_projection(X, XS, W, Scales, Minimum, DownExperts, P, Codes, Headers, expert_start, N: gl.constexpr, K: gl.constexpr, M: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, PITCH: gl.constexpr, SCALE_CODEC: gl.constexpr, LOCAL_EXPERTS: gl.constexpr):
    pid = gl.program_id(0)
    tile = pid // (N // BN)
    column = pid % (N // BN)
    descriptor = gl.load(DownExperts + tile)
    if descriptor >= 0:
        expert = (descriptor & 511).to(gl.int32)
        owned = (expert >= expert_start) & (expert < expert_start + LOCAL_EXPERTS)
        weight_expert = gl.where(owned, expert - expert_start, LOCAL_EXPERTS)
        valid_rows = (descriptor >> 9 & 127).to(gl.int32)
        block = (descriptor >> 16 & 65535).to(gl.int32)
        dense_base = (descriptor >> 32).to(gl.int32)
        if M < 8192 and valid_rows <= 16:
            _project_w2_tile(X, XS, W, Scales, Minimum, P, Codes, Headers, block, weight_expert, column, valid_rows, dense_base, N, K, BN, BK, 16, PITCH, SCALE_CODEC)
        elif valid_rows <= 32:
            _project_w2_tile(X, XS, W, Scales, Minimum, P, Codes, Headers, block, weight_expert, column, valid_rows, dense_base, N, K, BN, BK, 32, PITCH, SCALE_CODEC)
        else:
            _project_w2_tile(X, XS, W, Scales, Minimum, P, Codes, Headers, block, weight_expert, column, valid_rows, dense_base, N, K, BN, BK, 64, PITCH, SCALE_CODEC)

@gluon.jit
def _grouped_tile(pid, ROW_TILES: gl.constexpr, COL_TILES: gl.constexpr, GROUP_M: gl.constexpr=8):
    full_blocks: gl.constexpr = ROW_TILES // GROUP_M * GROUP_M
    tail: gl.constexpr = ROW_TILES % GROUP_M
    group = pid // (GROUP_M * COL_TILES)
    within = pid % (GROUP_M * COL_TILES)
    block = group * GROUP_M + within % GROUP_M
    column = within // GROUP_M
    if tail != 0:
        if pid >= full_blocks * COL_TILES:
            tail_pid = pid - full_blocks * COL_TILES
            block = full_blocks + tail_pid % tail
            column = tail_pid // tail
    return (block, column)

@gluon.jit
def _shared_reduce(X, XS, W, Scales, Offsets, P, Codes, Headers, Weights, Inverse, Y, M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, PITCH: gl.constexpr, route_count, LOCAL_EXPERTS: gl.constexpr):
    block, column = _grouped_tile(gl.program_id(0), gl.cdiv(M, BM), H // BN)
    first = block * BM
    row_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    row = first + gl.arange(0, BM, row_layout)
    source_row = gl.load(Offsets + 256) + gl.minimum(row, M - 1)
    acc = _native_down_projection(X, XS, W, Scales, source_row, gl.full((), LOCAL_EXPERTS, gl.int32), column, H, K, BM, BN, 256, False)
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    shared = gl.convert_layout(acc.to(gl.bfloat16), ep)
    rm = first + gl.arange(0, BM, gl.SliceLayout(1, ep))
    cn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, ep))
    value = gl.full((BM, BN), 0, gl.float32, ep)
    metadata_layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    metadata_row = first + gl.arange(0, BM, gl.SliceLayout(1, metadata_layout))
    route_id = gl.arange(0, 8, gl.SliceLayout(0, metadata_layout))
    dense_rows = gl.load(Inverse + metadata_row[:, None] * 8 + route_id[None, :], metadata_row[:, None] < M, 0)
    route_weights = gl.load(Weights + metadata_row[:, None] * 8 + route_id[None, :], metadata_row[:, None] < M, 0)
    route_headers = gl.load(Headers + dense_rows * (H // 256) + column * BN // 256, metadata_row[:, None] < M, 127)
    for pair in range(route_count // 2):
        dense0, weight0, header0 = _route_metadata(dense_rows, route_weights, route_headers, 2 * pair, gl.SliceLayout(1, ep))
        dense1, weight1, header1 = _route_metadata(dense_rows, route_weights, route_headers, 2 * pair + 1, gl.SliceLayout(1, ep))
        offset0 = dense0[:, None] * PITCH + cn[None, :]
        offset1 = dense1[:, None] * PITCH + cn[None, :]
        code0 = gl.amd.cdna4.buffer_load(Codes, offset0, mask=rm[:, None] < M, other=0, cache='.cg')
        code1 = gl.amd.cdna4.buffer_load(Codes, offset1, mask=rm[:, None] < M, other=0, cache='.cg')
        part0 = _decode_w2_panel(P, code0, header0, offset0, rm[:, None] < M)
        value += part0 * weight0[:, None]
        part1 = _decode_w2_panel(P, code1, header1, offset1, rm[:, None] < M)
        value += part1 * weight1[:, None]
    value += shared.to(gl.float32)
    gl.store(Y + rm[:, None] * H + cn[None, :], value, rm[:, None] < M)

class _Workspace:

    def __init__(self, x, intermediate, local_experts):
        m, h = x.shape
        self.intermediate = intermediate
        block_m = self.block_m = 128
        self.routes = 9 * m
        self.capacity = triton.cdiv(self.routes + 257 * (block_m - 1), block_m) * block_m
        self.chunks = triton.cdiv(self.routes, 256)
        self.down_tiles = triton.cdiv(8 * m + 256 * 63, 64)

        def empty(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=x.device)
        self.payload_pitch = h + 64
        self.parts = empty((8 * m, self.payload_pitch), torch.float32)
        self.codes = empty((8 * m, self.payload_pitch), torch.int16)
        cursor = 0

        def early(shape, dtype):
            nonlocal cursor
            base = self.parts.view(dtype).view(-1)
            item_bytes = base.element_size()
            size = prod(shape)
            begin = triton.cdiv(cursor, 256) * 256
            cursor = begin + size * item_bytes
            return base.narrow(0, begin // item_bytes, size).view(shape)
        self.logits = early((m, 256), torch.bfloat16)
        self.ids = early((m, 9), torch.int32)
        self.weights = empty((m, 8), torch.float32)
        self.partial_counts = early((257, self.chunks), torch.int32)
        self.counts = early((257,), torch.int32)
        self.offsets = empty((2 * 257,), torch.int32)
        self.inverse = empty((m, 8), torch.int32)
        self.sorted_routes = early((self.capacity,), torch.int32)
        self.experts = empty((self.capacity // self.block_m,), torch.int32)
        self.down_experts = empty((self.down_tiles,), torch.int64)
        self.xq = empty((2 * m, h // 2), torch.uint8)
        self.xs = empty((2 * m, h // 32), torch.uint8)
        self.aq = empty((self.capacity, intermediate // 2), torch.uint8)
        self.aqs = empty((self.capacity, intermediate // 32), torch.uint8)
        self.headers = empty((8 * m, h // 256), torch.uint8)
        self.output = empty((m, h))
        self.scale_codec = m >= 8192 and local_experts == 256
        self.weight_minimum = empty((256, h // 256), torch.uint8) if self.scale_codec else None

def _route_and_pack(x, router, correction_bias, work, w2_scale, expert_start, local_experts, routed_scaling_factor):
    m, h = x.shape
    router_rows = 128 if 16383 <= m <= 16384 else 32 if m < 8192 else 64
    router_columns = 128 if m > 16384 else 64
    router_k = 64 if m > 16384 else 128
    if 8191 <= m <= 8193:
        router_k = 256
    quantize_groups, quantize_values = (256, 32)
    _router_projection[triton.cdiv(m, router_rows), 256 // router_columns](x, router, work.logits, m, h, x.stride(0), router_rows, router_k, router_columns)
    _router[m,](work.logits, correction_bias, work.ids, work.weights, work.inverse, expert_start, local_experts, routed_scaling_factor, num_warps=1)
    _chunk_counts[max(work.chunks, triton.cdiv(work.capacity, 1024)),](work.ids, work.partial_counts, work.sorted_routes, work.experts, work.down_experts, work.routes, work.chunks, work.capacity, work.block_m, work.down_tiles)
    _chunk_prefix[257,](work.partial_counts, work.partial_counts, work.counts, work.chunks, triton.next_power_of_2(work.chunks))
    _build_expert_blocks[257,](work.counts, work.offsets, work.experts, work.down_experts, work.block_m, triton.next_power_of_2(triton.cdiv(m, work.block_m)), triton.next_power_of_2(triton.cdiv(m, 64)))
    minimum_tiles = h if work.scale_codec else 0
    _scatter_quantize[work.chunks + triton.cdiv(m * h // 32, quantize_groups) + minimum_tiles,](work.ids, work.offsets, work.partial_counts, work.sorted_routes, work.inverse, x, work.xq, work.xs, work.routes, work.chunks, m, h, x.stride(0), quantize_groups, quantize_values, w2_scale, work.weight_minimum, work.scale_codec)

def _project_experts(w13, w13_scale, w2, w2_scale, work, expert_start, local_experts):
    m, h = work.output.shape
    intermediate = work.intermediate
    up_columns, up_k = (256, 256)
    down_columns, down_k = (256, 512)
    _w13_projection[work.capacity // work.block_m * (2 * intermediate // up_columns),](work.xq, work.xs, w13, w13_scale, work.sorted_routes, work.experts, work.aq, work.aqs, expert_start, 2 * intermediate, h, m, work.block_m, up_columns, up_k, local_experts, enable_fp_fusion=False)
    _w2_projection[work.down_tiles * (h // down_columns),](work.aq, work.aqs, w2, w2_scale, work.weight_minimum, work.down_experts, work.parts, work.codes, work.headers, expert_start, h, intermediate, m, down_columns, down_k, work.payload_pitch, work.scale_codec, local_experts, enable_fp_fusion=False, waves_per_eu=2)

def _finish(w2, w2_scale, work, local_experts):
    m, h = work.output.shape
    reduce_rows, reduce_columns = (32, 256)
    _shared_reduce[triton.cdiv(m, reduce_rows) * (h // reduce_columns),](work.aq, work.aqs, w2, w2_scale, work.offsets, work.parts, work.codes, work.headers, work.weights, work.inverse, work.output, m, h, work.intermediate, reduce_rows, reduce_columns, work.payload_pitch, 8, local_experts, enable_fp_fusion=False)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, expert_start=0, routed_scaling_factor=2.5, fuse_shared_expert=False):
    intermediate = w13.shape[1] // 2
    local_experts = w13.shape[0] - 1
    assert w2.shape == (local_experts + 1, x.shape[1], intermediate // 2)
    work = _Workspace(x, intermediate, local_experts)
    _route_and_pack(x, router, correction_bias, work, w2_scale, expert_start, local_experts, routed_scaling_factor)
    _project_experts(w13, w13_scale, w2, w2_scale, work, expert_start, local_experts)
    _finish(w2, w2_scale, work, local_experts)
    return work.output
