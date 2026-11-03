# Created by OpenAI
# fmt: off
"""Experimental GLM-5.2 TP8 fused MoE with runtime M and bucketed codegen."""

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
def _grouped_tile(pid, blocks: gl.constexpr, COLUMNS: gl.constexpr, GROUP: gl.constexpr):
    p = pid.to(gl.uint32)
    first = p // (GROUP * COLUMNS) * GROUP
    within = p % (GROUP * COLUMNS)
    block = first + within % GROUP
    column = within // GROUP
    if blocks % GROUP != 0:
        tail: gl.constexpr = blocks % GROUP
        last = first == blocks // GROUP * GROUP
        block = gl.where(last, first + within % tail, block)
        column = gl.where(last, within // tail, column)
    return (block.to(gl.int32), column.to(gl.int32))

@gluon.jit(do_not_specialize=['blocks'])
def _grouped_tile_runtime(pid, blocks, COLUMNS: gl.constexpr, GROUP: gl.constexpr):
    """Grouped tile mapping whose row-block count remains a runtime value."""
    p = pid.to(gl.uint32)
    blocks = blocks.to(gl.uint32)
    first = p // (GROUP * COLUMNS) * GROUP
    within = p % (GROUP * COLUMNS)
    block = first + within % GROUP
    column = within // GROUP
    tail = blocks % GROUP
    safe_tail = gl.maximum(tail, 1)
    last = (tail != 0) & (first == blocks // GROUP * GROUP)
    block = gl.where(last, first + within % safe_tail, block)
    column = gl.where(last, within // safe_tail, column)
    return (block.to(gl.int32), column.to(gl.int32))

@gluon.jit
def _split_columns(tile):
    return gl.split(tile.reshape((tile.shape[0], 2, tile.shape[1] // 2)).permute((0, 2, 1)))

@gluon.jit
def _router_projection_tile(X, W, Y, row_tile, col_tile, actual_m, K: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BK, BK // 8], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [BK // 8, 512 // BK], [1, 4], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    rows = row_tile * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    cols = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    a = gl.load(X + rows[:, None] * SX + ak[None, :], rows[:, None] < actual_m, 0)
    b = gl.load(W + cols[None, :] * K + bk[:, None])
    for k in range(K // BK - 1):
        aa = gl.convert_layout(a, ad)
        bb = gl.convert_layout(b, bd)
        next_k = k + 1
        a = gl.load(X + rows[:, None] * SX + next_k * BK + ak[None, :], rows[:, None] < actual_m, 0)
        b = gl.load(W + cols[None, :] * K + next_k * BK + bk[:, None])
        acc = gl.amd.cdna4.mfma(aa, bb, acc)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(a, ad), gl.convert_layout(b, bd), acc)
    rm = row_tile * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    cn = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(Y + rm[:, None] * 256 + cn[None, :], acc, rm[:, None] < actual_m)

@gluon.jit
def _router(Logits, Bias, Ids, Weights, SCALE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    m = gl.program_id(0)
    e = gl.arange(0, 256, layout)
    probability = _sigmoid(gl.load(Logits + m * 256 + e).to(gl.float32))
    score = probability + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    selected = gl.full((256,), 0.0, gl.float32, layout)
    total = 0.0
    for j in range(8):
        maximum = gl.max(score, 0)
        idx = gl.min(gl.where(available & (score == maximum), e, 256), 0)
        if idx >= 256:
            idx = gl.min(gl.where(available, e, 256), 0)
        index = gl.full((1,), idx, gl.int32, layout)
        prob = gl.sum(gl.gather(probability, index, 0), 0)
        gl.store(Ids + m * 8 + j, idx)
        selected = gl.where(e == j, prob, selected)
        total += prob
        available = available & (e != idx)
        score = gl.where(e == idx, -float('inf'), score)
    gl.store(Weights + m * 8 + e, selected / total * SCALE, e < 8)

@gluon.jit(do_not_specialize=['routes'])
def _chunk_counts(Ids, Counts, Sorted, Experts, routes, CHUNKS: gl.constexpr, CAPACITY: gl.constexpr, BM: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    chunk = gl.program_id(0)
    r = chunk * 256 + gl.arange(0, 256, layout)
    e = gl.load(Ids + r, r < routes, 257)
    histogram = gl.histogram(e, 256, mask=e < 256, layout=layout)
    expert = gl.arange(0, 256, layout)
    gl.store(Counts + expert * CHUNKS + chunk, histogram, chunk < CHUNKS)
    r = chunk * 1024 + gl.arange(0, 1024, layout)
    gl.store(Sorted + r, -1, r < CAPACITY)
    gl.store(Experts + r, -1, r < CAPACITY // BM)

@gluon.jit(do_not_specialize=['actual_m'])
def _chunk_prefix(Counts, Prefix, Totals, actual_m, CHUNKS: gl.constexpr, BLOCK: gl.constexpr):
    e = gl.program_id(0)
    if e < 256:
        c = gl.arange(0, BLOCK, gl.BlockedLayout([1], [64], [4], [0]))
        counts = gl.load(Counts + e * CHUNKS + c, c < CHUNKS, 0)
        prefix = gl.associative_scan(counts, 0, _add) - counts
        gl.store(Prefix + e * CHUNKS + c, prefix, c < CHUNKS)
        gl.store(Totals + e, gl.sum(counts, 0))
    else:
        gl.store(Totals + e, actual_m)

@gluon.jit
def _build_expert_blocks(Counts, Offsets, Experts, BM: gl.constexpr, BLOCK: gl.constexpr):
    e = gl.program_id(0)
    count = gl.load(Counts + e)
    all_e = gl.arange(0, 512, gl.BlockedLayout([1], [64], [4], [0]))
    all_counts = gl.load(Counts + all_e, all_e < 257, 0)
    offset = gl.sum(gl.where(all_e < e, gl.cdiv(all_counts, BM), 0), 0)
    gl.store(Offsets + e, offset * BM)
    compact = gl.sum(gl.where(all_e < e, all_counts, 0), 0)
    gl.store(Offsets + 257 + e, compact - offset * BM)
    i = gl.arange(0, BLOCK, gl.BlockedLayout([1], [64], [4], [0]))
    rows = gl.minimum(count - i * BM, BM)
    descriptor = (compact + i * BM).to(gl.int64) << 17 | e + rows * 512
    gl.store(Experts + offset + i, descriptor, i < gl.cdiv(count, BM))

@gluon.jit
def _exchange_keys(keys, lane, distance: gl.constexpr):
    if distance == 1:
        return gl.inline_asm_elementwise('s_nop 1\n\tv_mov_b32_dpp $0, $1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1', constraints='=v,v', args=[keys], dtype=gl.int32, is_pure=True, pack=1)
    elif distance == 2:
        return gl.inline_asm_elementwise('s_nop 1\n\tv_mov_b32_dpp $0, $1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1', constraints='=v,v', args=[keys], dtype=gl.int32, is_pure=True, pack=1)
    elif distance == 4:
        return gl.inline_asm_elementwise('s_nop 1\n\tv_mov_b32_dpp $0, $1 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1\n\ts_nop 1\n\tv_mov_b32_dpp $0, $0 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1', constraints='=v,v', args=[keys], dtype=gl.int32, is_pure=True, pack=1)
    elif distance == 8:
        return gl.inline_asm_elementwise('s_nop 1\n\tv_mov_b32_dpp $0, $1 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1', constraints='=v,v', args=[keys], dtype=gl.int32, is_pure=True, pack=1)
    elif distance < 64:
        return gl.inline_asm_elementwise('ds_bpermute_b32 $0, $1, $2\n\ts_waitcnt lgkmcnt(0)', constraints='=v,v,v', args=[(lane ^ distance) % 64 * 4, keys], dtype=gl.int32, is_pure=True, pack=1)
    else:
        return gl.gather(keys, lane ^ distance, 0)

@gluon.jit(do_not_specialize=['routes'])
def _scatter(Ids, Offsets, Prefix, Sorted, Inverse, routes, CHUNKS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    chunk = gl.program_id(0)
    lane = gl.arange(0, 256, layout)
    r = chunk * 256 + lane
    e = gl.load(Ids + r, r < routes, 257)
    keys = e * 256 + lane
    for stage in gl.static_range(1, 9):
        for step in gl.static_range(stage):
            distance = 1 << stage - 1 - step
            other = _exchange_keys(keys, lane, 1 << stage - 1 - step)
            take_min = (lane & 1 << stage == 0) == (lane & distance == 0)
            keys = gl.where(take_min, gl.minimum(keys, other), gl.maximum(keys, other))
    expert = keys // 256
    previous = gl.gather(expert, gl.maximum(lane - 1, 0), 0)
    starts = gl.where((lane == 0) | (expert != previous), lane, 0)
    starts = gl.associative_scan(starts, 0, _maximum)
    route = chunk * 256 + keys % 256
    valid = (expert < 256) & (route < routes)
    offset = gl.load(Offsets + expert, valid, 0)
    prefix = gl.load(Prefix + expert * CHUNKS + chunk, valid, 0)
    sorted_row = offset + prefix + lane - starts
    gl.store(Sorted + sorted_row, route, valid)
    delta = gl.load(Offsets + 257 + expert, valid, 0)
    gl.store(Inverse + route, sorted_row + delta, valid)

@gluon.jit
def _quantize_values(x, shared):
    peak = gl.max(gl.abs(x), 1)
    divided = gl.div_rn(peak, 6.0)
    bits = divided.to(gl.uint32, bitcast=True)
    exponent = (bits >> 23 & 255).to(gl.int32) - 127 + (bits & 8388607 != 0)
    peak_bits = peak.to(gl.uint32, bitcast=True)
    floor_exp = (peak_bits >> 23 & 255).to(gl.int32) - 127
    threshold = gl.exp2(floor_exp.to(gl.float32)) * 1.75
    even_exp = floor_exp - 2 + (peak >= threshold).to(gl.int32)
    exponent = gl.where(shared, even_exp, exponent)
    exponent = gl.maximum(-127, gl.minimum(127, exponent))
    scale = gl.exp2(exponent.to(gl.float32))
    inverse_scale = gl.div_rn(1.0, scale)
    a = gl.abs(x * inverse_scale[:, None])
    a = gl.where(a == a, a, float('inf'))
    low, high = gl.split(a.reshape((x.shape[0], 16, 2)))
    unit = gl.full(low.shape, 1.0, gl.float32, low.type.layout)
    packed = gl.inline_asm_elementwise('v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3', constraints='=v,v,v,v', args=[low, high, unit], dtype=gl.uint32, is_pure=True, pack=1).to(gl.uint8)
    signs = gl.where(x < 0, 8, 0).to(gl.uint8)
    sign_low, sign_high = gl.split(signs.reshape((x.shape[0], 16, 2)))
    return (packed | sign_low | sign_high << 4, (exponent + 127).to(gl.uint8))

@gluon.jit
def _quantize_input_tile(X, Q, QScale, tile, actual_m, M_BUCKET: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, VALUES: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, VALUES], [2 * VALUES, 32 // VALUES], [4, 1], [1, 0])
    group = tile * GROUPS + gl.arange(0, GROUPS, gl.SliceLayout(1, layout))
    k = gl.arange(0, 32, gl.SliceLayout(0, layout))
    row = group // (K // 32)
    col = (group % (K // 32))[:, None] * 32 + k[None, :]
    x = gl.load(X + row[:, None] * SX + col, row[:, None] < actual_m, 0).to(gl.float32)
    routed, routed_scale = _quantize_values(x, False)
    shared, shared_scale = _quantize_values(x, True)
    packed_layout: gl.constexpr = routed.type.layout
    r = gl.convert_layout(row, gl.SliceLayout(1, packed_layout))
    g = gl.convert_layout(group, gl.SliceLayout(1, packed_layout))
    b = gl.arange(0, 16, gl.SliceLayout(0, packed_layout))
    dest = g[:, None] * 16 + b[None, :]
    gl.store(Q + dest, routed, r[:, None] < actual_m)
    gl.store(Q + M_BUCKET * (K // 2) + dest, shared, r[:, None] < actual_m)
    gl.store(QScale + group, routed_scale, row < actual_m)
    gl.store(QScale + M_BUCKET * (K // 32) + group, shared_scale, row < actual_m)

@gluon.jit(do_not_specialize=['actual_m'])
def _router_project_quantize(X, W, Logits, Q, QScale, actual_m, M_BUCKET: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr, GROUPS: gl.constexpr, VALUES: gl.constexpr):
    tile = gl.program_id(0)
    rows: gl.constexpr = gl.cdiv(M_BUCKET, BM)
    projection_tiles: gl.constexpr = rows * (256 // BN)
    if tile < projection_tiles:
        _router_projection_tile(X, W, Logits, tile % rows, tile // rows, actual_m, K, SX, BM, BK, BN)
    else:
        _quantize_input_tile(X, Q, QScale, tile - projection_tiles, actual_m, M_BUCKET, K, SX, GROUPS, VALUES)

@gluon.jit
def _store_w13_activation(acc, Y, YScale, expert, start_row, column, N: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr):
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    gate, up = _split_columns(acc)
    if expert == 256:
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
    activated = (gate * _sigmoid(gate) * up).to(gl.bfloat16)
    activated = gl.convert_layout(activated, ep).reshape((BM * BN // 64, 32)).to(gl.float32)
    packed, scale = _quantize_values(activated, expert == 256)
    packed = gl.convert_layout(packed.reshape((BM, BN // 4)), ep)
    scale = gl.convert_layout(scale.reshape((BM, BN // 64)), ep)
    rm = start_row + gl.arange(0, BM, gl.SliceLayout(1, ep))
    cn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, ep))
    sg = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, ep))
    gl.store(Y + rm[:, None] * (N // 4) + cn[None, :], packed)
    gl.store(YScale + rm[:, None] * (N // 64) + sg[None, :], scale)

@gluon.jit
def _reconstruction_bits(value, code, quantum):
    reconstructed = code.to(gl.float32) * quantum[:, None]
    mismatch = reconstructed.to(gl.uint32, bitcast=True) ^ value.to(gl.uint32, bitcast=True)
    return gl.max(mismatch, 1)

@gluon.jit
def _store_w2_panel(acc, raw_low, raw_high, code_low, code_high, route_start, route_split, header_base, column, valid_rows, N: gl.constexpr, BN: gl.constexpr, ROW_BASE: gl.constexpr, CODE_STRIDE: gl.constexpr):
    peak = gl.max(gl.abs(acc), 1)
    exponent = (peak.to(gl.uint32, bitcast=True) >> 23 & 255).to(gl.int32) - 141
    exponent = gl.maximum(-126, gl.minimum(112, exponent))
    quantum = (exponent + 127 << 23).to(gl.float32, bitcast=True)
    inverse = (127 - exponent << 23).to(gl.float32, bitcast=True)
    biased = gl.fma(acc, inverse[:, None], 12582912.0)
    code = biased.to(gl.uint32, bitcast=True).to(gl.int16)
    if ROW_BASE == 0:
        acc_low, acc_high = _split_columns(acc)
        code_half_low, code_half_high = _split_columns(code)
        half_quantum = gl.convert_layout(quantum, gl.SliceLayout(1, code_half_low.type.layout), assert_trivial=True)
        low_bits = _reconstruction_bits(acc_low, code_half_low, half_quantum)
        escaped = low_bits | _reconstruction_bits(acc_high, code_half_high, half_quantum) != 0
        escaped = gl.convert_layout(escaped, gl.SliceLayout(1, acc.type.layout), assert_trivial=True)
    else:
        escaped = _reconstruction_bits(acc, code, quantum) != 0
    native_rows = ROW_BASE + gl.arange(0, acc.shape[0], gl.SliceLayout(1, acc.type.layout))
    gl.store(header_base + native_rows * (N // BN) + column, gl.where(escaped, 0, exponent + 127).to(gl.uint8), native_rows < valid_rows)
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    codes_tile = gl.allocate_shared_memory(gl.int16, acc.shape, shared, code)
    codes = codes_tile.load(ep)
    rows = ROW_BASE + gl.arange(0, acc.shape[0], gl.SliceLayout(1, ep))
    columns = column * BN + gl.arange(0, BN, gl.SliceLayout(0, ep))
    global_rows = route_start + rows
    code_mask = rows[:, None] < valid_rows
    gl.amd.cdna4.buffer_store(stored_value=codes, ptr=code_low, offsets=global_rows[:, None] * CODE_STRIDE + columns[None, :], mask=code_mask & (global_rows[:, None] < route_split), cache='.cs')
    gl.amd.cdna4.buffer_store(stored_value=codes, ptr=code_high, offsets=(global_rows[:, None] - route_split) * CODE_STRIDE + columns[None, :], mask=code_mask & (global_rows[:, None] >= route_split), cache='.cs')
    gl.barrier()
    if gl.sum((escaped & (native_rows < valid_rows)).to(gl.int32), 0) != 0:
        raw_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])
        raw_tile = gl.allocate_shared_memory(gl.float32, acc.shape, raw_layout, acc)
        raw = raw_tile.load(ep)
        escape_rows = gl.convert_layout(escaped, gl.SliceLayout(1, ep))
        raw_mask = (rows[:, None] < valid_rows) & escape_rows[:, None]
        gl.amd.cdna4.buffer_store(stored_value=raw, ptr=raw_low, offsets=global_rows[:, None] * N + columns[None, :], mask=raw_mask & (global_rows[:, None] < route_split), cache='.cs')
        gl.amd.cdna4.buffer_store(stored_value=raw, ptr=raw_high, offsets=(global_rows[:, None] - route_split) * N + columns[None, :], mask=raw_mask & (global_rows[:, None] >= route_split), cache='.cs')
    gl.barrier()

@gluon.jit
def _store_w2_panels(acc, raw_low, raw_high, code_low, code_high, route_start, route_split, header_base, column, valid_rows, N: gl.constexpr, BN: gl.constexpr, CODE_STRIDE: gl.constexpr):
    low, high = gl.split(acc.reshape((2, 64, BN)).permute((1, 2, 0)))
    _store_w2_panel(low, raw_low, raw_high, code_low, code_high, route_start, route_split, header_base, column, valid_rows, N, BN, 0, CODE_STRIDE)
    _store_w2_panel(high, raw_low, raw_high, code_low, code_high, route_start, route_split, header_base, column, valid_rows, N, BN, 64, CODE_STRIDE)

@gluon.jit
def _fp4_gemm(X, XScale, W, WScale, Sorted, expert, start_row, column, valid_rows, N: gl.constexpr, K: gl.constexpr, UP: gl.constexpr, M_STORAGE: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[4, 1] if not UP and BM >= 64 else [2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0]) if UP or BM >= 64 else gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([16, 1], [1, 64], [2, 2], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [BM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    mi = gl.arange(0, BM, gl.SliceLayout(1, al))
    row = start_row + mi
    if UP:
        if expert == 256:
            row += M_STORAGE
        else:
            route = gl.load(Sorted + row)
            row = gl.maximum(route // 8, 0)
        row = gl.where(mi < valid_rows, row, 0)
    if UP:
        ak = gl.arange(0, BK // 2, gl.SliceLayout(0, al))
        bk = gl.arange(0, BK // 2, gl.SliceLayout(1, bl))
        ni = gl.arange(0, BN, gl.SliceLayout(0, bl))
        wn = column * (BN // 2) + ni % (BN // 2) + ni // (BN // 2) * (N // 2)
        a_offset = row[:, None] * (K // 2) + ak[None, :]
        b_offset = wn[None, :] // 16 * (K * 8) + bk[:, None] // 16 * 256 + wn[None, :] % 16 * 16 + bk[:, None] % 16
    w_base = W + expert * (N * K // 2)
    ws_base = WScale + expert * (N * gl.cdiv(K // 32, 8) * 8)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    if UP:
        gl.static_assert(BK == 128 and BN == 256)
        a_shared: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, order=[1, 0])
        b_shared: gl.constexpr = gl.SharedLinearLayout(offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0]], block_bases=[], alignment=16)
        a_tile = gl.allocate_shared_memory(gl.uint8, (BM, BK // 2), a_shared)
        b_tile = gl.allocate_shared_memory(gl.uint8, (BK // 2, BN), b_shared)
        a_odd_tile = gl.allocate_shared_memory(gl.uint8, (BM, BK // 2), a_shared)
        b_odd_tile = gl.allocate_shared_memory(gl.uint8, (BK // 2, BN), b_shared)
        a_word_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 2], [4, 1], [1, 0])
        b_word_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
        word_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
        b_scale_shared: gl.constexpr = gl.SharedLinearLayout(offset_bases=[[16, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [32, 0], [64, 0], [128, 0]], block_bases=[], alignment=4)
        xa_words = gl.allocate_shared_memory(gl.uint32, (BM, 2), word_shared)
        wb_words = gl.allocate_shared_memory(gl.uint32, (BN // 32, 64), word_shared)
        xa_tile = xa_words._reinterpret(gl.uint8, (BM, 8), word_shared)
        wb_tile = wb_words._reinterpret(gl.uint8, (BN, 8), b_scale_shared)
        panel_row = gl.convert_layout(row, gl.SliceLayout(1, a_word_layout))
        ag = gl.arange(0, 2, gl.SliceLayout(0, a_word_layout))
        nt = gl.arange(0, BN // 32, gl.SliceLayout(1, b_word_layout))
        word = gl.arange(0, 64, gl.SliceLayout(0, b_word_layout))
        n_tile = column * (BN // 64) + nt % (BN // 64) + nt // (BN // 64) * (N // 64)
        xa_offset = panel_row[:, None] * (K // 128) + ag[None, :]
        wb_offset = n_tile[:, None] * (K // 256 * 64) + word[None, :]
        xa_ptr = XScale.to(gl.pointer_type(gl.uint32))
        wb_ptr = ws_base.to(gl.pointer_type(gl.uint32))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(xa_words, xa_ptr, xa_offset)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(wb_words, wb_ptr, wb_offset)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_tile, X, a_offset)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_tile, w_base, b_offset)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_odd_tile, X + 64, a_offset)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_odd_tile, w_base + 1024, b_offset)
        gl.amd.cdna4.async_copy.commit_group()
        for panel in range(K // 256 - 1):
            gl.amd.cdna4.async_copy.wait_group(1)
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_tile, ad)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_tile, bd)
            a_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(xa_tile.slice(0, 4, 1), asl)
            b_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(wb_tile.slice(0, 4, 1), bsl)
            a_next_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(xa_tile.slice(4, 4, 1), asl)
            b_next_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(wb_tile.slice(4, 4, 1), bsl)
            gl.barrier()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(xa_words, xa_ptr + (panel + 1) * 2, xa_offset)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(wb_words, wb_ptr + (panel + 1) * 64, wb_offset)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(a_tile, X + (panel + 1) * 128, a_offset)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(b_tile, w_base + (panel + 1) * 2048, b_offset)
            gl.amd.cdna4.async_copy.commit_group()
            acc = gl.amd.cdna4.mfma_scaled(a, a_scale, 'e2m1', b, b_scale, 'e2m1', acc)
            gl.amd.cdna4.async_copy.wait_group(1)
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_odd_tile, ad)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_odd_tile, bd)
            gl.barrier()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(a_odd_tile, X + (panel + 1) * 128 + 64, a_offset)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(b_odd_tile, w_base + (panel + 1) * 2048 + 1024, b_offset)
            gl.amd.cdna4.async_copy.commit_group()
            acc = gl.amd.cdna4.mfma_scaled(a, a_next_scale, 'e2m1', b, b_next_scale, 'e2m1', acc)
        gl.amd.cdna4.async_copy.wait_group(1)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_tile, ad)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_tile, bd)
        a_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(xa_tile.slice(0, 4, 1), asl)
        b_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(wb_tile.slice(0, 4, 1), bsl)
        acc = gl.amd.cdna4.mfma_scaled(a, a_scale, 'e2m1', b, b_scale, 'e2m1', acc)
        gl.amd.cdna4.async_copy.wait_group(0)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_odd_tile, ad)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_odd_tile, bd)
        a_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(xa_tile.slice(4, 4, 1), asl)
        b_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(wb_tile.slice(4, 4, 1), bsl)
        acc = gl.amd.cdna4.mfma_scaled(a, a_scale, 'e2m1', b, b_scale, 'e2m1', acc)
    else:
        gl.static_assert(BK == 256 and BN == 256)
        a_shared: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, order=[1, 0])
        b_shared: gl.constexpr = gl.SharedLinearLayout(offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0], [64, 0]], block_bases=[], alignment=16)
        a_words_shared: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 4, order=[1, 0])
        a_words = gl.allocate_shared_memory(gl.uint32, (BM, BK // 8), a_words_shared)
        a_tile = a_words._reinterpret(gl.uint8, (BM, BK // 2), a_shared)
        b_words_shared: gl.constexpr = gl.SharedLinearLayout(offset_bases=[[1, 0], [2, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [4, 0], [8, 0], [16, 0]], block_bases=[], alignment=16)
        b_words = gl.allocate_shared_memory(gl.uint32, (BK // 8, BN), b_words_shared)
        b_tile = b_words._reinterpret(gl.uint8, (BK // 2, BN), b_shared)
        a_word_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 2], [4, 1], [1, 0])
        b_word_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
        word_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
        b_scale_shared: gl.constexpr = gl.SharedLinearLayout(offset_bases=[[16, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [32, 0], [64, 0], [128, 0]], block_bases=[], alignment=4)
        xa_words = gl.allocate_shared_memory(gl.uint32, (BM, 2), word_shared)
        wb_words = gl.allocate_shared_memory(gl.uint32, (BN // 32, 64), word_shared)
        xa_tile = xa_words._reinterpret(gl.uint8, (BM, 8), word_shared)
        wb_tile = wb_words._reinterpret(gl.uint8, (BN, 8), b_scale_shared)
        panel_row = gl.convert_layout(row, gl.SliceLayout(1, a_word_layout))
        ag = gl.arange(0, 2, gl.SliceLayout(0, a_word_layout))
        nt = gl.arange(0, BN // 32, gl.SliceLayout(1, b_word_layout))
        word = gl.arange(0, 64, gl.SliceLayout(0, b_word_layout))
        xa_offset = panel_row[:, None] * (K // 128) + ag[None, :]
        wb_offset = (column * (BN // 32) + nt[:, None]) * (K // 256 * 64) + word[None, :]
        xa_ptr = XScale.to(gl.pointer_type(gl.uint32))
        wb_ptr = ws_base.to(gl.pointer_type(gl.uint32))
        a_copy_layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
        copy_rows = start_row + gl.arange(0, BM, gl.SliceLayout(1, a_copy_layout))
        copy_k = gl.arange(0, BK // 8, gl.SliceLayout(0, a_copy_layout))
        copy_offset = copy_rows[:, None] * (K // 8) + copy_k[None, :]
        a_ptr = X.to(gl.pointer_type(gl.uint32))
        b_copy_layout: gl.constexpr = gl.BlockedLayout([4, 1], [1, 64], [2, 2], [0, 1])
        b_word_k = gl.arange(0, BK // 8, gl.SliceLayout(1, b_copy_layout))
        b_word_n = column * BN + gl.arange(0, BN, gl.SliceLayout(0, b_copy_layout))
        b_word_offset = b_word_n[None, :] // 16 * (K * 2) + b_word_k[:, None] // 4 * 64 + b_word_n[None, :] % 16 * 4 + b_word_k[:, None] % 4
        b_ptr = w_base.to(gl.pointer_type(gl.uint32))
        for step in range(K // BK):
            gl.amd.cdna4.async_copy.buffer_load_to_shared(xa_words, xa_ptr, xa_offset + step * 2)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(wb_words, wb_ptr, wb_offset + step * 64)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(a_words, a_ptr, copy_offset + step * (BK // 8))
            gl.amd.cdna4.async_copy.buffer_load_to_shared(b_words, b_ptr, b_word_offset + step * (BK * 2))
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.wait_group(0)
            a = a_tile.load(ad)
            b = b_tile.load(bd)
            a_scale = xa_tile.load(asl)
            b_scale = wb_tile.load(bsl)
            acc = gl.amd.cdna4.mfma_scaled(a, a_scale, 'e2m1', b, b_scale, 'e2m1', acc)
    return acc

@gluon.jit(do_not_specialize=['actual_m'])
def _expert_projection(X, XScale, W, Scales, Sorted, Experts, Y, Y2, YScale, Codes, Codes2, Headers, actual_m, N: gl.constexpr, K: gl.constexpr, UP: gl.constexpr, M_BUCKET: gl.constexpr, P_SPLIT: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr):
    pid = gl.program_id(0)
    blocks: gl.constexpr = gl.cdiv((9 if UP else 8) * M_BUCKET + (257 if UP else 256) * (BM - 1), BM)
    block, column = _grouped_tile(pid, blocks, N // BN, GROUP_M)
    descriptor = gl.load(Experts + block)
    active = descriptor >= 0
    if not UP:
        active = active & (descriptor & 511 != 256)
    if active:
        expert = (descriptor & 511).to(gl.int32)
        valid_rows = (descriptor >> 9 & 255).to(gl.int32)
        dense_base = (descriptor >> 17).to(gl.int32)
        source_row = gl.where(UP & (expert == 256), dense_base - 8 * actual_m, block * BM)
        if UP:
            up = _fp4_gemm(X, XScale, W, Scales, Sorted, expert, source_row, column, valid_rows, N, K, True, M_BUCKET, BM, BN, BK)
            _store_w13_activation(up, Y, YScale, expert, block * BM, column, N, BM, BN)
        elif valid_rows <= 64:
            small_down = _fp4_gemm(X, XScale, W, Scales, Sorted, expert, source_row, column, valid_rows, N, K, False, M_BUCKET, 64, BN, BK)
            header_base = Headers + dense_base * (N // BN)
            _store_w2_panel(small_down, Y, Y2, Codes, Codes2, dense_base, P_SPLIT, header_base, column, valid_rows, N, BN, 0, N + 64)
        else:
            large_down = _fp4_gemm(X, XScale, W, Scales, Sorted, expert, source_row, column, valid_rows, N, K, False, M_BUCKET, BM, BN, BK)
            header_base = Headers + dense_base * (N // BN)
            _store_w2_panels(large_down, Y, Y2, Codes, Codes2, dense_base, P_SPLIT, header_base, column, valid_rows, N, BN, N + 64)

@gluon.jit(do_not_specialize=['actual_m'])
def _shared_reduce(X, XScale, W, Scales, Offsets, P0, P1, Codes0, Codes1, Headers, Weights, Inverse, Y, actual_m, H: gl.constexpr, K: gl.constexpr, M_BUCKET: gl.constexpr, P_SPLIT: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, route_count=8, GROUP: gl.constexpr=1):
    blocks = gl.cdiv(actual_m, BM)
    block, column = _grouped_tile_runtime(gl.program_id(0), blocks, H // BN, GROUP)
    first = block * BM
    source_row = gl.load(Offsets + 256) + first
    acc = _fp4_gemm(X, XScale, W, Scales, Offsets, 256, source_row, column, gl.minimum(actual_m - first, BM), H, K, False, M_BUCKET, BM, BN, 256)
    values_per_lane: gl.constexpr = 4 if M_BUCKET < 6144 else 8
    lane_rows: gl.constexpr = 8 if M_BUCKET >= 12288 else 4
    ep: gl.constexpr = gl.BlockedLayout([1, values_per_lane], [lane_rows, 64 // lane_rows], [4, 1], [1, 0])
    shared = gl.convert_layout(acc.to(gl.bfloat16), ep)
    rm = first + gl.arange(0, BM, gl.SliceLayout(1, ep))
    cn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, ep))
    value = gl.full((BM, BN), 0, gl.float32, ep)
    metadata_layout: gl.constexpr = gl.BlockedLayout([1, 1], [lane_rows, 64 // lane_rows], [4, 1], [1, 0])
    metadata_row = first + gl.arange(0, BM, gl.SliceLayout(1, metadata_layout))
    route_id = gl.arange(0, 8, gl.SliceLayout(0, metadata_layout))
    dense_rows = gl.load(Inverse + metadata_row[:, None] * 8 + route_id[None, :], metadata_row[:, None] < actual_m, 0)
    route_weights = gl.load(Weights + metadata_row[:, None] * 8 + route_id[None, :], metadata_row[:, None] < actual_m, 0)
    route_headers = gl.load(Headers + dense_rows * (H // 256) + column // (256 // BN), metadata_row[:, None] < actual_m, 127)
    for j in range(route_count):
        index = gl.full((BM, 1), j, gl.int32, metadata_layout)
        dense_row = gl.gather(dense_rows, index, 1).reshape((BM,))
        weight = gl.gather(route_weights, index, 1).reshape((BM,))
        dense_row = gl.convert_layout(dense_row, gl.SliceLayout(1, ep))
        weight = gl.convert_layout(weight, gl.SliceLayout(1, ep))
        header_bits = gl.gather(route_headers, index, 1).reshape((BM,))
        header_bits = gl.convert_layout(header_bits, gl.SliceLayout(1, ep))
        header = (header_bits.to(gl.uint32) << 23).to(gl.float32, bitcast=True)
        code_mask = rm[:, None] < actual_m
        code_offset = dense_row[:, None] * (H + 64) + cn[None, :]
        code_high_offset = (dense_row[:, None] - P_SPLIT) * (H + 64) + cn[None, :]
        code = gl.amd.cdna4.buffer_load(ptr=Codes0, offsets=code_offset, mask=code_mask & (dense_row[:, None] < P_SPLIT), other=0, cache='.cg')
        code = gl.amd.cdna4.buffer_load(ptr=Codes1, offsets=code_high_offset, mask=code_mask & (dense_row[:, None] >= P_SPLIT), other=code, cache='.cg')
        part = code.to(gl.float32) * header[:, None]
        raw_mask = (rm[:, None] < actual_m) & (header[:, None] == 0)
        low_offset = dense_row[:, None] * H + cn[None, :]
        high_offset = (dense_row[:, None] - P_SPLIT) * H + cn[None, :]
        part = gl.load(P0 + low_offset, raw_mask & (dense_row[:, None] < P_SPLIT), part, cache_modifier='.cg')
        part = gl.load(P1 + high_offset, raw_mask & (dense_row[:, None] >= P_SPLIT), part, cache_modifier='.cg')
        value += part * weight[:, None]
    value += shared.to(gl.float32)
    gl.store(Y + rm[:, None] * H + cn[None, :], value, rm[:, None] < actual_m)

class _Workspace:

    def __init__(self, x, intermediate, storage_m, capacity, block_m, chunks):
        m, h = x.shape

        def empty(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=x.device)
        # gfx950 buffer operations have a 4-GiB addressing window.  Split the
        # FP32 routed partials so each allocation stays below that limit at
        # M=32768; dense route ids select the half in the kernels.
        self.parts_low = empty((4 * storage_m, h), torch.float32)
        self.parts_high = empty((4 * storage_m, h), torch.float32)
        self.codes_low = empty((4 * storage_m, h + 64), torch.int16)
        self.codes_high = empty((4 * storage_m, h + 64), torch.int16)
        early_bf16 = self.parts_low.view(torch.bfloat16)
        early_i32 = self.parts_low.view(torch.int32)
        cursor = 0

        def early(shape, base, item_bytes):
            nonlocal cursor
            elements = 1
            for dimension in shape:
                elements *= dimension
            begin = triton.cdiv(cursor, 256) * 256
            cursor = begin + elements * item_bytes
            return base.view(-1).narrow(0, begin // item_bytes, elements).view(shape)
        self.logits = early((m, 256), early_bf16, 2)
        self.ids = early((m, 8), early_i32, 4)
        self.weights = empty((m, 8), torch.float32)
        self.partial_counts = early((256, chunks), early_i32, 4)
        self.counts = early((257,), early_i32, 4)
        self.offsets = empty((2 * 257,), torch.int32)
        self.inverse = empty((m, 8), torch.int32)
        self.sorted_routes = early((capacity,), early_i32, 4)
        self.experts = empty((capacity // block_m,), torch.int64)
        self.xq = self.codes_low.view(torch.uint8).as_strided((2 * storage_m, h // 2), (h // 2, 1))
        self.xs = empty((2 * storage_m, h // 32), torch.uint8)
        self.aq = empty((capacity, intermediate // 2), torch.uint8)
        self.aqs = empty((capacity, intermediate // 32), torch.uint8)
        self.headers = empty((8 * storage_m, h // 256), torch.uint8)
        self.output = empty((m, h))

def _route_inputs(x, router, correction_bias, work, storage_m, block_m, routed_scaling_factor):
    m, h = x.shape
    capacity = work.sorted_routes.numel()
    chunks = work.partial_counts.shape[1]
    routes = 8 * m
    router_rows = 32 if storage_m < 6144 else 64 if storage_m < 16384 else 128
    router_columns = 128 if 12288 <= storage_m < 16384 else 64
    router_k = 256 if storage_m < 6144 else 128
    quantize_groups, quantize_values = (128, 16) if storage_m < 12288 else (256, 32)
    projection_tiles = triton.cdiv(storage_m, router_rows) * (256 // router_columns)
    quantize_tiles = triton.cdiv(storage_m * h // 32, quantize_groups)
    _router_project_quantize[projection_tiles + quantize_tiles,](x, router, work.logits, work.xq, work.xs, m, storage_m, h, x.stride(0), router_rows, router_k, router_columns, quantize_groups, quantize_values)
    _router[m,](work.logits, correction_bias, work.ids, work.weights, routed_scaling_factor, num_warps=1)
    _chunk_counts[max(chunks, triton.cdiv(capacity, 1024)),](work.ids, work.partial_counts, work.sorted_routes, work.experts, routes, chunks, capacity, block_m)
    _chunk_prefix[257,](work.partial_counts, work.partial_counts, work.counts, m, chunks, triton.next_power_of_2(chunks))
    _build_expert_blocks[257,](work.counts, work.offsets, work.experts, block_m, triton.next_power_of_2(triton.cdiv(m, block_m)))
    _scatter[chunks,](work.ids, work.offsets, work.partial_counts, work.sorted_routes, work.inverse, routes, chunks)

def _padded_capacity(routes, experts, block_m):
    return triton.cdiv(routes + experts * (block_m - 1), block_m) * block_m

def _finish(work, w2, w2_scale, m, storage_m, h, intermediate):
    reduce_columns, reduce_rows = (256, 32)
    _shared_reduce[triton.cdiv(m, reduce_rows) * (h // reduce_columns),](work.aq, work.aqs, w2, w2_scale, work.offsets, work.parts_low, work.parts_high, work.codes_low, work.codes_high, work.headers, work.weights, work.inverse, work.output, m, h, intermediate, storage_m, 4 * storage_m, reduce_rows, reduce_columns, 8, 8 if storage_m >= 6144 else 1, enable_fp_fusion=False)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, routed_scaling_factor=2.5):
    m, h = x.shape
    assert 4193 <= m <= 32768
    storage_m = triton.cdiv(m, 1024) * 1024
    intermediate = w13.shape[1] // 2
    block_m = 128
    capacity = _padded_capacity(9 * storage_m, 257, block_m)
    work = _Workspace(x, intermediate, storage_m, capacity, block_m, triton.cdiv(8 * storage_m, 256))
    _route_inputs(x, router, correction_bias, work, storage_m, block_m, routed_scaling_factor)
    up_columns, up_k = (256, 128)
    _expert_projection[capacity // block_m * (2 * intermediate // up_columns),](work.xq, work.xs, w13, w13_scale, work.sorted_routes, work.experts, work.aq, work.aq, work.aqs, work.codes_low, work.codes_low, work.headers, m, 2 * intermediate, h, True, storage_m, 0, block_m, up_columns, up_k, 8, enable_fp_fusion=False)
    routed_capacity = _padded_capacity(8 * storage_m, 256, block_m)
    _expert_projection[routed_capacity // block_m * (h // 256),](work.aq, work.aqs, w2, w2_scale, work.sorted_routes, work.experts, work.parts_low, work.parts_high, work.aqs, work.codes_low, work.codes_high, work.headers, m, h, intermediate, False, storage_m, 4 * storage_m, block_m, 256, 256, 2, enable_fp_fusion=False, waves_per_eu=2)
    _finish(work, w2, w2_scale, m, storage_m, h, intermediate)
    return work.output
