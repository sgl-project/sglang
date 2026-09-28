# fmt: off
"""GLM-5.2 TP8 MTP fused MoE specialization for M=1..16."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _weight_offset(expert, n, k, N: gl.constexpr, K: gl.constexpr):
    byte = k // 2
    return (((expert * (N // 16) + n // 16) * (K // 64) + byte // 32) * 2 + byte // 16 % 2) * 256 + n % 16 * 16 + byte % 16

@gluon.jit
def _scale_offset(expert, n, k, N: gl.constexpr, K: gl.constexpr):
    group = k // 32
    groups: gl.constexpr = triton.cdiv(K // 32, 8) * 8
    return (((((expert * (N // 32) + n // 32) * (groups // 8) + group // 8) * 4 + group % 4) * 16 + n % 16) * 2 + group // 4 % 2) * 2 + n // 16 % 2

@gluon.jit
def _quantize_mxfp4_codes(x, shared):
    peak = gl.max(gl.abs(x), 1)
    peak_bits = peak.to(gl.uint32, bitcast=True)
    floor_exponent = (peak_bits >> 23 & 255).to(gl.int32) - 127
    threshold = gl.exp2(floor_exponent.to(gl.float32)) * 1.75
    even_exponent = floor_exponent - 2 + (peak >= threshold).to(gl.int32)
    divided = gl.div_rn(peak, 6.0)
    bits = divided.to(gl.uint32, bitcast=True)
    ceil_exponent = (bits >> 23 & 255).to(gl.int32) - 127 + (bits & 8388607 != 0)
    exponent = gl.where(shared, even_exponent, ceil_exponent)
    exponent = gl.maximum(-127, gl.minimum(127, exponent))
    scale = gl.exp2(exponent.to(gl.float32))
    a = gl.abs(x / scale[:, None])
    code = gl.where(a <= 0.25, 0, gl.where(a < 0.75, 1, gl.where(a <= 1.25, 2, gl.where(a < 1.75, 3, gl.where(a <= 2.5, 4, gl.where(a < 3.5, 5, gl.where(a <= 5.0, 6, 7)))))))
    code = (code | gl.where(x < 0, 8, 0)).to(gl.uint8)
    return (code, (exponent + 127).to(gl.uint8))

@gluon.jit
def _store_input_quantized(Q, values, row, group, H: gl.constexpr, M: gl.constexpr, SHARED: gl.constexpr):
    code, scales = _quantize_mxfp4_codes(values, SHARED)
    pack_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [1, 0])
    code = gl.convert_layout(code, pack_layout)
    lo, hi = gl.split(code.reshape((values.shape[0], 16, 2)))
    packed = lo | hi << 4
    pg = gl.convert_layout(group, gl.SliceLayout(1, packed.type.layout))
    pk = gl.arange(0, 16, gl.SliceLayout(0, packed.type.layout))
    pitch: gl.constexpr = H // 2 + H // 32
    output_row = row + (M if SHARED else 0)
    gl.store(Q + output_row * pitch + pg[:, None] * 16 + pk[None, :], packed)
    gl.store(Q + output_row * pitch + H // 2 + group, scales)

@gluon.jit
def _router_linear(X, W, Y, Q, Groups, H: gl.constexpr, SX: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr, GROUPED: gl.constexpr, QVEC: gl.constexpr, QGROUPS: gl.constexpr):
    program = gl.program_id(0)
    if program < 256 // BN * SPLITS:
        tile = program % (256 // BN)
        split = program // (256 // BN)
        mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
        al: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [0, 1])
        bl: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 1], [1, 0])
        mi = gl.arange(0, 16, gl.SliceLayout(1, al))
        input_m = gl.minimum(mi, M - 1)
        ak = gl.arange(0, BK, gl.SliceLayout(0, al))
        n = tile * BN + gl.arange(0, BN, gl.SliceLayout(1, bl))
        bk = gl.arange(0, BK, gl.SliceLayout(0, bl))
        acc = gl.zeros((16, BN), gl.float32, mma)
        for base in range(H // SPLITS // BK):
            start = split * (H // SPLITS) + base * BK
            a = gl.load(X + input_m[:, None] * SX + start + ak[None, :])
            b = gl.load(W + n[:, None] * H + start + bk[None, :])
            acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b.T, gl.DotOperandLayout(1, mma, 8)), acc)
        om = gl.arange(0, 16, gl.SliceLayout(1, mma))
        on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
        gl.store(Y + (om[:, None] * SPLITS + split) * 256 + on[None, :], acc, om[:, None] < M)
    elif GROUPED and program == 256 // BN * SPLITS:
        e = gl.arange(0, 512, gl.BlockedLayout([1], [64], [1], [0]))
        gl.store(Groups + e, 0, e < 257)
    else:
        quant = program - 256 // BN * SPLITS - (1 if GROUPED else 0)
        row = quant // (H // (32 * QGROUPS))
        tile_q = quant % (H // (32 * QGROUPS))
        layout: gl.constexpr = gl.BlockedLayout([1, QVEC], [2 * QVEC, 32 // QVEC], [1, 1], [1, 0])
        group = tile_q * QGROUPS + gl.arange(0, QGROUPS, gl.SliceLayout(1, layout))
        lane = gl.arange(0, 32, gl.SliceLayout(0, layout))
        k = group[:, None] * 32 + lane[None, :]
        values = gl.load(X + row * SX + k).to(gl.float32)
        _store_input_quantized(Q, values, row, group, H, M, False)
        _store_input_quantized(Q, values, row, group, H, M, True)

@gluon.jit
def _select_routes(Logits, Bias, Ids, Weights, Groups, SPLITS: gl.constexpr, GROUPED: gl.constexpr, SCALE: gl.constexpr):
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    record_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    e = gl.arange(0, 256, layout)
    logits = gl.load(Logits + row * SPLITS * 256 + e)
    for part in gl.static_range(1, SPLITS):
        logits += gl.load(Logits + (row * SPLITS + part) * 256 + e)
    logits = logits.to(gl.bfloat16).to(gl.float32)
    probability = 1.0 / (1.0 + gl.exp(-logits))
    score = probability + gl.load(Bias + e).to(gl.float32)
    probability_table = gl.allocate_shared_memory(gl.float32, [256], gl.SwizzledSharedLayout(1, 1, 1, [0]), probability)
    record_size: gl.constexpr = 16 if GROUPED else 256
    r = gl.arange(0, record_size, record_layout)
    available = gl.full((256,), True, gl.int1, layout)
    total = 0.0
    selected_ids = gl.full((record_size,), 256, gl.int32, record_layout)
    for j in gl.static_range(8):
        maximum = gl.max(score, 0)
        key = gl.where(available, e + gl.where(score == maximum, 0, 256), 512)
        local_key = gl.min(gl.reshape(key, (64, 4)), 1)
        local_key = gl.convert_layout(local_key, record_layout)
        valid, fallback = gl.inline_asm_elementwise('v_cmp_gt_u32_e64 $0, 1, $2\nv_cmp_gt_u32_e64 $1, 2, $2', constraints='=&s,=&s,v', args=[local_key >> 8], dtype=(gl.uint64, gl.uint64), is_pure=True, pack=1)
        mask = gl.where(valid != 0, valid, fallback)
        winning_lane = gl.inline_asm_elementwise('s_ff1_i32_b64 $0, $1', constraints='=s,s', args=[mask], dtype=gl.int32, is_pure=True, pack=1)
        elected = gl.inline_asm_elementwise('v_readlane_b32 $0, $1, $2', constraints='=s,v,s', args=[local_key, winning_lane], dtype=gl.int32, is_pure=True, pack=1)
        first = gl.full((1,), 0, gl.int32, record_layout)
        idx = gl.sum(gl.gather(elected, first, 0), 0) & 255
        selected_ids = gl.where(r == j, idx, selected_ids)
        available = available & (e != idx)
        score = gl.where(e == idx, -float('inf'), score)
    selected = probability_table.gather(gl.minimum(selected_ids, 255), 0)
    for j in gl.static_range(8):
        index = gl.full((1,), j, gl.int32, record_layout)
        total += gl.sum(gl.gather(selected, index, 0), 0)
    gl.store(Ids + row * 9 + r, selected_ids, r < 9)
    if GROUPED:
        membership = (r + 1).to(gl.uint64) << row * 4
        gl.amd.cdna4.buffer_atomic_or(Groups.to(gl.pointer_type(gl.int64)), selected_ids, membership.to(gl.int64, bitcast=True), r < 9, sem='relaxed')
    gl.store(Weights + row * 9 + r, selected / total * SCALE, r < 8)
    gl.store(Weights + row * 9 + 8, 1.0)

@gluon.jit
def _expert_coordinates(N: gl.constexpr, M: gl.constexpr, UP: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, SHARED_ONLY: gl.constexpr, STAGGER: gl.constexpr):
    if SHARED_ONLY:
        route = gl.cast(8, gl.int32)
        if STAGGER:
            tile = gl.program_id(0)
            split = gl.cast(0, gl.int32)
        else:
            work = gl.program_id(0) - M
            tile = work % (N // BN)
            split = work // (N // BN)
    else:
        if STAGGER:
            work = gl.program_id(0) - M * (N // 128)
            job = work // (N // BN) % (M * 8) if M <= 2 else work // (SPLITS * (N // BN))
            tile = work % (N // BN) if M <= 2 else work // SPLITS % (N // BN)
            split = work // (M * 8 * (N // BN)) if M <= 2 else work % SPLITS
        else:
            job = gl.program_id(1) if UP else gl.program_id(0)
            tile = gl.program_id(2) if UP else gl.program_id(1)
            split = gl.program_id(0) if UP else gl.program_id(2)
        if STAGGER:
            route = job // 8 * 9 + job % 8
        else:
            route = gl.where(job < M * 8, job // 8 * 9 + job % 8, 8)
        if UP and M == 2:
            linear = tile + N // BN * split
            tile = linear // 4 % (N // BN)
            split = linear % 4 + linear // (4 * (N // BN)) * 4
    return (route, tile, split)

@gluon.jit
def _word_to_bytes(word):
    b0 = word.to(gl.uint8)
    b1 = (word >> 8).to(gl.uint8)
    b2 = (word >> 16).to(gl.uint8)
    b3 = (word >> 24).to(gl.uint8)
    return gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape(word.shape + (4,))

@gluon.jit
def _expert_projection(X, W, Scales, Ids, Groups, Y, N: gl.constexpr, K: gl.constexpr, M: gl.constexpr, UP: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, SHARED_ONLY: gl.constexpr=False, STAGGER: gl.constexpr=False):
    BK: gl.constexpr = 128
    GROUPED: gl.constexpr = M >= 2
    route, tile, split = _expert_coordinates(N, M, UP, SPLITS, BN, SHARED_ONLY, STAGGER)
    expert = gl.cast(256, gl.uint32) if SHARED_ONLY else gl.load(Ids + route).to(gl.uint32)
    if GROUPED:
        if SHARED_ONLY:
            members = gl.cast(((1 << M * 4) - 1) // 15 * 9, gl.uint64)
        else:
            members = gl.load(Groups + expert)
        earlier = gl.cast(1, gl.uint64) << route // 9 * 4
        owner = members & earlier - 1 == 0
    else:
        owner = True
    if owner:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 1])
        al: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, 1], [1, 0])
        bl: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 1], [0, 1])
        ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
        bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
        asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, BK // 32])
        bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
        mi = gl.arange(0, 16, gl.SliceLayout(1, al))
        if GROUPED:
            rank = (members >> mi * 4 & 15).to(gl.int32) - 1
            destination = gl.where(rank >= 0, mi * 9 + rank, -1)
            source_route = gl.maximum(destination, route).to(gl.uint32)
        else:
            destination = gl.full((16,), route, gl.int32, gl.SliceLayout(1, al))
            source_route = destination.to(gl.uint32)
        row = source_route // 9 + gl.where(expert == 256, M, 0) if UP else source_route
        ak = gl.arange(0, BK // 2, gl.SliceLayout(0, al))
        bn = tile * BN + gl.arange(0, BN, gl.SliceLayout(1, bl))
        bk = gl.arange(0, BK // 8, gl.SliceLayout(0, bl))
        ar = gl.convert_layout(row, gl.SliceLayout(1, asl))
        ag = gl.arange(0, BK // 32, gl.SliceLayout(0, asl))
        sn = tile * BN + gl.arange(0, BN, gl.SliceLayout(1, bsl))
        sg = gl.arange(0, BK // 32, gl.SliceLayout(0, bsl))
        pitch: gl.constexpr = K // 2 + K // 32
        acc = gl.zeros((16, BN), gl.float32, mma)
        for base in gl.static_range(K // SPLITS // BK):
            start = split * (K // SPLITS) + base * BK
            a = gl.load(X + row[:, None] * pitch + start // 2 + ak[None, :])
            b_offset = _weight_offset(expert, bn[:, None], start + 8 * bk[None, :], N, K) // 4
            b_offset = gl.max_contiguous(gl.multiple_of(b_offset, [1, 4]), [1, 4])
            words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), b_offset.to(gl.uint32))
            b = _word_to_bytes(words).reshape((BN, BK // 2))
            a_scale = gl.load(X + ar[:, None] * pitch + K // 2 + start // 32 + ag[None, :])
            b_scale = gl.load(Scales + _scale_offset(expert, sn[:, None], start + sg[None, :] * 32, N, K))
            acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), a_scale, 'e2m1', gl.convert_layout(b.T, bd), b_scale, 'e2m1', acc)
        om = gl.arange(0, 16, gl.SliceLayout(1, mma))
        on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
        output_route = gl.convert_layout(destination, gl.SliceLayout(1, mma))
        valid = output_route >= 0 if GROUPED else om == 0
        offsets = (output_route[:, None] * SPLITS + split) * N + on[None, :]
        if STAGGER and SHARED_ONLY:
            offsets = output_route[:, None] // 9 * N + on[None, :]
        gl.store(Y + offsets, acc, valid[:, None])

@gluon.jit
def _select_and_shared(Logits, Bias, Ids, Weights, Groups, X, W, Scales, GU, M: gl.constexpr, H: gl.constexpr, I: gl.constexpr, ROUTER_SPLITS: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, SCALE: gl.constexpr):
    if gl.program_id(0) < M:
        _select_routes(Logits, Bias, Ids, Weights, Groups, ROUTER_SPLITS, M >= 2, SCALE)
    else:
        _expert_projection(X, W, Scales, Ids, Groups, GU, 2 * I, H, M, True, SPLITS, BN, SHARED_ONLY=True)

@gluon.jit
def _activation_tile(GU, Q, route, tile, I: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [WARPS, 1], [1, 0])
    group = tile * (2 * WARPS) + gl.arange(0, 2 * WARPS, gl.SliceLayout(1, layout))
    lane = gl.arange(0, 32, gl.SliceLayout(0, layout))
    n = group[:, None] * 32 + lane[None, :]
    gate = gl.load(GU + route * SPLITS * 2 * I + n)
    up = gl.load(GU + route * SPLITS * 2 * I + I + n)
    for split in gl.static_range(1, SPLITS):
        gate += gl.load(GU + (route * SPLITS + split) * 2 * I + n)
        up += gl.load(GU + (route * SPLITS + split) * 2 * I + I + n)
    shared = route % 9 == 8
    gate = gl.where(shared, gate.to(gl.bfloat16).to(gl.float32), gate)
    up = gl.where(shared, up.to(gl.bfloat16).to(gl.float32), up)
    a = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
    code, scales = _quantize_mxfp4_codes(a, shared)
    pack_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [WARPS, 1], [1, 0])
    code = gl.convert_layout(code, pack_layout)
    lo, hi = gl.split(code.reshape((2 * WARPS, 16, 2)))
    packed = lo | hi << 4
    pg = gl.convert_layout(group, gl.SliceLayout(1, packed.type.layout))
    pk = gl.arange(0, 16, gl.SliceLayout(0, packed.type.layout))
    pitch: gl.constexpr = I // 2 + I // 32
    gl.store(Q + route * pitch + pg[:, None] * 16 + pk[None, :], packed)
    gl.store(Q + route * pitch + I // 2 + group, scales)

@gluon.jit
def _activate_quantize(GU, Q, I: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr):
    _activation_tile(GU, Q, gl.program_id(0), gl.program_id(1), I, SPLITS, WARPS)

@gluon.jit
def _up_and_shared_activation(X, W, Scales, Ids, Groups, GU, AQ, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr):
    program = gl.program_id(0)
    if program < M * (I // 64):
        route = program // (I // 64) * 9 + 8
        tile = program % (I // 64)
        _activation_tile(GU, AQ, route, tile, I, SPLITS, 1)
    else:
        _expert_projection(X, W, Scales, Ids, Groups, GU, 2 * I, H, M, True, SPLITS, BN, STAGGER=True)

@gluon.jit
def _activation_and_shared_down(GU, AQ, W, Scales, Ids, Groups, Shared, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr):
    BN: gl.constexpr = 32 if M <= 2 else 16
    program = gl.program_id(0)
    if program < H // BN:
        _expert_projection(AQ, W, Scales, Ids, Groups, Shared, H, I, M, False, 1, BN, SHARED_ONLY=True, STAGGER=True)
    else:
        work = program - H // BN
        job = work // (I // 64)
        route = job // 8 * 9 + job % 8
        tile = work % (I // 64)
        _activation_tile(GU, AQ, route, tile, I, SPLITS, 1)

@gluon.jit
def _routed_down_batch(X, W, Scales, Ids, token, tile, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr):
    BK: gl.constexpr = 128
    CN: gl.constexpr = 8 * BN
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, 8])
    al: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, 8], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [8, 1], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [CN, BK // 32])
    am = gl.arange(0, 16, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK // 2, gl.SliceLayout(0, al))
    col = gl.arange(0, CN, gl.SliceLayout(1, bl))
    bk = gl.arange(0, BK // 8, gl.SliceLayout(0, bl))
    expert = gl.load(Ids + token * 9 + col // BN)
    bn = tile * BN + col % BN
    sm = gl.arange(0, 16, gl.SliceLayout(1, asl))
    ag = gl.arange(0, BK // 32, gl.SliceLayout(0, asl))
    sc = gl.arange(0, CN, gl.SliceLayout(1, bsl))
    sg = gl.arange(0, BK // 32, gl.SliceLayout(0, bsl))
    se = gl.load(Ids + token * 9 + sc // BN)
    sn = tile * BN + sc % BN
    pitch: gl.constexpr = K // 2 + K // 32
    acc = gl.zeros((16, CN), gl.float32, mma)
    for base in gl.static_range(K // BK):
        start = base * BK
        a = gl.load(X + (token * 9 + am[:, None] % 8) * pitch + start // 2 + ak[None, :])
        b_offset = _weight_offset(expert[:, None], bn[:, None], start + 8 * bk[None, :], N, K) // 4
        b_offset = gl.max_contiguous(gl.multiple_of(b_offset, [1, 4]), [1, 4])
        words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), b_offset.to(gl.uint32))
        b = _word_to_bytes(words).reshape((CN, BK // 2))
        a_scale = gl.load(X + (token * 9 + sm[:, None] % 8) * pitch + K // 2 + start // 32 + ag[None, :])
        b_scale = gl.load(Scales + _scale_offset(se[:, None], sn[:, None], start + sg[None, :] * 32, N, K))
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), a_scale, 'e2m1', gl.convert_layout(b.T, bd), b_scale, 'e2m1', acc)
    expanded = gl.reshape(acc, (16, 8, BN))
    result_layout: gl.constexpr = expanded.type.layout
    rank = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(2, result_layout)))
    index = gl.full((1, 8, BN), 0, gl.int32, result_layout) + rank[None, :, None]
    return gl.reshape(gl.gather(expanded, index, 0), (8, BN))

@gluon.jit
def _down_combine(X, W, Scales, Ids, Weights, Shared, Y, H: gl.constexpr, I: gl.constexpr, BN: gl.constexpr):
    tile = gl.program_id(0)
    token = gl.program_id(1)
    parts = _routed_down_batch(X, W, Scales, Ids, token, tile, H, I, BN)
    layout: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[1, 0], [2, 0], [4, 0]] + ([[0, 16]] if BN == 32 else []), lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 0], [0, 0]], warp_bases=[[0, 0]] * 3, block_bases=[], shape=[8, BN])
    parts = gl.convert_layout(parts, layout)
    value = gl.full((BN,), 0.0, gl.float32, gl.SliceLayout(0, layout))
    for rank in gl.static_range(8):
        part = gl.sum(gl.amd.slice(parts, [1, BN], [rank, 0]), 0)
        value += part * gl.load(Weights + token * 9 + rank)
    n = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, layout))
    shared_value = gl.load(Shared + token * H + n)
    shared_value = shared_value.to(gl.bfloat16).to(gl.float32)
    value += shared_value * gl.load(Weights + token * 9 + 8)
    gl.store(Y + token * H + n, value)

@gluon.jit
def _combine(P, Weights, Y, H: gl.constexpr, BLOCK: gl.constexpr, VEC: gl.constexpr, WARPS: gl.constexpr):
    row = gl.program_id(0)
    h = gl.program_id(1) * BLOCK + gl.arange(0, BLOCK, gl.BlockedLayout([VEC], [64], [WARPS], [0]))
    value = gl.full((BLOCK,), 0.0, gl.float32, gl.BlockedLayout([VEC], [64], [WARPS], [0]))
    for j in gl.static_range(9):
        part = gl.load(P + (row * 9 + j) * H + h)
        if j == 8:
            part = part.to(gl.bfloat16).to(gl.float32)
        value += part * gl.load(Weights + row * 9 + j)
    gl.store(Y + row * H + h, value)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, routed_scaling_factor=2.5) -> torch.Tensor:
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    routes = m * 9
    grouped = m >= 2
    stagger = m <= 8
    splits = 24 if m == 1 else 8 if 3 <= m <= 8 else 12
    router_splits = 12
    router_block = 512
    quant_vector = 2 if 3 <= m <= 8 else 1
    quant_groups = 2 * quant_vector
    up_width = 16 if 3 <= m <= 8 else 32
    down_width = 32

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=x.device)
    logits = empty((m, router_splits, 256), torch.float32)
    ids = empty((routes,), torch.int32)
    groups = empty((257,), torch.uint64)
    weights = empty((routes,), torch.float32)
    xq = empty((2 * m, h // 2 + h // 32), torch.uint8)
    gu = empty((routes, splits, 2 * intermediate), torch.float32)
    aq = empty((routes, intermediate // 2 + intermediate // 32), torch.uint8)
    down_output = empty((m if stagger else routes, h), torch.float32)
    out = empty((m, h))
    _router_linear[16 * router_splits + m * (h // (32 * quant_groups)) + int(grouped),](x, router, logits, xq, groups, h, x.stride(0), m, router_splits, router_block, 16, grouped, quant_vector, quant_groups, num_warps=1)
    if stagger:
        _select_and_shared[m + splits * (2 * intermediate // up_width),](logits, correction_bias, ids, weights, groups, xq, w13, w13_scale, gu, m, h, intermediate, router_splits, splits, up_width, routed_scaling_factor, num_warps=1)
    else:
        _select_routes[m,](logits, correction_bias, ids, weights, groups, router_splits, grouped, routed_scaling_factor, num_warps=1)
    if stagger:
        up_jobs = m * (intermediate // 64) + m * 8 * splits * (2 * intermediate // up_width)
        _up_and_shared_activation[up_jobs,](xq, w13, w13_scale, ids, groups, gu, aq, h, intermediate, m, splits, up_width, num_warps=1, enable_fp_fusion=False)
        shared_width = 32 if m <= 2 else 16
        _activation_and_shared_down[h // shared_width + m * 8 * (intermediate // 64),](gu, aq, w2, w2_scale, ids, groups, down_output, h, intermediate, m, splits, num_warps=1, enable_fp_fusion=False)
        tail_width = 16 if m == 2 else 32
        _down_combine[h // tail_width, m](aq, w2, w2_scale, ids, weights, down_output, out, h, intermediate, tail_width, num_warps=8, enable_fp_fusion=False)
    else:
        jobs = m * 8 + 1
        activation_warps = 4
        up_grid = (splits, jobs, 2 * intermediate // up_width)
        _expert_projection[up_grid](xq, w13, w13_scale, ids, groups, gu, 2 * intermediate, h, m, True, splits, up_width, num_warps=1)
        _activate_quantize[routes, intermediate // (64 * activation_warps)](gu, aq, intermediate, splits, activation_warps, num_warps=activation_warps, enable_fp_fusion=False)
        down_grid = (jobs, h // down_width, 1)
        _expert_projection[down_grid](aq, w2, w2_scale, ids, groups, down_output, h, intermediate, m, False, 1, down_width, num_warps=1)
        _combine[m, h // 256](down_output, weights, out, h, 256, 1, 4, num_warps=4, enable_fp_fusion=False)
    return out
