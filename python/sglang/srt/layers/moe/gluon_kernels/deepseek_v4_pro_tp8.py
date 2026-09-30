# fmt: off
"""DeepSeek-V4 Pro TP8 fused MoE specialization for c=1 decode."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _unpack_weight_words(words, N: gl.constexpr, BK: gl.constexpr, layout: gl.constexpr):
    b0, b1 = (words.to(gl.uint8), (words >> 8).to(gl.uint8))
    b2, b3 = ((words >> 16).to(gl.uint8), (words >> 24).to(gl.uint8))
    packed = gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((N, BK // 2))
    return gl.convert_layout(packed, layout)

@gluon.jit
def _quantized_bytes(x, a, NATIVE: gl.constexpr=False):
    pack_layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [gl.num_warps(), 1], [1, 0])
    a = gl.where(a == a, a, 6.0)
    if NATIVE:
        a = gl.where(x < 0, -a, a)
    a = gl.convert_layout(a, pack_layout)
    lo, hi = gl.split(a.reshape((x.shape[0], 16, 2)))
    packed = gl.inline_asm_elementwise('v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0 op_sel:[0,0,0]', constraints='=v,v,v', args=[lo, hi], dtype=gl.uint32, is_pure=True, pack=1).to(gl.uint8)
    if NATIVE:
        return packed
    sign = gl.convert_layout(gl.where(x < 0, 8, 0).to(gl.uint8), pack_layout)
    sign_lo, sign_hi = gl.split(sign.reshape((x.shape[0], 16, 2)))
    return packed | sign_lo | sign_hi << 4

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
def _quantize_groups(x, shared, NATIVE: gl.constexpr=False):
    peak = gl.max(gl.abs(x), 1)
    divided = gl.div_rn(peak, 6.0)
    bits = divided.to(gl.uint32, bitcast=True)
    exponent = (bits >> 23 & 255).to(gl.int32) - 127 + (bits & 8388607 != 0)
    peak_bits = peak.to(gl.uint32, bitcast=True)
    floor_exponent = (peak_bits >> 23 & 255).to(gl.int32) - 127
    threshold = gl.exp2(floor_exponent.to(gl.float32)) * 1.75
    even_exponent = floor_exponent - 2 + (peak >= threshold).to(gl.int32)
    exponent = gl.where(shared, even_exponent, exponent)
    exponent = gl.maximum(-127, gl.minimum(127, exponent))
    scale = gl.exp2(exponent.to(gl.float32))
    a = gl.abs(x / scale[:, None])
    packed = _quantized_bytes(x, a, NATIVE)
    codes = gl.convert_layout((exponent + 127).to(gl.uint8), gl.SliceLayout(1, packed.type.layout))
    return (packed, codes)

@gluon.jit
def _router_linear(X, W, Y, Q, QS, Groups, H: gl.constexpr, SX: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr, GROUPED: gl.constexpr, QVEC: gl.constexpr, QGROUPS: gl.constexpr, NATIVE: gl.constexpr=False):
    program = gl.program_id(0)
    if program < 384 // BN * SPLITS:
        tile = program % (384 // BN)
        split = program // (384 // BN)
        mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
        al: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [0, 1])
        bl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
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
        gl.store(Y + (om[:, None] * SPLITS + split) * 512 + on[None, :], acc, om[:, None] < M)
    elif GROUPED and program == 384 // BN * SPLITS:
        e = gl.arange(0, 512, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
        shared_members = gl.full((512,), ((1 << M * 4) - 1) // 15 * 9, gl.uint64, e.type.layout)
        gl.store(Groups + e, 0, e < 384)
    else:
        quant = program - 384 // BN * SPLITS - (1 if GROUPED else 0)
        row = quant // (H // (32 * QGROUPS))
        tile_q = quant % (H // (32 * QGROUPS))
        layout: gl.constexpr = gl.BlockedLayout([1, QVEC], [2 * QVEC, 32 // QVEC], [1, 1], [1, 0])
        group = tile_q * QGROUPS + gl.arange(0, QGROUPS, gl.SliceLayout(1, layout))
        lane = gl.arange(0, 32, gl.SliceLayout(0, layout))
        k = group[:, None] * 32 + lane[None, :]
        values = gl.load(X + row * SX + k).to(gl.float32)
        routed, routed_scale = _quantize_groups(values, False, NATIVE)
        shared, shared_scale = _quantize_groups(values, True, NATIVE)
        qg = gl.convert_layout(group, gl.SliceLayout(1, routed.type.layout))
        qk = qg[:, None] * 16 + gl.arange(0, 16, gl.SliceLayout(0, routed.type.layout))[None, :]
        gl.store(Q + row * (H // 2) + qk, routed)
        gl.store(Q + (row + M) * (H // 2) + qk, shared)
        gl.store(QS + row * (H // 32) + qg, routed_scale)
        gl.store(QS + (row + M) * (H // 32) + qg, shared_scale)

@gluon.jit
def _routing_scores(Logits, Bias, row, SPLITS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
    e = gl.arange(0, 512, layout)
    logits = gl.load(Logits + row * SPLITS * 512 + e, e < 384, other=0.0)
    for part in gl.static_range(1, SPLITS):
        logits += gl.load(
            Logits + (row * SPLITS + part) * 512 + e,
            e < 384,
            other=0.0,
        )
    softplus = gl.maximum(logits, 0.0) + gl.log(1.0 + gl.exp(-gl.abs(logits)))
    probability = gl.sqrt(softplus)
    score = probability + gl.load(Bias + e, e < 384, other=-float('inf')).to(gl.float32)
    score = gl.where(e < 384, score, -float('inf'))
    return (e, probability, score)

@gluon.jit
def _next_expert(score, available, e):
    record_layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    maximum = gl.max(score, 0)
    key = gl.where(available, e + gl.where(score == maximum, 0, 512), 1024)
    local_key = gl.min(gl.reshape(key, (64, 8)), 1)
    local_key = gl.convert_layout(local_key, record_layout)
    valid, fallback = gl.inline_asm_elementwise('v_cmp_gt_u32_e64 $0, 1, $2\nv_cmp_gt_u32_e64 $1, 2, $2', constraints='=&s,=&s,v', args=[local_key >> 9], dtype=(gl.uint64, gl.uint64), is_pure=True, pack=1)
    mask = gl.where(valid != 0, valid, fallback)
    winning_lane = gl.inline_asm_elementwise('s_ff1_i32_b64 $0, $1', constraints='=s,s', args=[mask], dtype=gl.int32, is_pure=True, pack=1)
    elected = gl.inline_asm_elementwise('v_readlane_b32 $0, $1, $2', constraints='=s,v,s', args=[local_key, winning_lane], dtype=gl.int32, is_pure=True, pack=1)
    first = gl.full((1,), 0, gl.int32, record_layout)
    idx = gl.sum(gl.gather(elected, first, 0), 0) & 511
    return idx

@gluon.jit
def _select_routes(Logits, Bias, Ids, Weights, Groups, SPLITS: gl.constexpr, GROUPED: gl.constexpr, SCALE: gl.constexpr):
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
    record_layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    e, probability, score = _routing_scores(Logits, Bias, row, SPLITS)
    probability_table = gl.allocate_shared_memory(gl.float32, [512], gl.SwizzledSharedLayout(1, 1, 1, [0]), probability)
    record_size: gl.constexpr = 16 if GROUPED else 512
    r = gl.arange(0, record_size, record_layout)
    available = e < 384
    total = 0.0
    selected_ids = gl.full((record_size,), 0, gl.int32, record_layout)
    for j in gl.static_range(6):
        idx = _next_expert(score, available, e)
        selected_ids = gl.where(r == j, idx, selected_ids)
        available = available & (e != idx)
        score = gl.where(e == idx, -float('inf'), score)
    selected = probability_table.gather(gl.minimum(selected_ids, 383), 0)
    for j in gl.static_range(6):
        index = gl.full((1,), j, gl.int32, record_layout)
        total += gl.sum(gl.gather(selected, index, 0), 0)
    gl.store(Ids + row * 9 + r, selected_ids, r < 9)
    if GROUPED:
        membership = (r + 1).to(gl.uint64) << row * 4
        gl.amd.cdna4.buffer_atomic_or(Groups.to(gl.pointer_type(gl.int64)), selected_ids, membership.to(gl.int64, bitcast=True), r < 8, sem='relaxed')
    gl.store(
        Weights + row * 9 + r,
        gl.where(r < 6, selected / total * SCALE, 0.0),
        r < 9,
    )

@gluon.jit
def _expert_at_rank(Logits, Bias, row, rank, SPLITS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
    if rank >= 6:
        return gl.cast(0, gl.int32)
    e, _, score = _routing_scores(Logits, Bias, row, SPLITS)
    available = e < 384
    idx = _next_expert(score, available, e)
    for j in range(rank):
        available = available & (e != idx)
        score = gl.where(e == idx, -float('inf'), score)
        idx = _next_expert(score, available, e)
    return idx

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
            job = work // (SPLITS * (N // BN))
            tile = work // SPLITS % (N // BN)
            split = work % SPLITS
        else:
            job = gl.program_id(1) if UP else gl.program_id(0)
            tile = gl.program_id(2) if UP else gl.program_id(1)
            split = gl.program_id(0) if UP else gl.program_id(2)
        if STAGGER:
            route = job // 8 * 9 + job % 8
        else:
            route = gl.where(job < M * 8, job // 8 * 9 + job % 8, 8)
    return (route, tile, split)

@gluon.jit
def _expert_projection(X, XS, W, Scales, Ids, Groups, Y, N: gl.constexpr, K: gl.constexpr, M: gl.constexpr, UP: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, SHARED_ONLY: gl.constexpr=False, STAGGER: gl.constexpr=False, RECOMPUTE_ROUTE: gl.constexpr=False, Logits=None, Bias=None, ROUTER_SPLITS: gl.constexpr=12, BK: gl.constexpr=128, WEIGHT_CACHE: gl.constexpr='buffer'):
    PREFETCH: gl.constexpr = RECOMPUTE_ROUTE and M == 1
    if RECOMPUTE_ROUTE:
        work = gl.program_id(0) - M
        job = work // (SPLITS * (N // BN))
        if M == 1:
            route = gl.where(job < 8, 7 - job, 8)
        else:
            route = job % M * 9 + job // M
        tile = work // SPLITS % (N // BN)
        split = work % SPLITS
        if route % 9 == 8:
            expert = gl.cast(0, gl.uint32)
        else:
            expert = _expert_at_rank(Logits, Bias, route // 9, route % 9, ROUTER_SPLITS).to(gl.uint32)
    else:
        route, tile, split = _expert_coordinates(N, M, UP, SPLITS, BN, SHARED_ONLY, STAGGER)
        if SHARED_ONLY:
            expert = gl.cast(0, gl.uint32)
        else:
            expert = gl.load(Ids + route).to(gl.uint32)
    if not RECOMPUTE_ROUTE:
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
        data_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, 1], [0, 1])
        scale_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, 1], [0, 1])
        ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
        bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
        asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, BK // 32])
        bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
        mi = gl.arange(0, 16, gl.SliceLayout(1, data_layout))
        if not RECOMPUTE_ROUTE:
            rank = (members >> mi * 4 & 15).to(gl.int32) - 1
            destination = gl.where(rank >= 0, mi * 9 + rank, -1)
            source = gl.maximum(destination, route).to(gl.uint32)
        else:
            destination = gl.full((16,), route, gl.int32, gl.SliceLayout(1, data_layout))
            source = destination.to(gl.uint32)
        row = source // 9 + gl.where(route % 9 == 8, M, 0) if UP else source
        k = gl.arange(0, BK // 2, gl.SliceLayout(0, data_layout)).to(gl.uint32)
        word_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 1], [0, 1])
        wn = (tile * BN + gl.arange(0, BN, gl.SliceLayout(1, word_layout))).to(gl.uint32)
        wrow = gl.where(wn < N // 2, 2 * wn, 2 * (wn - N // 2) + 1) if UP else wn
        wk = gl.arange(0, BK // 8, gl.SliceLayout(0, word_layout)).to(gl.uint32) * 8
        sr = gl.convert_layout(row, gl.SliceLayout(1, scale_layout))
        sn = (tile * BN + gl.arange(0, BN, gl.SliceLayout(1, scale_layout))).to(gl.uint32)
        srow = gl.where(sn < N // 2, 2 * sn, 2 * (sn - N // 2) + 1) if UP else sn
        sg = gl.arange(0, BK // 32, gl.SliceLayout(0, scale_layout)).to(gl.uint32)
        weight_base = W.to(gl.pointer_type(gl.uint32)) + expert.to(gl.int64) * (N * K // 8)
        scale_base = Scales + expert * (N * gl.cdiv(K // 32, 8) * 8)
        if PREFETCH:
            gl.static_assert(BK == 256 and RECOMPUTE_ROUTE)
            first_start = (split * (K // SPLITS)).to(gl.uint32)
            first_offset = _weight_offset(0, wrow[:, None], first_start + wk[None, :], N, K) // 4
            first_offset = gl.max_contiguous(gl.multiple_of(first_offset, [1, 4]), [1, 4])
            if WEIGHT_CACHE == 'buffer':
                next_words = gl.amd.cdna4.buffer_load(weight_base, first_offset)
            else:
                next_words = gl.load(weight_base + first_offset, cache_modifier=WEIGHT_CACHE)
            next_scales = gl.amd.cdna4.buffer_load(scale_base.to(gl.pointer_type(gl.uint32)), _scale_offset(0, srow[:, None], first_start + 32 * sg[None, :], N, K) // 4)
        acc = gl.zeros((16, BN), gl.float32, mma)
        for base in gl.static_range(K // SPLITS // BK):
            start = (split * (K // SPLITS) + base * BK).to(gl.uint32)
            if PREFETCH:
                words = next_words
                scale_words = next_scales
                if base + 1 < K // SPLITS // BK:
                    next_offset = _weight_offset(0, wrow[:, None], start + BK + wk[None, :], N, K) // 4
                    next_offset = gl.max_contiguous(gl.multiple_of(next_offset, [1, 4]), [1, 4])
                    if WEIGHT_CACHE == 'buffer':
                        next_words = gl.amd.cdna4.buffer_load(weight_base, next_offset)
                    else:
                        next_words = gl.load(weight_base + next_offset, cache_modifier=WEIGHT_CACHE)
                    next_scales = gl.amd.cdna4.buffer_load(scale_base.to(gl.pointer_type(gl.uint32)), _scale_offset(0, srow[:, None], start + BK + 32 * sg[None, :], N, K) // 4)
            a = gl.load(X + row[:, None] * (K // 2) + start // 2 + k[None, :])
            if not PREFETCH:
                offsets = _weight_offset(0, wrow[:, None], start + wk[None, :], N, K) // 4
                offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
                if WEIGHT_CACHE == 'buffer':
                    words = gl.amd.cdna4.buffer_load(weight_base, offsets)
                else:
                    words = gl.load(weight_base + offsets, cache_modifier=WEIGHT_CACHE)
            b = _unpack_weight_words(words, BN, BK, data_layout)
            if RECOMPUTE_ROUTE:
                input_scale_word = gl.load(XS.to(gl.pointer_type(gl.uint32)) + sr[:, None] * (K // 128) + start // 128 + sg[None, :] // 4)
                ax = (input_scale_word >> sg[None, :] % 4 * 8).to(gl.uint8)
            else:
                ax = gl.load(XS + sr[:, None] * (K // 32) + start // 32 + sg[None, :])
            if RECOMPUTE_ROUTE:
                if not PREFETCH:
                    scale_words = gl.amd.cdna4.buffer_load(scale_base.to(gl.pointer_type(gl.uint32)), _scale_offset(0, srow[:, None], start + 32 * sg[None, :], N, K) // 4)
                scale_shift = srow[:, None] // 16 % 2 * 8 + sg[None, :] // 4 % 2 * 16
                bx = (scale_words >> scale_shift).to(gl.uint8)
            else:
                bx = gl.amd.cdna4.buffer_load(scale_base, _scale_offset(0, srow[:, None], start + 32 * sg[None, :], N, K))
            acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), gl.convert_layout(ax, asl), 'e2m1', gl.convert_layout(b.T, bd), gl.convert_layout(bx, bsl), 'e2m1', acc)
        om = gl.arange(0, 16, gl.SliceLayout(1, mma))
        on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
        dest = gl.convert_layout(destination, gl.SliceLayout(1, mma))
        valid = dest >= 0 if not RECOMPUTE_ROUTE else om == 0
        output_row = dest // 9 if SHARED_ONLY and (not UP) else dest
        gl.store(Y + (output_row[:, None] * SPLITS + split) * N + on[None, :], acc, valid[:, None])

@gluon.jit
def _select_and_shared(Logits, Bias, Ids, Weights, Groups, X, XS, W, Scales, GU, M: gl.constexpr, H: gl.constexpr, I: gl.constexpr, ROUTER_SPLITS: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, SCALE: gl.constexpr):
    if gl.program_id(0) < M:
        _select_routes(Logits, Bias, Ids, Weights, Groups, ROUTER_SPLITS, True, SCALE)
    else:
        _expert_projection(X, XS, W, Scales, Ids, Groups, GU, 2 * I, H, M, True, SPLITS, BN, SHARED_ONLY=True)

@gluon.jit
def _select_and_up(Logits, Bias, Ids, Weights, Groups, X, XS, W, Scales, GU, M: gl.constexpr, H: gl.constexpr, I: gl.constexpr, ROUTER_SPLITS: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, SCALE: gl.constexpr, BK: gl.constexpr=128, WEIGHT_CACHE: gl.constexpr='buffer'):
    if gl.program_id(0) < M:
        _select_routes(Logits, Bias, Ids, Weights, Groups, ROUTER_SPLITS, False, SCALE)
    else:
        _expert_projection(X, XS, W, Scales, Ids, Groups, GU, 2 * I, H, M, True, SPLITS, BN, RECOMPUTE_ROUTE=True, Logits=Logits, Bias=Bias, ROUTER_SPLITS=ROUTER_SPLITS, BK=BK, WEIGHT_CACHE=WEIGHT_CACHE)

@gluon.jit
def _activation_tile(GU, Q, QS, route, tile, I: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr, VEC: gl.constexpr=2, NATIVE: gl.constexpr=False, LIMIT: gl.constexpr=10.0):
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [2 * VEC, 32 // VEC], [WARPS, 1], [1, 0])
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
    gate = gl.minimum(gate, LIMIT)
    up = gl.maximum(-LIMIT, gl.minimum(up, LIMIT))
    a = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
    q, scale = _quantize_groups(a, shared, NATIVE)
    qg = gl.convert_layout(group, gl.SliceLayout(1, q.type.layout))
    qk = qg[:, None] * 16 + gl.arange(0, 16, gl.SliceLayout(0, q.type.layout))[None, :]
    gl.store(Q + route * (I // 2) + qk, q)
    gl.store(QS + route * (I // 32) + qg, scale)

@gluon.jit
def _activate_quantize(GU, Q, QS, I: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr, VEC: gl.constexpr=2, NATIVE: gl.constexpr=False, LIMIT: gl.constexpr=10.0):
    _activation_tile(GU, Q, QS, gl.program_id(0), gl.program_id(1), I, SPLITS, WARPS, VEC, NATIVE, LIMIT)

@gluon.jit
def _up_and_shared_activation(X, XS, W, Scales, Ids, Groups, GU, AQ, AQS, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr):
    program = gl.program_id(0)
    if program < M * (I // 64):
        route = program // (I // 64) * 9 + 8
        tile = program % (I // 64)
        _activation_tile(GU, AQ, AQS, route, tile, I, SPLITS, 1)
    else:
        _expert_projection(X, XS, W, Scales, Ids, Groups, GU, 2 * I, H, M, True, SPLITS, BN, STAGGER=True)

@gluon.jit
def _activation_and_shared_down(GU, AQ, AQS, W, Scales, Ids, Groups, Shared, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr):
    BN: gl.constexpr = 16
    program = gl.program_id(0)
    if program < H // BN:
        _expert_projection(AQ, AQS, W, Scales, Ids, Groups, Shared, H, I, M, False, 1, BN, SHARED_ONLY=True, STAGGER=True)
    else:
        work = program - H // BN
        job = work // (I // 64)
        route = job // 8 * 9 + job % 8
        tile = work % (I // 64)
        _activation_tile(GU, AQ, AQS, route, tile, I, SPLITS, 1)

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

@gluon.jit
def _fused_down_combine(X, XS, W, Scales, Ids, Weights, Shared, Y, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr):
    token = gl.program_id(0)
    tile = gl.program_id(1)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, WARPS])
    data_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [WARPS, 1], [0, 1])
    scale_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [WARPS, 1], [0, 1])
    word_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WARPS, 1], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, 8])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, 8])
    k = gl.arange(0, 128, gl.SliceLayout(0, data_layout)).to(gl.uint32)
    wn = (tile * BN + gl.arange(0, BN, gl.SliceLayout(1, word_layout))).to(gl.uint32)
    wk = gl.arange(0, 32, gl.SliceLayout(0, word_layout)).to(gl.uint32) * 8
    sn = (tile * BN + gl.arange(0, BN, gl.SliceLayout(1, scale_layout))).to(gl.uint32)
    sg = gl.arange(0, 8, gl.SliceLayout(0, scale_layout)).to(gl.uint32)
    result = gl.full((16, BN), 0.0, gl.float32, mma)
    for j in range(8):
        route = token * 9 + j
        expert = gl.load(Ids + route).to(gl.uint32)
        row = gl.full((16,), route, gl.int32, gl.SliceLayout(1, data_layout))
        sr = gl.convert_layout(row, gl.SliceLayout(1, scale_layout))
        weight_base = W.to(gl.pointer_type(gl.uint32)) + expert.to(gl.int64) * (N * K // 8)
        scale_base = Scales + expert * (N * gl.cdiv(K // 32, 8) * 8)
        acc = gl.zeros((16, BN), gl.float32, mma)
        for base in gl.static_range(K // 256):
            start = base * 256
            a = gl.load(X + row[:, None] * (K // 2) + start // 2 + k[None, :])
            offsets = _weight_offset(0, wn[:, None], start + wk[None, :], N, K) // 4
            offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
            words = gl.amd.cdna4.buffer_load(weight_base, offsets)
            b = _unpack_weight_words(words, BN, 256, data_layout)
            ax = gl.load(XS + sr[:, None] * (K // 32) + start // 32 + sg[None, :])
            scale_words = gl.amd.cdna4.buffer_load(scale_base.to(gl.pointer_type(gl.uint32)), _scale_offset(0, sn[:, None], start + 32 * sg[None, :], N, K) // 4)
            scale_shift = sn[:, None] // 16 % 2 * 8 + sg[None, :] // 4 % 2 * 16
            bx = (scale_words >> scale_shift).to(gl.uint8)
            acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), gl.convert_layout(ax, asl), 'e2m1', gl.convert_layout(b.T, bd), gl.convert_layout(bx, bsl), 'e2m1', acc)
        result += acc * gl.load(Weights + route)
    om = gl.arange(0, 16, gl.SliceLayout(1, mma))
    on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    shared = gl.load(Shared + token * N + on).to(gl.bfloat16).to(gl.float32)
    result += shared[None, :] * gl.load(Weights + token * 9 + 8)
    gl.store(Y + token * N + om[:, None] * 0 + on[None, :], result, om[:, None] == 0)

@gluon.jit
def _down_accumulator(X, XS, W, Scales, Ids, token, tile, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, BANDS: gl.constexpr, STATIC_SHARED: gl.constexpr=False):
    CN: gl.constexpr = BANDS * BN
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, WARPS])
    data_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [WARPS, 1], [0, 1])
    word_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WARPS, 1], [0, 1])
    scale_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [WARPS, 1], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, 4])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [CN, 4])
    if BANDS == 1:
        ai = gl.arange(0, 16, gl.SliceLayout(1, data_layout))
        ar = (token + ai % 2) * 9 + 8
    elif BANDS == 16:
        ar = token * 9 + gl.minimum(gl.arange(0, 16, gl.SliceLayout(1, data_layout)), 8)
    else:
        ar = token * 9 + gl.arange(0, 16, gl.SliceLayout(1, data_layout)) % 8
    k = gl.arange(0, 64, gl.SliceLayout(0, data_layout)).to(gl.uint32)
    wn = gl.arange(0, CN, gl.SliceLayout(1, word_layout)).to(gl.uint32)
    if BANDS == 1:
        expert = gl.full((CN,), 0, gl.uint32, gl.SliceLayout(1, word_layout))
    elif BANDS == 16:
        expert = gl.load(Ids + token * 9 + gl.minimum(wn // BN, 8), mask=wn // BN < 8 if STATIC_SHARED else True, other=0).to(gl.uint32)
    else:
        expert = gl.load(Ids + token * 9 + wn // BN).to(gl.uint32)
    wk = gl.arange(0, 16, gl.SliceLayout(0, word_layout)).to(gl.uint32) * 8
    n = tile * BN + wn % BN
    if BANDS == 1:
        sr = gl.convert_layout(ar, gl.SliceLayout(1, scale_layout))
    elif BANDS == 16:
        sr = token * 9 + gl.minimum(gl.arange(0, 16, gl.SliceLayout(1, scale_layout)), 8)
    else:
        sr = token * 9 + gl.arange(0, 16, gl.SliceLayout(1, scale_layout)) % 8
    sn = gl.arange(0, CN, gl.SliceLayout(1, scale_layout)).to(gl.uint32)
    if BANDS == 1:
        se = gl.full((CN,), 0, gl.uint32, gl.SliceLayout(1, scale_layout))
    elif BANDS == 16:
        se = gl.load(Ids + token * 9 + gl.minimum(sn // BN, 8), mask=sn // BN < 8 if STATIC_SHARED else True, other=0).to(gl.uint32)
    else:
        se = gl.load(Ids + token * 9 + sn // BN).to(gl.uint32)
    sg = gl.arange(0, 4, gl.SliceLayout(0, scale_layout)).to(gl.uint32)
    scale_n = tile * BN + sn % BN
    acc = gl.zeros((16, CN), gl.float32, mma)
    for panel in gl.static_range(K // 128):
        start = panel * 128
        offsets = _weight_offset(expert[:, None], n[:, None], start + wk[None, :], N, K) // 4
        offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
        words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offsets)
        b = _unpack_weight_words(words, CN, 128, data_layout)
        a = gl.load(X + ar[:, None] * (K // 2) + start // 2 + k[None, :])
        ax = gl.load(XS + sr[:, None] * (K // 32) + start // 32 + sg[None, :])
        sw = gl.amd.cdna4.buffer_load(Scales.to(gl.pointer_type(gl.uint32)), _scale_offset(se[:, None], scale_n[:, None], start + 32 * sg[None, :], N, K) // 4)
        shift = scale_n[:, None] // 16 % 2 * 8 + panel % 2 * 16
        bx = (sw >> shift).to(gl.uint8)
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), gl.convert_layout(ax, asl), 'e2m1', gl.convert_layout(b.T, bd), gl.convert_layout(bx, bsl), 'e2m1', acc)
    return acc

@gluon.jit
def _route_down(X, XS, W, Scales, Ids, P, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr):
    token = gl.program_id(0)
    rank = gl.program_id(1)
    tile = gl.program_id(2)
    route = token * 9 + rank
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, 1])
    data_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, 1], [0, 1])
    word_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 1], [0, 1])
    scale_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, 1], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, 4])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, 4])
    ar = gl.full((16,), route, gl.uint32, gl.SliceLayout(1, data_layout))
    expert = gl.load(Ids + route).to(gl.uint32)
    k = gl.arange(0, 64, gl.SliceLayout(0, data_layout)).to(gl.uint32)
    wn = gl.arange(0, BN, gl.SliceLayout(1, word_layout)).to(gl.uint32)
    wk = gl.arange(0, 16, gl.SliceLayout(0, word_layout)).to(gl.uint32) * 8
    n = tile * BN + wn
    sr = gl.full((16,), route, gl.uint32, gl.SliceLayout(1, scale_layout))
    sn = gl.arange(0, BN, gl.SliceLayout(1, scale_layout)).to(gl.uint32)
    sg = gl.arange(0, 4, gl.SliceLayout(0, scale_layout)).to(gl.uint32)
    scale_n = tile * BN + sn
    acc = gl.zeros((16, BN), gl.float32, mma)
    for panel in gl.static_range(K // 128):
        start = panel * 128
        offsets = _weight_offset(expert, n[:, None], start + wk[None, :], N, K) // 4
        offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
        words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offsets)
        b = _unpack_weight_words(words, BN, 128, data_layout)
        a = gl.load(X + ar[:, None] * (K // 2) + start // 2 + k[None, :])
        ax = gl.load(XS + sr[:, None] * (K // 32) + start // 32 + sg[None, :])
        sw = gl.amd.cdna4.buffer_load(Scales.to(gl.pointer_type(gl.uint32)), _scale_offset(expert, scale_n[:, None], start + 32 * sg[None, :], N, K) // 4)
        shift = scale_n[:, None] // 16 % 2 * 8 + panel % 2 * 16
        bx = (sw >> shift).to(gl.uint8)
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), gl.convert_layout(ax, asl), 'e2m1', gl.convert_layout(b.T, bd), gl.convert_layout(bx, bsl), 'e2m1', acc)
    om = gl.arange(0, 16, gl.SliceLayout(1, mma))
    on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(P + route * N + om[:, None] * 0 + on[None, :], acc, om[:, None] == 0)

@gluon.jit
def _diagonal_down(X, XS, W, Scales, Ids, Weights, Shared, Y, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr):
    token = gl.program_id(0)
    tile = gl.program_id(1)
    CN: gl.constexpr = 8 * BN
    acc = _down_accumulator(X, XS, W, Scales, Ids, token, tile, N, K, BN, WARPS, 8)
    gather_layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, WARPS], [1, 0])
    acc = gl.convert_layout(acc, gather_layout)
    ci = gl.arange(0, CN, gl.SliceLayout(0, gather_layout))
    parts = gl.gather(acc, (ci // BN)[None, :], 0).reshape((8, BN))
    sum_layout: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, WARPS], [1, 0])
    parts = gl.convert_layout(parts, sum_layout)
    result = gl.full((1, BN), 0.0, gl.float32, sum_layout)
    for j in gl.static_range(8):
        part = gl.gather(parts, gl.full((1, BN), j, gl.int32, sum_layout), 0)
        result += part * gl.load(Weights + token * 9 + j)
    on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, sum_layout))
    shared = gl.load(Shared + token * N + on)[None, :].to(gl.bfloat16).to(gl.float32)
    result += shared * gl.load(Weights + token * 9 + 8)
    gl.store(Y + token * N + on[None, :], result)

@gluon.jit
def _padded_down(X, XS, W, Scales, Ids, Weights, Y, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr):
    token = gl.program_id(0)
    tile = gl.program_id(1)
    CN: gl.constexpr = 16 * BN
    acc = _down_accumulator(X, XS, W, Scales, Ids, token, tile, N, K, BN, WARPS, 16)
    gather_layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, WARPS], [1, 0])
    acc = gl.convert_layout(acc, gather_layout)
    ci = gl.arange(0, CN, gl.SliceLayout(0, gather_layout))
    parts = gl.gather(acc, (ci // BN)[None, :], 0).reshape((16, BN))
    sum_layout: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, WARPS], [1, 0])
    parts = gl.convert_layout(parts, sum_layout)
    result = gl.full((1, BN), 0.0, gl.float32, sum_layout)
    for j in gl.static_range(9):
        part = gl.amd.slice(parts, [1, BN], [j, 0])
        if j == 8:
            part = part.to(gl.bfloat16).to(gl.float32)
        result += part * gl.load(Weights + token * 9 + j)
    on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, sum_layout))
    gl.store(Y + token * N + on[None, :], result)

@gluon.jit
def _token_local_down(X, XS, W, Scales, Ids, Weights, Y, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, TILE_GROUP: gl.constexpr=96):
    if TILE_GROUP == 96:
        token = gl.program_id(1)
        tile = gl.program_id(0)
    else:
        token = gl.program_id(0) // TILE_GROUP
        tile = gl.program_id(1) * TILE_GROUP + gl.program_id(0) % TILE_GROUP
    acc = _down_accumulator(X, XS, W, Scales, Ids, token, tile, N, K, BN, WARPS, 16, STATIC_SHARED=True)
    layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, WARPS], [1, 0])
    result = gl.full((1, BN), 0.0, gl.float32, layout)
    for j in gl.static_range(8):
        band = gl.amd.slice(acc, [16, BN], [0, j * BN])
        band = gl.convert_layout(band, layout)
        part = gl.gather(band, gl.full((1, BN), j, gl.int32, layout), 0)
        result += part * gl.load(Weights + token * 9 + j)
    band = gl.amd.slice(acc, [16, BN], [0, 8 * BN])
    band = gl.convert_layout(band, layout)
    shared = gl.gather(band, gl.full((1, BN), 8, gl.int32, layout), 0)
    shared = shared.to(gl.bfloat16).to(gl.float32)
    result += shared * gl.load(Weights + token * 9 + 8)
    n = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, layout))
    gl.store(Y + token * N + n[None, :], result)

@gluon.jit
def _fused_up_body(X, XS, W, Scales, AQ, AQS, Logits, Bias, token, rank, tile, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, SHARED: gl.constexpr, ROUTER_SPLITS: gl.constexpr, NATIVE_QUANT: gl.constexpr, LIMIT: gl.constexpr=10.0):
    BK: gl.constexpr = 128
    WINDOW: gl.constexpr = 2
    CN: gl.constexpr = 4 * BN
    if SHARED:
        expert = gl.cast(0, gl.uint32)
    else:
        expert = _expert_at_rank(Logits, Bias, token, rank, ROUTER_SPLITS).to(gl.uint32)
    route = token * 9 + rank
    row = token + (M if SHARED else 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, WARPS])
    dl: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [WARPS, 1], [0, 1])
    wl: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WARPS, 1], [0, 1])
    sl: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [WARPS, 1], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [CN, BK // 32])
    am = gl.arange(0, 16, gl.SliceLayout(1, dl)).to(gl.uint32)
    ak = gl.arange(0, BK // 2, gl.SliceLayout(0, dl)).to(gl.uint32)
    wn = gl.arange(0, CN, gl.SliceLayout(1, wl)).to(gl.uint32)
    wk = gl.arange(0, BK // 8, gl.SliceLayout(0, wl)).to(gl.uint32) * 8
    n = tile * BN + wn % BN
    split = wn // BN
    sm = gl.arange(0, 16, gl.SliceLayout(1, sl)).to(gl.uint32)
    sn = gl.arange(0, CN, gl.SliceLayout(1, sl)).to(gl.uint32)
    sg = gl.arange(0, BK // 32, gl.SliceLayout(0, sl)).to(gl.uint32)
    scale_n = tile * BN + sn % BN
    scale_split = sn // BN
    wb = W.to(gl.pointer_type(gl.uint32)) + expert.to(gl.int64) * (2 * I * H // 8)
    sb = Scales.to(gl.pointer_type(gl.uint32)) + expert * (2 * I * (H // 32) // 4)
    gacc = gl.zeros((16, CN), gl.float32, mma)
    uacc = gl.zeros((16, CN), gl.float32, mma)
    gl.static_assert(H % (4 * BK * WINDOW) == 0)
    for panel in range(H // 4 // BK // WINDOW):
        for step in gl.static_range(WINDOW):
            base = panel * WINDOW + step
            start = gl.multiple_of(base * BK, BK)
            a = gl.load(X + row * (H // 2) + am[:, None] % 4 * (H // 8) + start // 2 + ak[None, :])
            offsets = _weight_offset(0, n[:, None], split[:, None] * (H // 4) + start + wk[None, :], 2 * I, H) // 4
            offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
            words = gl.amd.cdna4.buffer_load(wb, offsets)
            uwords = gl.amd.cdna4.buffer_load(wb + I * H // 8, offsets)
            b = _unpack_weight_words(words, CN, BK, dl)
            ub = _unpack_weight_words(uwords, CN, BK, dl)
            aw = gl.load(XS.to(gl.pointer_type(gl.uint32)) + row * (H // 128) + sm % 4 * (H // 512) + start // 128)
            ax = (aw[:, None] >> sg[None, :] * 8).to(gl.uint8)
            if step % 2 == 0:
                sw = gl.amd.cdna4.buffer_load(sb, _scale_offset(0, scale_n[:, None], scale_split[:, None] * (H // 4) + start + sg[None, :] * 32, 2 * I, H) // 4)
                usw = gl.amd.cdna4.buffer_load(sb + I * (H // 32) // 4, _scale_offset(0, scale_n[:, None], scale_split[:, None] * (H // 4) + start + sg[None, :] * 32, 2 * I, H) // 4)
            shift = scale_n[:, None] // 16 % 2 * 8 + base % 2 * 16
            bx = (sw >> shift).to(gl.uint8)
            ux = (usw >> shift).to(gl.uint8)
            gacc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), gl.convert_layout(ax, asl), 'e2m1', gl.convert_layout(b.T, bd), gl.convert_layout(bx, bsl), 'e2m1', gacc)
            uacc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), gl.convert_layout(ax, asl), 'e2m1', gl.convert_layout(ub.T, bd), gl.convert_layout(ux, bsl), 'e2m1', uacc)
    gather_layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, WARPS], [1, 0])
    gate = gl.full((1, BN), 0.0, gl.float32, gather_layout)
    up = gl.full((1, BN), 0.0, gl.float32, gather_layout)
    for part in gl.static_range(4):
        gband = gl.amd.slice(gacc, [16, BN], [0, part * BN])
        uband = gl.amd.slice(uacc, [16, BN], [0, part * BN])
        gband = gl.convert_layout(gband, gather_layout)
        uband = gl.convert_layout(uband, gather_layout)
        idx = gl.full((1, BN), part, gl.int32, gather_layout)
        gpart = gl.gather(gband, idx, 0)
        upart = gl.gather(uband, idx, 0)
        if part == 0:
            gate = gpart
            up = upart
        else:
            gate += gpart
            up += upart
    if SHARED:
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
    gate = gl.minimum(gate, LIMIT)
    up = gl.maximum(-LIMIT, gl.minimum(up, LIMIT))
    activated = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
    qlayout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [WARPS, 1], [1, 0])
    activated = gl.convert_layout(activated.reshape((BN // 32, 32)), qlayout)
    q, scale = _quantize_groups(activated, SHARED, NATIVE=NATIVE_QUANT)
    g = tile * (BN // 32) + gl.arange(0, BN // 32, gl.SliceLayout(1, q.type.layout))
    k = g[:, None] * 16 + gl.arange(0, 16, gl.SliceLayout(0, q.type.layout))[None, :]
    gl.store(AQ + route * (I // 2) + k, q)
    gl.store(AQS + route * (I // 32) + g, scale)

@gluon.jit
def _select_fused_up(Logits, Bias, Ids, Weights, X, XS, W, Scales, AQ, AQS, M: gl.constexpr, H: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, ROUTER_SPLITS: gl.constexpr, NATIVE_QUANT: gl.constexpr, PAIR_ORDER: gl.constexpr, SCALE: gl.constexpr, LIMIT: gl.constexpr=10.0):
    pid = gl.program_id(0)
    if pid < M:
        _select_routes(Logits, Bias, Ids, Weights, None, ROUTER_SPLITS, False, SCALE)
    else:
        work = pid - M
        tile = work % (I // BN)
        job = work // (I // BN)
        token = job % M
        rank = job // M
        if PAIR_ORDER:
            rank = gl.where(rank < 8, gl.where(rank % 2 == 0, rank // 2, 7 - rank // 2), 8)
        if rank == 8:
            _fused_up_body(X, XS, W, Scales, AQ, AQS, Logits, Bias, token, 8, tile, H, I, M, BN, WARPS, True, ROUTER_SPLITS, NATIVE_QUANT, LIMIT)
        else:
            _fused_up_body(X, XS, W, Scales, AQ, AQS, Logits, Bias, token, rank, tile, H, I, M, BN, WARPS, False, ROUTER_SPLITS, NATIVE_QUANT, LIMIT)

@gluon.jit
def _static_single_down(X, XS, W, Scales, Ids, Weights, Y, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr):
    token = 0
    tile = gl.program_id(1)
    acc = _down_accumulator(X, XS, W, Scales, Ids, token, tile, N, K, BN, WARPS, 16, STATIC_SHARED=True)
    layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, WARPS], [1, 0])
    result = gl.full((1, BN), 0.0, gl.float32, layout)
    for j in gl.static_range(9):
        band = gl.amd.slice(acc, [16, BN], [0, j * BN])
        band = gl.convert_layout(band, layout)
        part = gl.gather(band, gl.full((1, BN), j, gl.int32, layout), 0)
        if j == 8:
            part = part.to(gl.bfloat16).to(gl.float32)
            result += part * gl.load(Weights + token * 9 + j)
        else:
            result += part * gl.load(Weights + token * 9 + j)
    n = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, layout))
    gl.store(Y + token * N + n[None, :], result)

@gluon.jit
def _paired_panel_accumulator(X, XS, W, Scales, Ids, token, tile, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, RANK_BASE: gl.constexpr):
    CN: gl.constexpr = 8 * BN
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, WARPS])
    data_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [WARPS, 1], [0, 1])
    word_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WARPS, 1], [0, 1])
    scale_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [WARPS, 1], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, K // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [CN, K // 32])
    ai = gl.arange(0, 16, gl.SliceLayout(1, data_layout))
    ar = (token + ai // 4 % 2) * 9 + RANK_BASE + ai % 4
    k = gl.arange(0, K // 2, gl.SliceLayout(0, data_layout)).to(gl.uint32)
    wn = gl.arange(0, CN, gl.SliceLayout(1, word_layout)).to(gl.uint32)
    expert = gl.load(Ids + (token + wn // (4 * BN)) * 9 + RANK_BASE + wn // BN % 4).to(gl.uint32)
    wk = gl.arange(0, K // 8, gl.SliceLayout(0, word_layout)).to(gl.uint32) * 8
    n = tile * BN + wn % BN
    offsets = _weight_offset(expert[:, None], n[:, None], wk[None, :], N, K) // 4
    offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offsets)
    b = _unpack_weight_words(words, CN, K, data_layout)
    a = gl.load(X + ar[:, None] * (K // 2) + k[None, :])
    sr = gl.convert_layout(ar, gl.SliceLayout(1, scale_layout))
    sn = gl.arange(0, CN, gl.SliceLayout(1, scale_layout)).to(gl.uint32)
    se = gl.load(Ids + (token + sn // (4 * BN)) * 9 + RANK_BASE + sn // BN % 4).to(gl.uint32)
    sg = gl.arange(0, K // 32, gl.SliceLayout(0, scale_layout)).to(gl.uint32)
    ax = gl.load(XS + sr[:, None] * (K // 32) + sg[None, :])
    scale_n = tile * BN + sn % BN
    sw = gl.amd.cdna4.buffer_load(Scales.to(gl.pointer_type(gl.uint32)), _scale_offset(se[:, None], scale_n[:, None], 32 * sg[None, :], N, K) // 4)
    shift = scale_n[:, None] // 16 % 2 * 8 + sg[None, :] // 4 % 2 * 16
    bx = (sw >> shift).to(gl.uint8)
    acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), gl.convert_layout(ax, asl), 'e2m1', gl.convert_layout(b.T, bd), gl.convert_layout(bx, bsl), 'e2m1', gl.zeros((16, CN), gl.float32, mma))
    return acc

@gluon.jit
def _stream_paired_down(X, XS, W, Scales, Ids, Weights, Y, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, TILE_GROUP: gl.constexpr=96):
    if TILE_GROUP == 96:
        tile = gl.program_id(0)
        token = gl.program_id(1) * 2
    else:
        tile = gl.program_id(1) * TILE_GROUP + gl.program_id(0) % TILE_GROUP
        token = gl.program_id(0) // TILE_GROUP * 2
    layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, WARPS], [1, 0])
    result0 = gl.full((1, BN), 0.0, gl.float32, layout)
    result1 = gl.full((1, BN), 0.0, gl.float32, layout)
    shared_acc = _down_accumulator(X, XS, W, Scales, Ids, token, tile, N, K, BN, WARPS, 1)
    shared_acc = gl.convert_layout(shared_acc, layout)
    shared0 = gl.gather(shared_acc, gl.full((1, BN), 0, gl.int32, layout), 0).to(gl.bfloat16).to(gl.float32)
    shared1 = gl.gather(shared_acc, gl.full((1, BN), 1, gl.int32, layout), 0).to(gl.bfloat16).to(gl.float32)
    for panel in gl.static_range(2):
        acc = _paired_panel_accumulator(X, XS, W, Scales, Ids, token, tile, N, K, BN, WARPS, panel * 4)
        for j in gl.static_range(4):
            band0 = gl.amd.slice(acc, [16, BN], [0, j * BN])
            band1 = gl.amd.slice(acc, [16, BN], [0, (4 + j) * BN])
            band0 = gl.convert_layout(band0, layout)
            band1 = gl.convert_layout(band1, layout)
            part0 = gl.gather(band0, gl.full((1, BN), j, gl.int32, layout), 0)
            part1 = gl.gather(band1, gl.full((1, BN), 4 + j, gl.int32, layout), 0)
            result0 += part0 * gl.load(Weights + token * 9 + panel * 4 + j)
            result1 += part1 * gl.load(Weights + (token + 1) * 9 + panel * 4 + j)
    result0 += shared0 * gl.load(Weights + token * 9 + 8)
    result1 += shared1 * gl.load(Weights + (token + 1) * 9 + 8)
    n = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, layout))
    gl.store(Y + token * N + n[None, :], result0)
    gl.store(Y + (token + 1) * N + n[None, :], result1)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, routed_scaling_factor=2.5, swiglu_limit=10.0) -> torch.Tensor:
    m, h = x.shape
    if m not in (1, 4, 6):
        raise RuntimeError(f"DeepSeek-V4 Pro Gluon MoE does not support M={m}")
    if h != 7168 or router.shape != (384, 7168):
        raise RuntimeError("DeepSeek-V4 Pro Gluon MoE requires router [384, 7168]")
    if w13.shape != (384, 768, 3584) or w2.shape != (384, 7168, 192):
        raise RuntimeError("DeepSeek-V4 Pro Gluon MoE received incompatible FP4 weights")
    if swiglu_limit != 10.0:
        raise RuntimeError("DeepSeek-V4 Pro Gluon MoE requires swiglu_limit=10")
    intermediate = w13.shape[1] // 2
    routes = m * 9
    direct = m in (1, 4, 6)
    grouped = not direct
    stagger = m <= 8
    splits = 4 if direct else 12
    router_splits = 14
    router_block = 512
    quant_vector = 2
    quant_groups = 4 if m == 16 else 8
    up_width = 16 if m <= 2 else 32
    up_panel = 256 if m <= 2 else 128
    up_cache = '.cg' if m <= 2 else 'buffer'
    down_width = 32

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=x.device)
    logits = empty((m, router_splits, 512), torch.float32)
    ids = empty((routes,), torch.int32)
    groups = None if direct else empty((257,), torch.uint64)
    weights = empty((routes,), torch.float32)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    gu = empty((routes, splits, 2 * intermediate), torch.float32)
    aq = empty((routes, intermediate // 2), torch.uint8)
    aqs = empty((routes, intermediate // 32), torch.uint8)
    down_output = empty((routes, h), torch.float32)
    out = empty((m, h))
    _router_linear[24 * router_splits + m * (h // (32 * quant_groups)) + int(grouped),](x, router, logits, xq, xs, groups, h, x.stride(0), m, router_splits, router_block, 16, grouped, quant_vector, quant_groups, NATIVE=False, num_warps=1)
    if direct:
        _select_and_up[m + routes * splits * (2 * intermediate // up_width),](logits, correction_bias, ids, weights, groups, xq, xs, w13, w13_scale, gu, m, h, intermediate, router_splits, splits, up_width, routed_scaling_factor, up_panel, up_cache, num_warps=1)
        _activate_quantize[routes, intermediate // 64](gu, aq, aqs, intermediate, splits, 1, VEC=1 if m == 1 else 2, NATIVE=True, LIMIT=swiglu_limit, num_warps=1, enable_fp_fusion=False)
        _route_down[m, 9, h // 16](aq, aqs, w2, w2_scale, ids, down_output, h, intermediate, 16, num_warps=1, enable_fp_fusion=False)
        _combine[m, h // 256](down_output, weights, out, h, 256, 1, 4, num_warps=4, enable_fp_fusion=False)
        return out
    if stagger:
        _select_and_shared[m + splits * (2 * intermediate // up_width),](logits, correction_bias, ids, weights, groups, xq, xs, w13, w13_scale, gu, m, h, intermediate, router_splits, splits, up_width, routed_scaling_factor, num_warps=1)
        up_jobs = m * (intermediate // 64) + m * 8 * splits * (2 * intermediate // up_width)
        _up_and_shared_activation[up_jobs,](xq, xs, w13, w13_scale, ids, groups, gu, aq, aqs, h, intermediate, m, splits, up_width, num_warps=1, enable_fp_fusion=False)
        shared_width = 16
        _activation_and_shared_down[h // shared_width + m * 8 * (intermediate // 64),](gu, aq, aqs, w2, w2_scale, ids, groups, down_output, h, intermediate, m, splits, num_warps=1, enable_fp_fusion=False)
        if m == 3:
            tail_width, tail_warps = (64, 4)
            _diagonal_down[m, h // tail_width](aq, aqs, w2, w2_scale, ids, weights, down_output, out, h, intermediate, tail_width, tail_warps, num_warps=tail_warps, enable_fp_fusion=False)
        else:
            _fused_down_combine[m, h // 64](aq, aqs, w2, w2_scale, ids, weights, down_output, out, h, intermediate, 64, 2, num_warps=2, enable_fp_fusion=False)
    else:
        _select_routes[m,](logits, correction_bias, ids, weights, groups, router_splits, grouped, routed_scaling_factor, num_warps=1)
        jobs = m * 8 + 1
        activation_warps = 4
        up_grid = (splits, jobs, 2 * intermediate // up_width)
        _expert_projection[up_grid](xq, xs, w13, w13_scale, ids, groups, gu, 2 * intermediate, h, m, True, splits, up_width, BK=up_panel, num_warps=1)
        _activate_quantize[routes, intermediate // (64 * activation_warps)](gu, aq, aqs, intermediate, splits, activation_warps, num_warps=activation_warps, enable_fp_fusion=False)
        down_grid = (jobs, h // down_width, 1)
        _expert_projection[down_grid](aq, aqs, w2, w2_scale, ids, groups, down_output, h, intermediate, m, False, 1, down_width, num_warps=1)
        combine_vector, combine_warps = (1, 4)
        _combine[m, h // 256](down_output, weights, out, h, 256, combine_vector, combine_warps, num_warps=combine_warps, enable_fp_fusion=False)
    return out
