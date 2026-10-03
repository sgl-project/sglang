"""Generated Kimi-K3 MLA paged prefill attention kernel for gfx950.

Source: OpenAI-Partners/artemis-kernel-integrations PR 17,
commit 35b249f7a551278946a81b7da1d58c286c41fb8f.
"""

# ruff: noqa
# fmt: off

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _value_offset(token, channel, BN: gl.constexpr):
    return channel // 32 * (BN * 32) + token // 16 * 512 + token % 16 // 8 * 256 + channel % 32 * 8 + token % 8

@gluon.jit
def _store_native_key(PK, values, stripe, WIDTH: gl.constexpr, START: gl.constexpr, BN: gl.constexpr, CHUNK_PACKED: gl.constexpr):
    values = values.reshape((16, WIDTH // 16, 2, 8))
    values = values.permute((1, 2, 0, 3)).reshape((16 * WIDTH,))
    flat: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
    values = gl.convert_layout(values, flat)
    x = gl.arange(0, 16 * WIDTH, flat)
    if CHUNK_PACKED and WIDTH == 512:
        chunk = x // 2048
        local = x % 2048
        offset = chunk * (128 * BN) + stripe // 2 * 4096 + local // 128 * 256 + stripe % 2 * 128 + local % 128
    else:
        offset = stripe // 2 * (32 * WIDTH) + x // 128 * 256 + stripe % 2 * 128 + x % 128 + START * BN
    gl.store(PK + offset, values)

@gluon.jit
def _store_native_value(PV, values, stripe, DV: gl.constexpr, BN: gl.constexpr):
    values = values.reshape((2, 8, DV // 32, 32))
    values = values.permute((2, 0, 3, 1)).reshape((16 * DV,))
    flat: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
    values = gl.convert_layout(values, flat)
    x = gl.arange(0, 16 * DV, flat)
    offset = x // 512 * (BN * 32) + stripe * 512 + x % 512
    gl.store(PV + offset, values)

@gluon.jit
def _pack_tile(K, V, IND, PK, PV, first, length, block, stripe, head, K0: gl.constexpr, K1: gl.constexpr, V0: gl.constexpr, V1: gl.constexpr, IS: gl.constexpr, D: gl.constexpr, DV: gl.constexpr, CACHED: gl.constexpr, BN: gl.constexpr, CHUNK_PACKED: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    n = stripe * 16 + gl.arange(0, 16, gl.SliceLayout(1, layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, layout))
    valid = block * BN + n < length
    logical = first + block * BN + n
    if CACHED:
        slot = gl.load(IND + logical.to(gl.int64) * IS, valid, 0).to(gl.int64)
    else:
        slot = logical.to(gl.int64)
    k = gl.load(K + slot[:, None] * K0 + head * K1 + d[None, :], valid[:, None] & (d[None, :] < D), 0.0).to(gl.bfloat16)
    _store_native_key(PK, k, stripe, 512, 0, BN, CHUNK_PACKED)
    if D > 512:
        tail = 512 + gl.arange(0, 64, gl.SliceLayout(0, layout))
        kt = gl.load(K + slot[:, None] * K0 + head * K1 + tail[None, :], valid[:, None] & (tail[None, :] < D), 0.0).to(gl.bfloat16)
        _store_native_key(PK, kt, stripe, 64, 512, BN, CHUNK_PACKED)
    v = gl.load(V + slot[:, None] * V0 + head * V1 + d[None, :], valid[:, None] & (d[None, :] < DV), 0.0).to(PV.dtype.element_ty)
    _store_native_value(PV, v, stripe, DV, BN)

@gluon.jit
def _pack_all(K, V, KC, VC, QPTR, KPTR, IND, PK, PV, FK, FV, K0: gl.constexpr, K1: gl.constexpr, V0: gl.constexpr, V1: gl.constexpr, KC0: gl.constexpr, KC1: gl.constexpr, VC0: gl.constexpr, VC1: gl.constexpr, IS: gl.constexpr, D: gl.constexpr, DV: gl.constexpr, CP: gl.constexpr, FP: gl.constexpr, SEQUENCES: gl.constexpr, BN: gl.constexpr, GUARD_TILES: gl.constexpr, CHUNK_PACKED: gl.constexpr):
    task, head = (gl.program_id(0), gl.program_id(1))
    stripe = task % (BN // 16)
    tile = task // (BN // 16)
    if tile < CP:
        seq = 0
        first = gl.load(KPTR)
        for candidate in gl.static_range(1, SEQUENCES):
            boundary = gl.load(KPTR + candidate)
            take = tile >= boundary // BN + candidate
            seq = gl.where(take, candidate, seq)
            first = gl.where(take, boundary, first)
        length = gl.load(KPTR + seq + 1) - first
        block = tile - (first // BN + seq)
        if GUARD_TILES:
            active = (block >= 0) & (block * BN <= length)
        else:
            active = (block >= 0) & (block * BN < length)
        if active:
            packed = (head * CP + tile).to(gl.int64)
            _pack_tile(KC, VC, IND, PK + packed * (BN * D), PV + packed * (BN * DV), first, length, block, stripe, head, KC0, KC1, VC0, VC1, IS, D, DV, True, BN, CHUNK_PACKED)
    else:
        fresh_tile = tile - CP
        fresh_seq = 0
        fresh_first = gl.load(QPTR)
        for fresh_candidate in gl.static_range(1, SEQUENCES):
            fresh_boundary = gl.load(QPTR + fresh_candidate)
            fresh_take = fresh_tile >= fresh_boundary // BN + fresh_candidate
            fresh_seq = gl.where(fresh_take, fresh_candidate, fresh_seq)
            fresh_first = gl.where(fresh_take, fresh_boundary, fresh_first)
        fresh_length = gl.load(QPTR + fresh_seq + 1) - fresh_first
        fresh_block = fresh_tile - (fresh_first // BN + fresh_seq)
        if GUARD_TILES:
            fresh_active = (fresh_block >= 0) & (fresh_block * BN <= fresh_length)
        else:
            fresh_active = (fresh_block >= 0) & (fresh_block * BN < fresh_length)
        if fresh_active:
            fresh_packed = (head * FP + fresh_tile).to(gl.int64)
            _pack_tile(K, V, IND, FK + fresh_packed * (BN * D), FV + fresh_packed * (BN * DV), fresh_first, fresh_length, fresh_block, stripe, head, K0, K1, V0, V1, IS, D, DV, False, BN, CHUNK_PACKED)

@gluon.jit
def _load_native_key(PK, CHANNELS: gl.constexpr, START: gl.constexpr, BN: gl.constexpr, CHUNK_PACKED: gl.constexpr):
    gl.static_assert(BN == 128)
    if CHANNELS == 128:
        physical: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[1], [2], [4], [512], [1024], [2048]], lane_bases=[[8], [16], [32], [64], [128], [256]], warp_bases=[[4096], [8192]], block_bases=[], shape=[16384])
    else:
        gl.static_assert(CHANNELS == 64)
        physical: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[1], [2], [4], [512], [1024]], lane_bases=[[8], [16], [32], [64], [128], [256]], warp_bases=[[2048], [4096]], block_bases=[], shape=[8192])
    x = gl.arange(0, CHANNELS * BN, physical)
    if CHANNELS == 128 and (not CHUNK_PACKED):
        offset = x // 4096 * 16384 + x % 4096 + START * 32
    else:
        offset = x + START * BN
    values = gl.amd.cdna4.buffer_load(PK, offset).to(gl.bfloat16)
    values = values.reshape((BN // 32, CHANNELS // 16, 2, 32, 8))
    values = values.permute((1, 2, 4, 0, 3)).reshape((CHANNELS, BN))
    parent: gl.constexpr = gl.amd.AMDMFMALayout(4, [32, 32, 16], True, [1, 4])
    return gl.convert_layout(values, gl.DotOperandLayout(1, parent, 8), assert_trivial=True)

@gluon.jit
def _load_native_value(PV, BN: gl.constexpr, DV: gl.constexpr):
    gl.static_assert(BN == 128 and DV == 512)
    physical: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[1], [2], [4], [512], [1024], [2048], [16384], [32768]], lane_bases=[[8], [16], [32], [64], [128], [256]], warp_bases=[[4096], [8192]], block_bases=[], shape=[65536])
    x = gl.arange(0, BN * DV, physical)
    values = gl.amd.cdna4.buffer_load(PV, x).to(gl.bfloat16)
    values = values.reshape((DV // 32, BN // 16, 2, 32, 8))
    values = values.permute((1, 2, 4, 0, 3)).reshape((BN, DV))
    parent: gl.constexpr = gl.amd.AMDMFMALayout(4, [32, 32, 16], True, [1, 4])
    return gl.convert_layout(values, gl.DotOperandLayout(1, parent, 8), assert_trivial=True)

@gluon.jit
def _sum_within_key_wave(probability, BN: gl.constexpr):
    stripes = probability.reshape((64, BN // 128, 4, 32))
    return gl.sum(gl.sum(stripes, 3), 1)

@gluon.jit
def _maximum_across_key_waves(score):
    local = gl.max(score.reshape((64, 4, 32)), 2)
    shared = gl.allocate_shared_memory(gl.float32, (64, 4), gl.SwizzledSharedLayout(1, 1, 1, [0, 1]), local)
    readers: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [4, 1], [0, 1])
    values = shared.load(readers)
    maximum = gl.max(values, 1)
    shared._keep_alive()
    return gl.convert_layout(maximum, gl.SliceLayout(1, score.type.layout))

@gluon.jit
def _attention_step(q, qt, PK, PV, maximum, denominator, acc, query_pos, key_block, key_length, scale, D: gl.constexpr, DV: gl.constexpr, CAUSAL: gl.constexpr, BN: gl.constexpr, FULL: gl.constexpr=False, CHUNK_PACKED: gl.constexpr=False, TAIL_REGISTER: gl.constexpr=False, RESIDENT_QUERY: gl.constexpr=3, WAVE_MAX: gl.constexpr=False, NATIVE_VALUE: gl.constexpr=False, PREFETCH: gl.constexpr=False, CARRY: gl.constexpr=False, next_key=None, next_PK=None):
    qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(4, [32, 32, 16], True, [1, 4])
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(4, [32, 32, 16], True, [1, 4])
    vb_layout: gl.constexpr = gl.DotOperandLayout(1, pv_layout, 8)
    score = gl.zeros((64, BN), gl.float32, qk_layout)
    for chunk in gl.static_range(4):
        if chunk != RESIDENT_QUERY:
            query_fragment = q[chunk].load(gl.DotOperandLayout(0, qk_layout, 8))
        else:
            query_fragment = q[chunk]
        if (PREFETCH or CARRY) and chunk == 0:
            key_fragment = next_key
        else:
            key_fragment = _load_native_key(PK, 128, chunk * 128, BN, CHUNK_PACKED)
        score = gl.amd.cdna4.mfma(query_fragment, key_fragment, score)
    if D > 512:
        kt = _load_native_key(PK, 64, 512, BN, CHUNK_PACKED)
        if TAIL_REGISTER:
            query_tail = qt
        else:
            query_tail = qt.load(gl.DotOperandLayout(0, qk_layout, 8))
        score = gl.amd.cdna4.mfma(query_tail, kt, score)
    if NATIVE_VALUE:
        value = _load_native_value(PV, BN, DV)
    else:
        vn = gl.arange(0, BN, gl.SliceLayout(1, vb_layout))
        vd = gl.arange(0, DV, gl.SliceLayout(0, vb_layout))
        value = gl.load(PV + _value_offset(vn[:, None], vd[None, :], BN)).to(gl.bfloat16)
    if FULL:
        score = score * scale
    else:
        cols = key_block * BN + gl.arange(0, BN, gl.SliceLayout(0, qk_layout))
        valid = cols[None, :] < key_length
        if CAUSAL:
            valid = valid & (cols[None, :] <= query_pos[:, None])
        score = gl.where(valid, score * scale, -float('inf'))
    if WAVE_MAX:
        tile_max = _maximum_across_key_waves(score)
    else:
        tile_max = gl.max(score, 1)
    new_max = gl.maximum(maximum, tile_max)
    alpha = gl.exp(maximum - new_max)
    probability = gl.exp(score - new_max[:, None])
    if not FULL:
        probability = gl.where(valid, probability, 0.0)
    local_sum = _sum_within_key_wave(probability, BN)
    local_alpha = gl.convert_layout(alpha, gl.SliceLayout(1, local_sum.type.layout))
    denominator = denominator * local_alpha[:, None] + local_sum
    probability_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    probability_shared = gl.allocate_shared_memory(gl.bfloat16, (64, BN), probability_layout)
    probability_shared.store(probability.to(gl.bfloat16))
    probability = probability_shared.load(gl.DotOperandLayout(0, pv_layout, 8))
    rescale = gl.convert_layout(alpha, gl.SliceLayout(1, pv_layout), assert_trivial=True)
    probability_shared._keep_alive()
    acc = acc * rescale[:, None]
    if PREFETCH:
        next_key = _load_native_key(next_PK, 128, 0, BN, CHUNK_PACKED)
    acc = gl.amd.cdna4.mfma(probability, value, acc)
    return (new_max, denominator, acc, next_key)

@gluon.jit
def _attention(Q, QPTR, KPTR, PK, PV, FK, FV, OUT, PART, STATS, scale, M: gl.constexpr, H: gl.constexpr, HK: gl.constexpr, D: gl.constexpr, DV: gl.constexpr, Q0: gl.constexpr, Q1: gl.constexpr, CP: gl.constexpr, FP: gl.constexpr, SPLITS: gl.constexpr, SEQUENCES: gl.constexpr, HEAD_GROUP: gl.constexpr, BN: gl.constexpr, GUARD_LOOKAHEAD: gl.constexpr, CARRY_FRESH: gl.constexpr, CHUNK_PACKED: gl.constexpr):
    tile = gl.program_id(0) // HEAD_GROUP
    head = gl.program_id(1) * HEAD_GROUP + gl.program_id(0) % HEAD_GROUP
    seq = gl.program_id(2) // SPLITS
    split = gl.program_id(2) % SPLITS
    if SEQUENCES == 2:
        p0, p1, p2 = (gl.load(KPTR), gl.load(KPTR + 1), gl.load(KPTR + 2))
        qstart, qmiddle, qend = (gl.load(QPTR), gl.load(QPTR + 1), gl.load(QPTR + 2))
        work0 = 2 * (p1 - p0) + qmiddle - qstart
        work1 = 2 * (p2 - p1) + qend - qmiddle
        seq = gl.where(work1 > work0, 1 - seq, seq)
    first = gl.load(QPTR + seq)
    length = gl.load(QPTR + seq + 1) - first
    if tile * 64 >= length:
        return
    tile = gl.cdiv(length, 64) - 1 - tile
    pfirst = gl.load(KPTR + seq)
    plen = gl.load(KPTR + seq + 1) - pfirst
    kvhead = 0 if M == 4096 and HK == 1 else head // (H // HK)
    qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(4, [32, 32, 16], True, [1, 4])
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(4, [32, 32, 16], True, [1, 4])
    wave_max: gl.constexpr = M == 1024 or M == 4096
    resident_query: gl.constexpr = 0 if wave_max else 3
    tail_register: gl.constexpr = SPLITS == 1 or wave_max
    qload: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    qr = tile * 64 + gl.arange(0, 64, gl.SliceLayout(1, qload))
    qd0 = 0 + gl.arange(0, 128, gl.SliceLayout(0, qload))
    q0 = gl.load(Q + (first + qr[:, None]).to(gl.int64) * Q0 + head * Q1 + qd0[None, :], qr[:, None] < length, 0)
    if resident_query == 0:
        q0 = gl.convert_layout(q0, gl.DotOperandLayout(0, qk_layout, 8))
    else:
        q0 = gl.allocate_shared_memory(gl.bfloat16, (64, 128), gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), q0)
    qd1 = 128 + gl.arange(0, 128, gl.SliceLayout(0, qload))
    q1 = gl.load(Q + (first + qr[:, None]).to(gl.int64) * Q0 + head * Q1 + qd1[None, :], qr[:, None] < length, 0)
    q1 = gl.allocate_shared_memory(gl.bfloat16, (64, 128), gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), q1)
    qd2 = 256 + gl.arange(0, 128, gl.SliceLayout(0, qload))
    q2 = gl.load(Q + (first + qr[:, None]).to(gl.int64) * Q0 + head * Q1 + qd2[None, :], qr[:, None] < length, 0)
    q2 = gl.allocate_shared_memory(gl.bfloat16, (64, 128), gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), q2)
    qd3 = 384 + gl.arange(0, 128, gl.SliceLayout(0, qload))
    q3 = gl.load(Q + (first + qr[:, None]).to(gl.int64) * Q0 + head * Q1 + qd3[None, :], qr[:, None] < length, 0)
    if resident_query == 3:
        q3 = gl.convert_layout(q3, gl.DotOperandLayout(0, qk_layout, 8))
    else:
        q3 = gl.allocate_shared_memory(gl.bfloat16, (64, 128), gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), q3)
    q = (q0, q1, q2, q3)
    if D > 512:
        td = 512 + gl.arange(0, 64, gl.SliceLayout(0, qload))
        qt = gl.load(Q + (first + qr[:, None]).to(gl.int64) * Q0 + head * Q1 + td[None, :], qr[:, None] < length, 0)
        if tail_register:
            qt = gl.convert_layout(qt, gl.DotOperandLayout(0, qk_layout, 8))
        else:
            qt = gl.allocate_shared_memory(gl.bfloat16, (64, 64), gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), qt)
    else:
        qt = q
    pos = tile * 64 + gl.arange(0, 64, gl.SliceLayout(1, qk_layout))
    maximum = gl.full((64,), -float('inf'), gl.float32, gl.SliceLayout(1, qk_layout))
    denominator = _sum_within_key_wave(gl.zeros((64, BN), gl.float32, qk_layout), BN)
    acc = gl.zeros((64, DV), gl.float32, pv_layout)
    prefix_blocks = gl.cdiv(plen, BN)
    fresh_blocks = gl.cdiv(gl.minimum((tile + 1) * 64, length), BN)
    if SPLITS > 1:
        target = gl.cdiv(prefix_blocks + fresh_blocks, SPLITS)
        first_count = gl.maximum(target - fresh_blocks, 0)
        other_count = gl.cdiv(prefix_blocks - first_count, SPLITS - 1)
        lo = gl.where(split == 0, 0, first_count + (split - 1) * other_count)
        hi = gl.minimum(gl.where(split == 0, first_count, lo + other_count), prefix_blocks)
    else:
        lo, hi = (0, prefix_blocks)
    packed_base = (kvhead * CP + pfirst // BN + seq).to(gl.int64)
    full_hi = gl.minimum(hi, plen // BN)
    if lo < full_hi:
        next_key = _load_native_key(PK + (packed_base + lo) * (BN * D), 128, 0, BN, CHUNK_PACKED)
        for block in range(lo, full_hi):
            if GUARD_LOOKAHEAD:
                next_block = block + 1
            else:
                next_block = gl.minimum(block + 1, full_hi - 1)
            maximum, denominator, acc, next_key = _attention_step(q, qt, PK + (packed_base + block) * (BN * D), PV + (packed_base + block) * (BN * DV), maximum, denominator, acc, pos, block, plen, scale, D, DV, False, BN, True, CHUNK_PACKED=CHUNK_PACKED, TAIL_REGISTER=tail_register, PREFETCH=True, RESIDENT_QUERY=resident_query, WAVE_MAX=wave_max, NATIVE_VALUE=M == 4096, next_key=next_key, next_PK=PK + (packed_base + next_block) * (BN * D))
    if (lo <= full_hi) & (full_hi < hi):
        maximum, denominator, acc, _ = _attention_step(q, qt, PK + (packed_base + full_hi) * (BN * D), PV + (packed_base + full_hi) * (BN * DV), maximum, denominator, acc, pos, full_hi, plen, scale, D, DV, False, BN, False, CHUNK_PACKED=CHUNK_PACKED, TAIL_REGISTER=tail_register, RESIDENT_QUERY=resident_query, WAVE_MAX=wave_max, NATIVE_VALUE=M == 4096)
    if split == 0:
        fresh_base = (kvhead * FP + first // BN + seq).to(gl.int64)
        if CARRY_FRESH:
            next_key = _load_native_key(FK + fresh_base * (BN * D), 128, 0, BN, CHUNK_PACKED)
        if fresh_blocks > 1:
            if not CARRY_FRESH:
                next_key = _load_native_key(FK + fresh_base * (BN * D), 128, 0, BN, CHUNK_PACKED)
            for fresh_block in range(fresh_blocks - 1):
                if CARRY_FRESH:
                    next_block = fresh_block + 1
                else:
                    next_block = gl.minimum(fresh_block + 1, fresh_blocks - 2)
                maximum, denominator, acc, next_key = _attention_step(q, qt, FK + (fresh_base + fresh_block) * (BN * D), FV + (fresh_base + fresh_block) * (BN * DV), maximum, denominator, acc, pos, fresh_block, length, scale, D, DV, False, BN, True, CHUNK_PACKED=CHUNK_PACKED, TAIL_REGISTER=tail_register, PREFETCH=True, RESIDENT_QUERY=resident_query, WAVE_MAX=wave_max, NATIVE_VALUE=M == 4096, next_key=next_key, next_PK=FK + (fresh_base + next_block) * (BN * D))
        fresh_block = fresh_blocks - 1
        if CARRY_FRESH:
            maximum, denominator, acc, _ = _attention_step(q, qt, FK + (fresh_base + fresh_block) * (BN * D), FV + (fresh_base + fresh_block) * (BN * DV), maximum, denominator, acc, pos, fresh_block, length, scale, D, DV, True, BN, False, CHUNK_PACKED=CHUNK_PACKED, TAIL_REGISTER=tail_register, RESIDENT_QUERY=resident_query, WAVE_MAX=wave_max, NATIVE_VALUE=M == 4096, CARRY=True, next_key=next_key)
        else:
            maximum, denominator, acc, _ = _attention_step(q, qt, FK + (fresh_base + fresh_block) * (BN * D), FV + (fresh_base + fresh_block) * (BN * DV), maximum, denominator, acc, pos, fresh_block, length, scale, D, DV, True, BN, False, CHUNK_PACKED=CHUNK_PACKED, TAIL_REGISTER=tail_register, RESIDENT_QUERY=resident_query, WAVE_MAX=wave_max, NATIVE_VALUE=M == 4096)
    denominator = gl.sum(denominator, 1)
    store_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    rows = first + tile * 64 + gl.arange(0, 64, gl.SliceLayout(1, store_layout))
    cols = gl.arange(0, DV, gl.SliceLayout(0, store_layout))
    live = rows < first + length
    if SPLITS == 1:
        den = gl.convert_layout(denominator, gl.SliceLayout(1, pv_layout))
        result = gl.convert_layout((acc / den[:, None]).to(gl.bfloat16), store_layout)
        gl.store(OUT + (rows[:, None].to(gl.int64) * H + head) * DV + cols[None, :], result, live[:, None])
    else:
        partial = gl.convert_layout(acc, store_layout)
        gl.store(PART + ((split * M + rows[:, None].to(gl.int64)) * H + head) * DV + cols[None, :], partial, live[:, None])
        m = gl.convert_layout(maximum, gl.SliceLayout(1, store_layout))
        l = gl.convert_layout(denominator, gl.SliceLayout(1, store_layout))
        stat = (split * M + rows) * H + head
        gl.store(STATS + stat * 2, m, live)
        gl.store(STATS + stat * 2 + 1, l, live)

@gluon.jit
def _merge(PART, STATS, OUT, M: gl.constexpr, H: gl.constexpr, DV: gl.constexpr, SPLITS: gl.constexpr, ROWS: gl.constexpr):
    if ROWS == 8:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 32], [4, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    col = gl.arange(0, DV, gl.SliceLayout(0, layout))
    gl.static_assert(SPLITS == 4)
    stat = row * 2
    m0 = gl.load(STATS + stat, row < M * H, -float('inf'))
    m1 = gl.load(STATS + stat + 2 * M * H, row < M * H, -float('inf'))
    m2 = gl.load(STATS + stat + 4 * M * H, row < M * H, -float('inf'))
    m3 = gl.load(STATS + stat + 6 * M * H, row < M * H, -float('inf'))
    l0 = gl.load(STATS + stat + 1, row < M * H, 0)
    l1 = gl.load(STATS + stat + 2 * M * H + 1, row < M * H, 0)
    l2 = gl.load(STATS + stat + 4 * M * H + 1, row < M * H, 0)
    l3 = gl.load(STATS + stat + 6 * M * H + 1, row < M * H, 0)
    maximum = gl.maximum(gl.maximum(m0, m2), gl.maximum(m1, m3))
    w0 = gl.exp(m0 - maximum)
    w1 = gl.exp(m1 - maximum)
    w2 = gl.exp(m2 - maximum)
    w3 = gl.exp(m3 - maximum)
    offset = row[:, None].to(gl.int64) * DV + col[None, :]
    live = row[:, None] < M * H
    v0 = gl.load(PART + offset, live, 0).to(gl.float32)
    v1 = gl.load(PART + offset + M * H * DV, live, 0).to(gl.float32)
    v2 = gl.load(PART + offset + 2 * M * H * DV, live, 0).to(gl.float32)
    v3 = gl.load(PART + offset + 3 * M * H * DV, live, 0).to(gl.float32)
    acc = v0 * w0[:, None] + v2 * w2[:, None] + (v1 * w1[:, None] + v3 * w3[:, None])
    den = l0 * w0 + l2 * w2 + (l1 * w1 + l3 * w3)
    gl.store(OUT + offset, (acc / den[:, None]).to(gl.bfloat16), live)

def paged_attention_prefill(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, *, scale: float, max_query_length: int, output_tensor=None) -> torch.Tensor:

    m, h, d = query.shape
    hk, dv = (key.shape[1], value.shape[2])
    assert d == 576 and dv == 512 and (h % hk == 0)
    sequences = qo_indptr.numel() - 1
    bn = 128
    cp = triton.cdiv(kv_indices.numel(), bn) + sequences
    fp = triton.cdiv(m, bn) + sequences
    packed_key = torch.empty((hk * cp * bn * d,), dtype=torch.bfloat16, device=query.device)
    packed_value = torch.empty((hk * cp * bn * dv,), dtype=torch.bfloat16, device=query.device)
    fresh_key = torch.empty((hk * fp * bn * d,), dtype=torch.bfloat16, device=query.device)
    fresh_value = torch.empty((hk * fp * bn * dv,), dtype=torch.bfloat16, device=query.device)
    if output_tensor is None:
        out = torch.empty((m, h, dv), dtype=torch.bfloat16, device=query.device)
    else:
        assert tuple(output_tensor.shape) == (m, h, dv)
        assert output_tensor.dtype == torch.bfloat16 and output_tensor.device == query.device
        assert output_tensor.is_contiguous()
        assert output_tensor.untyped_storage().data_ptr() not in {x.untyped_storage().data_ptr() for x in (query, key, value, key_cache, value_cache, qo_indptr, kv_indptr, kv_indices)}
        out = output_tensor
    splits = 4 if m <= 2048 else 1
    guard_lookahead = m in (4096, 6144)
    carry_fresh = m == 6144
    chunk_packed = m in (4096, 8192)
    merge_rows = 8 if m <= 1024 else 4
    if splits > 1:
        partial = torch.empty((splits, m, h, dv), dtype=torch.float32, device=query.device)
        stats = torch.empty((splits, m, h, 2), dtype=torch.float32, device=query.device)
    else:
        partial, stats = (out, out)
    _pack_all[(cp + fp) * (bn // 16), hk](key, value, key_cache, value_cache, qo_indptr, kv_indptr, kv_indices, packed_key, packed_value, fresh_key, fresh_value, key.stride(0), key.stride(1), value.stride(0), value.stride(1), key_cache.stride(0), key_cache.stride(1), value_cache.stride(0), value_cache.stride(1), kv_indices.stride(0), d, dv, cp, fp, sequences, bn, guard_lookahead, chunk_packed, num_warps=4)
    nt = triton.cdiv(max_query_length, 64)
    head_group = h // hk
    _attention[nt * head_group, h // head_group, sequences * splits](query, qo_indptr, kv_indptr, packed_key, packed_value, fresh_key, fresh_value, out, partial, stats, scale, m, h, hk, d, dv, query.stride(0), query.stride(1), cp, fp, splits, sequences, head_group, bn, guard_lookahead, carry_fresh, chunk_packed, num_warps=4, num_stages=1, llvm_fn_attrs=[['amdgpu-sched-strategy', 'iterative-ilp']])
    if splits > 1:
        _merge[triton.cdiv(m * h, merge_rows),](partial, stats, out, m, h, dv, splits, merge_rows, num_warps=4)
    return out
