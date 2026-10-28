# Created by OpenAI
# fmt: off
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_artifact_next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)

@gluon.jit
def _add(a, b):
    return a + b

@gluon.jit
def _encode_groups(x, shared):
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
    low, high = gl.split(a.reshape((x.shape[0], 16, 2)))
    packed = gl.inline_asm_elementwise('v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3;', constraints='=v,v,v,v', args=[low, high, gl.full(low.shape, 1.0, gl.float32, low.type.layout)], dtype=gl.uint32, is_pure=True, pack=1)
    signs = gl.where(x < 0, 8, 0).to(gl.uint8)
    low_sign, high_sign = gl.split(signs.reshape((x.shape[0], 16, 2)))
    return (packed.to(gl.uint8) & 119 | low_sign | high_sign << 4, (exponent + 127).to(gl.uint8))

@gluon.jit(do_not_specialize=['M', 'CTA_OFFSET'])
def _quantize_input(X, Q, QS, M, H: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, CTA_OFFSET=0, WARPS: gl.constexpr=1):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [WARPS, 1], [1, 0])
    g = (gl.program_id(0) - CTA_OFFSET) * GROUPS + gl.arange(0, GROUPS, gl.SliceLayout(1, layout))
    k = gl.arange(0, 32, gl.SliceLayout(0, layout))
    x = gl.load(X + (g // (H // 32))[:, None] * SX + (g % (H // 32))[:, None] * 32 + k[None, :], g[:, None] < M * (H // 32), 0).to(gl.float32)
    rq, rs = _encode_groups(x, False)
    sq, ss = _encode_groups(x, True)
    pk = gl.arange(0, 16, gl.SliceLayout(0, rq.type.layout))
    pg = gl.convert_layout(g, gl.SliceLayout(1, rq.type.layout))
    gl.store(Q + pg[:, None] * 16 + pk[None, :], rq, pg[:, None] < M * (H // 32))
    gl.store(Q + M * H // 2 + pg[:, None] * 16 + pk[None, :], sq, pg[:, None] < M * (H // 32))
    gl.store(QS + g, rs, g < M * (H // 32))
    gl.store(QS + M * H // 32 + g, ss, g < M * (H // 32))

@gluon.jit(do_not_specialize=['M'])
def _router_linear(X, W, L, M, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, Counts, WARPS: gl.constexpr=4, INIT_SHARDS: gl.constexpr=8, ROW_MAJOR: gl.constexpr=False):
    if ROW_MAJOR:
        row_pid = gl.program_id(0) // (256 // BN)
        column_pid = gl.program_id(0) % (256 // BN)
    else:
        row_pid = gl.program_id(0) % gl.cdiv(M, BM)
        column_pid = gl.program_id(0) // gl.cdiv(M, BM)
    init_shard = row_pid * (256 // BN) + column_pid
    if init_shard < INIT_SHARDS:
        counter = gl.arange(0, 256, gl.BlockedLayout([1], [64], [WARPS], [0]))
        gl.store(Counts + init_shard * 256 + counter, 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1] if WARPS == 1 else [2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WARPS, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, WARPS], [0, 1])
    mi = row_pid * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    ni = column_pid * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for base in range(H // BK):
        a = gl.load(X + mi[:, None] * SX + (base * BK + ak)[None, :], mi[:, None] < M, 0)
        b = gl.load(W + ni[None, :] * H + (base * BK + bk)[:, None])
        acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8)), acc)
    mm = row_pid * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    nn = column_pid * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(L + mm[:, None] * 256 + nn[None, :], acc, mm[:, None] < M)

@gluon.jit(do_not_specialize=['M'])
def _front_end(X, W, L, Q, QS, Counts, M, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, WARPS: gl.constexpr, GROUPS: gl.constexpr, SHARDS: gl.constexpr, ROW_MAJOR: gl.constexpr=False):
    router_ctas = gl.cdiv(M, BM) * (256 // BN)
    if gl.program_id(0) < router_ctas:
        _router_linear(X, W, L, M, H, SX, BM, BN, BK, Counts, WARPS, SHARDS, ROW_MAJOR)
    else:
        _quantize_input(X, Q, QS, M, H, SX, GROUPS, router_ctas, WARPS)

@gluon.jit
def _select_routes(L, Bias, Ids, Weights, Counts, SHARDS: gl.constexpr, TICKET_STRIDE: gl.constexpr, SCALE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    m = gl.program_id(0)
    e = gl.arange(0, 256, layout)
    prob = 1.0 / (1.0 + gl.exp(-gl.load(L + m * 256 + e).to(gl.float32)))
    score = prob + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    selected_prob = gl.full((256,), 0.0, gl.float32, layout)
    selected_id = gl.full((256,), 0, gl.int32, layout)
    total = 0.0
    for j in gl.static_range(8):
        maximum = gl.max(score, 0)
        idx = gl.min(gl.where(available & (score == maximum), e, 256), 0)
        idx = gl.where(idx < 256, idx, gl.min(gl.where(available, e, 256), 0))
        p = gl.sum(gl.where(e == idx, prob, 0.0), 0)
        total += p
        selected_prob = gl.where(e == j, p, selected_prob)
        selected_id = gl.where(e == j, idx, selected_id)
        available &= e != idx
        score = gl.where(e == idx, -float('inf'), score)
    ticket = gl.atomic_add(Counts + m // 64 % SHARDS * 256 + selected_id, 1, e < 8, sem='relaxed')
    gl.store(Ids + m * 8 + e, selected_id * TICKET_STRIDE + ticket, e < 8)
    gl.store(Weights + m * 8 + e, selected_prob / total * SCALE, e < 8)

@gluon.jit(do_not_specialize=['M'])
def _write_height_descriptors(UpInfo, e, experts, counts, offset, count, M, BM: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(4192, BM)), layout)
    live = gl.minimum(BM, count - b * BM)
    remainder = counts % BM
    destination = gl.full(b.shape, 0, gl.int32, layout)
    for level in gl.static_range(3 + (BM == 128)):
        height = gl.constexpr(BM >> level)
        lower = gl.constexpr(0 if height == 16 else height // 2)
        same = ((remainder > lower) & (remainder <= height)).to(gl.int32)
        if height == BM:
            same += counts // BM
            before = gl.sum(gl.where(experts < e, same, 0), 0)
            local = b
        else:
            higher = counts // BM + (remainder > height).to(gl.int32)
            shared_higher = M // BM + (M % BM > height)
            before = gl.sum(higher, 0) + shared_higher + gl.sum(gl.where(experts < e, same, 0), 0)
            local = gl.full(b.shape, 0, gl.int32, layout)
        destination = gl.where((live > lower) & (live <= height), before + local, destination)
    descriptor = e | live << 9 | offset + b << 17
    gl.store(UpInfo + destination, descriptor, b < gl.cdiv(count, BM))

@gluon.jit
def _sum_shard_counts(Counts, SHARDS: gl.constexpr, before=0, WITH_PREFIX: gl.constexpr=False):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [1, 0])
    shard = gl.arange(0, SHARDS, gl.SliceLayout(1, layout))
    expert = gl.arange(0, 256, gl.SliceLayout(0, layout))
    partial = gl.load(Counts + shard[:, None] * 256 + expert[None, :])
    native: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    totals = gl.convert_layout(gl.sum(partial, 0), native)
    if WITH_PREFIX:
        prefix = gl.sum(gl.where(shard[:, None] < before, partial, 0), 0)
        return (totals, gl.convert_layout(prefix, native))
    else:
        return totals

@gluon.jit(do_not_specialize=['M'])
def _write_down_tiles(Jobs, e, experts, counts, offset, count, arena_start, M, BM: gl.constexpr, WIDE: gl.constexpr=32):
    gl.static_assert(9 * 4192 + 256 * (BM // 2) + BM < 65536)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    remainder = counts % BM
    sizes = counts // BM * 2 + gl.where(remainder > WIDE, 2, (remainder > 0).to(gl.int32))
    start = gl.sum(gl.where(experts < e, sizes, 0), 0)
    b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(4192, BM)), layout)
    live = gl.minimum(BM, count - b * BM)
    dense_start = gl.sum(gl.where(experts < e, counts, 0), 0)
    descriptor = (e | live << 9 | offset + b << 17).to(gl.uint64)
    descriptor |= (dense_start + b * BM).to(gl.uint64) << 32
    descriptor |= (arena_start + b * BM).to(gl.uint64) << 48
    gl.store(Jobs + start + b * 2, descriptor, b < gl.cdiv(count, BM))
    gl.store(Jobs + start + b * 2 + 1, descriptor | 1 << 29, (b < gl.cdiv(count, BM)) & (live > WIDE))

@gluon.jit(do_not_specialize=['M', 'CHUNKS', 'ROUTED_BLOCKS', 'SCHEDULED', 'DOWN_SCHEDULED'])
def _prepare_routes(Codes, RouteWeights, Counts, Sorted, UpInfo, Jobs, M, CHUNKS, BM: gl.constexpr, ROUTED_BLOCKS, SCHEDULED, DOWN_SCHEDULED, WIDE: gl.constexpr=32, SHARDS: gl.constexpr=8, TICKET_STRIDE: gl.constexpr=1024):
    pid = gl.program_id(0)
    if pid < CHUNKS:
        chunk = pid
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        lane = gl.arange(0, 256, layout)
        route = chunk * 256 + lane
        code = gl.load(Codes + route, route < M * 8, 0).to(gl.int32)
        expert, ticket = (code // TICKET_STRIDE, code % TICKET_STRIDE)
        counts, prefix_counts = _sum_shard_counts(Counts, SHARDS, chunk // 2 % SHARDS, True)
        tiles = gl.cdiv(counts, BM)
        dense_offsets = gl.associative_scan(counts, 0, _add) - counts
        dense = gl.gather(dense_offsets, expert, 0)
        offsets = (gl.associative_scan(tiles, 0, _add) - tiles) * BM
        offset = gl.gather(offsets, expert, 0)
        prefix = gl.gather(prefix_counts, expert, 0)
        gl.store(Sorted + offset + prefix + ticket, route, route < M * 8)
        weight = gl.load(RouteWeights + route, route < M * 8, 0)
        record = (dense + prefix + ticket).to(gl.uint64)
        record |= weight.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
        gl.store(Codes + route, record, route < M * 8)
    else:
        e = pid - CHUNKS
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        experts = gl.arange(0, 256, layout)
        counts = _sum_shard_counts(Counts, SHARDS)
        tiles = gl.cdiv(counts, BM)
        if e == 257:
            active = gl.sum(tiles, 0)
            hole = gl.arange(0, 1024, layout)
            gl.store(UpInfo + hole, 0, (hole >= active + gl.cdiv(M, BM)) & (hole < SCHEDULED))
            remainder = counts % BM
            sizes = counts // BM * 2 + gl.where(remainder > WIDE, 2, (remainder > 0).to(gl.int32))
            shared_tiles = M // BM * 2 + gl.where(M % BM > WIDE, 2, (M % BM > 0).to(gl.int32))
            active_tiles = gl.sum(sizes, 0) + shared_tiles
            holes = gl.arange(0, 1024, layout)
            gl.store(Jobs + holes, 0, (holes >= active_tiles) & (holes < DOWN_SCHEDULED))
        else:
            if e < 256:
                offset = gl.sum(gl.where(experts < e, tiles, 0), 0)
                count = gl.sum(gl.where(experts == e, counts, 0), 0)
            else:
                offset = ROUTED_BLOCKS
                count = M
            _write_height_descriptors(UpInfo, e, experts, counts, offset, count, M, BM)
            remainder = counts % BM
            tail = gl.where(remainder > 64, 128, gl.where(remainder > 32, 64, gl.where(remainder > 16, 32, gl.where(remainder > 0, 16, 0))))
            arena_sizes = counts // BM * BM + tail
            arena_start = gl.sum(gl.where(experts < e, arena_sizes, 0), 0)
            _write_down_tiles(Jobs, e, experts, counts, offset, count, arena_start, M, BM, WIDE)
            b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(4192, BM)), layout)
            gl.store(Sorted + ROUTED_BLOCKS * BM + offset + b, arena_start + b * BM, b < gl.cdiv(count, BM))

@gluon.jit
def _weight_offset(e, n, k, N: gl.constexpr, K: gl.constexpr):
    byte = k // 2
    return ((((e * (N // 16) + n // 16) * (K // 64) + byte // 32) * 2 + byte // 16 % 2) * 16 + n % 16) * 16 + byte % 16

@gluon.jit
def _load_packed_weight(W, S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, CACHE: gl.constexpr=''):
    W = W + expert * (N * K // 2)
    S = S + expert * (N * (triton.cdiv(K // 32, 8) * 8))
    packed: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [4, 1], [1, 0])
    n = gl.arange(0, BN, gl.SliceLayout(1, packed))
    if UP:
        n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)
    else:
        n = column * BN + n
    k = base * BK + 8 * gl.arange(0, BK // 8, gl.SliceLayout(0, packed))
    offset = _weight_offset(0, n[:, None], k[None, :], N, K) // 4
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset, cache=CACHE)
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    idx = gl.arange(0, BN * BK // 128, sl)
    nb = idx // (BK // 4)
    if UP:
        nb = column * (BN // 64) + nb % (BN // 64) + nb // (BN // 64) * (N // 64)
    else:
        nb = column * (BN // 32) + nb
    kg = idx // 64 % (BK // 256)
    inner = idx % 64
    sw = gl.amd.cdna4.buffer_load(S.to(gl.pointer_type(gl.uint32)), nb * (K // 4) + base * (BK // 4) + kg * 64 + inner)
    raw = sw.reshape((BN // 32, BK // 256, 4, 16))
    b0, b1 = (raw.to(gl.uint8), (raw >> 8).to(gl.uint8))
    b2, b3 = ((raw >> 16).to(gl.uint8), (raw >> 24).to(gl.uint8))
    packed_scales = gl.join(gl.join(b0, b2), gl.join(b1, b3))
    scale_byte = gl.permute(packed_scales, (0, 5, 3, 1, 4, 2)).reshape((BN, BK // 32))
    return (words, scale_byte)

@gluon.jit
def _word_bytes(words):
    b0 = words.to(gl.uint8)
    b1 = (words >> 8).to(gl.uint8)
    b2 = (words >> 16).to(gl.uint8)
    b3 = (words >> 24).to(gl.uint8)
    return gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((words.shape[0], words.shape[1] * 4))

@gluon.jit(do_not_specialize=['M', 'ROUTED_BLOCKS'])
def _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, column, expert, live, packed_rows, M, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, ROUTED_BLOCKS, UP: gl.constexpr, TM: gl.constexpr, EARLY_A: gl.constexpr, EARLY_AS: gl.constexpr, PACK_A: gl.constexpr, WEIGHT_CACHE: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4] if TM <= 32 else [2, 2])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [TM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
    mi = gl.arange(0, TM, gl.SliceLayout(1, al))
    shared = expert == 256
    if UP:
        arena_row = gl.load(Sorted + ROUTED_BLOCKS * BM + block)
        if shared:
            route = ((block - ROUTED_BLOCKS) * BM + mi) * 8
        else:
            route = gl.load(Sorted + block * BM + mi, mi < live, 0)
        row = gl.where(mi < live, route // 8 + gl.where(shared, M, 0), 0)
    else:
        arena_row = (packed_rows >> 16).to(gl.int32)
        dense_base = (packed_rows & 65535).to(gl.int32)
        row = arena_row + mi
    ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
    a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    b_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [0, 1])
    a_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 2], a_shared_layout)
    b_shared = gl.allocate_shared_memory(gl.uint8, [BK // 2, BN], b_shared_layout)
    gl.static_assert(BK == 256)
    bs_layout: gl.constexpr = gl.SharedLinearLayout([[0, 4], [16, 0], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [32, 0], [64, 0]] + ([[128, 0]] if BN == 256 else []))
    if PACK_A:
        as_layout: gl.constexpr = gl.SharedLinearLayout([[0, 4], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0]] + ([[16, 0]] if TM >= 32 else []) + ([[32, 0]] if TM >= 64 else []) + ([[64, 0]] if TM >= 128 else []))
    else:
        as_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    bs_shared = gl.allocate_shared_memory(gl.uint8, [BN, BK // 32], bs_layout)
    as_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 32], as_layout)
    as_load_layout: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [4, 1], [1, 0])
    scale_row = gl.convert_layout(row, gl.SliceLayout(1, as_load_layout))
    scale_word = gl.arange(0, BK // 128, gl.SliceLayout(0, as_load_layout))
    acc = gl.zeros((TM, BN), gl.float32, mma)
    for base in range(K // BK):
        a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
        if EARLY_A:
            a_shared.store(_word_bytes(a_words))
        sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + base * (BK // 128) + scale_word[None, :])
        if EARLY_AS:
            as_shared.store(_word_bytes(sa_words))
        words, scales = _load_packed_weight(W, WS, expert, column, base, N, K, BN, BK, UP, WEIGHT_CACHE)
        if not EARLY_A:
            a_shared.store(_word_bytes(a_words))
        if not EARLY_AS:
            as_shared.store(_word_bytes(sa_words))
        b = _word_bytes(words).T
        b_shared.store(b)
        bs_shared.store(scales)
        sa = as_shared.load(asl)
        a = a_shared.load(ad)
        b = b_shared.load(bd)
        sb = bs_shared.load(bsl)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    if UP:
        gate, up = gl.split(gl.permute(gl.reshape(acc, (TM, 2, BN // 2)), (0, 2, 1)))
        if shared:
            gate = gate.to(gl.bfloat16).to(gl.float32)
            up = up.to(gl.bfloat16).to(gl.float32)
        activated = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
        ep: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
        activated = gl.convert_layout(activated, ep)
        codes, scales = _encode_groups(gl.reshape(activated, (TM * (BN // 64), 32)), shared)
        codes = codes.reshape((TM, BN // 4))
        scales = scales.reshape((TM, BN // 64))
        rr = gl.arange(0, TM, gl.SliceLayout(1, codes.type.layout))
        nn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, codes.type.layout))
        gl.store(Q + (arena_row + rr[:, None]) * (N // 4) + nn[None, :], codes)
        rr_s = gl.arange(0, TM, gl.SliceLayout(1, scales.type.layout))
        nn_s = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, scales.type.layout))
        gl.store(QS + (arena_row + rr_s[:, None]) * (N // 64) + nn_s[None, :], scales)
    else:
        store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        result = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
        rr = gl.arange(0, TM, gl.SliceLayout(1, store_layout))
        nn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, store_layout))
        if shared:
            token = (block - ROUTED_BLOCKS) * BM + rr
            gl.store(Y + token[:, None] * N + nn[None, :], result, rr[:, None] < live)
        else:
            part_base = Parts + column * M * 8 * BN + dense_base * 128
            address = rr[:, None] * 128 + nn[None, :] % 128
            if BN > 128:
                address += nn[None, :] % BN // 128 * M * 8 * 128
            gl.amd.cdna4.buffer_store(result.to(Parts.dtype.element_ty), part_base, address, rr[:, None] < live)

@gluon.jit(do_not_specialize=['M', 'ROUTED_BLOCKS'])
def _scaled_experts(X, XS, W, WS, Sorted, Info, Q, QS, Parts, Y, M, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, ROUTED_BLOCKS, GROUP: gl.constexpr, UP: gl.constexpr, UP_BN: gl.constexpr=128, EARLY_A: gl.constexpr=True, EARLY_AS: gl.constexpr=True, PACK_A: gl.constexpr=False, WEIGHT_CACHE: gl.constexpr=''):
    pid = gl.program_id(0)
    COLS: gl.constexpr = N // (UP_BN if UP else 256)
    tile = pid // (GROUP * COLS) * GROUP + pid % GROUP
    panel = pid // GROUP % COLS
    descriptor = gl.load(Info + tile)
    info = descriptor.to(gl.int32)
    packed_rows = (descriptor.to(gl.uint64) >> 32).to(gl.uint32)
    if info != 0:
        expert = info & 511
        live = info >> 9 & 255
        block = info >> 17 & 4095
        column = panel if UP else panel * 2 + (info >> 29)
        if UP:
            if live <= 16:
                _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, UP_BN, 256, ROUTED_BLOCKS, True, 16, EARLY_A, EARLY_AS, PACK_A, WEIGHT_CACHE)
            elif live <= 32:
                _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, UP_BN, 256, ROUTED_BLOCKS, True, 32, EARLY_A, EARLY_AS, PACK_A, WEIGHT_CACHE)
            elif live <= 64:
                _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, UP_BN, 256, ROUTED_BLOCKS, True, 64, EARLY_A, EARLY_AS, PACK_A, WEIGHT_CACHE)
            else:
                _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, column, expert, live, packed_rows, M, N, K, BM, UP_BN, 256, ROUTED_BLOCKS, True, BM, EARLY_A, EARLY_AS, PACK_A, WEIGHT_CACHE)
        elif live <= 16:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, 256, 256, ROUTED_BLOCKS, False, 16, EARLY_A, EARLY_AS, PACK_A, WEIGHT_CACHE)
        elif live <= 32:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, 256, 256, ROUTED_BLOCKS, False, 32, EARLY_A, EARLY_AS, PACK_A, WEIGHT_CACHE)
        elif live <= 64:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, 256, 256, ROUTED_BLOCKS, False, 64, EARLY_A, EARLY_AS, PACK_A, WEIGHT_CACHE)
        else:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, column, expert, live, packed_rows, M, N, K, BM, 128, 256, ROUTED_BLOCKS, False, BM, EARLY_A, EARLY_AS, PACK_A, WEIGHT_CACHE)

@gluon.jit(do_not_specialize=['M'])
def _reduce_parts(P, Y, Records, M, H: gl.constexpr, BLOCK: gl.constexpr, WARPS: gl.constexpr, CACHE: gl.constexpr='', VECTOR: gl.constexpr=1):
    gl.static_assert(BLOCK % 128 == 0)
    m = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([VECTOR], [64], [WARPS], [0])
    inner = gl.arange(0, BLOCK, layout)
    h = gl.program_id(1) * BLOCK + inner
    part_base = P + gl.program_id(1) * M * 8 * BLOCK
    value = gl.full((BLOCK,), 0.0, gl.float32, layout)
    for rank in gl.static_range(8):
        record = gl.load(Records + m * 8 + rank).to(gl.uint64)
        route = record.to(gl.int32)
        weight = (record >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
        address = inner // 128 * M * 8 * 128 + route * 128 + inner % 128
        contribution = gl.amd.cdna4.buffer_load(part_base, address, cache=CACHE).to(gl.float32)
        value += contribution * weight
    value += gl.load(Y + m * H + h).to(gl.float32)
    gl.store(Y + m * H + h, value)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, routed_scaling_factor=2.5):
    m, h = x.shape
    assert 1024 <= m <= 4192
    intermediate = w13.shape[1] // 2
    block_m = 128
    routed_blocks = triton.cdiv(m * 8, block_m) + 256
    blocks = routed_blocks + triton.cdiv(m, block_m)
    chunks = triton.cdiv(m * 8, 256)
    shards = 8
    ticket_stride = _artifact_next_power_of_2(triton.cdiv(m, shards * 64) * 64)
    scheduled_blocks = triton.cdiv(blocks, 8) * 8
    router_n = 16 if m <= 1536 else 32 if m <= 3072 else 64
    router_m = 16 if m <= 1536 else 32
    router_k = 512 if m <= 3072 else 256
    router_warps = 1 if m <= 1536 else 4
    group_up = 2 if m > 4096 else 4 if m < 3072 else 8
    group_down = 4 if m <= 1536 else 2
    wide_down = 64
    down_blocks = triton.cdiv(triton.cdiv(m * 8, 64) + 256 + triton.cdiv(m, 64), 8) * 8

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, device=x.device, dtype=dtype)
    parts = empty((h // 128, m * 8, 128), torch.bfloat16)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    logits = empty((m, 256))
    weights = empty((m, 8), torch.float32)
    records = empty((m, 8), torch.int64)
    partial_counts = empty((shards, 256), torch.int32)
    up_info = empty((scheduled_blocks,), torch.int32)
    jobs = empty((down_blocks,), torch.int64)
    sorted_routes = empty((routed_blocks * block_m + blocks,), torch.int32)
    arena_rows = m * 8 + 256 * 64 + triton.cdiv(m, block_m) * block_m
    aq = empty((arena_rows, intermediate // 2), torch.uint8)
    aqs = empty((arena_rows, intermediate // 32), torch.uint8)
    out = empty((m, h))
    quant_groups = 16 if m <= 1536 else 256 if 2560 < m <= 3072 else 128
    quant_ctas = triton.cdiv(m * (h // 32), quant_groups)
    router_ctas = triton.cdiv(m, router_m) * (256 // router_n)
    _front_end[router_ctas + quant_ctas,](x, router, logits, xq, xs, partial_counts, m, h, x.stride(0), router_m, router_n, router_k, router_warps, quant_groups, shards, ROW_MAJOR=m > 4096, num_warps=router_warps, enable_fp_fusion=False)
    _select_routes[m,](logits, correction_bias, records, weights, partial_counts, shards, ticket_stride, routed_scaling_factor, num_warps=1, enable_fp_fusion=False)
    _prepare_routes[chunks + 258,](records, weights, partial_counts, sorted_routes, up_info, jobs, m, chunks, block_m, routed_blocks, scheduled_blocks, down_blocks, wide_down, shards, ticket_stride, num_warps=1, enable_fp_fusion=False)
    up_bn = 128 if 1536 < m <= 2560 else 256
    up_early_data = m <= 1536
    up_early_scales = m <= 3072 or m > 4096
    up_columns = 2 * intermediate // up_bn
    _scaled_experts[scheduled_blocks * up_columns,](xq, xs, w13, w13_scale, sorted_routes, up_info, aq, aqs, parts, out, m, 2 * intermediate, h, block_m, routed_blocks, group_up, True, up_bn, up_early_data, up_early_scales, m > 1536, WEIGHT_CACHE='.cg' if m <= 1536 else '', enable_fp_fusion=False)
    _scaled_experts[down_blocks * (h // 256),](aq, aqs, w2, w2_scale, sorted_routes, jobs, aq, aqs, parts, out, m, h, intermediate, block_m, routed_blocks, group_down, False, PACK_A=m > 1536, WEIGHT_CACHE='.cg' if m <= 1536 else '')
    _reduce_parts[m, h // 128](parts, out, records, m, h, 128, 1, '.cg', 2, num_warps=1, enable_fp_fusion=False)
    return out
