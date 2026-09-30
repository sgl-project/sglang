# fmt: off
"""Shared TP4 fused-MoE specialization for active batches M=128 through M=4192."""

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

@gluon.jit
def _quantize_input(X, Q, QS, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, CTA_OFFSET: gl.constexpr=0, WARPS: gl.constexpr=1):
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

@gluon.jit
def _router_linear(X, W, L, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, Counts, pid_m, pid_n, WARPS: gl.constexpr=4, INIT_SHARDS: gl.constexpr=8):
    init_shard = pid_m * (256 // BN) + pid_n
    if init_shard < INIT_SHARDS:
        counter = gl.arange(0, 256, gl.BlockedLayout([1], [64], [WARPS], [0]))
        gl.store(Counts + init_shard * 256 + counter, 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1] if WARPS == 1 else [2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WARPS, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, WARPS], [0, 1])
    mi = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    ni = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for base in range(H // BK):
        a = gl.load(X + mi[:, None] * SX + (base * BK + ak)[None, :], mi[:, None] < M, 0)
        b = gl.load(W + ni[None, :] * H + (base * BK + bk)[:, None])
        acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8)), acc)
    mm = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    nn = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(L + mm[:, None] * 256 + nn[None, :], acc, mm[:, None] < M)

@gluon.constexpr_function
def _router_grid(m, block_m, block_n):
    rows = triton.cdiv(m, block_m)
    grouped = m > 3584 and rows % 32 != 0
    padded_rows = triton.cdiv(rows, 32) * 32 if grouped else rows
    return (rows, grouped, padded_rows * (256 // block_n))

@gluon.jit
def _router_quantize(X, W, L, Counts, Q, QS, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, WARPS: gl.constexpr, SHARDS: gl.constexpr, GROUPS: gl.constexpr):
    pid = gl.program_id(0)
    grid: gl.constexpr = _router_grid(M, BM, BN)
    ROWS: gl.constexpr = grid[0]
    GROUP_ROUTER: gl.constexpr = grid[1]
    ROUTER_CTAS: gl.constexpr = grid[2]
    if pid < ROUTER_CTAS:
        if GROUP_ROUTER:
            router_row = pid // (32 * (256 // BN)) * 32 + pid % 32
            router_column = pid // 32 % (256 // BN)
            if router_row * BM < M:
                _router_linear(X, W, L, M, H, SX, BM, BN, BK, Counts, router_row, router_column, WARPS, SHARDS)
        else:
            _router_linear(X, W, L, M, H, SX, BM, BN, BK, Counts, pid % ROWS, pid // ROWS, WARPS, SHARDS)
    else:
        _quantize_input(X, Q, QS, M, H, SX, GROUPS, ROUTER_CTAS, WARPS)

@gluon.jit
def _select_routes(L, Bias, Ids, Counts, expert_start, SHARDS: gl.constexpr, TICKET_STRIDE: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, SCALE: gl.constexpr):
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
        priority = gl.where(available, e + gl.where(score == maximum, 0, 256), 512)
        idx = gl.min(priority, 0) % 256
        p = gl.sum(gl.gather(prob, gl.full((1,), idx, gl.int32, layout), 0), 0)
        total += p
        selected_prob = gl.where(e == j, p, selected_prob)
        selected_id = gl.where(e == j, idx, selected_id)
        available &= e != idx
        score = gl.where(e == idx, -float('inf'), score)
    ticket = gl.atomic_add(Counts + m // 64 % SHARDS * 256 + selected_id, 1, e < 8, sem='relaxed')
    owned = (selected_id >= expert_start) & (selected_id < expert_start + LOCAL_EXPERTS)
    weight = gl.where(owned, selected_prob / total * SCALE, 0.0)
    record = (selected_id * TICKET_STRIDE + ticket).to(gl.uint64)
    record |= weight.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
    gl.store(Ids + m * 8 + e, record, e < 8)

@gluon.jit
def _write_height_descriptors(UpInfo, e, experts, counts, offset, count, M: gl.constexpr, BM: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(M, BM)), layout)
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
            shared_higher = gl.constexpr(M // BM + (M % BM > height))
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

@gluon.jit
def _write_down_tiles(Jobs, e, experts, counts, offset, count, arena_start, M: gl.constexpr, BM: gl.constexpr, WIDE: gl.constexpr=32):
    gl.static_assert(9 * M + 256 * (BM // 2) + BM < 65536)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    remainder = counts % BM
    sizes = counts // BM * 2 + gl.where(remainder > WIDE, 2, (remainder > 0).to(gl.int32))
    start = gl.sum(gl.where(experts < e, sizes, 0), 0)
    b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(M, BM)), layout)
    live = gl.minimum(BM, count - b * BM)
    dense_start = gl.sum(gl.where(experts < e, counts, 0), 0)
    descriptor = (e | live << 9 | offset + b << 17).to(gl.uint64)
    descriptor |= (dense_start + b * BM).to(gl.uint64) << 32
    descriptor |= (arena_start + b * BM).to(gl.uint64) << 48
    gl.store(Jobs + start + b * 2, descriptor, b < gl.cdiv(count, BM))
    gl.store(Jobs + start + b * 2 + 1, descriptor | 1 << 29, (b < gl.cdiv(count, BM)) & (live > WIDE))

@gluon.jit
def _prepare_tickets(Codes, Counts, Sorted, UpInfo, Jobs, M: gl.constexpr, CHUNKS: gl.constexpr, BM: gl.constexpr, ROUTED_BLOCKS: gl.constexpr, SCHEDULED: gl.constexpr, DOWN_SCHEDULED: gl.constexpr, WIDE: gl.constexpr=32, SHARDS: gl.constexpr=8, TICKET_STRIDE: gl.constexpr=1024, SKIP_SHARED_DOWN: gl.constexpr=False):
    pid = gl.program_id(0)
    if pid < CHUNKS:
        chunk = pid
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        lane = gl.arange(0, 256, layout)
        route = chunk * 256 + lane
        ticket_record = gl.load(Codes + route, route < M * 8, 0).to(gl.uint64)
        code = ticket_record.to(gl.int32)
        expert, ticket = (code // TICKET_STRIDE, code % TICKET_STRIDE)
        counts, prefix_counts = _sum_shard_counts(Counts, SHARDS, chunk // 2 % SHARDS, True)
        tiles = gl.cdiv(counts, BM)
        dense_offsets = gl.associative_scan(counts, 0, _add) - counts
        dense = gl.gather(dense_offsets, expert, 0)
        offsets = (gl.associative_scan(tiles, 0, _add) - tiles) * BM
        offset = gl.gather(offsets, expert, 0)
        prefix = gl.gather(prefix_counts, expert, 0)
        gl.store(Sorted + offset + prefix + ticket, route, route < M * 8)
        record = (dense + prefix + ticket).to(gl.uint64)
        record |= ticket_record & 18446744069414584320
        gl.store(Codes + route, record, route < M * 8)
    else:
        e = pid - CHUNKS
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        experts = gl.arange(0, 256, layout)
        counts = _sum_shard_counts(Counts, SHARDS)
        tiles = gl.cdiv(counts, BM)
        if e == 257:
            active = gl.sum(tiles, 0)
            hole = gl.arange(0, _artifact_next_power_of_2(SCHEDULED), layout)
            gl.store(UpInfo + hole, 0, (hole >= active + triton.cdiv(M, BM)) & (hole < SCHEDULED))
            remainder = counts % BM
            sizes = counts // BM * 2 + gl.where(remainder > WIDE, 2, (remainder > 0).to(gl.int32))
            shared_tiles: gl.constexpr = M // BM * 2 + (2 if M % BM > WIDE else 1 if M % BM > 0 else 0)
            active_tiles = gl.sum(sizes, 0)
            if not SKIP_SHARED_DOWN:
                active_tiles += shared_tiles
            holes = gl.arange(0, _artifact_next_power_of_2(DOWN_SCHEDULED), layout)
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
            if (not SKIP_SHARED_DOWN) | (e < 256):
                _write_down_tiles(Jobs, e, experts, counts, offset, count, arena_start, M, BM, WIDE)
            b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(M, BM)), layout)
            gl.store(Sorted + ROUTED_BLOCKS * BM + offset + b, arena_start + b * BM, b < gl.cdiv(count, BM))

@gluon.constexpr_function
def _packed_weight_layout(rows, block_k):
    bases = [[1, 0], [2, 0], [4, 0], [8, 0]]
    bases += [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0]]
    bases += [[32 << bit, 0] for bit in range((block_k // 64).bit_length() - 1)]
    bases += [[0, 16 << bit] for bit in range((rows // 16).bit_length() - 1)]
    return gl.SharedLinearLayout(bases)

@gluon.jit
def _load_packed_weight(W, S, BShared, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, CACHE: gl.constexpr=''):
    W = W + expert * (N * K // 2)
    S = S + expert * (N * (triton.cdiv(K // 32, 8) * 8))
    packed: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    word = gl.arange(0, BN * BK // 8, packed)
    panel = word // (BK * 2)
    if UP:
        panel = column * (BN // 32) + panel % (BN // 32) + panel // (BN // 32) * (N // 32)
    else:
        panel = column * (BN // 16) + panel
    offset = panel * (K * 2) + base * (BK * 2) + word % (BK * 2)
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset, cache=CACHE)
    BShared.store(words)
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
    sw = sw & 4278190335 | sw >> 8 & 65280 | sw << 8 & 16711680
    return sw

@gluon.jit
def _prefetch_packed_weight(W, BShared, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, CACHE: gl.constexpr):
    packed: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    word = gl.arange(0, BN * BK // 8, packed)
    panel = word // (BK * 2)
    panel = column * (BN // 32) + panel % (BN // 32) + panel // (BN // 32) * (N // 32)
    offset = panel * (K * 2) + base * (BK * 2) + word % (BK * 2)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(BShared, (W + expert * (N * K // 2)).to(gl.pointer_type(gl.uint32)), offset, cache_modifier=CACHE)
    gl.amd.cdna4.async_copy.commit_group()

@gluon.jit
def _load_up_weight_scales(S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    S = S + expert * (N * (triton.cdiv(K // 32, 8) * 8))
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    idx = gl.arange(0, BN * BK // 128, sl)
    nb = idx // (BK // 4)
    nb = column * (BN // 64) + nb % (BN // 64) + nb // (BN // 64) * (N // 64)
    kg = idx // 64 % (BK // 256)
    inner = idx % 64
    sw = gl.amd.cdna4.buffer_load(S.to(gl.pointer_type(gl.uint32)), nb * (K // 4) + base * (BK // 4) + kg * 64 + inner)
    return sw & 4278190335 | sw >> 8 & 65280 | sw << 8 & 16711680

@gluon.jit
def _word_bytes(words):
    b0 = words.to(gl.uint8)
    b1 = (words >> 8).to(gl.uint8)
    b2 = (words >> 16).to(gl.uint8)
    b3 = (words >> 24).to(gl.uint8)
    return gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((words.shape[0], words.shape[1] * 4))

@gluon.constexpr_function
def _paired_scale_layout(rows, block_k, weight):
    if weight:
        bases = [[0, 4], [16, 0], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]]
        bases += [[0, 8 << bit] for bit in range((block_k // 256).bit_length() - 1)]
        bases += [[32 << bit, 0] for bit in range((rows // 32).bit_length() - 1)]
    else:
        bases = [[0, 4], [0, 1], [0, 2]]
        bases += [[0, 8 << bit] for bit in range((block_k // 256).bit_length() - 1)]
        bases += [[1 << bit, 0] for bit in range(rows.bit_length() - 1)]
    return gl.SharedLinearLayout(bases)

@gluon.jit
def _store_up_projection(acc, Q, QS, arena_row, column, shared, N: gl.constexpr, BN: gl.constexpr, TM: gl.constexpr, BF16_HANDOFF: gl.constexpr=False):
    gate, up = gl.split(gl.permute(gl.reshape(acc, (TM, 2, BN // 2)), (0, 2, 1)))
    if shared:
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
    activated = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16)
    if not BF16_HANDOFF:
        activated = activated.to(gl.float32)
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    activated = gl.convert_layout(activated, ep).to(gl.float32)
    codes, scales = _encode_groups(gl.reshape(activated, (TM * (BN // 64), 32)), shared)
    codes = codes.reshape((TM, BN // 4))
    scales = scales.reshape((TM, BN // 64))
    rr = gl.arange(0, TM, gl.SliceLayout(1, codes.type.layout))
    nn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, codes.type.layout))
    gl.store(Q + (arena_row + rr[:, None]) * (N // 4) + nn[None, :], codes)
    rr_s = gl.arange(0, TM, gl.SliceLayout(1, scales.type.layout))
    nn_s = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, scales.type.layout))
    gl.store(QS + (arena_row + rr_s[:, None]) * (N // 64) + nn_s[None, :], scales)

@gluon.jit
def _store_down_projection(acc, Parts, Y, block, column, live, dense_base, shared, M: gl.constexpr, N: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, TM: gl.constexpr, ROUTED_BLOCKS: gl.constexpr):
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
        gl.amd.cdna4.buffer_store(result.to(Parts.dtype.element_ty), part_base, address, rr[:, None] < live, cache='.wt' if M <= 2560 else '')

@gluon.constexpr_function
def _direct_weight_words_layout(block_n, block_k):
    registers = [[1, 0], [2, 0]]
    registers += [[16 << bit, 0] for bit in range((block_k // 128).bit_length() - 1)]
    registers += [[0, 64 << bit] for bit in range((block_n // 64).bit_length() - 1)]
    return gl.DistributedLinearLayout(reg_bases=[[n, k] for k, n in registers], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp_bases=[[16, 0], [32, 0]], block_bases=[], shape=[block_n, block_k // 8])

@gluon.jit
def _load_direct_weight(W, S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BD: gl.constexpr, CACHE: gl.constexpr, UP: gl.constexpr=False):
    layout: gl.constexpr = _direct_weight_words_layout(BN, BK)
    kw = gl.arange(0, BK // 8, gl.SliceLayout(0, layout))
    nn = gl.arange(0, BN, gl.SliceLayout(1, layout))
    if UP:
        nn = column * (BN // 2) + nn % (BN // 2) + nn // (BN // 2) * (N // 2)
    else:
        nn = column * BN + nn
    offsets = nn[:, None] // 16 * (K * 2) + base * (BK * 2) + kw[None, :] // 8 * 128 + kw[None, :] // 4 % 2 * 64 + nn[:, None] % 16 * 4 + kw[None, :] % 4
    offsets = gl.max_contiguous(gl.multiple_of(offsets.reshape((BN * BK // 8,)), 4), 4)
    words = gl.amd.cdna4.buffer_load((W + expert * (N * K // 2)).to(gl.pointer_type(gl.uint32)), offsets, cache=CACHE)
    payload = _word_bytes(words.reshape((BN, BK // 8))).permute((1, 0))
    b = gl.convert_layout(payload, BD, assert_trivial=True)
    if UP:
        return (b, _load_up_weight_scales(S, expert, column, base, N, K, BN, BK))
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    idx = gl.arange(0, BN * BK // 128, sl)
    nb = column * (BN // 32) + idx // (BK // 4)
    kg = idx // 64 % (BK // 256)
    sw = gl.amd.cdna4.buffer_load((S + expert * (N * (triton.cdiv(K // 32, 8) * 8))).to(gl.pointer_type(gl.uint32)), nb * (K // 4) + base * (BK // 4) + kg * 64 + idx % 64)
    scales = sw & 4278190335 | sw >> 8 & 65280 | sw << 8 & 16711680
    return (b, scales)

@gluon.jit
def _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, column, expert, live, packed_rows, expert_start, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, ROUTED_BLOCKS: gl.constexpr, UP: gl.constexpr, TM: gl.constexpr, LOCAL_EXPERTS: gl.constexpr):
    DIRECT_UP: gl.constexpr = UP and TM > 64 and (M > 1536)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=not (UP and M > 2560), warps_per_cta=[1, 4] if DIRECT_UP or TM <= 64 else [2, 2], tiles_per_warp=[1, 2] if UP and (not DIRECT_UP) and (M <= 1536 or M > 2560) else [1, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [TM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
    mi = gl.arange(0, TM, gl.SliceLayout(1, al))
    shared = expert == 256
    owned = (expert >= expert_start) & (expert < expert_start + LOCAL_EXPERTS)
    weight_expert = gl.where(owned, expert - expert_start, LOCAL_EXPERTS)
    if UP:
        arena_row = gl.load(Sorted + ROUTED_BLOCKS * BM + block)
        if shared:
            route = ((block - ROUTED_BLOCKS) * BM + mi) * 8
        else:
            route = gl.load(Sorted + block * BM + mi, mi < live, 0)
        row = gl.where(mi < live, route // 8 + gl.where(shared, M, 0), 0)
        dense_base = 0
    else:
        arena_row = (packed_rows >> 16).to(gl.int32)
        dense_base = (packed_rows & 65535).to(gl.int32)
        row = arena_row + mi
    ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
    if M <= 1536:
        a_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BK // 2, 32]], [TM, BK // 2], [1, 0])
    else:
        a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    a_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 2], a_shared_layout)
    DIRECT_B: gl.constexpr = not UP and TM <= 64 or DIRECT_UP
    PIPELINED: gl.constexpr = UP and TM <= 64
    if PIPELINED:
        b_slots = gl.allocate_shared_memory(gl.uint32, [2, BN * BK // 8], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    elif not DIRECT_B:
        b_shared = gl.allocate_shared_memory(gl.uint32, [BN * BK // 8], gl.SwizzledSharedLayout(1, 1, 1, [0]))
        b_native = b_shared._reinterpret(gl.uint8, [BK // 2, BN], _packed_weight_layout(BN, BK))
    bs_shared = gl.allocate_shared_memory(gl.uint32, [BN * BK // 128], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    bs_layout: gl.constexpr = _paired_scale_layout(BN, BK, True)
    bs_native = bs_shared._reinterpret(gl.uint8, [BN, BK // 32], bs_layout)
    if M > 1536:
        as_layout: gl.constexpr = _paired_scale_layout(TM, BK, False)
    else:
        as_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    as_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 32], as_layout)
    as_load_layout: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [4, 1], [1, 0])
    scale_row = gl.convert_layout(row, gl.SliceLayout(1, as_load_layout))
    scale_word = gl.arange(0, BK // 128, gl.SliceLayout(0, as_load_layout))
    acc = gl.zeros((TM, BN), gl.float32, mma)
    if PIPELINED:
        if M <= 1536:
            a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + ki[None, :])
            sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + scale_word[None, :])
            scales = _load_up_weight_scales(WS, weight_expert, column, 0, N, K, BN, BK)
            _prefetch_packed_weight(W, b_slots.index(0), weight_expert, column, 0, N, K, BN, BK, '.cg')
            for base in range(K // BK):
                gl.amd.cdna4.async_copy.wait_group(0)
                gl.barrier()
                a_shared.store(_word_bytes(a_words))
                as_shared.store(_word_bytes(sa_words))
                bs_shared.store(scales)
                b_current = b_slots.index(base % 2)._reinterpret(gl.uint8, [BK // 2, BN], _packed_weight_layout(BN, BK))
                a = a_shared.load(ad)
                sa = as_shared.load(asl)
                b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_current, bd)
                sb = bs_native.load(bsl)
                if base + 1 < K // BK:
                    a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + (base + 1) * (BK // 8) + ki[None, :])
                    sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + (base + 1) * (BK // 128) + scale_word[None, :])
                    scales = _load_up_weight_scales(WS, weight_expert, column, base + 1, N, K, BN, BK)
                    _prefetch_packed_weight(W, b_slots.index((base + 1) % 2), weight_expert, column, base + 1, N, K, BN, BK, '.cg')
                acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
        else:
            _prefetch_packed_weight(W, b_slots.index(0), weight_expert, column, 0, N, K, BN, BK, '.cg')
            for base in range(K // BK):
                a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
                sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + base * (BK // 128) + scale_word[None, :])
                scales = _load_up_weight_scales(WS, weight_expert, column, base, N, K, BN, BK)
                gl.amd.cdna4.async_copy.wait_group(0)
                gl.barrier()
                if base + 1 < K // BK:
                    _prefetch_packed_weight(W, b_slots.index((base + 1) % 2), weight_expert, column, base + 1, N, K, BN, BK, '.cg')
                a_shared.store(_word_bytes(a_words))
                as_shared.store(_word_bytes(sa_words))
                bs_shared.store(scales)
                b_current = b_slots.index(base % 2)._reinterpret(gl.uint8, [BK // 2, BN], _packed_weight_layout(BN, BK))
                a = a_shared.load(ad)
                sa = as_shared.load(asl)
                b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_current, bd)
                sb = bs_native.load(bsl)
                acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    else:
        for base in range(K // BK):
            a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
            a = _word_bytes(a_words)
            sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + base * (BK // 128) + scale_word[None, :])
            sa_bytes = _word_bytes(sa_words)
            if M > 1536:
                a_shared.store(a)
                as_shared.store(sa_bytes)
            CACHE: gl.constexpr = '.cg' if M <= 1536 or (not UP and (M <= 2560 or TM == 16 or (M > 3584 and M <= 4096 and (TM <= 64)))) else ''
            if DIRECT_B:
                b, scales = _load_direct_weight(W, WS, weight_expert, column, base, N, K, BN, BK, bd, CACHE, UP)
            else:
                scales = _load_packed_weight(W, WS, b_shared, weight_expert, column, base, N, K, BN, BK, UP, CACHE)
            if M <= 1536:
                a_shared.store(a)
            bs_shared.store(scales)
            if M <= 1536:
                as_shared.store(sa_bytes)
            sa = as_shared.load(asl)
            a = a_shared.load(ad)
            if not DIRECT_B:
                b = b_native.load(bd)
            sb = bs_native.load(bsl)
            acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    if UP:
        _store_up_projection(acc, Q, QS, arena_row, column, shared, N, BN, TM, BF16_HANDOFF=M > 2560)
    else:
        _store_down_projection(acc, Parts, Y, block, column, live, dense_base, shared, M, N, BM, BN, TM, ROUTED_BLOCKS)

@gluon.jit
def _scaled_experts(X, XS, W, WS, Sorted, Info, Q, QS, Parts, Y, expert_start, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, ROUTED_BLOCKS: gl.constexpr, GROUP: gl.constexpr, UP: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, UP_BN: gl.constexpr=128):
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
        SHORT_BN: gl.constexpr = UP_BN if UP else 256
        TALL_BN: gl.constexpr = UP_BN if UP else 128
        if live <= 16:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, expert_start, M, N, K, BM, SHORT_BN, 256, ROUTED_BLOCKS, UP, 16, LOCAL_EXPERTS)
        elif live <= 32:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, expert_start, M, N, K, BM, SHORT_BN, 256, ROUTED_BLOCKS, UP, 32, LOCAL_EXPERTS)
        elif live <= 64:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, expert_start, M, N, K, BM, SHORT_BN, 256, ROUTED_BLOCKS, UP, 64, LOCAL_EXPERTS)
        else:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, column, expert, live, packed_rows, expert_start, M, N, K, BM, TALL_BN, 256, ROUTED_BLOCKS, UP, BM, LOCAL_EXPERTS)

@gluon.jit
def _sum_routed_ranks(Parts, records, total, column, M: gl.constexpr, TM: gl.constexpr, BN: gl.constexpr, BEGIN: gl.constexpr, END: gl.constexpr):
    output_layout: gl.constexpr = total.type.layout
    record_layout: gl.constexpr = records.type.layout
    nn = gl.arange(0, BN, gl.SliceLayout(0, output_layout))
    for rank in gl.static_range(BEGIN, END):
        rank_index = gl.full((TM, 1), rank, gl.int32, record_layout)
        record = gl.gather(records, rank_index, 1).reshape((TM,))
        record = gl.convert_layout(record, gl.SliceLayout(1, output_layout)).to(gl.uint64)
        dense_row = record.to(gl.int32)
        weight = (record >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
        address = dense_row[:, None] * 128 + nn[None, :]
        part = gl.amd.cdna4.buffer_load(Parts + column * M * 8 * 128, address, cache='.cg').to(gl.float32)
        total += part * weight[:, None]
    return total

@gluon.jit
def _shared_finish(X, XS, W, WS, Sorted, Parts, Records, Y, M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, ROUTED_BLOCKS: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, TM: gl.constexpr=32, BN: gl.constexpr=128):
    WARPS: gl.constexpr = 4
    DIRECT: gl.constexpr = M <= 1536
    EARLY: gl.constexpr = 0 if M <= 1536 else 8
    EARLY_POSITION: gl.constexpr = 1 if M > 1536 and M <= 2560 else 0
    ROW_LANES: gl.constexpr = 4 if M > 3584 else 8
    ELEMENTS: gl.constexpr = 8
    gl.static_assert(BN == 128)
    BK: gl.constexpr = 512 if K % 512 == 0 else 256
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, WARPS])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [TM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [WARPS, 1], [1, 0])
    output_layout: gl.constexpr = gl.BlockedLayout([1, ELEMENTS], [ROW_LANES, 64 // ROW_LANES], [WARPS, 1], [1, 0])
    record_layout: gl.constexpr = gl.BlockedLayout([1, 1], [ROW_LANES, 64 // ROW_LANES], [WARPS, 1], [1, 0])
    column = gl.program_id(1)
    if EARLY > 0:
        mt = gl.program_id(0) * TM + gl.arange(0, TM, gl.SliceLayout(1, record_layout))
        ranks = gl.arange(0, 8, gl.SliceLayout(0, record_layout))
        records = gl.load(Records + mt[:, None] * 8 + ranks[None, :], mt[:, None] < M, 0)
        total = gl.zeros((TM, BN), gl.float32, output_layout)
        if EARLY_POSITION == 0:
            total = _sum_routed_ranks(Parts, records, total, column, M, TM, BN, 0, EARLY)
    token = gl.program_id(0) * TM + gl.arange(0, TM, gl.SliceLayout(1, al))
    shared_start = gl.load(Sorted + ROUTED_BLOCKS * BM + ROUTED_BLOCKS)
    row = shared_start + gl.minimum(token, M - 1)
    ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
    a_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 2], gl.SwizzledSharedLayout(16, 1, 8, [1, 0]))
    if not DIRECT:
        b_shared = gl.allocate_shared_memory(gl.uint32, [BN * BK // 8], gl.SwizzledSharedLayout(1, 1, 1, [0]))
        b_native = b_shared._reinterpret(gl.uint8, [BK // 2, BN], _packed_weight_layout(BN, BK))
    bs_shared = gl.allocate_shared_memory(gl.uint32, [BN * BK // 128], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    bs_native = bs_shared._reinterpret(gl.uint8, [BN, BK // 32], _paired_scale_layout(BN, BK, True))
    as_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 32], _paired_scale_layout(TM, BK, False))
    as_load: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [WARPS, 1], [1, 0])
    scale_row = gl.convert_layout(row, gl.SliceLayout(1, as_load))
    scale_word = gl.arange(0, BK // 128, gl.SliceLayout(0, as_load))
    acc = gl.zeros((TM, BN), gl.float32, mma)
    for base in range(K // BK):
        a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
        sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + base * (BK // 128) + scale_word[None, :])
        a_shared.store(_word_bytes(a_words))
        as_shared.store(_word_bytes(sa_words))
        if DIRECT:
            b, scales = _load_direct_weight(W, WS, LOCAL_EXPERTS, column, base, H, K, BN, BK, bd, '')
        else:
            scales = _load_packed_weight(W, WS, b_shared, LOCAL_EXPERTS, column, base, H, K, BN, BK, False, '')
        bs_shared.store(scales)
        if not DIRECT:
            b = b_native.load(bd)
        if EARLY > 0 and EARLY_POSITION == 1:
            if base == 0:
                total = _sum_routed_ranks(Parts, records, total, column, M, TM, BN, 0, EARLY)
        acc = gl.amd.cdna4.mfma_scaled(a_shared.load(ad), as_shared.load(asl), 'e2m1', b, bs_native.load(bsl), 'e2m1', acc)
    if EARLY == 0:
        mt = gl.program_id(0) * TM + gl.arange(0, TM, gl.SliceLayout(1, record_layout))
        ranks = gl.arange(0, 8, gl.SliceLayout(0, record_layout))
        records = gl.load(Records + mt[:, None] * 8 + ranks[None, :], mt[:, None] < M, 0)
        total = gl.zeros((TM, BN), gl.float32, output_layout)
    total = _sum_routed_ranks(Parts, records, total, column, M, TM, BN, EARLY, 8)
    total += gl.convert_layout(acc.to(gl.bfloat16).to(gl.float32), output_layout)
    mm = gl.program_id(0) * TM + gl.arange(0, TM, gl.SliceLayout(1, output_layout))
    nn = gl.arange(0, BN, gl.SliceLayout(0, output_layout))
    gl.amd.cdna4.buffer_store(total.to(Y.dtype.element_ty), Y + column * BN, mm[:, None] * H + nn[None, :], mm[:, None] < M)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, expert_start=0, routed_scaling_factor=2.5, fuse_shared_expert=False):
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    local_experts = w13.shape[0] - 1
    block_m = 128
    routed_blocks = triton.cdiv(m * 8, block_m) + 256
    blocks = routed_blocks + triton.cdiv(m, block_m)
    chunks = triton.cdiv(m * 8, 256)
    shards = 8
    ticket_stride = _artifact_next_power_of_2(triton.cdiv(m, shards * 64) * 64)
    scheduled_blocks = triton.cdiv(blocks, 8) * 8
    router_n = 32 if m <= 3072 else 64
    router_m = 32
    router_k = 512 if m <= 3072 else 256
    router_warps = 4
    quant_groups = 256 if 2560 < m <= 3584 else 128
    up_bn = 128 if 1536 < m <= 2560 else 256
    group_up = 8 if m <= 2560 else 1 if m <= 3584 else 2
    group_down = 2
    wide_down = 64
    down_blocks = triton.cdiv(triton.cdiv(m * 8, 64) + 256, 8) * 8

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, device=x.device, dtype=dtype)
    parts = empty((h // 128, m * 8, 128), torch.bfloat16)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    logits = empty((m, 256))
    records = empty((m, 8), torch.int64)
    partial_counts = empty((shards, 256), torch.int32)
    up_info = empty((scheduled_blocks,), torch.int32)
    jobs = empty((down_blocks,), torch.int64)
    sorted_routes = empty((routed_blocks * block_m + blocks,), torch.int32)
    arena_rows = m * 8 + 256 * 64 + triton.cdiv(m, block_m) * block_m
    aq = empty((arena_rows, intermediate // 2), torch.uint8)
    aqs = empty((arena_rows, intermediate // 32), torch.uint8)
    out = empty((m, h))
    quant_ctas = triton.cdiv(m * (h // 32), quant_groups)
    _, _, router_ctas = _router_grid(m, router_m, router_n)
    _router_quantize[router_ctas + quant_ctas,](x, router, logits, partial_counts, xq, xs, m, h, x.stride(0), router_m, router_n, router_k, router_warps, shards, quant_groups, num_warps=router_warps, enable_fp_fusion=False)
    _select_routes[m,](logits, correction_bias, records, partial_counts, expert_start, shards, ticket_stride, local_experts, routed_scaling_factor, num_warps=1, enable_fp_fusion=False)
    _prepare_tickets[chunks + 258,](records, partial_counts, sorted_routes, up_info, jobs, m, chunks, block_m, routed_blocks, scheduled_blocks, down_blocks, wide_down, shards, ticket_stride, SKIP_SHARED_DOWN=True, num_warps=1, enable_fp_fusion=False)
    up_columns = 2 * intermediate // up_bn
    _scaled_experts[scheduled_blocks * up_columns,](xq, xs, w13, w13_scale, sorted_routes, up_info, aq, aqs, parts, out, expert_start, m, 2 * intermediate, h, block_m, routed_blocks, group_up, True, local_experts, up_bn, enable_fp_fusion=False)
    _scaled_experts[down_blocks * (h // 256),](aq, aqs, w2, w2_scale, sorted_routes, jobs, aq, aqs, parts, out, expert_start, m, h, intermediate, block_m, routed_blocks, group_down, False, local_experts)
    _shared_finish[triton.cdiv(m, 32), h // 128](aq, aqs, w2, w2_scale, sorted_routes, parts, records, out, m, h, intermediate, block_m, routed_blocks, local_experts, enable_fp_fusion=False)
    return out
