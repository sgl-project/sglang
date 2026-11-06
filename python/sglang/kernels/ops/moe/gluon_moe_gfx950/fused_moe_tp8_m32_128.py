# Created by OpenAI
# fmt: off
"""GLM-5.2 TP8 fused MoE specialization for M=32..128."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_artifact_next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)

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
def _quantize_input(X, Q, QS, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, CTA_OFFSET: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [1, 0])
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
def _router_linear(X, W, L, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, Counts, Sorted, Jobs, ROUTED_CAPACITY: gl.constexpr, pid_m, pid_n, SPLITS: gl.constexpr, split):
    ROUTE_STRIDE: gl.constexpr = triton.cdiv(M, 16) * 16
    init_shard = pid_m * (256 // BN) + pid_n
    if (split == 0) & (init_shard < ROUTE_STRIDE):
        slots = init_shard * 256 + gl.arange(0, 256, gl.BlockedLayout([1], [64], [1], [0]))
        gl.store(Sorted + slots, -1)
    if (init_shard < 1) & (split == 0):
        counter = gl.arange(0, 256, gl.BlockedLayout([1], [64], [1], [0]))
        gl.store(Counts + init_shard * 256 + counter, 0)
    if (init_shard == 0) & (split == 0):
        gl.store(Counts + 256, 0)
        jobs = gl.arange(0, _artifact_next_power_of_2(ROUTED_CAPACITY), gl.BlockedLayout([1], [64], [1], [0]))
        gl.store(Jobs + jobs, -1, jobs < ROUTED_CAPACITY)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    mi = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    ni = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for step in range(H // BK // SPLITS):
        base = split * (H // BK // SPLITS) + step
        a = gl.load(X + mi[:, None] * SX + (base * BK + ak)[None, :], mi[:, None] < M, 0)
        b = gl.load(W + ni[None, :] * H + (base * BK + bk)[:, None])
        acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8)), acc)
    mm = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    nn = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(L + split * M * 256 + mm[:, None] * 256 + nn[None, :], acc, mm[:, None] < M)

@gluon.jit
def _router_and_quantize(X, W, L, Q, QS, Counts, Sorted, Jobs, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUPS: gl.constexpr, SPLITS: gl.constexpr, ROUTED_CAPACITY: gl.constexpr):
    ROUTER_ROWS: gl.constexpr = triton.cdiv(M, BM)
    ROUTER_CTAS: gl.constexpr = ROUTER_ROWS * (256 // BN) * SPLITS
    pid = gl.program_id(0)
    if pid < ROUTER_CTAS:
        _router_linear(X, W, L, M, H, SX, BM, BN, BK, Counts, Sorted, Jobs, ROUTED_CAPACITY, pid % ROUTER_ROWS, pid // ROUTER_ROWS % (256 // BN), SPLITS, pid // (ROUTER_ROWS * (256 // BN)))
    else:
        _quantize_input(X, Q, QS, M, H, SX, GROUPS, ROUTER_CTAS)

@gluon.jit
def _select_routes(L, Bias, Weights, Counts, Sorted, Jobs, M: gl.constexpr, SPLITS: gl.constexpr, SCALE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    publication_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    m = gl.program_id(0).to(gl.uint32)
    e = gl.arange(0, 256, layout).to(gl.uint32)
    rank = gl.arange(0, 64, publication_layout)
    logit = gl.amd.cdna4.buffer_load(L, m * 256 + e)
    for split in gl.static_range(1, SPLITS):
        logit += gl.amd.cdna4.buffer_load(L, split * M * 256 + m * 256 + e)
    logit = logit.to(gl.bfloat16).to(gl.float32)
    prob = 1.0 / (1.0 + gl.exp(-logit))
    score = prob + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    selected_prob = gl.full((64,), 0.0, gl.float32, publication_layout)
    selected_id = gl.full((64,), 0, gl.int32, publication_layout)
    total = 0.0
    for j in gl.static_range(8):
        maximum = gl.max(score, 0)
        priority = gl.where(available, e + gl.where(score == maximum, 0, 256), 512)
        idx = (gl.min(priority, 0) % 256).to(gl.int32)
        p = gl.sum(gl.gather(prob, gl.full((1,), idx, gl.int32, layout), 0), 0)
        total += p
        selected_prob = gl.where(rank == j, p, selected_prob)
        selected_id = gl.where(rank == j, idx, selected_id)
        available &= e != idx
        score = gl.where(e == idx, -float('inf'), score)
    ticket = gl.atomic_add(Counts + selected_id, 1, rank < 8, sem='relaxed')
    gl.store(Sorted + selected_id * (triton.cdiv(M, 16) * 16) + ticket, m * 8 + rank, rank < 8)
    gl.store(Weights + m * 8 + rank, selected_prob / total * SCALE, rank < 8)
    publish = (rank < 8) & (ticket % 16 == 0)
    number = gl.sum(publish.to(gl.int32), 0)
    if number > 0:
        reservation = gl.atomic_add(Counts + 256 + gl.full((64,), 0, gl.int32, publication_layout), gl.full((64,), number, gl.int32, publication_layout), rank == 0, sem='relaxed')
        base = gl.inline_asm_elementwise('v_readfirstlane_b32 $0, $1;', constraints='=s,v', args=[reservation], dtype=gl.int32, is_pure=True, pack=1)
        mask = gl.inline_asm_elementwise('v_cmp_ne_u32_e64 $0, 0, $1;', constraints='=s,v', args=[publish.to(gl.uint32)], dtype=gl.uint64, is_pure=True, pack=1)
        preceding = mask.to(gl.uint32) & (1 << (rank & 31)) - 1
        offset = gl.inline_asm_elementwise('v_bcnt_u32_b32 $0, $1, 0;', constraints='=v,v', args=[preceding], dtype=gl.int32, is_pure=True, pack=1)
        descriptor = selected_id + ticket // 16 * 256
        gl.store(Jobs + base + offset, descriptor, publish)

@gluon.jit
def _weight_offset(n, k, K: gl.constexpr):
    byte = k // 2
    return (((n // 16 * (K // 64) + byte // 32) * 2 + byte // 16 % 2) * 16 + n % 16) * 16 + byte % 16

@gluon.jit
def _unpack_words(words):
    b0 = words.to(gl.uint8)
    b1 = (words >> 8).to(gl.uint8)
    b2 = (words >> 16).to(gl.uint8)
    b3 = (words >> 24).to(gl.uint8)
    return gl.join(gl.join(b0, b2), gl.join(b1, b3))

@gluon.jit
def _load_packed_weight(W, S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, WEIGHT_CACHE: gl.constexpr, WARPS: gl.constexpr, DIRECT_WEIGHT: gl.constexpr):
    W = W + expert * (N * K // 2)
    S = S + expert * (N * (triton.cdiv(K // 32, 8) * 8))
    packed: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4] if DIRECT_WEIGHT else [32, 2], [WARPS, 1], [0, 1])
    n = gl.arange(0, BN, gl.SliceLayout(1, packed))
    if UP:
        n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)
    else:
        n = column * BN + n
    if UP and BN == 128:
        kw = gl.arange(0, BK // 8, gl.SliceLayout(0, packed))
        offset = n[:, None] // 16 * (2 * K) + (base * (BK // 32) + kw[None, :] // 4) * 64 + n[:, None] % 16 * 4 + kw[None, :] % 4
    else:
        k = base * BK + 8 * gl.arange(0, BK // 8, gl.SliceLayout(0, packed))
        offset = _weight_offset(n[:, None], k[None, :], K) // 4
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset, cache=WEIGHT_CACHE)
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [WARPS], [0])
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
    packed_scales = _unpack_words(raw)
    scale_byte = gl.permute(packed_scales, (0, 5, 3, 1, 4, 2)).reshape((BN, BK // 32))
    return (words, scale_byte)

@gluon.jit
def _scaled_matmul(X, XS, W, WS, Sorted, job, column, expert, live, expert_tile, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, WARPS: gl.constexpr, DIRECT_WEIGHT: gl.constexpr, WEIGHT_CACHE: gl.constexpr, SHARED: gl.constexpr):
    TM: gl.constexpr = 16
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, WARPS])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [TM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [WARPS, 1], [1, 0])
    mi = gl.arange(0, TM, gl.SliceLayout(1, al))
    shared: gl.constexpr = SHARED
    arena_row = job * 16
    if UP:
        if shared:
            token = expert_tile * 16 + mi
        else:
            route = gl.load(Sorted + expert * (triton.cdiv(M, 16) * 16) + expert_tile * 16 + mi)
            token = gl.maximum(route, 0) // 8
        row = gl.where(mi < live, token + gl.where(shared, M, 0), 0)
    else:
        row = arena_row + mi
    ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
    a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    b_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [0, 1])
    a_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 2], a_shared_layout)
    if not DIRECT_WEIGHT:
        b_shared = gl.allocate_shared_memory(gl.uint8, [BK // 2, BN], b_shared_layout)
    if M <= 32:
        bs_shared = gl.allocate_shared_memory(gl.uint8, [WARPS, 4, 16, BN // (16 * WARPS), BK // 128], gl.SwizzledSharedLayout(1, 1, 1, [4, 3, 2, 1, 0]))
        bs_logical = bs_shared.permute((3, 0, 2, 4, 1)).reshape((BN, BK // 32))
    elif not UP:
        bs_shared = gl.allocate_shared_memory(gl.uint8, [BN, BK // 32], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        bs_logical = bs_shared
    as_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 32], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    as_load_layout: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [WARPS, 1], [1, 0])
    scale_row = gl.convert_layout(row, gl.SliceLayout(1, as_load_layout))
    scale_word = gl.arange(0, BK // 128, gl.SliceLayout(0, as_load_layout))
    acc = gl.zeros((TM, BN), gl.float32, mma)
    for base in range(K // BK):
        a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
        a = _unpack_words(a_words).reshape((TM, BK // 2))
        sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + base * (BK // 128) + scale_word[None, :])
        sa_bytes = _unpack_words(sa_words).reshape((TM, BK // 32))
        words, scales = _load_packed_weight(W, WS, expert, column, base, N, K, BN, BK, UP, WEIGHT_CACHE, WARPS, DIRECT_WEIGHT)
        if M > 32 and UP:
            sb = gl.convert_layout(scales, bsl)
        b = _unpack_words(words).reshape((BN, BK // 2)).T
        a_shared.store(a)
        if not DIRECT_WEIGHT:
            b_shared.store(b)
        if M <= 32:
            bs_shared.store(scales.reshape((BN // (16 * WARPS), WARPS, 16, BK // 128, 4)).permute((1, 4, 2, 0, 3)))
        elif not UP:
            bs_shared.store(scales)
        as_shared.store(sa_bytes)
        sa = as_shared.load(asl)
        a = a_shared.load(ad)
        if DIRECT_WEIGHT:
            b = gl.convert_layout(b, bd, assert_trivial=True)
        else:
            b = b_shared.load(bd)
        if M <= 32 or not UP:
            sb = bs_logical.load(bsl)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    return acc

@gluon.jit
def _store_activation(acc, Q, QS, arena_row, column, shared, N: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr):
    TM: gl.constexpr = 16
    gate, up = gl.split(gl.permute(gl.reshape(acc, (TM, 2, BN // 2)), (0, 2, 1)))
    if shared:
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
    activated = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
    ep: gl.constexpr = gl.BlockedLayout([1, 4] if BN == 64 else [1, 8], [8, 8], [WARPS, 1], [1, 0])
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

@gluon.jit
def _store_output(acc, Sorted, Parts, Y, column, expert, expert_tile, live, M: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, SHARED: gl.constexpr):
    TM: gl.constexpr = 16
    shared: gl.constexpr = SHARED
    store_layout: gl.constexpr = acc.type.layout if M <= 32 else gl.BlockedLayout([1, 4], [4, 16], [WARPS, 1], [1, 0])
    result = gl.convert_layout(acc, store_layout)
    rr = gl.arange(0, TM, gl.SliceLayout(1, store_layout))
    nn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, store_layout))
    if shared:
        token = expert_tile * 16 + rr
        gl.store(Y + token[:, None] * N + nn[None, :], result, rr[:, None] < live)
    else:
        route = gl.load(Sorted + expert * (triton.cdiv(M, 16) * 16) + expert_tile * 16 + rr)
        part_base = Parts + column * BN // 128 * M * 8 * 128
        address = route[:, None] * 128 + nn[None, :] % 128
        if BN > 128:
            address += nn[None, :] % BN // 128 * M * 8 * 128
        gl.amd.cdna4.buffer_store(result, part_base, address, route[:, None] >= 0)

@gluon.jit
def _run_expert_job(X, XS, W, WS, Sorted, Q, QS, Parts, Y, job, column, expert, expert_tile, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, WARPS: gl.constexpr, DIRECT_WEIGHT: gl.constexpr, WEIGHT_CACHE: gl.constexpr, SHARED: gl.constexpr):
    live = gl.minimum(16, M - expert_tile * 16) if SHARED else 16
    acc = _scaled_matmul(X, XS, W, WS, Sorted, job, column, expert, live, expert_tile, M, N, K, BN, BK, UP, WARPS, DIRECT_WEIGHT, WEIGHT_CACHE, SHARED)
    if UP:
        _store_activation(acc, Q, QS, job * 16, column, SHARED, N, BN, WARPS)
    else:
        _store_output(acc, Sorted, Parts, Y, column, expert, expert_tile, live, M, N, BN, WARPS, SHARED)

@gluon.jit
def _scaled_experts(X, XS, W, WS, Sorted, Jobs, Q, QS, Parts, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUP: gl.constexpr, UP: gl.constexpr, WARPS: gl.constexpr, DIRECT_WEIGHT: gl.constexpr, WEIGHT_CACHE: gl.constexpr, ROUTED_CAPACITY: gl.constexpr):
    pid = gl.program_id(0)
    COLS: gl.constexpr = N // BN
    SHARED_CTAS: gl.constexpr = triton.cdiv(M, 16) * COLS
    if pid < SHARED_CTAS:
        expert_tile = pid // COLS
        column = pid % COLS
        job = ROUTED_CAPACITY + expert_tile
        _run_expert_job(X, XS, W, WS, Sorted, Q, QS, Parts, Y, job, column, 256, expert_tile, M, N, K, BN, BK, UP, WARPS, DIRECT_WEIGHT, WEIGHT_CACHE, True)
    else:
        routed_pid = pid - SHARED_CTAS
        job = routed_pid // (GROUP * COLS) * GROUP + routed_pid % GROUP
        column = routed_pid // GROUP % COLS
        descriptor = gl.load(Jobs + job)
        if descriptor >= 0:
            expert = descriptor % 256
            expert_tile = descriptor // 256
            _run_expert_job(X, XS, W, WS, Sorted, Q, QS, Parts, Y, job, column, expert, expert_tile, M, N, K, BN, BK, UP, WARPS, DIRECT_WEIGHT, WEIGHT_CACHE, False)

@gluon.jit
def _reduce_parts(P, Y, Weights, M: gl.constexpr, H: gl.constexpr):
    m = gl.program_id(0)
    panel = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    n = gl.arange(0, 128, layout)
    value = gl.full((128,), 0.0, gl.float32, layout)
    for rank in gl.static_range(8):
        weight = gl.load(Weights + m * 8 + rank)
        address = panel * M * 8 * 128 + (m * 8 + rank) * 128 + n
        contribution = gl.amd.cdna4.buffer_load(P, address, cache='.cg')
        value += contribution * weight
    address = m * H + panel * 128 + n
    value += gl.load(Y + address).to(gl.float32)
    gl.store(Y + address, value)

def _projection_config(m, up_k):
    if m <= 32:
        return ((128, up_k, 2, 2, True, ''), (256, 256, 4, 1, True, ''))
    return ((64, 256, 2, 1, False, ''), (128, 256, 2, 1, True, ''))

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, routed_scaling_factor=2.5):
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    router_splits = (12 if m <= 32 else 8) if h % 6144 == 0 else 1
    router_m, router_n = (16, 16)
    router_k = 512 if m <= 32 and h % 512 == 0 else 256
    up_config, down_config = _projection_config(m, 512 if h % 512 == 0 else 256)
    up_n, up_k, up_warps, up_group, up_direct, up_cache = up_config
    down_n, down_k, down_warps, down_group, down_direct, down_cache = down_config
    quant_groups = 16
    active_expert_bound = min(256, m * 8)
    routed_job_bound = active_expert_bound + (m * 8 - active_expert_bound) // 16
    routed_capacity = triton.cdiv(routed_job_bound, 4) * 4
    jobs = routed_capacity + triton.cdiv(m, 16)

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, device=x.device, dtype=dtype)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    logits = empty((router_splits, m, 256), torch.float32)
    weights = empty((m, 8), torch.float32)
    counts = empty((257,), torch.int32)
    sorted_routes = empty((256, triton.cdiv(m, 16) * 16), torch.int32)
    job_info = empty((jobs,), torch.int32)
    aq = empty((jobs * 16, intermediate // 2), torch.uint8)
    aqs = empty((jobs * 16, intermediate // 32), torch.uint8)
    parts = empty((h // 128, m * 8, 128), torch.float32)
    out = empty((m, h))
    quant_ctas = triton.cdiv(m * (h // 32), quant_groups)
    router_ctas = triton.cdiv(m, router_m) * (256 // router_n) * router_splits
    _router_and_quantize[router_ctas + quant_ctas,](x, router, logits, xq, xs, counts, sorted_routes, job_info, m, h, x.stride(0), router_m, router_n, router_k, quant_groups, router_splits, routed_capacity, num_warps=1, enable_fp_fusion=False)
    _select_routes[m,](logits, correction_bias, weights, counts, sorted_routes, job_info, m, router_splits, routed_scaling_factor, num_warps=1, enable_fp_fusion=False)
    _scaled_experts[jobs * (2 * intermediate // up_n),](xq, xs, w13, w13_scale, sorted_routes, job_info, aq, aqs, parts, out, m, 2 * intermediate, h, up_n, up_k, up_group, True, up_warps, up_direct, up_cache, routed_capacity, num_warps=up_warps, enable_fp_fusion=False)
    _scaled_experts[jobs * (h // down_n),](aq, aqs, w2, w2_scale, sorted_routes, job_info, aq, aqs, parts, out, m, h, intermediate, down_n, down_k, down_group, False, down_warps, down_direct, down_cache, routed_capacity, num_warps=down_warps)
    _reduce_parts[m, h // 128](parts, out, weights, m, h, num_warps=1, enable_fp_fusion=False)
    return out
