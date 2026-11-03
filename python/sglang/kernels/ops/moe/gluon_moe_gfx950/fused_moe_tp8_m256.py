# Created by OpenAI
# fmt: off
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
def _router_linear(X, W, L, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, pid_m, pid_n, SPLITS: gl.constexpr, partition):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    mi = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    ni = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for iteration in range(H // (BK * SPLITS)):
        base = partition * (H // (BK * SPLITS)) + iteration
        a = gl.load(X + mi[:, None] * SX + (base * BK + ak)[None, :], mi[:, None] < M, 0)
        b = gl.load(W + ni[None, :] * H + (base * BK + bk)[:, None])
        acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8)), acc)
    mm = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    nn = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(L + (partition * M + mm[:, None]) * 256 + nn[None, :], acc, mm[:, None] < M)

@gluon.jit
def _router_projection(X, W, L, Counts, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, SPLITS: gl.constexpr, COUNT_STRIDE: gl.constexpr):
    ROUTER_ROWS: gl.constexpr = triton.cdiv(M, BM)
    ROUTER_BASE: gl.constexpr = ROUTER_ROWS * (256 // BN)
    pid = gl.program_id(0)
    if pid == 0:
        jl: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        e = gl.arange(0, 256, jl)
        gl.store(Counts + e * COUNT_STRIDE, 0)
    _router_linear(X, W, L, M, H, SX, BM, BN, BK, pid % ROUTER_ROWS, pid % ROUTER_BASE // ROUTER_ROWS, SPLITS, pid // ROUTER_BASE)

@gluon.jit
def _select_routes(L, Bias, Weights, Counts, Sorted, M: gl.constexpr, SPLITS: gl.constexpr, PAD: gl.constexpr, COUNT_STRIDE: gl.constexpr, SCALE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    m = gl.program_id(0)
    e = gl.arange(0, 256, layout)
    logit = gl.full((256,), 0.0, gl.float32, layout)
    for split in gl.static_range(SPLITS):
        logit += gl.load(L + (split * M + m) * 256 + e)
    logit = logit.to(gl.bfloat16).to(gl.float32)
    prob = 1.0 / (1.0 + gl.exp(-logit))
    score = prob + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    selected_prob = gl.full((256,), 0.0, gl.float32, layout)
    selected_id = gl.full((256,), 0, gl.int32, layout)
    total = 0.0
    maximum = gl.max(score, 0)
    candidate = gl.where(score == maximum, e, e + 256)
    idx = gl.min(gl.where(available, candidate, 512), 0) % 256
    for j in gl.static_range(8):
        available &= e != idx
        score = gl.where(e == idx, -float('inf'), score)
        if j < 7:
            maximum = gl.max(score, 0)
            candidate = gl.where(score == maximum, e, e + 256)
            next_idx = gl.min(gl.where(available, candidate, 512), 0) % 256
        local_prob = gl.sum(gl.where(e // 64 == idx // 64, prob, 0.0).reshape((4, 64)), 0)
        index = gl.full((1,), idx % 64, gl.int32, local_prob.type.layout)
        p = gl.sum(gl.gather(local_prob, index, 0), 0)
        total += p
        selected_prob = gl.where(e == j, p, selected_prob)
        selected_id = gl.where(e == j, idx, selected_id)
        if j < 7:
            idx = next_idx
    ticket = gl.atomic_add(Counts + selected_id * COUNT_STRIDE, 1, e < 8, sem='relaxed')
    arena = selected_id * PAD + ticket
    gl.store(Sorted + arena, m * 8 + e, e < 8)
    gl.store(Weights + m * 8 + e, selected_prob / total * SCALE, e < 8)

@gluon.jit
def _select_and_quantize(X, Q, QS, L, Bias, Weights, Counts, Sorted, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, SPLITS: gl.constexpr, PAD: gl.constexpr, COUNT_STRIDE: gl.constexpr, GROUPS: gl.constexpr, SCALE: gl.constexpr):
    if gl.program_id(0) < M:
        _select_routes(L, Bias, Weights, Counts, Sorted, M, SPLITS, PAD, COUNT_STRIDE, SCALE)
    else:
        _quantize_input(X, Q, QS, M, H, SX, GROUPS, M)

@gluon.jit
def _weight_offset(e, n, k, N: gl.constexpr, K: gl.constexpr):
    byte = k // 2
    return ((((e * (N // 16) + n // 16) * (K // 64) + byte // 32) * 2 + byte // 16 % 2) * 16 + n % 16) * 16 + byte % 16

@gluon.jit
def _load_packed_weight(W, S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, NATIVE: gl.constexpr, CACHE: gl.constexpr=''):
    WARPS: gl.constexpr = gl.num_warps()
    W = W + expert * (N * K // 2)
    S = S + expert * (N * (triton.cdiv(K // 32, 8) * 8))
    packed: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2] if NATIVE else [16, 4], [WARPS, 1], [0, 1])
    n = gl.arange(0, BN, gl.SliceLayout(1, packed))
    if UP:
        n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)
    else:
        n = column * BN + n
    k = base * BK + 8 * gl.arange(0, BK // 8, gl.SliceLayout(0, packed))
    offset = _weight_offset(0, n[:, None], k[None, :], N, K) // 4
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset, cache=CACHE)
    if UP and NATIVE:
        sl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [WARPS, 1], [1, 0])
        block_n = gl.arange(0, BN // 32, gl.SliceLayout(1, sl))
        block_k = gl.arange(0, BK // 4, gl.SliceLayout(0, sl))
        idx = (block_n[:, None] * (BK // 4) + block_k[None, :]).reshape((BN * BK // 128,))
    else:
        sl: gl.constexpr = gl.BlockedLayout([1], [64], [WARPS], [0])
        idx = gl.arange(0, BN * BK // 128, sl)
    nb = idx // (BK // 4)
    if UP:
        nb = column * (BN // 64) + nb % (BN // 64) + nb // (BN // 64) * (N // 64)
    else:
        nb = column * (BN // 32) + nb
    kg = idx // 64 % (BK // 256)
    inner = idx % 64
    sw = gl.amd.cdna4.buffer_load(S.to(gl.pointer_type(gl.uint32)), nb * (K // 4) + base * (BK // 4) + kg * 64 + inner, cache='' if UP or not NATIVE else '.cg')
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

@gluon.jit
def _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, arena_row, scratch_row, column, expert, live, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, PAD: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, TM: gl.constexpr, NATIVE: gl.constexpr, ROWS: gl.constexpr):
    WARPS: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 64] if NATIVE else [16, 16, 128], transposed=True, warps_per_cta=[1, WARPS])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [TM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    if not UP:
        al: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2] if NATIVE else [16, 4], [WARPS, 1], [0, 1])
    else:
        al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [WARPS, 1], [1, 0])
    mi = gl.arange(0, TM, gl.SliceLayout(1, al))
    shared = expert == 256
    if UP:
        if shared:
            route = (arena_row - 256 * PAD + mi) * 8
        else:
            route = gl.load(Sorted + arena_row + mi, mi < live, 0).to(gl.uint32)
        row = gl.where(mi < live, route // 8 + gl.where(shared, M, 0), 0)
    else:
        row = scratch_row + gl.where(mi < live, mi, 0)
    ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
    if UP and NATIVE:
        a_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BK // 2, 16]], [TM, BK // 2], [1, 0])
    else:
        a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    if UP:
        a_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 2], a_shared_layout)
        if not NATIVE:
            as_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 32], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    if not (UP and NATIVE):
        bs_shared = gl.allocate_shared_memory(gl.uint8, [BN, BK // 32], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    as_load_layout: gl.constexpr = gl.BlockedLayout([1, 8] if UP and NATIVE else [1, 2], [32, 2], [WARPS, 1], [1, 0])
    if UP and NATIVE:
        scale_m = gl.arange(0, TM, gl.SliceLayout(1, as_load_layout))
        if shared:
            scale_route = (arena_row - 256 * PAD + scale_m) * 8
        else:
            scale_route = gl.load(Sorted + arena_row + scale_m, scale_m < live, 0).to(gl.uint32)
        scale_row = gl.where(scale_m < live, scale_route // 8 + gl.where(shared, M, 0), 0)
    else:
        scale_row = gl.convert_layout(row, gl.SliceLayout(1, as_load_layout))
    scale_word = gl.arange(0, BK // 128, gl.SliceLayout(0, as_load_layout))
    acc = gl.zeros((TM, BN), gl.float32, mma)
    for base in range(K // BK):
        if not UP:
            a_offset = ((base * (BK // 32) + ki[None, :] // 4) * ROWS + row[:, None]) * 4
            a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), a_offset + ki[None, :] % 4)
            a = gl.convert_layout(_word_bytes(a_words), ad, assert_trivial=True)
            sr = scratch_row + gl.arange(0, TM, gl.SliceLayout(1, asl))
            sr = gl.where(sr - scratch_row < live, sr, scratch_row)
            sg = base * (BK // 32) + gl.arange(0, BK // 32, gl.SliceLayout(0, asl))
            sa = gl.load(XS + sg[None, :] * ROWS + sr[:, None])
        else:
            a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
            a = _word_bytes(a_words)
            a_shared.store(a)
            sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + base * (BK // 128) + scale_word[None, :])
            sa_bytes = _word_bytes(sa_words)
            if NATIVE:
                sa = gl.convert_layout(sa_bytes, asl)
        words, scales = _load_packed_weight(W, WS, expert, column, base, N, K, BN, BK, UP, NATIVE, '.cg' if UP else '')
        b = _word_bytes(words).T
        b = gl.convert_layout(b, bd, assert_trivial=True)
        if UP and NATIVE:
            sb = gl.convert_layout(scales, bsl)
        else:
            bs_shared.store(scales)
        if UP:
            if not NATIVE:
                as_shared.store(sa_bytes)
                sa = as_shared.load(asl)
            a = a_shared.load(ad)
        if not (UP and NATIVE):
            sb = bs_shared.load(bsl)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    if UP:
        if NATIVE:
            split_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [WARPS, 1], [1, 0])
            acc = gl.convert_layout(acc, split_layout)
        gate, up = gl.split(gl.permute(gl.reshape(acc, (TM, 2, BN // 2)), (0, 2, 1)))
        if shared:
            gate = gate.to(gl.bfloat16).to(gl.float32)
            up = up.to(gl.bfloat16).to(gl.float32)
        activated = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
        ep: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [WARPS, 1], [1, 0])
        activated = gl.convert_layout(activated, ep)
        codes, scales = _encode_groups(gl.reshape(activated, (TM * (BN // 64), 32)), shared)
        codes = codes.reshape((TM, BN // 4))
        scales = scales.reshape((TM, BN // 64))
        rr = gl.arange(0, TM, gl.SliceLayout(1, codes.type.layout))
        nn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, codes.type.layout))
        q_offset = (nn[None, :] // 16 * ROWS + scratch_row + rr[:, None]) * 16 + nn[None, :] % 16
        gl.store(Q + q_offset, codes, rr[:, None] < live)
        rr_s = gl.arange(0, TM, gl.SliceLayout(1, scales.type.layout))
        nn_s = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, scales.type.layout))
        scale_offset = nn_s[None, :] * ROWS + scratch_row + rr_s[:, None]
        gl.store(QS + scale_offset, scales, rr_s[:, None] < live)
    else:
        if NATIVE:
            store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WARPS, 1], [1, 0])
        else:
            store_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, WARPS], [1, 0])
        rr = gl.arange(0, TM, gl.SliceLayout(1, store_layout))
        nn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, store_layout))
        if shared:
            shared_result = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
            token = arena_row - 256 * PAD + rr
            gl.store(Y + token[:, None] * N + nn[None, :], shared_result, rr[:, None] < live)
        else:
            routed_result = gl.convert_layout(acc.to(Parts.dtype.element_ty), store_layout)
            route = gl.load(Sorted + arena_row + rr, rr < live, 0).to(gl.uint32)
            if NATIVE:
                route = route // 128 * 128 + route % 8 * 16 + route // 8 % 16
            PART_ROWS: gl.constexpr = triton.cdiv(M, 16) * 16 if NATIVE else M
            part_base = Parts + column * PART_ROWS * 8 * BN
            SLAB: gl.constexpr = 128 if NATIVE else 64
            address = route[:, None] * SLAB + nn[None, :] % SLAB
            if BN > SLAB:
                address += nn[None, :] % BN // SLAB * PART_ROWS * 8 * SLAB
            gl.amd.cdna4.buffer_store(routed_result.to(Parts.dtype.element_ty), part_base, address, rr[:, None] < live)

@gluon.jit
def _job_for_tile(Counts, tile, M: gl.constexpr, PAD: gl.constexpr, TM: gl.constexpr, COUNT_STRIDE: gl.constexpr):
    shared_jobs: gl.constexpr = triton.cdiv(M, TM)
    if tile < shared_jobs:
        descriptor = (256 * PAD + tile * TM) * 32 + gl.minimum(TM, M - tile * TM) - 1
    else:
        scan_layout: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
        e = gl.arange(0, 256, scan_layout)
        count = gl.load(Counts + e * COUNT_STRIDE)
        chunks = gl.cdiv(count, TM)
        end = gl.associative_scan(chunks, 0, _add)
        begin = end - chunks
        chunk = tile - shared_jobs - begin
        packed = (e * PAD + chunk * TM) * 32 + gl.minimum(TM, count - chunk * TM) - 1
        descriptor = gl.max(gl.where((chunk >= 0) & (chunk < chunks), packed, -1), 0)
    return gl.inline_asm_elementwise('v_readfirstlane_b32 $0, $1;', '=s,v', [descriptor], dtype=gl.int32, is_pure=True, pack=1)

@gluon.jit
def _expert_tiles(X, XS, W, WS, Sorted, Jobs, Counts, Q, QS, Parts, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, PAD: gl.constexpr, GROUP: gl.constexpr, UP: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, TM: gl.constexpr, NATIVE: gl.constexpr, ROWS: gl.constexpr, COUNT_STRIDE: gl.constexpr):
    pid = gl.program_id(0).to(gl.uint32)
    COLS: gl.constexpr = N // BN
    tile = pid // (GROUP * COLS) * GROUP + pid % GROUP
    column = pid // GROUP % COLS
    if UP:
        descriptor = _job_for_tile(Counts, tile.to(gl.int32), M, PAD, TM, COUNT_STRIDE)
        if column == 0:
            gl.store(Jobs + tile, descriptor)
    else:
        descriptor = gl.load(Jobs + tile)
    if descriptor >= 0:
        arena = descriptor.to(gl.uint32) >> 5
        live = (descriptor & 31) + 1
        expert = arena // PAD
        scratch_row = tile * TM
        _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, arena, scratch_row, column, expert, live, M, N, K, PAD, BN, BK, UP, TM, NATIVE, ROWS)

@gluon.jit
def _reduce_parts(P, Y, Weights, M: gl.constexpr, H: gl.constexpr, BLOCK: gl.constexpr, WARPS: gl.constexpr, CACHE: gl.constexpr, VECTOR: gl.constexpr):
    gl.static_assert(BLOCK % 128 == 0)
    m = gl.program_id(0).to(gl.uint32)
    layout: gl.constexpr = gl.BlockedLayout([VECTOR], [64], [WARPS], [0])
    inner = gl.arange(0, BLOCK, layout).to(gl.uint32)
    h = gl.program_id(1) * BLOCK + inner
    PART_ROWS: gl.constexpr = triton.cdiv(M, 16) * 16 if M > 128 else M
    SLAB: gl.constexpr = 64 if M <= 128 else 128
    part_base = P + gl.program_id(1) * PART_ROWS * 8 * BLOCK
    value = gl.full((BLOCK,), 0.0, gl.float32, layout)
    for rank in gl.static_range(8):
        route = m * 8 + rank
        weight = gl.load(Weights + route)
        if M > 128:
            route = m // 16 * 128 + rank * 16 + m % 16
        address = inner // SLAB * PART_ROWS * 8 * SLAB + route * SLAB + inner % SLAB
        contribution = gl.amd.cdna4.buffer_load(part_base, address, cache=CACHE).to(gl.float32)
        value += contribution * weight
    value += gl.load(Y + m * H + h).to(gl.float32)
    gl.store(Y + m * H + h, value)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, routed_scaling_factor=2.5):
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    small_batch = m <= 128
    tile_rows = 16 if small_batch else 32
    native = not small_batch
    up_n = 64 if small_batch else 128
    up_k = 1024 if small_batch else 2048
    up_warps = 2 if small_batch else 4
    down_n = 256 if small_batch else 128
    down_warps = 4
    down_group = 1
    quant_groups = 32 if small_batch else 64
    router_mn = 32
    router_splits = 24 if small_batch else 8
    pad = triton.cdiv(m, tile_rows) * tile_rows
    active_bound = min(256, m * 8)
    job_bound = active_bound + (m * 8 - active_bound) // tile_rows + triton.cdiv(m, tile_rows)
    jobs_count = triton.cdiv(job_bound, 8) * 8

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=x.device)
    router_k = 128 if small_batch else 256
    jobs = empty((jobs_count,), torch.int32)
    count_stride = 32
    counts = empty((256 * count_stride,), torch.int32)
    sorted_routes = empty((256 * pad,), torch.int32)
    weights = empty((m, 8), torch.float32)
    part_rows = m if small_batch else triton.cdiv(m, 16) * 16
    slab = 64 if small_batch else 128
    parts = empty((h // slab, part_rows * 8, slab), torch.bfloat16)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    logits = empty((router_splits, m, 256), torch.float32)
    activation_rows = jobs_count * tile_rows
    aq = empty((activation_rows, intermediate // 2), torch.uint8)
    aqs = empty((activation_rows, intermediate // 32), torch.uint8)
    out = empty((m, h))
    router_ctas = triton.cdiv(m, router_mn) * (256 // router_mn) * router_splits
    quant_ctas = triton.cdiv(m * (h // 32), quant_groups)
    _router_projection[router_ctas,](x, router, logits, counts, m, h, x.stride(0), router_mn, router_mn, router_k, router_splits, count_stride, num_warps=1, enable_fp_fusion=False)
    _select_and_quantize[m + quant_ctas,](x, xq, xs, logits, correction_bias, weights, counts, sorted_routes, m, h, x.stride(0), router_splits, pad, count_stride, quant_groups, routed_scaling_factor, num_warps=1, enable_fp_fusion=False)
    _expert_tiles[jobs_count * (2 * intermediate // up_n),](xq, xs, w13, w13_scale, sorted_routes, jobs, counts, aq, aqs, parts, out, m, 2 * intermediate, h, pad, 1, True, up_n, up_k, tile_rows, native, activation_rows, count_stride, num_warps=up_warps, enable_fp_fusion=False)
    _expert_tiles[jobs_count * (h // down_n),](aq, aqs, w2, w2_scale, sorted_routes, jobs, counts, aq, aqs, parts, out, m, h, intermediate, pad, down_group, False, down_n, 256, tile_rows, native, activation_rows, count_stride, num_warps=down_warps)
    reduce_block = 512 if small_batch else 128
    reduce_vector = 4 if small_batch else 2
    reduce_warps = 2 if small_batch else 1
    _reduce_parts[m, h // reduce_block](parts, out, weights, m, h, reduce_block, reduce_warps, '.cg', reduce_vector, num_warps=reduce_warps, enable_fp_fusion=False)
    return out
