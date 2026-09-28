# fmt: off
"""GLM-5.2 TP8 MTP fused MoE for selected M=24..768 product shapes."""

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
    WARPS: gl.constexpr = gl.num_warps()
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
def _router_and_quantize(X, W, L, Q, QS, Counts, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUPS: gl.constexpr, SPLITS: gl.constexpr):
    ROUTER_ROWS: gl.constexpr = triton.cdiv(M, BM)
    ROUTER_BASE: gl.constexpr = ROUTER_ROWS * (256 // BN)
    ROUTER_CTAS: gl.constexpr = ROUTER_BASE * SPLITS
    pid = gl.program_id(0)
    if pid == 0:
        jl: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        e = gl.arange(0, 256, jl)
        gl.store(Counts + e, 0)
    if pid < ROUTER_CTAS:
        _router_linear(X, W, L, M, H, SX, BM, BN, BK, pid % ROUTER_ROWS, pid % ROUTER_BASE // ROUTER_ROWS, SPLITS, pid // ROUTER_BASE)
    else:
        _quantize_input(X, Q, QS, M, H, SX, GROUPS, ROUTER_CTAS)

@gluon.jit
def _select_routes(L, Bias, Weights, Counts, Sorted, M: gl.constexpr, SPLITS: gl.constexpr, PAD: gl.constexpr, SCALE: gl.constexpr):
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
    for j in gl.static_range(8):
        maximum = gl.max(score, 0)
        candidate = gl.where(score == maximum, e, e + 256)
        idx = gl.min(gl.where(available, candidate, 512), 0) % 256
        local_prob = gl.sum(gl.where(e // 64 == idx // 64, prob, 0.0).reshape((4, 64)), 0)
        index = gl.full((1,), idx % 64, gl.int32, local_prob.type.layout)
        p = gl.sum(gl.gather(local_prob, index, 0), 0)
        total += p
        selected_prob = gl.where(e == j, p, selected_prob)
        selected_id = gl.where(e == j, idx, selected_id)
        available &= e != idx
        score = gl.where(e == idx, -float('inf'), score)
    ticket = gl.atomic_add(Counts + selected_id, 1, e < 8, sem='relaxed')
    arena = selected_id * PAD + ticket
    gl.store(Sorted + arena, m * 8 + e, e < 8)
    gl.store(Weights + m * 8 + e, selected_prob / total * SCALE, e < 8)

@gluon.jit
def _weight_offset(n, k, K: gl.constexpr):
    byte = k // 2
    return (((n // 16 * (K // 64) + byte // 32) * 2 + byte // 16 % 2) * 16 + n % 16) * 16 + byte % 16

@gluon.jit
def _load_packed_weight(W, S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr):
    WARPS: gl.constexpr = gl.num_warps()
    W = W + expert * (N * K // 2)
    S = S + expert * (N * (triton.cdiv(K // 32, 8) * 8))
    packed: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WARPS, 1], [0, 1])
    n = gl.arange(0, BN, gl.SliceLayout(1, packed))
    if UP:
        n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)
    else:
        n = column * BN + n
    k = base * BK + 8 * gl.arange(0, BK // 8, gl.SliceLayout(0, packed))
    offset = _weight_offset(n[:, None], k[None, :], K) // 4
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset, cache='.cg' if UP else '')
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [WARPS], [0])
    idx = gl.arange(0, BN * BK // 128, sl)
    nb = idx // (BK // 4)
    if UP:
        nb = column * (BN // 64) + nb % (BN // 64) + nb // (BN // 64) * (N // 64)
    else:
        nb = column * (BN // 32) + nb
    kg = idx // 64 % (BK // 256)
    inner = idx % 64
    sw = gl.amd.cdna4.buffer_load(S.to(gl.pointer_type(gl.uint32)), nb * (K // 4) + base * (BK // 4) + kg * 64 + inner, cache='' if UP else '.cg')
    raw = sw.reshape((BN // 32, BK // 256, 4, 16))
    packed_scales = _unpack_words(raw)
    scale_byte = gl.permute(packed_scales, (0, 5, 3, 1, 4, 2)).reshape((BN, BK // 32))
    return (words, scale_byte)

@gluon.jit
def _unpack_words(words):
    b0 = words.to(gl.uint8)
    b1 = (words >> 8).to(gl.uint8)
    b2 = (words >> 16).to(gl.uint8)
    b3 = (words >> 24).to(gl.uint8)
    return gl.join(gl.join(b0, b2), gl.join(b1, b3))

@gluon.jit
def _project_tile(X, XS, W, WS, row, column, expert, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr):
    TM: gl.constexpr = 16
    WARPS: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, WARPS])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [TM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [WARPS, 1], [1, 0])
    ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
    a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    a_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 2], a_shared_layout)
    bs_shared = gl.allocate_shared_memory(gl.uint8, [BN, BK // 32], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
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
        words, scales = _load_packed_weight(W, WS, expert, column, base, N, K, BN, BK, UP)
        b = _unpack_words(words).reshape((BN, BK // 2)).T
        a_shared.store(a)
        b = gl.convert_layout(b, bd, assert_trivial=True)
        bs_shared.store(scales)
        as_shared.store(sa_bytes)
        sa = as_shared.load(asl)
        a = a_shared.load(ad)
        sb = bs_shared.load(bsl)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    return acc

@gluon.jit
def _store_activation(acc, Q, QS, scratch_row, column, shared, N: gl.constexpr, BN: gl.constexpr):
    TM: gl.constexpr = 16
    WARPS: gl.constexpr = gl.num_warps()
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
    gl.store(Q + (scratch_row + rr[:, None]) * (N // 4) + nn[None, :], codes)
    rr_s = gl.arange(0, TM, gl.SliceLayout(1, scales.type.layout))
    nn_s = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, scales.type.layout))
    gl.store(QS + (scratch_row + rr_s[:, None]) * (N // 64) + nn_s[None, :], scales)

@gluon.jit
def _store_projection(acc, Sorted, Parts, Y, arena_row, column, live, shared, M: gl.constexpr, N: gl.constexpr, PAD: gl.constexpr, BN: gl.constexpr):
    TM: gl.constexpr = 16
    WARPS: gl.constexpr = gl.num_warps()
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WARPS, 1], [1, 0])
    rr = gl.arange(0, TM, gl.SliceLayout(1, store_layout))
    nn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, store_layout))
    if shared:
        shared_result = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
        token = arena_row - 256 * PAD + rr
        gl.store(Y + token[:, None] * N + nn[None, :], shared_result, rr[:, None] < live)
    else:
        routed_result = gl.convert_layout(acc, store_layout)
        route = gl.load(Sorted + arena_row + rr, rr < live, 0).to(gl.uint32)
        part_base = Parts + column * M * 8 * BN
        address = route[:, None] * 128 + nn[None, :] % 128
        address += nn[None, :] % BN // 128 * M * 8 * 128
        gl.amd.cdna4.buffer_store(routed_result.to(Parts.dtype.element_ty), part_base, address, rr[:, None] < live)

@gluon.jit
def _ordered_descriptor(Counts, tile, M: gl.constexpr, PAD: gl.constexpr):
    jl: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 1], [1, 64], [gl.num_warps(), 1], [1, 0]))
    e = gl.arange(0, 256, jl)
    counts = gl.load(Counts + e)
    sizes = (counts + 15) // 16
    ends = gl.sum(gl.associative_scan(sizes[None, :], 1, _add), 0)
    route_job = tile.to(gl.int32) - triton.cdiv(M, 16)
    expert = gl.min(gl.where(route_job < ends, e, 256), 0)
    begin = gl.sum(gl.where(e < expert, sizes, 0), 0)
    descriptor = gl.where(expert < 256, expert * PAD + (route_job - begin) * 16, -1)
    return gl.where(tile < triton.cdiv(M, 16), 256 * PAD + tile.to(gl.int32) * 16, descriptor)

@gluon.jit
def _expert_tiles(X, XS, W, WS, Sorted, Jobs, Counts, Q, QS, Parts, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, PAD: gl.constexpr, GROUP: gl.constexpr, UP: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    pid = gl.program_id(0).to(gl.uint32)
    COLS: gl.constexpr = N // BN
    tile = pid // (GROUP * COLS) * GROUP + pid % GROUP
    column = pid // GROUP % COLS
    if UP:
        descriptor = _ordered_descriptor(Counts, tile, M, PAD)
        if column == 0:
            gl.store(Jobs + tile, descriptor)
    else:
        descriptor = gl.load(Jobs + tile)
    if descriptor >= 0:
        arena = descriptor.to(gl.uint32)
        expert = arena // PAD
        start = arena % PAD
        if expert == 256:
            count = M
        else:
            count = gl.load(Counts + expert)
        live = gl.minimum(16, count - start)
        scratch_row = tile * 16
        shared = expert == 256
        al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [gl.num_warps(), 1], [1, 0])
        mi = gl.arange(0, 16, gl.SliceLayout(1, al))
        if UP:
            if shared:
                route = (arena - 256 * PAD + mi) * 8
            else:
                route = gl.load(Sorted + arena + mi, mi < live, 0).to(gl.uint32)
            row = gl.where(mi < live, route // 8 + gl.where(shared, M, 0), 0)
        else:
            row = scratch_row + mi
        acc = _project_tile(X, XS, W, WS, row, column, expert, N, K, BN, BK, UP)
        if UP:
            _store_activation(acc, Q, QS, scratch_row, column, shared, N, BN)
        else:
            _store_projection(acc, Sorted, Parts, Y, arena, column, live, shared, M, N, PAD, BN)

@gluon.jit
def _reduce_parts(P, Y, Weights, M: gl.constexpr, H: gl.constexpr):
    BLOCK: gl.constexpr = 128
    m = gl.program_id(0).to(gl.uint32)
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    inner = gl.arange(0, BLOCK, layout).to(gl.uint32)
    h = gl.program_id(1) * BLOCK + inner
    part_base = P + gl.program_id(1) * M * 8 * BLOCK
    value = gl.full((BLOCK,), 0.0, gl.float32, layout)
    for rank in gl.static_range(8):
        route = m * 8 + rank
        weight = gl.load(Weights + route)
        address = route * 128 + inner
        contribution = gl.amd.cdna4.buffer_load(part_base, address, cache='.cg').to(gl.float32)
        value += contribution * weight
    value += gl.load(Y + m * H + h).to(gl.float32)
    gl.store(Y + m * H + h, value)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, routed_scaling_factor=2.5):
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    pad = triton.cdiv(m, 16) * 16
    active_bound = min(256, m * 8)
    job_bound = active_bound + (m * 8 - active_bound) // 16 + triton.cdiv(m, 16)
    jobs_count = triton.cdiv(job_bound, 8) * 8

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=x.device)
    router_splits, router_k = (8, 256)
    up_n, up_k, up_group, up_warps = (64, 1024, 1, 2)
    down_n, down_k, down_group, down_warps = (256, 256, 1, 4)
    jobs = empty((jobs_count,), torch.int32)
    counts = empty((256,), torch.int32)
    sorted_routes = empty((256 * pad,), torch.int32)
    weights = empty((m, 8), torch.float32)
    parts = empty((h // 128, m * 8, 128), torch.float32)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    logits = empty((router_splits, m, 256), torch.float32)
    activation_rows = jobs_count * 16
    aq = empty((activation_rows, intermediate // 2), torch.uint8)
    aqs = empty((activation_rows, intermediate // 32), torch.uint8)
    out = empty((m, h))
    router_ctas = triton.cdiv(m, 16) * 16 * router_splits
    quant_ctas = triton.cdiv(m * (h // 32), 16)
    _router_and_quantize[router_ctas + quant_ctas,](x, router, logits, xq, xs, counts, m, h, x.stride(0), 16, 16, router_k, 16, router_splits, num_warps=1, enable_fp_fusion=False)
    _select_routes[m,](logits, correction_bias, weights, counts, sorted_routes, m, router_splits, pad, routed_scaling_factor, num_warps=1, enable_fp_fusion=False)
    _expert_tiles[jobs_count * (2 * intermediate // up_n),](xq, xs, w13, w13_scale, sorted_routes, jobs, counts, aq, aqs, parts, out, m, 2 * intermediate, h, pad, up_group, True, up_n, up_k, num_warps=up_warps, enable_fp_fusion=False)
    _expert_tiles[jobs_count * (h // down_n),](aq, aqs, w2, w2_scale, sorted_routes, jobs, counts, aq, aqs, parts, out, m, h, intermediate, pad, down_group, False, down_n, down_k, num_warps=down_warps)
    _reduce_parts[m, h // 128](parts, out, weights, m, h, num_warps=1, enable_fp_fusion=False)
    return out
