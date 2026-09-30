# fmt: off
"""Shared TP4 fused-MoE specialization for active batches 17 <= M <= 64."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
def _router_linear(X, W, L, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, Counts, pid_m, pid_n, split, SPLITS: gl.constexpr):
    init_shard = pid_m * (256 // BN) + pid_n
    if (init_shard == 0) & (split == 0):
        counter = gl.arange(0, 256, gl.BlockedLayout([1], [64], [1], [0]))
        gl.store(Counts + init_shard * 256 + counter, 0)
        gl.store(Counts + 256, 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    mi = pid_m * BM + gl.arange(0, 16, gl.SliceLayout(1, al)) % BM
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    ni = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((16, BN), gl.float32, mma)
    for base in range(H // (BK * SPLITS)):
        offset = split * (H // SPLITS) + base * BK
        a = gl.load(X + mi[:, None] * SX + (offset + ak)[None, :], mi[:, None] < M, 0)
        b = gl.load(W + ni[None, :] * H + (offset + bk)[:, None])
        acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8)), acc)
    local_m = gl.arange(0, 16, gl.SliceLayout(1, mma))
    mm = pid_m * BM + local_m
    nn = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(L + (split * M + mm[:, None]) * 256 + nn[None, :], acc, (mm[:, None] < M) & (local_m[:, None] < BM))

@gluon.jit
def _router_and_quantize(X, W, L, Q, QS, Counts, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUPS: gl.constexpr, SPLITS: gl.constexpr):
    ROUTER_ROWS: gl.constexpr = triton.cdiv(M, BM)
    ROUTER_CTAS: gl.constexpr = ROUTER_ROWS * (256 // BN) * SPLITS
    pid = gl.program_id(0)
    if pid < ROUTER_CTAS:
        _router_linear(X, W, L, M, H, SX, BM, BN, BK, Counts, pid // SPLITS % ROUTER_ROWS, pid // (SPLITS * ROUTER_ROWS), pid % SPLITS, SPLITS)
    else:
        _quantize_input(X, Q, QS, M, H, SX, GROUPS, ROUTER_CTAS)

@gluon.jit
def _select_routes(L, Bias, Sorted, Weights, Counts, Jobs, expert_start, M: gl.constexpr, TM: gl.constexpr, SPLITS: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, SCALE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    m = gl.program_id(0)
    e = gl.arange(0, 256, layout)
    rank = gl.arange(0, 8, layout)
    logits = gl.full((256,), 0.0, gl.float32, layout)
    for split in gl.static_range(SPLITS):
        logits += gl.load(L + (split * M + m) * 256 + e).to(gl.float32)
    logits = logits.to(gl.bfloat16).to(gl.float32)
    prob = 1.0 / (1.0 + gl.exp(-logits))
    score = prob + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    selected_prob = gl.full((8,), 0.0, gl.float32, layout)
    selected_id = gl.full((8,), 0, gl.int32, layout)
    total = 0.0
    for j in gl.static_range(8):
        maximum = gl.max(score, 0)
        idx = gl.min(gl.where(available & (score == maximum), e, 256), 0)
        if idx >= 256:
            idx = gl.min(gl.where(available, e, 256), 0)
        p = gl.sum(gl.where(e == idx, prob, 0.0), 0)
        total += p
        selected_prob = gl.where(rank == j, p, selected_prob)
        selected_id = gl.where(rank == j, idx, selected_id)
        available &= e != idx
        score = gl.where(e == idx, -float('inf'), score)
    owned = (selected_id >= expert_start) & (selected_id < expert_start + LOCAL_EXPERTS)
    selected_id = gl.where(owned, selected_id - expert_start, 0)
    ticket = gl.atomic_add(Counts + selected_id, 1, owned, sem='relaxed')
    gl.store(Sorted + selected_id * (triton.cdiv(M, TM) * TM) + ticket, m * 8 + rank, owned)
    gl.store(Weights + m * 8 + rank, gl.where(owned, selected_prob / total * SCALE, 0.0))
    publish = owned & (ticket % TM == 0)
    job = gl.atomic_add(Counts + 256 + gl.zeros_like(rank), 1, publish, sem='relaxed')
    gl.store(Jobs + job, selected_id | ticket // TM << 9, publish)

@gluon.jit
def _decode_job(Counts, Jobs, tile, M: gl.constexpr, TM: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, HAS_SHARED: gl.constexpr):
    index_type: gl.constexpr = gl.uint32 if TM == 16 else gl.int32
    SHARED_BLOCKS: gl.constexpr = triton.cdiv(M, TM) if HAS_SHARED else 0
    expert = LOCAL_EXPERTS
    block = tile.to(gl.int32)
    live = gl.minimum(TM, M - tile.to(gl.int32) * TM)
    if tile >= SHARED_BLOCKS:
        index = tile - SHARED_BLOCKS
        count = gl.load(Counts + 256)
        descriptor = gl.load(Jobs + index, index < count, 0)
        expert = descriptor & 511
        block = descriptor >> 9
        expert_count = gl.load(Counts + expert)
        live = gl.where(index < count, gl.minimum(TM, expert_count - block * TM), 0)
    return (expert.to(index_type), block.to(index_type), live)

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
def _load_packed_weight(W, S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, CACHE: gl.constexpr, WARPS: gl.constexpr, TM: gl.constexpr):
    index_type: gl.constexpr = gl.uint32 if TM == 16 else gl.int32
    W = W + expert * (N * K // 2)
    S = S + expert * (N * (triton.cdiv(K // 32, 8) * 8))
    native_32: gl.constexpr = not UP and TM == 32
    packed: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2] if native_32 else [16, 4], [WARPS, 1], [0, 1])
    n = gl.arange(0, BN, gl.SliceLayout(1, packed)).to(index_type)
    if UP:
        n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)
    else:
        n = column * BN + n
    k = base * BK + 8 * gl.arange(0, BK // 8, gl.SliceLayout(0, packed)).to(index_type)
    offset = _weight_offset(n[:, None], k[None, :], K) // 4
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset, cache=CACHE)
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [WARPS], [0])
    idx = gl.arange(0, BN * BK // 128, sl).to(index_type)
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
def _project_tile(X, XS, W, WS, Sorted, block, column, expert, live, tile, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, WARPS: gl.constexpr, WEIGHT_CACHE: gl.constexpr, DIRECT_A: gl.constexpr, TM: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, HAS_SHARED: gl.constexpr):
    index_type: gl.constexpr = gl.uint32 if TM == 16 else gl.int32
    native_32: gl.constexpr = not UP and TM == 32
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 64] if native_32 else [16, 16, 128], transposed=True, warps_per_cta=[1, WARPS])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [TM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    al: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [WARPS, 1], [0, 1]) if DIRECT_A else gl.BlockedLayout([1, 4], [8, 8], [WARPS, 1], [1, 0])
    mi = gl.arange(0, TM, gl.SliceLayout(1, al)).to(index_type)
    shared = HAS_SHARED & (expert == LOCAL_EXPERTS)
    arena_row = expert * (triton.cdiv(M, TM) * TM) + block * TM
    if UP:
        if shared:
            route = (block * TM + mi) * 8
        else:
            route = gl.load(Sorted + arena_row + mi, mi < live, 0).to(index_type)
        row = gl.where(mi < live, route // 8 + gl.where(shared, M, 0), 0)
    else:
        row = (tile * TM if TM == 32 else arena_row) + mi
    ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al)).to(index_type)
    a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    a_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 2], a_shared_layout)
    bs_shared = gl.allocate_shared_memory(gl.uint8, [BN, BK // 32], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    as_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 32], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    as_load_layout: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [WARPS, 1], [1, 0])
    scale_row = gl.convert_layout(row, gl.SliceLayout(1, as_load_layout))
    scale_word = gl.arange(0, BK // 128, gl.SliceLayout(0, as_load_layout)).to(index_type)
    acc = gl.zeros((TM, BN), gl.float32, mma)
    for base in range(K // BK):
        if UP:
            a_offset = row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :]
        else:
            a_offset = (tile * TM if TM == 32 else arena_row) * (K // 8) + (base * (BK // 8) + ki[None, :]) // 4 * (TM * 4) + mi[:, None] * 4 + ki[None, :] % 4
        a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), a_offset)
        a = _unpack_words(a_words).reshape((TM, BK // 2))
        if UP or TM == 16:
            sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + base * (BK // 128) + scale_word[None, :])
            sa_bytes = _unpack_words(sa_words).reshape((TM, BK // 32))
        else:
            sw_layout: gl.constexpr = gl.BlockedLayout([1], [64], [WARPS], [0])
            si = gl.arange(0, TM * BK // 128, sw_layout).to(index_type)
            sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), tile * TM * (K // 128) + base * (TM * BK // 128) + si)
            sa_bytes = _unpack_words(sa_words.reshape((BK // 32, TM // 4)))
            sa_bytes = sa_bytes.reshape((BK // 32, TM)).T
        words, scales = _load_packed_weight(W, WS, expert, column, base, N, K, BN, BK, UP, WEIGHT_CACHE, WARPS, TM)
        b = _unpack_words(words).reshape((BN, BK // 2)).T
        b = gl.convert_layout(b, bd)
        if DIRECT_A:
            a = gl.convert_layout(a, ad)
        else:
            a_shared.store(a)
        bs_shared.store(scales)
        as_shared.store(sa_bytes)
        sa = as_shared.load(asl)
        if not DIRECT_A:
            a = a_shared.load(ad)
        sb = bs_shared.load(bsl)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    return (acc, arena_row, shared)

@gluon.jit
def _store_activation(acc, Q, QS, arena_row, column, shared, N: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, TM: gl.constexpr):
    index_type: gl.constexpr = gl.uint32 if TM == 16 else gl.int32
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
    rr = gl.arange(0, TM, gl.SliceLayout(1, codes.type.layout)).to(index_type)
    nn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, codes.type.layout)).to(index_type)
    q_offset = arena_row * (N // 4) + nn[None, :] // 16 * (TM * 16) + rr[:, None] * 16 + nn[None, :] % 16
    gl.store(Q + q_offset, codes)
    rr_s = gl.arange(0, TM, gl.SliceLayout(1, scales.type.layout)).to(index_type)
    nn_s = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, scales.type.layout)).to(index_type)
    if TM == 32:
        scale_offset = arena_row * (N // 64) + nn_s[None, :] * TM + rr_s[:, None]
    else:
        scale_offset = (arena_row + rr_s[:, None]) * (N // 64) + nn_s[None, :]
    gl.store(QS + scale_offset, scales)

@gluon.jit
def _store_projection(acc, Sorted, Parts, Y, arena_row, block, column, shared, live, M: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, TM: gl.constexpr):
    index_type: gl.constexpr = gl.uint32 if TM == 16 else gl.int32
    gl.static_assert(BN == 64 or BN == 128)
    rr = gl.arange(0, TM, gl.SliceLayout(1, acc.type.layout)).to(index_type)
    nn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, acc.type.layout)).to(index_type)
    if shared:
        token = block * TM + rr
        gl.store(Y + token[:, None] * N + nn[None, :], acc, rr[:, None] < live)
    else:
        route = gl.load(Sorted + arena_row + rr, rr < live, 0).to(index_type)
        part_base = Parts + column * BN // 128 * M * 8 * 128
        address = route[:, None] * 128 + nn[None, :] % 128
        gl.amd.cdna4.buffer_store(acc.to(Parts.dtype.element_ty), part_base, address, rr[:, None] < live)

@gluon.jit
def _scaled_experts(X, XS, W, WS, Sorted, Counts, Jobs, Q, QS, Parts, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, GROUP: gl.constexpr, UP: gl.constexpr, BN: gl.constexpr, WARPS: gl.constexpr, BK: gl.constexpr, WEIGHT_CACHE: gl.constexpr, DIRECT_A: gl.constexpr, TM: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, HAS_SHARED: gl.constexpr):
    index_type: gl.constexpr = gl.uint32 if TM == 16 else gl.int32
    pid = gl.program_id(0).to(index_type)
    COLS: gl.constexpr = N // BN
    tile = pid // (GROUP * COLS) * GROUP + pid % GROUP
    panel = pid // GROUP % COLS
    expert, block, live = _decode_job(Counts, Jobs, tile, M, TM, LOCAL_EXPERTS, HAS_SHARED)
    if live > 0:
        acc, arena_row, shared = _project_tile(X, XS, W, WS, Sorted, block, panel, expert, live, tile, M, N, K, BN, BK, UP, WARPS, WEIGHT_CACHE, DIRECT_A, TM, LOCAL_EXPERTS, HAS_SHARED)
        if UP:
            _store_activation(acc, Q, QS, tile * TM if TM == 32 else arena_row, panel, shared, N, BN, WARPS, TM)
        else:
            _store_projection(acc, Sorted, Parts, Y, arena_row, block, panel, shared, live, M, N, BN, TM)

@gluon.jit
def _reduce_parts(P, Y, Weights, M: gl.constexpr, H: gl.constexpr, HAS_SHARED: gl.constexpr):
    index_type: gl.constexpr = gl.uint32 if M <= 32 else gl.int32
    m = gl.program_id(0).to(index_type)
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    inner = gl.arange(0, 128, layout).to(index_type)
    h = gl.program_id(1).to(index_type) * 128 + inner
    part_base = P + gl.program_id(1).to(index_type) * M * 8 * 128
    value = gl.full((128,), 0.0, gl.float32, layout)
    for rank in gl.static_range(8):
        route = m * 8 + rank
        weight = gl.load(Weights + route)
        address = route * 128 + inner
        contribution = gl.amd.cdna4.buffer_load(
            part_base,
            address,
            mask=weight != 0.0,
            other=0.0,
            cache='.cg',
        ).to(gl.float32)
        value += contribution * weight
    if HAS_SHARED:
        value += gl.load(Y + m * H + h).to(gl.float32)
    gl.store(Y + m * H + h, value)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, expert_start=0, routed_scaling_factor=2.5, fuse_shared_expert=False):
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    local_experts = w13.shape[0] - 1
    routes = m * 8
    tile_rows = 16 if m <= 32 else 32
    expert_rows = triton.cdiv(m, tile_rows) * tile_rows
    distinct = min(256, routes)
    routed_blocks = distinct + (routes - distinct) // tile_rows
    shared_blocks = triton.cdiv(m, tile_rows) if fuse_shared_expert else 0
    scheduled = triton.cdiv(routed_blocks + shared_blocks, 4) * 4

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, device=x.device, dtype=dtype)
    parts = empty((h // 128, routes, 128), torch.float32)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    router_splits = 12
    router_rows = 16
    logits = empty((router_splits, m, 256), torch.float32)
    weights = empty((m, 8), torch.float32)
    counts = empty((257,), torch.int32)
    jobs = empty((scheduled,), torch.int32)
    sorted_routes = empty(((local_experts + 1) * expert_rows,), torch.int32)
    activation_rows = scheduled * tile_rows if m > 32 else (local_experts + 1) * expert_rows
    aq = empty((activation_rows, intermediate // 2), torch.uint8)
    aqs = empty((activation_rows, intermediate // 32), torch.uint8)
    out = empty((m, h))
    quant_ctas = triton.cdiv(m * (h // 32), 16)
    router_ctas = triton.cdiv(m, router_rows) * 16 * router_splits
    _router_and_quantize[router_ctas + quant_ctas,](x, router, logits, xq, xs, counts, m, h, x.stride(0), BM=router_rows, BN=16, BK=128, GROUPS=16, SPLITS=router_splits, num_warps=1, enable_fp_fusion=False)
    _select_routes[m,](logits, correction_bias, sorted_routes, weights, counts, jobs, expert_start, m, tile_rows, router_splits, local_experts, routed_scaling_factor, num_warps=1, enable_fp_fusion=False)
    _scaled_experts[scheduled * (2 * intermediate // 64),](xq, xs, w13, w13_scale, sorted_routes, counts, jobs, aq, aqs, parts, out, m, 2 * intermediate, h, GROUP=1, UP=True, BN=64, WARPS=2, BK=512, WEIGHT_CACHE='.cg', DIRECT_A=False, TM=tile_rows, LOCAL_EXPERTS=local_experts, HAS_SHARED=fuse_shared_expert, num_warps=2, enable_fp_fusion=False)
    down_n = 64 if m <= 32 else 128
    down_warps = 2 if m <= 32 else 4
    _scaled_experts[scheduled * (h // down_n),](aq, aqs, w2, w2_scale, sorted_routes, counts, jobs, aq, aqs, parts, out, m, h, intermediate, GROUP=2, UP=False, BN=down_n, WARPS=down_warps, BK=512, WEIGHT_CACHE='', DIRECT_A=m > 32, TM=tile_rows, LOCAL_EXPERTS=local_experts, HAS_SHARED=fuse_shared_expert, num_warps=down_warps)
    _reduce_parts[m, h // 128](parts, out, weights, m, h, fuse_shared_expert, num_warps=1, enable_fp_fusion=False)
    return out
