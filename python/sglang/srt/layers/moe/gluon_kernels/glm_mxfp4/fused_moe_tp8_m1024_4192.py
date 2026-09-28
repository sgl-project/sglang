# fmt: off
"""GLM-5.2 TP8 fused MoE specialization for M=1024..4192."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_artifact_next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)

@gluon.jit
def _add(a, b):
    return a + b

@gluon.jit
def _quantize_packed(x, shared):
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
    inverse_scale = gl.exp2(-exponent.to(gl.float32))
    a = gl.abs(x * inverse_scale[:, None])
    a = gl.where(a == a, a, 6.0)
    a = gl.where(x < 0, -a, a)
    low, high = gl.split(a.reshape((x.shape[0], 16, 2)))
    packed = gl.inline_asm_elementwise('v_mov_b32 $0, 0\nv_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0 op_sel:[0,0,0]', constraints='=&v,v,v', args=[low, high], dtype=gl.uint32, is_pure=True, pack=1).to(gl.uint8)
    return (packed, (exponent + 127).to(gl.uint8))

@gluon.jit
def _quantize_input(X, XQ, pid, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, NW: gl.constexpr, QG: gl.constexpr):
    al: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [NW, 1], [1, 0])
    g = pid * QG + gl.arange(0, QG, gl.SliceLayout(1, al))
    k = gl.arange(0, 32, gl.SliceLayout(0, al))
    x = gl.load(X + (g // (H // 32))[:, None] * SX + (g % (H // 32))[:, None] * 32 + k[None, :], g[:, None] < M * (H // 32), 0).to(gl.float32)
    routed, rs = _quantize_packed(x, False)
    shared, ss = _quantize_packed(x, True)
    routed = gl.convert_layout(routed, al)
    shared = gl.convert_layout(shared, al)
    pk = gl.arange(0, 16, gl.SliceLayout(0, al))
    gl.store(XQ + g[:, None] * 16 + pk[None, :], routed, g[:, None] < M * (H // 32))
    gl.store(XQ + M * H // 2 + g[:, None] * 16 + pk[None, :], shared, g[:, None] < M * (H // 32))
    rs = gl.convert_layout(rs, gl.SliceLayout(1, al))
    ss = gl.convert_layout(ss, gl.SliceLayout(1, al))
    gl.store(XQ + M * H + g, rs, g < M * (H // 32))
    gl.store(XQ + M * H + M * (H // 32) + g, ss, g < M * (H // 32))

@gluon.jit
def _router_linear(X, W, L, Counts, pm, pn, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, NW: gl.constexpr):
    if (pm == 0) & (pn < 8):
        c = gl.arange(0, max(256, BN * 8), gl.BlockedLayout([1], [64], [NW], [0]))
        gl.store(Counts + pn * max(256, BN * 8) + c, 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1] if NW == 1 else [2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NW, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, NW], [0, 1])
    mi = pm * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    ni = pn * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    if M <= 1536:
        a_smem = gl.allocate_shared_memory(gl.bfloat16, (BM, BK), gl.SwizzledSharedLayout(8, 1, 8, [1, 0]))
        b_smem = gl.allocate_shared_memory(gl.bfloat16, (BK, BN), gl.SwizzledSharedLayout(8, 1, 8, [0, 1]))
        a = gl.load(X + mi[:, None] * SX + ak[None, :], mi[:, None] < M, 0)
        b = gl.load(W + ni[None, :] * H + bk[:, None])
        for base in range(H // BK):
            a_smem.store(a)
            b_smem.store(b)
            next_base = gl.minimum(base + 1, H // BK - 1)
            a = gl.load(X + mi[:, None] * SX + (next_base * BK + ak)[None, :], mi[:, None] < M, 0)
            b = gl.load(W + ni[None, :] * H + (next_base * BK + bk)[:, None])
            ad = a_smem.load(gl.DotOperandLayout(0, mma, 8))
            bd = b_smem.load(gl.DotOperandLayout(1, mma, 8))
            acc = gl.amd.cdna4.mfma(ad, bd, acc)
    else:
        for base in range(H // BK):
            a = gl.load(X + mi[:, None] * SX + (base * BK + ak)[None, :], mi[:, None] < M, 0)
            b = gl.load(W + ni[None, :] * H + (base * BK + bk)[:, None])
            acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8)), acc)
    mm = pm * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    nn = pn * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(L + mm[:, None] * 256 + nn[None, :], acc, mm[:, None] < M)

@gluon.jit
def _prologue(X, W, L, Counts, XQ, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, NW: gl.constexpr, QG: gl.constexpr):
    pid = gl.program_id(0)
    ROWS: gl.constexpr = triton.cdiv(M, BM)
    ROUTER_CTAS: gl.constexpr = ROWS * (256 // BN)
    if pid < ROUTER_CTAS:
        if M > 4096:
            GROUP: gl.constexpr = 8
            COLS: gl.constexpr = 256 // BN
            FULL: gl.constexpr = ROWS // GROUP * GROUP * COLS
            if ROWS % GROUP:
                if pid < FULL:
                    row = pid // (GROUP * COLS) * GROUP + pid % GROUP
                    column = pid // GROUP % COLS
                else:
                    row = ROWS // GROUP * GROUP + (pid - FULL) % (ROWS % GROUP)
                    column = (pid - FULL) // (ROWS % GROUP)
            else:
                row = pid // (GROUP * COLS) * GROUP + pid % GROUP
                column = pid // GROUP % COLS
        else:
            row, column = (pid % ROWS, pid // ROWS)
        _router_linear(X, W, L, Counts, row, column, M, H, SX, BM, BN, BK, NW)
    else:
        _quantize_input(X, XQ, pid - ROUTER_CTAS, M, H, SX, NW, QG)

@gluon.jit
def _select_routes(L, Bias, Codes, Weights, Counts, M: gl.constexpr, SCALE: gl.constexpr):
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
    ticket = gl.atomic_add(Counts + m // 64 % 8 * 256 + selected_id, 1, e < 8, sem='relaxed')
    gl.store(Codes + m * 8 + e, selected_id * M + ticket, e < 8)
    gl.store(Weights + m * 8 + e, selected_prob / total * SCALE, e < 8)

@gluon.jit
def _prepare(Codes, Counts, Sorted, Info, M: gl.constexpr, BM: gl.constexpr, CAP: gl.constexpr, SCHEDULED: gl.constexpr, CHUNKS: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    experts = gl.arange(0, 256, layout)
    shard_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [1, 0])
    shard = gl.arange(0, 8, gl.SliceLayout(1, shard_layout))
    se = gl.arange(0, 256, gl.SliceLayout(0, shard_layout))
    partial = gl.load(Counts + shard[:, None] * 256 + se[None, :])
    counts = gl.convert_layout(gl.sum(partial, 0), layout)
    tiles = gl.cdiv(counts, BM)
    offsets = gl.associative_scan(tiles, 0, _add) - tiles
    if pid < CHUNKS:
        route = pid * 256 + experts
        code = gl.load(Codes + route, route < M * 8, 0)
        expert, ticket = (code // M, code % M)
        prefix = gl.sum(gl.where(shard[:, None] < pid // 2 % 8, partial, 0), 0)
        prefix = gl.convert_layout(prefix, layout)
        ticket += gl.gather(prefix, expert, 0)
        offset = gl.gather(offsets, expert, 0)
        gl.store(Sorted + offset * BM + ticket, route, route < M * 8)
        dense_offsets = gl.associative_scan(counts, 0, _add) - counts
        dense = gl.gather(dense_offsets, expert, 0)
        gl.store(Codes + route, dense + ticket, route < M * 8)
    elif pid == CHUNKS + 257:
        active = gl.sum(tiles, 0) + triton.cdiv(M, BM)
        i = gl.arange(0, _artifact_next_power_of_2(SCHEDULED), layout)
        gl.store(Info + i, 0, (i >= active) & (i < SCHEDULED))
        if M > 1536:
            gl.store(Info + SCHEDULED + i, 0, (i >= active) & (i < SCHEDULED))
    else:
        expert = pid - CHUNKS
        if expert < 256:
            start = gl.sum(gl.where(experts == expert, offsets, 0), 0)
            count = gl.sum(gl.where(experts == expert, counts, 0), 0)
            row = start
            start += triton.cdiv(M, BM)
        else:
            start = 0
            count = M
            row = CAP
        b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(M, BM)), layout)
        live = gl.minimum(BM, count - b * BM)
        info = expert | live << 9 | row + b << 17
        gl.store(Info + start + b, info, b < gl.cdiv(count, BM))
        if M > 1536:
            if expert < 256:
                full = counts // BM
                tail = counts % BM
                full_prefix = gl.sum(gl.where(experts < expert, full, 0), 0)
                full_total = gl.sum(full, 0)
                own_tail = count % BM
                height = gl.where(tail > 64, 128, gl.where(tail > 32, 64, gl.where(tail > 16, 32, gl.where(tail > 0, 16, 0))))
                own_height = gl.where(own_tail > 64, 128, gl.where(own_tail > 32, 64, gl.where(own_tail > 16, 32, gl.where(own_tail > 0, 16, 0))))
                before_tail = gl.sum(((height > own_height) | (height == own_height) & (experts < expert)).to(gl.int32), 0)
                full_start = triton.cdiv(M, BM) + full_prefix
                tail_start = triton.cdiv(M, BM) + full_total + before_tail
                up_pos = gl.where(b < count // BM, full_start + b, tail_start)
            else:
                up_pos = b
            gl.store(Info + SCHEDULED + up_pos, info, b < gl.cdiv(count, BM))
        dense = gl.sum(gl.where(experts < expert, counts, 0), 0)
        gl.store(Sorted + CAP * BM + row + b, dense + b * BM, b < gl.cdiv(count, BM))

@gluon.jit
def _load_packed_weight(W, S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, STAGED_B: gl.constexpr):
    W += expert * (N * K // 2)
    S += expert * N * (triton.cdiv(K // 32, 8) * 8)
    if STAGED_B:
        weight_layout: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
        p = gl.arange(0, BN * BK // 8, weight_layout)
        panel = p // (2 * BK)
        panel = column * (BN // 32) + panel % (BN // 32) + panel // (BN // 32) * (N // 32)
        offset = (panel * (K // 64) + base * (BK // 64) + p // 128 % (BK // 64)) * 128 + p % 128
        packed = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset)
        words = gl.permute(packed.reshape((BN // 16, BK // 64, 2, 16, 4)), (0, 3, 1, 2, 4)).reshape((BN, BK // 8))
    else:
        word_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [0, 1])
        ni = gl.arange(0, BN, gl.SliceLayout(1, word_layout))
        ki = gl.arange(0, BK // 8, gl.SliceLayout(0, word_layout))
        if UP:
            ni = column * (BN // 2) + ni % (BN // 2) + ni // (BN // 2) * (N // 2)
        else:
            ni += column * BN
        kw = base * (BK // 8) + ki
        offset = (ni[:, None] // 16 * (K // 64) + kw[None, :] // 8) * 128 + kw[None, :] % 8 // 4 * 64 + ni[:, None] % 16 * 4 + kw[None, :] % 4
        words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset)
    physical: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    q = gl.arange(0, BN * (BK // 32) // 4, physical)
    group_n = q // (64 * (BK // 256))
    if UP:
        group_n = column * (BN // 64) + group_n % (BN // 64) + group_n // (BN // 64) * (N // 64)
    else:
        group_n += column * (BN // 32)
    offset_s = (group_n * (K // 256) + base * (BK // 256) + q // 64 % (BK // 256)) * 64 + q % 64
    sw = gl.amd.cdna4.buffer_load(S.to(gl.pointer_type(gl.uint32)), offset_s)
    s0, s1 = (sw.to(gl.uint8), (sw >> 8).to(gl.uint8))
    s2, s3 = ((sw >> 16).to(gl.uint8), (sw >> 24).to(gl.uint8))
    bytes_s = gl.join(gl.join(s0, s2), gl.join(s1, s3))
    scale = gl.permute(bytes_s.reshape((BN // 32, BK // 256, 4, 16, 2, 2)), (0, 5, 3, 1, 4, 2)).reshape((BN, BK // 32))
    return (words, scale)

@gluon.jit
def _broadcast_record(code, weight, rank: gl.constexpr, TM: gl.constexpr, BN: gl.constexpr):
    index = gl.full((TM, 1), rank, gl.int32, code.type.layout)
    dense = gl.gather(code, index, 1).reshape((TM,))
    scale = gl.gather(weight, index, 1).reshape((TM,))
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    dense = gl.convert_layout(dense, gl.SliceLayout(1, ep))
    scale = gl.convert_layout(scale, gl.SliceLayout(1, ep))
    fill = gl.full((TM, BN), 0, gl.int32, ep)
    dense, _ = gl.broadcast(dense[:, None], fill)
    scale, _ = gl.broadcast(scale[:, None], fill)
    return (dense, scale)

@gluon.jit
def _expert_tile(X, W, S, Sorted, AQ, Parts, Y, Weights, Codes, block, column, expert, live, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, CAP: gl.constexpr, UP: gl.constexpr, TM: gl.constexpr, FINAL: gl.constexpr=False, start=0):
    STAGED_B: gl.constexpr = UP and M > 1536
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 2] if STAGED_B and TM > 16 else [1, 4])
    al: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    sl: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [1, 0])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    scale_a: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_a, (TM, BK // 32))
    scale_b: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_b, (BN, BK // 32))
    mi = start + gl.arange(0, TM, gl.SliceLayout(1, al))
    shared = expert == 256
    if UP:
        if shared:
            route = ((block - CAP) * BM + mi) * 8
        else:
            route = gl.load(Sorted + block * BM + mi, mi < live, 0)
        row = gl.where(mi < live, route // 8 + gl.where(shared, M, 0), 0)
        XS = X + M * K
    else:
        row = block * BM + mi
        XS = X + (CAP + triton.cdiv(M, BM)) * BM * (K // 2)
    sr = gl.convert_layout(row, gl.SliceLayout(1, sl))
    sk = gl.arange(0, BK // 32, gl.SliceLayout(0, sl))
    ki = gl.arange(0, BK // 2, gl.SliceLayout(0, al))
    a_smem = gl.allocate_shared_memory(gl.uint8, (TM, BK // 2), gl.SwizzledSharedLayout(16, 1, 8, [1, 0]))
    if STAGED_B:
        b_smem = gl.allocate_shared_memory(gl.uint8, (BK // 2, BN), gl.SwizzledSharedLayout(16, 1, 8, [0, 1]))
    if UP:
        as_smem = gl.allocate_shared_memory(gl.uint8, (TM, BK // 32), gl.SwizzledSharedLayout(1, 1, 1, [1, 0] if STAGED_B else [0, 1]))
        bs_smem = gl.allocate_shared_memory(gl.uint8, (BN, BK // 32), gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
    acc = gl.zeros((TM, BN), gl.float32, mma)
    words, scales = _load_packed_weight(W, S, expert, column, 0, N, K, BN, BK, UP, STAGED_B)
    for base in range(K // BK):
        a = gl.amd.cdna4.buffer_load(X, row[:, None] * (K // 2) + (base * BK // 2 + ki)[None, :])
        sa = gl.amd.cdna4.buffer_load(XS, sr[:, None] * (K // 32) + (base * BK // 32 + sk)[None, :])
        b0, b1 = (words.to(gl.uint8), (words >> 8).to(gl.uint8))
        b2, b3 = ((words >> 16).to(gl.uint8), (words >> 24).to(gl.uint8))
        b = gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((BN, BK // 2)).T
        a_smem.store(a)
        if STAGED_B:
            b_smem.store(b)
        else:
            bd = gl.convert_layout(b, dot_b, assert_trivial=True)
        if UP:
            as_smem.store(sa)
            bs_smem.store(scales)
        else:
            sa = gl.convert_layout(sa, scale_a)
            sb = gl.convert_layout(scales, scale_b)
        next_base = gl.minimum(base + 1, K // BK - 1)
        words, scales = _load_packed_weight(W, S, expert, column, next_base, N, K, BN, BK, UP, STAGED_B)
        ad = a_smem.load(dot_a)
        if STAGED_B:
            bd = b_smem.load(dot_b)
        if UP:
            sa = as_smem.load(scale_a)
            sb = bs_smem.load(scale_b)
        acc = gl.amd.cdna4.mfma_scaled(ad, sa, 'e2m1', bd, sb, 'e2m1', acc)
    if UP:
        gate, up = gl.split(gl.permute(gl.reshape(acc, (TM, 2, BN // 2)), (0, 2, 1)))
        if shared:
            gate = gate.to(gl.bfloat16).to(gl.float32)
            up = up.to(gl.bfloat16).to(gl.float32)
        activated = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
        ep: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
        activated = gl.convert_layout(activated, ep)
        quantized, scales = _quantize_packed(gl.reshape(activated, (TM * (BN // 64), 32)), shared)
        quantized = gl.convert_layout(gl.reshape(quantized, (TM, BN // 4)), ep)
        rr = block * BM + gl.arange(0, TM, gl.SliceLayout(1, ep))
        nn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, ep))
        gl.store(AQ + rr[:, None] * (N // 4) + nn[None, :], quantized)
        out_scale_layout: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [4, 1], [1, 0])
        scales = gl.convert_layout(gl.reshape(scales, (TM, BN // 64)), out_scale_layout)
        sr = block * BM + gl.arange(0, TM, gl.SliceLayout(1, out_scale_layout))
        sn = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, out_scale_layout))
        scale_ptr = AQ + (CAP + triton.cdiv(M, BM)) * BM * (N // 4)
        gl.store(scale_ptr + sr[:, None] * (N // 64) + sn[None, :], scales)
    else:
        ep: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        result = gl.convert_layout(acc.to(gl.bfloat16), ep)
        rr = start + gl.arange(0, TM, gl.SliceLayout(1, ep))
        local_n = gl.arange(0, BN, gl.SliceLayout(0, ep))
        nn = column * BN + local_n
        if FINAL:
            valid = gl.full((TM,), True, gl.int1, gl.SliceLayout(1, ep)) if M % 16 == 0 else rr < live
            token = (block - CAP) * BM + rr
            shared_result = result.to(gl.float32)
            value = gl.full((TM, BN), 0.0, gl.float32, ep)
            part_base = Parts + column * M * 8 * BN
            records: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
            record_rows = gl.convert_layout(rr, gl.SliceLayout(1, records))
            record_tokens = (block - CAP) * BM + record_rows
            record_rank = gl.arange(0, 16, gl.SliceLayout(0, records)) % 8
            record_valid = gl.full((TM,), True, gl.int1, gl.SliceLayout(1, records)) if M % 16 == 0 else record_rows < live
            code = gl.load(Codes + record_tokens[:, None] * 8 + record_rank[None, :], record_valid[:, None], 0)
            scale = gl.load(Weights + record_tokens[:, None] * 8 + record_rank[None, :], record_valid[:, None], 0)
            dense, weight = _broadcast_record(code, scale, 0, TM, BN)
            part = gl.amd.cdna4.buffer_load(part_base, dense * BN + local_n[None, :], valid[:, None], 0, cache='.cg')
            for rank in gl.static_range(1, 8):
                next_dense, next_weight = _broadcast_record(code, scale, rank, TM, BN)
                next_part = gl.amd.cdna4.buffer_load(part_base, next_dense * BN + local_n[None, :], valid[:, None], 0, cache='.cg')
                value += part.to(gl.float32) * weight
                part = next_part
                weight = next_weight
            value += part.to(gl.float32) * weight
            gl.store(Y + token[:, None] * N + nn[None, :], value + shared_result, valid[:, None])
        else:
            dense_base = gl.load(Sorted + CAP * BM + block)
            part_base = Parts + column * M * 8 * BN + dense_base * BN
            gl.amd.cdna4.buffer_store(result, part_base, rr[:, None] * BN + nn[None, :] % BN, rr[:, None] < live)

@gluon.jit
def _experts(X, W, S, Sorted, Info, AQ, Parts, Y, Weights, Codes, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, CAP: gl.constexpr, UP: gl.constexpr, COLS: gl.constexpr):
    pid = gl.program_id(0)
    GROUP: gl.constexpr = (4 if M <= 1536 else 8) if UP else 1
    job = pid // (GROUP * COLS) * GROUP + pid % GROUP
    column = pid // GROUP % COLS
    if UP and M > 1536:
        SCHEDULED: gl.constexpr = triton.cdiv(CAP + triton.cdiv(M, BM), GROUP) * GROUP
        job += SCHEDULED
    elif not UP:
        job += triton.cdiv(M, BM)
    info = gl.load(Info + job)
    if info != 0:
        expert = info & 511
        live = info >> 9 & 255
        block = info >> 17
        if live <= 16:
            _expert_tile(X, W, S, Sorted, AQ, Parts, Y, Weights, Codes, block, column, expert, live, M, N, K, BM, BN, 512 if UP and M <= 1536 else BK, CAP, UP, 16)
        elif live <= 32:
            _expert_tile(X, W, S, Sorted, AQ, Parts, Y, Weights, Codes, block, column, expert, live, M, N, K, BM, BN, 512 if UP and M <= 1536 else BK, CAP, UP, 32)
        elif (live <= 64) | (BM == 64):
            _expert_tile(X, W, S, Sorted, AQ, Parts, Y, Weights, Codes, block, column, expert, live, M, N, K, BM, BN, 512 if UP and M <= 1536 else BK, CAP, UP, 64)
        else:
            _expert_tile(X, W, S, Sorted, AQ, Parts, Y, Weights, Codes, block, column, expert, live, M, N, K, BM, BN, BK, CAP, UP, 128)

@gluon.jit
def _shared_down(AQ, W, S, Sorted, Parts, Y, Weights, Codes, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, CAP: gl.constexpr):
    block = CAP + gl.program_id(0) * 16 // BM
    start = gl.program_id(0) * 16 % BM
    live = gl.minimum(BM, M - (block - CAP) * BM)
    _expert_tile(AQ, W, S, Sorted, AQ, Parts, Y, Weights, Codes, block, gl.program_id(1), 256, live, M, N, K, BM, 128, 256, CAP, False, 16, True, start)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, routed_scaling_factor=2.5):
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    block_m = 64 if m <= 1536 else 128
    routed_capacity = triton.cdiv(m * 8, block_m) + 256
    capacity = routed_capacity + triton.cdiv(m, block_m)
    up_group = 4 if m <= 1536 else 8
    scheduled = triton.cdiv(capacity, up_group) * up_group
    chunks = triton.cdiv(m * 8, 256)

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, device=x.device, dtype=dtype)
    logits = empty((m, 256))
    codes = empty((m, 8), torch.int32)
    weights = empty((m, 8), torch.float32)
    counts = empty((8, 256), torch.int32)
    sorted_routes = empty((routed_capacity * block_m + capacity,), torch.int32)
    info = empty(((2 if m > 1536 else 1) * scheduled,), torch.int32)
    xq = empty((2 * m * (h // 2 + h // 32),), torch.uint8)
    aq = empty((capacity * block_m * (intermediate // 2 + intermediate // 32),), torch.uint8)
    parts = empty((h // 128, m * 8, 128), torch.bfloat16)
    out = empty((m, h))
    router_warps = 4
    router_m = 32
    router_n = 32 if m <= 3072 else 64
    router_k = 512 if 1536 < m <= 2048 else 256
    quant_groups = 256 if 2048 < m <= 4096 else 64
    quant_ctas = triton.cdiv(m * (h // 32), quant_groups)
    router_ctas = triton.cdiv(m, router_m) * (256 // router_n)
    _prologue[router_ctas + quant_ctas,](x, router, logits, counts, xq, m, h, x.stride(0), router_m, router_n, router_k, router_warps, quant_groups, num_warps=router_warps, enable_fp_fusion=False)
    _select_routes[m,](logits, correction_bias, codes, weights, counts, m, routed_scaling_factor, num_warps=1, enable_fp_fusion=False)
    _prepare[chunks + 258,](codes, counts, sorted_routes, info, m, block_m, routed_capacity, scheduled, chunks, num_warps=1, enable_fp_fusion=False)
    _experts[scheduled * (2 * intermediate // 128),](xq, w13, w13_scale, sorted_routes, info, aq, parts, out, weights, codes, m, 2 * intermediate, h, block_m, 128, 256, routed_capacity, True, 2 * intermediate // 128, enable_fp_fusion=False)
    _experts[routed_capacity * (h // 128),](aq, w2, w2_scale, sorted_routes, info, aq, parts, out, weights, codes, m, h, intermediate, block_m, 128, 256, routed_capacity, False, h // 128, enable_fp_fusion=False)
    _shared_down[triton.cdiv(m, 16), h // 128](aq, w2, w2_scale, sorted_routes, parts, out, weights, codes, m, h, intermediate, block_m, routed_capacity, enable_fp_fusion=False)
    return out
