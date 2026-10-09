# Created by OpenAI
# fmt: off
"""Shared TP4 fused-MoE specialization for active batches M=1 through M=16."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.constexpr_function
def _scale_shared_layout(rows, groups):
    register_k = [[0, 1 << bit] for bit in range(2, groups.bit_length() - 1)]
    lane_n = [[1 << bit, 0] for bit in range(4)]
    lane_k = [[0, 1], [0, 2]]
    high_n = [[1 << bit, 0] for bit in range(4, rows.bit_length() - 1)]
    return gl.SharedLinearLayout(register_k + lane_n + lane_k + high_n)

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
    a = gl.abs(x / scale[:, None])
    a = gl.where(a == a, a, 6.0)
    low, high = gl.split(a.reshape((x.shape[0], 16, 2)))
    packed = gl.inline_asm_elementwise('v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0', constraints='=v,v,v', args=[low, high], dtype=gl.uint32, is_pure=True, pack=1).to(gl.uint8)
    sign_low, sign_high = gl.split((x < 0).to(gl.uint8).reshape((x.shape[0], 16, 2)))
    return (packed | sign_low << 3 | sign_high << 7, (exponent + 127).to(gl.uint8))

@gluon.jit
def _front_split(X, W, L, Q, QS, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, SPLITS: gl.constexpr):
    pid = gl.program_id(0)
    BASE_CTAS: gl.constexpr = triton.cdiv(M, 16) * 16
    ROUTER_CTAS: gl.constexpr = BASE_CTAS * SPLITS
    if pid < ROUTER_CTAS:
        tile = pid % BASE_CTAS
        split = pid // BASE_CTAS
        mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
        al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
        bl: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 1], [1, 0])
        m = tile // 16 * 16 + gl.arange(0, 16, gl.SliceLayout(1, al))
        ak = gl.arange(0, 256, gl.SliceLayout(0, al))
        n = tile % 16 * 16 + gl.arange(0, 16, gl.SliceLayout(0, bl))
        bk = gl.arange(0, 256, gl.SliceLayout(1, bl))
        acc = gl.zeros((16, 16), gl.float32, mma)
        for base in range(H // (256 * SPLITS)):
            if M == 1:
                a = gl.load(X + (split * (H // SPLITS) + base * 256 + ak)[None, :] + gl.full((16, 1), 0, gl.int32, al))
            else:
                a = gl.load(X + m[:, None] * SX + (split * (H // SPLITS) + base * 256 + ak)[None, :], m[:, None] < M, 0)
            b = gl.load(W + n[None, :] * H + (split * (H // SPLITS) + base * 256 + bk)[:, None])
            acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8), assert_trivial=True), acc)
        mm = tile // 16 * 16 + gl.arange(0, 16, gl.SliceLayout(1, mma))
        nn = tile % 16 * 16 + gl.arange(0, 16, gl.SliceLayout(0, mma))
        gl.store(L + (split * M + mm[:, None]) * 256 + nn[None, :], acc, mm[:, None] < M)
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [1, 0])
        g = (pid - ROUTER_CTAS) * 16 + gl.arange(0, 16, gl.SliceLayout(1, layout))
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
def _local_expert(expert, expert_start, LOCAL_EXPERTS: gl.constexpr):
    owned = (expert >= expert_start) & (expert < expert_start + LOCAL_EXPERTS)
    return gl.where(owned, expert - expert_start, LOCAL_EXPERTS), owned

@gluon.jit
def _select_split(L, Bias, Ids, Weights, expert_start, M: gl.constexpr, SPLITS: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, SCALE: gl.constexpr):
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
    low_available = 255
    for j in gl.static_range(8):
        maximum = gl.max(score, 0)
        idx = gl.min(gl.where(available & (score == maximum), e, 256), 0)
        fallback = gl.inline_asm_elementwise('v_ffbl_b32 $0, $1', constraints='=v,v', args=[low_available], dtype=gl.int32, is_pure=True, pack=1)
        idx = gl.where(idx < 256, idx, fallback)
        low_available &= ~gl.where(idx < 8, 1 << (idx & 7), 0)
        p = gl.sum(gl.where(e == idx, prob, 0.0), 0)
        total += p
        selected_prob = gl.where(e == j, p, selected_prob)
        selected_id = gl.where(e == j, idx, selected_id)
        available &= e != idx
        score = gl.where(e == idx, -float('inf'), score)
    local_id, owned = _local_expert(selected_id, expert_start, LOCAL_EXPERTS)
    gl.store(Ids + m * 8 + e, local_id, e < 8)
    gl.store(Weights + m * 8 + e, gl.where(owned, selected_prob / total * SCALE, 0.0), e < 8)

@gluon.jit
def _first_score_match(eligible, e):
    local_id = gl.min(gl.where(eligible, e, -1).to(gl.uint32).reshape((64, 4)), 1)
    _, wave_id = gl.inline_asm_elementwise('v_cmp_ne_u32_e64 $0, -1, $2\ns_ff1_i32_b64 $1, $0\nv_readlane_b32 $1, $2, $1', constraints='=&s,=&s,v,~{scc}', args=[local_id], dtype=(gl.uint64, gl.int32), is_pure=True, pack=1)
    zero = gl.full((1,), 0, gl.int32, wave_id.type.layout)
    return gl.gather(wave_id & 511, zero, 0).reshape(())

@gluon.jit
def _select_for_projection(L, Bias, Ids, Weights, expert_start, rank, column, M: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, SCALE: gl.constexpr):
    token = 0 if M == 1 else rank // 9
    rank = rank if M == 1 else rank % 9
    expert = 256
    if rank < 8:
        layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [1, 64], [WARPS, 1], [1, 0]))
        e = gl.arange(0, 256, layout)
        logit = gl.full((256,), 0.0, gl.float32, layout)
        for split in gl.static_range(SPLITS):
            logit += gl.load(L + (split * M + token) * 256 + e)
        logit = logit.to(gl.bfloat16).to(gl.float32)
        prob = 1.0 / (1.0 + gl.exp(-logit))
        score = prob + gl.load(Bias + e).to(gl.float32)
        available = gl.full((256,), True, gl.int1, layout)
        low_available = 255
        if (rank == 0) & (column == 0):
            selected_prob = gl.full((256,), 0.0, gl.float32, layout)
            selected_id = gl.full((256,), 0, gl.int32, layout)
            total = 0.0
            for j in gl.static_range(8):
                maximum = gl.max(score, 0)
                idx = _first_score_match(available & (score == maximum), e)
                fallback = gl.inline_asm_elementwise('v_ffbl_b32 $0, $1', constraints='=v,v', args=[low_available], dtype=gl.int32, is_pure=True, pack=1)
                idx = gl.where(idx < 256, idx, fallback)
                low_available &= ~gl.where(idx < 8, 1 << (idx & 7), 0)
                p = gl.sum(gl.where(e == idx, prob, 0.0), 0)
                total += p
                selected_prob = gl.where(e == j, p, selected_prob)
                selected_id = gl.where(e == j, idx, selected_id)
                expert = gl.where(j == 0, idx, expert)
                available &= e != idx
                score = gl.where(e == idx, -float('inf'), score)
            local_id, owned = _local_expert(selected_id, expert_start, LOCAL_EXPERTS)
            gl.store(Ids + token * 8 + e, local_id, e < 8)
            gl.store(Weights + token * 8 + e, gl.where(owned, selected_prob / total * SCALE, 0.0), e < 8)
        else:
            for j in range(rank + 1):
                maximum = gl.max(score, 0)
                idx = _first_score_match(available & (score == maximum), e)
                fallback = gl.inline_asm_elementwise('v_ffbl_b32 $0, $1', constraints='=v,v', args=[low_available], dtype=gl.int32, is_pure=True, pack=1)
                idx = gl.where(idx < 256, idx, fallback)
                low_available &= ~gl.where(idx < 8, 1 << (idx & 7), 0)
                expert = idx
                available &= e != idx
                score = gl.where(e == idx, -float('inf'), score)
    expert = gl.inline_asm_elementwise('v_readfirstlane_b32 $0, $1', constraints='=s,v', args=[expert], dtype=gl.int32, is_pure=True, pack=1)
    local_id, _ = _local_expert(expert, expert_start, LOCAL_EXPERTS)
    return local_id

@gluon.jit
def _weight_offset(n, k, K: gl.constexpr):
    byte = k // 2
    return (((n // 16 * (K // 64) + byte // 32) * 2 + byte // 16 % 2) * 16 + n % 16) * 16 + byte % 16

@gluon.jit
def _word_bytes(words):
    b0, b1 = (words.to(gl.uint8), (words >> 8).to(gl.uint8))
    b2, b3 = ((words >> 16).to(gl.uint8), (words >> 24).to(gl.uint8))
    return gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((words.shape[0], words.shape[1] * 4))

@gluon.jit
def _packed_weight(W, WS, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, WARPS: gl.constexpr, CACHE: gl.constexpr, REGISTER_B: gl.constexpr, DIRECT_SCALES: gl.constexpr, SCALE_LAYOUT: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WARPS, 1], [0, 1]) if REGISTER_B else gl.BlockedLayout([1, 4], [32, 2], [WARPS, 1], [1, 0])
    n = gl.arange(0, BN, gl.SliceLayout(1, layout))
    if UP:
        n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)
    else:
        n = column * BN + n
    k = base * BK + 8 * gl.arange(0, BK // 8, gl.SliceLayout(0, layout))
    words = gl.amd.cdna4.buffer_load((W + expert * (N * K // 2)).to(gl.pointer_type(gl.uint32)), _weight_offset(n[:, None], k[None, :], K) // 4, cache=CACHE)
    if DIRECT_SCALES:
        sn = column * BN + gl.arange(0, BN, gl.SliceLayout(1, SCALE_LAYOUT))
        sk = base * (BK // 32) + gl.arange(0, BK // 32, gl.SliceLayout(0, SCALE_LAYOUT))
        offset = sn[:, None] // 32 * (K // 4) + sk[None, :] // 8 * 64
        offset += sk[None, :] % 4 * 16 + sn[:, None] % 16
        sw = gl.amd.cdna4.buffer_load((WS + expert * N * (triton.cdiv(K // 32, 8) * 8)).to(gl.pointer_type(gl.uint32)), offset)
        shift = (sk[None, :] // 4 % 2 * 2 + sn[:, None] // 16 % 2) * 8
        scales = (sw >> shift).to(gl.uint8)
    else:
        scale_layout: gl.constexpr = gl.BlockedLayout([1], [64], [WARPS], [0])
        idx = gl.arange(0, BN * BK // 128, scale_layout)
        nb = idx // (BK // 4)
        if UP:
            nb = column * (BN // 64) + nb % (BN // 64) + nb // (BN // 64) * (N // 64)
        else:
            nb = column * (BN // 32) + nb
        kg = idx // 64 % (BK // 256)
        sw = gl.amd.cdna4.buffer_load((WS + expert * N * (triton.cdiv(K // 32, 8) * 8)).to(gl.pointer_type(gl.uint32)), nb * (K // 4) + base * (BK // 4) + kg * 64 + idx % 64)
        raw = sw.reshape((BN // 32, BK // 256, 4, 16))
        b0, b1 = (raw.to(gl.uint8), (raw >> 8).to(gl.uint8))
        b2, b3 = ((raw >> 16).to(gl.uint8), (raw >> 24).to(gl.uint8))
        scales = gl.permute(gl.join(gl.join(b0, b2), gl.join(b1, b3)), (0, 5, 3, 1, 4, 2))
        scales = scales.reshape((BN, BK // 32))
    return (words, scales)

@gluon.jit
def _expert(X, XS, W, WS, Ids, Q, QS, P, expert_start, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, WARPS: gl.constexpr=4, CACHE: gl.constexpr='.cg', COLUMN_GROUP: gl.constexpr=1, REGISTER_A: gl.constexpr=False, REGISTER_B: gl.constexpr=False, DIRECT_SCALES: gl.constexpr=False, FUSED_SELECT: gl.constexpr=False, Logits=None, Bias=None, Weights=None, ROUTER_SPLITS: gl.constexpr=8, PACK_SCALES: gl.constexpr=False, ROUTED_SCALE: gl.constexpr=2.5):
    SINGLE_TOKEN: gl.constexpr = M <= 4 or M == 8
    route = gl.program_id(0) // COLUMN_GROUP
    column = gl.program_id(1) * COLUMN_GROUP + gl.program_id(0) % COLUMN_GROUP
    shared = route % 9 == 8
    if FUSED_SELECT:
        expert = _select_for_projection(Logits, Bias, Ids, Weights, expert_start, route, column, M, ROUTER_SPLITS, WARPS, LOCAL_EXPERTS, ROUTED_SCALE)
    else:
        expert = gl.load(Ids + route // 9 * 8 + route % 9, ~shared, LOCAL_EXPERTS)
    route_layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [WARPS, 1], [0, 1])
    token_base = 0 if M == 1 else route // 9 if M > 1 and SINGLE_TOKEN else route // (16 * 9) * 16
    token = token_base + gl.arange(0, 16, gl.SliceLayout(1, route_layout))
    if SINGLE_TOKEN:
        slot = gl.full((16,), 0, gl.int32, gl.SliceLayout(1, route_layout)) + route % 9
        owner = route
    else:
        rank = gl.arange(0, 8, gl.SliceLayout(0, route_layout))
        route_ids = gl.load(Ids + token[:, None] * 8 + rank[None, :], token[:, None] < M, -1)
        match = route_ids == expert
        slot = gl.max(gl.where(match, rank[None, :], -1), 1)
        first = gl.min(gl.min(gl.where(match, token[:, None] * 9 + rank[None, :], M * 9), 1), 0)
        owner = gl.where(shared, token_base * 9 + 8, first)
        slot = gl.where(shared, 8, slot)
        # Non-local routed experts map to one zero sentinel.  Keep each route
        # independent so no uninitialized activation can leak through a zero
        # route weight.
        sentinel = (expert == LOCAL_EXPERTS) & ~shared
        owner = gl.where(sentinel, route, owner)
        slot = gl.where(sentinel, route % 9, slot)
    if M > 1 and SINGLE_TOKEN:
        valid = token == route // 9
    else:
        valid = (token < M) & (slot >= 0)
        if not SINGLE_TOKEN:
            valid = gl.where(sentinel, token == route // 9, valid)
    routes = token * 9 + gl.maximum(slot, 0)
    if route == owner:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, WARPS])
        ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
        bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
        asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, BK // 32])
        bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
        al: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WARPS, 1], [0, 1]) if REGISTER_A else gl.BlockedLayout([1, 4], [8, 8], [WARPS, 1], [1, 0])
        sl: gl.constexpr = gl.BlockedLayout([1, 2], [16, 4], [WARPS, 1], [1, 0])
        mi = gl.arange(0, 16, gl.SliceLayout(1, al))
        if UP:
            row = token_base + mi + gl.where(shared, M, 0)
        else:
            row = gl.convert_layout(routes, gl.SliceLayout(1, al))
        valid_a = gl.convert_layout(valid, gl.SliceLayout(1, al))
        if SINGLE_TOKEN:
            row = gl.full((16,), 0, gl.int32, gl.SliceLayout(1, al))
            row += (gl.where(shared, 1, 0) if M == 1 else route // 9 + gl.where(shared, M, 0)) if UP else route
        else:
            row = gl.where(valid_a, row, 0)
        ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
        sr = gl.convert_layout(row, gl.SliceLayout(1, sl))
        if not REGISTER_A:
            a_shared = gl.allocate_shared_memory(gl.uint8, [16, BK // 2], gl.SwizzledSharedLayout(16, 1, 8, [1, 0]))
        if not REGISTER_B:
            b_shared = gl.allocate_shared_memory(gl.uint8, [BK // 2, BN], gl.SwizzledSharedLayout(16, 1, 8, [0, 1]))
        if not DIRECT_SCALES:
            as_shared_layout: gl.constexpr = _scale_shared_layout(16, BK // 32) if PACK_SCALES else gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            bs_shared_layout: gl.constexpr = _scale_shared_layout(BN, BK // 32) if PACK_SCALES else gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            as_shared = gl.allocate_shared_memory(gl.uint8, [16, BK // 32], as_shared_layout)
            bs_shared = gl.allocate_shared_memory(gl.uint8, [BN, BK // 32], bs_shared_layout)
        acc = gl.zeros((16, BN), gl.float32, mma)
        for base in range(K // BK):
            aw = gl.load(X.to(gl.pointer_type(gl.uint32)) + row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
            bw, sb = _packed_weight(W, WS, expert, column, base, N, K, BN, BK, UP, WARPS, CACHE, REGISTER_B, DIRECT_SCALES, bsl)
            if DIRECT_SCALES:
                sr_native = gl.convert_layout(row, gl.SliceLayout(1, asl))
                sk_native = gl.arange(0, BK // 32, gl.SliceLayout(0, asl))
                sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), sr_native[:, None] * (K // 128) + base * (BK // 128) + sk_native[None, :] // 4)
                sa = (sa_words >> sk_native[None, :] % 4 * 8).to(gl.uint8)
            else:
                ak_scale = gl.arange(0, BK // 128, gl.SliceLayout(0, sl))
                sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), sr[:, None] * (K // 128) + base * (BK // 128) + ak_scale[None, :])
                sa = _word_bytes(sa_words)
            if REGISTER_A:
                a = gl.convert_layout(_word_bytes(aw), ad, assert_trivial=True)
            else:
                a_shared.store(_word_bytes(aw))
            if REGISTER_B:
                b = gl.convert_layout(_word_bytes(bw).T, bd, assert_trivial=True)
            else:
                b_shared.store(_word_bytes(bw).T)
            if not DIRECT_SCALES:
                as_shared.store(sa)
                bs_shared.store(sb)
            if not REGISTER_A and (not REGISTER_B) and (not DIRECT_SCALES):
                acc = gl.amd.cdna4.mfma_scaled(a_shared.load(ad), as_shared.load(asl), 'e2m1', b_shared.load(bd), bs_shared.load(bsl), 'e2m1', acc)
            else:
                if not REGISTER_A:
                    a = a_shared.load(ad)
                if not REGISTER_B:
                    b = b_shared.load(bd)
                if not DIRECT_SCALES:
                    sa = as_shared.load(asl)
                    sb = bs_shared.load(bsl)
                acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
        if UP:
            EP_M: gl.constexpr = 1 if SINGLE_TOKEN else 16
            if SINGLE_TOKEN:
                er = gl.arange(0, EP_M, gl.SliceLayout(1, mma))
                selected = gl.gather(acc, er[:, None] + gl.full((EP_M, BN), 0, gl.int32, mma), 0)
            else:
                selected = acc
            gate, up = gl.split(gl.permute(selected.reshape((EP_M, 2, BN // 2)), (0, 2, 1)))
            ep: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [WARPS, 1], [1, 0]) if SINGLE_TOKEN else gl.BlockedLayout([1, 8], [8, 8], [WARPS, 1], [1, 0])
            if SINGLE_TOKEN:
                gate = gl.convert_layout(gate, ep)
                up = gl.convert_layout(up, ep)
            gate = gl.where(shared, gate.to(gl.bfloat16).to(gl.float32), gate)
            up = gl.where(shared, up.to(gl.bfloat16).to(gl.float32), up)
            activated = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
            activated = gl.convert_layout(activated, ep)
            codes, scales = _encode_groups(activated.reshape((EP_M * (BN // 64), 32)), shared)
            codes = codes.reshape((EP_M, BN // 4))
            scales = scales.reshape((EP_M, BN // 64))
            if SINGLE_TOKEN:
                eri = gl.arange(0, EP_M, gl.SliceLayout(1, route_layout))
                ep_routes = gl.gather(routes, eri, 0)
                ep_valid = gl.gather(valid, eri, 0)
            else:
                ep_routes = routes
                ep_valid = valid
            nn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, codes.type.layout))
            qr = gl.convert_layout(ep_routes, gl.SliceLayout(1, codes.type.layout))
            qv = gl.convert_layout(ep_valid, gl.SliceLayout(1, codes.type.layout))
            gl.store(Q + qr[:, None] * (N // 4) + nn[None, :], codes, qv[:, None])
            nn_s = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, scales.type.layout))
            sr_out = gl.convert_layout(ep_routes, gl.SliceLayout(1, scales.type.layout))
            sv_out = gl.convert_layout(ep_valid, gl.SliceLayout(1, scales.type.layout))
            gl.store(QS + sr_out[:, None] * (N // 64) + nn_s[None, :], scales, sv_out[:, None])
        else:
            result = gl.where(shared, acc.to(gl.bfloat16).to(gl.float32), acc)
            nn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
            pr = gl.convert_layout(routes, gl.SliceLayout(1, mma))
            pv = gl.convert_layout(valid, gl.SliceLayout(1, mma))
            gl.store(P + pr[:, None] * N + nn[None, :], result, pv[:, None])

@gluon.jit
def _reduce_parts(P, Weights, Y, H: gl.constexpr):
    m = gl.program_id(0)
    h = gl.program_id(1) * 256 + gl.arange(0, 256, gl.BlockedLayout([1], [64], [1], [0]))
    value = gl.full((256,), 0.0, gl.float32, gl.BlockedLayout([1], [64], [1], [0]))
    for rank in gl.static_range(8):
        weight = gl.load(Weights + m * 8 + rank)
        contribution = gl.load(P + (m * 9 + rank) * H + h)
        value += contribution * weight
    value += gl.load(P + (m * 9 + 8) * H + h)
    gl.store(Y + m * H + h, value)

@gluon.jit
def _tiny_projection(X, XS, W, WS, Ids, token, column, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, WARPS: gl.constexpr, SHARED: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, CACHE: gl.constexpr):
    TN: gl.constexpr = BN if SHARED else 8 * BN
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, WARPS])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [TN, BK // 32])
    load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WARPS, 1], [0, 1])
    mi = gl.arange(0, 16, gl.SliceLayout(1, load_layout))
    row = token * 9 + (mi * 0 + 8 if SHARED else mi % 8)
    ki = gl.arange(0, BK // 8, gl.SliceLayout(0, load_layout))
    vn = gl.arange(0, TN, gl.SliceLayout(1, load_layout))
    n = column * BN + vn % BN
    expert = LOCAL_EXPERTS if SHARED else gl.load(Ids + token * 8 + vn // BN)
    ar = gl.arange(0, 16, gl.SliceLayout(1, asl))
    ar = token * 9 + (ar * 0 + 8 if SHARED else ar % 8)
    ak = gl.arange(0, BK // 32, gl.SliceLayout(0, asl))
    bvn = gl.arange(0, TN, gl.SliceLayout(1, bsl))
    bn = column * BN + bvn % BN
    be = LOCAL_EXPERTS if SHARED else gl.load(Ids + token * 8 + bvn // BN)
    bk = gl.arange(0, BK // 32, gl.SliceLayout(0, bsl))
    acc = gl.zeros((16, TN), gl.float32, mma)
    for base in range(K // BK):
        aw = gl.load(X.to(gl.pointer_type(gl.uint32)) + row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
        offset = _weight_offset(n[:, None], base * BK + 8 * ki[None, :], K) // 4
        if SHARED:
            offset += LOCAL_EXPERTS * (N * K // 8)
        else:
            offset += expert[:, None] * (N * K // 8)
        offset = gl.max_contiguous(gl.multiple_of(offset, (1, 4)), (1, 4))
        bw = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset, cache=CACHE)
        asw = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), ar[:, None] * (K // 128) + base * (BK // 128) + ak[None, :] // 4)
        sa = (asw >> ak[None, :] % 4 * 8).to(gl.uint8)
        sk = base * (BK // 32) + bk
        so = bn[:, None] // 32 * (K // 4) + sk[None, :] // 8 * 64
        so += sk[None, :] % 4 * 16 + bn[:, None] % 16
        if SHARED:
            so += LOCAL_EXPERTS * N * (K // 128)
        else:
            so += be[:, None] * N * (K // 128)
        bsw = gl.amd.cdna4.buffer_load(WS.to(gl.pointer_type(gl.uint32)), so)
        shift = (sk[None, :] // 4 % 2 * 2 + bn[:, None] // 16 % 2) * 8
        sb = (bsw >> shift).to(gl.uint8)
        a = gl.convert_layout(_word_bytes(aw), ad, assert_trivial=True)
        b = gl.convert_layout(_word_bytes(bw).T, bd, assert_trivial=True)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    return acc

@gluon.jit
def _down_reduce_tiny(X, XS, W, WS, Ids, Weights, Y, N: gl.constexpr, K: gl.constexpr, GRID_TRANSPOSE: gl.constexpr, LOCAL_EXPERTS: gl.constexpr, BN: gl.constexpr=32, BK: gl.constexpr=512, WARPS: gl.constexpr=2, CACHE: gl.constexpr='.ca', COLUMN_GROUP: gl.constexpr=0):
    gl.static_assert((WARPS == 1 or WARPS == 2) and BN == 16 * WARPS)
    if COLUMN_GROUP:
        token = gl.program_id(0) // COLUMN_GROUP
        column = gl.program_id(1) * COLUMN_GROUP + gl.program_id(0) % COLUMN_GROUP
    else:
        token = gl.program_id(1) if GRID_TRANSPOSE else gl.program_id(0)
        column = gl.program_id(0) if GRID_TRANSPOSE else gl.program_id(1)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, WARPS])
    acc = _tiny_projection(X, XS, W, WS, Ids, token, column, N, K, BN, BK, WARPS, False, LOCAL_EXPERTS, CACHE)
    vn = gl.arange(0, 8 * BN, gl.SliceLayout(0, mma))
    diagonal = gl.gather(acc, vn[None, :] // BN, 0).reshape((8, BN))
    ep: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[1, 0], [2, 0], [4, 0]], lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 0], [0, 0]], warp_bases=[[0, 16]] if WARPS == 2 else [], block_bases=[], shape=[8, BN])
    diagonal = gl.convert_layout(diagonal, ep)
    value = gl.full((1, BN), 0.0, gl.float32, ep)
    for rank in gl.static_range(8):
        part = gl.gather(diagonal, gl.full((1, BN), rank, gl.int32, ep), 0)
        weight = gl.load(Weights + token * 8 + rank)
        value += part * weight
    shared_acc = _tiny_projection(X, XS, W, WS, Ids, token, column, N, K, BN, BK, WARPS, True, LOCAL_EXPERTS, CACHE)
    shared = gl.gather(shared_acc, gl.full((1, BN), 0, gl.int32, mma), 0)
    shared = gl.convert_layout(shared, ep).to(gl.bfloat16).to(gl.float32)
    value += shared
    n = column * BN + gl.arange(0, BN, gl.SliceLayout(0, ep))
    gl.store(Y + token * N + n[None, :], value)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale, expert_start=0, routed_scaling_factor=2.5, fuse_shared_expert=False):
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    local_experts = w13.shape[0] - 1

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, device=x.device, dtype=dtype)
    splits = 12 if m <= 2 else 8
    logits = empty((splits, m, 256), torch.float32)
    ids = empty((m, 8), torch.int32)
    weights = empty((m, 8), torch.float32)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    aq = empty((m * 9, intermediate // 2), torch.uint8)
    aqs = empty((m * 9, intermediate // 32), torch.uint8)
    parts = empty((m * 9, h), torch.float32) if m > 4 and m not in (8, 16) else None
    out = empty((m, h))
    _front_split[triton.cdiv(m, 16) * 16 * splits + triton.cdiv(m * (h // 32), 16),](x, router, logits, xq, xs, m, h, x.stride(0), splits, num_warps=1, enable_fp_fusion=False)
    if m > 4 and m != 8:
        _select_split[m,](logits, correction_bias, ids, weights, expert_start, m, splits, local_experts, routed_scaling_factor, num_warps=1, enable_fp_fusion=False)
    up_warps = 1 if m in (4, 8) else 2
    up_k = 512 if m == 4 else 1024 if m <= 4 or m == 16 else 256
    up_group = 16 if m in (2, 4, 8) else 8 if m == 16 else 1
    up_cache = '.ca' if m in (2, 4, 8) else '.cg'
    _expert[m * 9 * up_group, 2 * intermediate // (64 * up_group)](xq, xs, w13, w13_scale, ids, aq, aqs, parts, expert_start, m, 2 * intermediate, h, 64, up_k, True, local_experts, up_warps, CACHE=up_cache, COLUMN_GROUP=up_group, REGISTER_A=m <= 4 or m == 8, REGISTER_B=m <= 4 or m in (8, 16), FUSED_SELECT=m <= 4 or m == 8, Logits=logits, Bias=correction_bias, Weights=weights, ROUTER_SPLITS=splits, PACK_SCALES=m == 1, ROUTED_SCALE=routed_scaling_factor, num_warps=up_warps, enable_fp_fusion=False)
    if m <= 4 or m in (8, 16):
        down_width = 16 if m == 2 else 32
        down_warps = 1 if m == 2 else 2
        transpose = m > 1
        down_group = 16 if m in (8, 16) else 0
        grid = (m * down_group, h // (down_width * down_group)) if down_group else (h // down_width, m) if transpose else (m, h // down_width)
        _down_reduce_tiny[grid](aq, aqs, w2, w2_scale, ids, weights, out, h, intermediate, transpose, local_experts, down_width, 512, down_warps, COLUMN_GROUP=down_group, num_warps=down_warps, enable_fp_fusion=False)
    else:
        down_n = 64
        down_warps = 2
        down_k = 256 if m == 7 else 512
        down_group = 8 if m >= 8 else 1
        down_cache = '.ca' if m > 8 else '.cg'
        down_registers = m >= 8
        _expert[m * 9 * down_group, h // (down_n * down_group)](aq, aqs, w2, w2_scale, ids, aq, aqs, parts, expert_start, m, h, intermediate, down_n, down_k, False, local_experts, down_warps, down_cache, down_group, REGISTER_A=down_registers, REGISTER_B=down_registers, DIRECT_SCALES=down_registers, num_warps=down_warps, enable_fp_fusion=False)
        _reduce_parts[m, h // 256](parts, weights, out, h, num_warps=1, enable_fp_fusion=False)
    return out
