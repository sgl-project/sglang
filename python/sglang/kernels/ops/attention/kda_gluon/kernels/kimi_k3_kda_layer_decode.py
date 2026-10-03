# ruff: noqa: E741, F841
# fmt: off
"""Selected Kimi-K3 whole-layer KDA decode schedules for gfx950.

Ported from OpenAI-Partners/artemis-kernel-integrations#17 at commit
35b249f7a551278946a81b7da1d58c286c41fb8f.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import range as loop_range
_CONV = gl.constexpr(gl.BlockedLayout([1, 1], [1, 64], [4, 2], [1, 0]))
_OUTPUT = gl.constexpr(gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0]))
_VECTOR = gl.constexpr(gl.BlockedLayout([2], [64], [8], [0]))


@gluon.jit
def _m1_projection_tile(
    X, weight, Y, tile, M: gl.constexpr, XM: gl.constexpr,
    weight_stride, TEMPORAL: gl.constexpr,
):

    CN: gl.constexpr = 16
    BK: gl.constexpr = 128
    WEIGHT_WINDOW: gl.constexpr = 5
    ACTIVATION_WINDOW: gl.constexpr = 30 if M == 2 else 20
    K: gl.constexpr = 7168
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[1, 1],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    load_a: gl.constexpr = (
        gl.BlockedLayout([1, 2], [1, 64], [1, 1], [1, 0]) if M == 1
        else gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    )
    load_b: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    rows = gl.arange(0, M, gl.SliceLayout(1, load_a))
    ak = gl.arange(0, BK, gl.SliceLayout(0, load_a))
    cols = gl.arange(0, CN, gl.SliceLayout(0, load_b))
    bk = gl.arange(0, BK, gl.SliceLayout(1, load_b))
    ao = rows[:, None] * XM + ak[None, :]
    bo = cols[None, :] * weight_stride + bk[:, None]
    acc = gl.zeros((M, CN), gl.float32, mma)
    inputs = ()
    weights = ()

    if M == 2:
        for block in gl.static_range(WEIGHT_WINDOW):
            weights += (gl.amd.cdna4.buffer_load(weight + block * BK, bo, cache=".cg"),)
            for a_panel in gl.static_range(6):
                inputs += (gl.amd.cdna4.buffer_load(X + (block * 6 + a_panel) * BK, ao),)
    elif M == 4:
        for block in gl.static_range(WEIGHT_WINDOW):
            for a_panel in gl.static_range(4):
                inputs += (gl.amd.cdna4.buffer_load(X + (block * 4 + a_panel) * BK, ao),)
            weights += (gl.amd.cdna4.buffer_load(
                weight + block * BK, bo,
                cache="" if TEMPORAL or block < 2 else ".cg",
            ),)
    else:

        for block in gl.static_range(WEIGHT_WINDOW):
            inputs += (gl.amd.cdna4.buffer_load(X + block * BK, ao),)
            weights += (gl.amd.cdna4.buffer_load(weight + block * BK, bo, cache=".cg"),)
        for block in gl.static_range(WEIGHT_WINDOW, ACTIVATION_WINDOW):
            inputs += (gl.amd.cdna4.buffer_load(X + block * BK, ao),)
    for block in gl.static_range(K // BK):
        a, b = inputs[0], weights[0]
        inputs, weights = inputs[1:], weights[1:]
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a, dot_a), gl.convert_layout(b, dot_b), acc,
        )
        if block + ACTIVATION_WINDOW < K // BK:
            inputs += (gl.amd.cdna4.buffer_load(
                X + (block + ACTIVATION_WINDOW) * BK, ao,
            ),)
        if block + WEIGHT_WINDOW < K // BK:
            weights += (gl.amd.cdna4.buffer_load(
                weight + (block + WEIGHT_WINDOW) * BK, bo, cache="" if TEMPORAL else ".cg",
            ),)
    out_rows = gl.arange(0, M, gl.SliceLayout(1, mma))
    out_cols = tile * CN + gl.arange(0, CN, gl.SliceLayout(0, mma))
    gl.store(Y + out_rows[:, None] * 6336 + out_cols[None, :], acc.to(gl.bfloat16))


@gluon.jit
def _m1_input_projections(
    X, WQ, WB, Y, M: gl.constexpr,
    XM: gl.constexpr, WN: gl.constexpr, WBN: gl.constexpr,
):
    TILES: gl.constexpr = 393
    pid = gl.program_id(0)

    if M == 1:
        tile = pid * 49 % TILES
    else:
        BANDS: gl.constexpr = 16 if M == 4 else 8
        band = pid % BANDS
        tile = band * (TILES // BANDS) + gl.minimum(band, TILES % BANDS) + pid // BANDS
    if M == 4:


        if tile < 384:
            _m1_projection_tile(X, WQ + tile * 16 * WN, Y, tile, M, XM, WN, False)
        else:
            _m1_projection_tile(X, WB + (tile - 384) * 16 * WBN, Y, tile, M, XM, WBN, True)
    else:
        is_q = tile < 384
        weight = gl.where(is_q, WQ, WB)
        weight_stride = gl.where(is_q, WN, WBN)
        weight_tile = gl.where(is_q, tile, tile - 384)
        weight += weight_tile * 16 * weight_stride
        _m1_projection_tile(X, weight, Y, tile, M, XM, weight_stride, False)


@gluon.jit
def _m1_sigmoid(x):
    return 1.0 / (1.0 + gl.exp2(x * -1.4426950408889634))


@gluon.jit
def _m1_convolve(X, CW, CS, row, head, slot, SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr):
    group = gl.arange(0, 4, gl.SliceLayout(1, _CONV))
    channel = gl.arange(0, 128, gl.SliceLayout(0, _CONV))
    offset = (group[:, None] * 12 + head) * 128 + channel[None, :]
    valid = group[:, None] < 3
    history = CS + slot * SC0
    old0 = gl.amd.cdna4.buffer_load(history, offset * SC2, valid, 0).to(gl.float32)
    old1 = gl.amd.cdna4.buffer_load(history, SC1 + offset * SC2, valid, 0)
    old2 = gl.amd.cdna4.buffer_load(history, 2 * SC1 + offset * SC2, valid, 0)
    x = gl.load(X + row * 6336 + offset)
    wl: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 64, 1], [4, 2, 1], [2, 1, 0])
    wo = gl.convert_layout(offset, gl.SliceLayout(2, wl), assert_trivial=True)
    wvalid = gl.convert_layout(valid, gl.SliceLayout(2, wl), assert_trivial=True)
    tap = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    weights = gl.amd.cdna4.buffer_load(CW, wo[:, :, None] * 4 + tap[None, None, :], wvalid[:, :, None], 0)
    even, odd = gl.split(gl.reshape(weights, (4, 128, 2, 2)))
    w0, w2 = gl.split(even)
    w1, w3 = gl.split(odd)
    w0 = gl.convert_layout(w0, _CONV)
    w1 = gl.convert_layout(w1, _CONV)
    w2 = gl.convert_layout(w2, _CONV)
    w3 = gl.convert_layout(w3, _CONV)
    z = old0 * w0 + old1.to(gl.float32) * w1 + old2.to(gl.float32) * w2 + x.to(gl.float32) * w3
    z = gl.where(group[:, None] == 3, x.to(gl.float32), z)
    sigmoid = _m1_sigmoid(z)
    z = gl.where(group[:, None] == 3, sigmoid, (z * sigmoid).to(gl.bfloat16).to(gl.float32))
    return z, (old1, old2, x, history, offset * SC2, valid)


@gluon.jit
def _m1_forget(X, FW, m, h, SW: gl.constexpr):
    pl: gl.constexpr = gl.BlockedLayout([1, 1, 16], [8, 8, 1], [1, 8, 1], [2, 0, 1])
    al: gl.constexpr = gl.BlockedLayout([1, 1, 1], [8, 8, 1], [1, 8, 1], [0, 1, 2])
    split = gl.arange(0, 8, gl.SliceLayout(1, gl.SliceLayout(2, pl)))
    row = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, pl)))
    kk = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, pl)))
    wk = split[:, None, None] * 16 + kk[None, None, :]
    w = gl.amd.cdna4.buffer_load(FW + h * 128 * SW, row[None, :, None] * SW + wk)
    a = gl.amd.cdna4.buffer_load(X + m * 6336 + 6144, wk)
    w = gl.convert_layout(w, gl.DotOperandLayout(0, al, 0))
    a = gl.convert_layout(gl.permute(a, (0, 2, 1)), gl.DotOperandLayout(1, al, 0))
    acc = gl.dot_fma(w, a, gl.zeros((8, 128, 1), gl.float32, al))
    f = gl.sum(gl.sum(acc, 2), 0)
    return f.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _m1_split_vector(v, ST: gl.constexpr):
    left, right = gl.split(gl.permute(gl.reshape(v, (2, 64)), (1, 0)))
    v0, v1 = gl.split(gl.permute(gl.reshape(left, (2, 32)), (1, 0)))
    v2, v3 = gl.split(gl.permute(gl.reshape(right, (2, 32)), (1, 0)))
    layout: gl.constexpr = gl.SliceLayout(0, ST)
    return (gl.convert_layout(v0, layout, assert_trivial=True),
            gl.convert_layout(v1, layout, assert_trivial=True),
            gl.convert_layout(v2, layout, assert_trivial=True),
            gl.convert_layout(v3, layout, assert_trivial=True))


@gluon.jit
def _m1_contract(packets, vectors, ORDER: gl.constexpr):
    partial = packets[ORDER[0]] * vectors[ORDER[0]][None, :]
    for j in gl.static_range(1, 4):
        partial = gl.fma(packets[ORDER[j]], vectors[ORDER[j]][None, :], partial)
    return gl.sum(partial, 1)


@gluon.jit
def _m1_recurrent_head(X, CW, FW, CS, S, IDX, A, DT, NW, O,
                    M: gl.constexpr, SW: gl.constexpr, SI: gl.constexpr,
                    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                    SS: gl.constexpr, LOWER, EPS):

    ST: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [8, 1], [1, 0])
    ST_T: gl.constexpr = gl.BlockedLayout([4, 1], [8, 8], [1, 8], [0, 1])

    LC: gl.constexpr = '.cg' if M <= 2 else ''
    SC: gl.constexpr = '.cs' if M <= 2 else '.wt'
    pid = gl.program_id(0)
    if M == 2:
        m = pid % M
        h = pid // M
    else:
        m = pid // 12
        h = pid % 12
    mh = m * 12 + h
    i = gl.arange(0, 128, _VECTOR)
    slot = gl.load(IDX + m * SI).to(gl.int64)
    if slot < 0:
        gl.store(O + (m * 6336 + h * 128 if M == 2 else mh * 128) + i, 0)
    else:
        nw = gl.load(NW + i).to(gl.float32)
        beta = _m1_sigmoid(gl.load(X + m * 6336 + 6272 + h).to(gl.float32))
        if M == 1:
            r = gl.arange(0, 128, gl.SliceLayout(1, ST))
            c = gl.arange(0, 32, gl.SliceLayout(0, ST))
            sb = S + slot * SS + h * 16384
            so = r[:, None] * 128 + c[None, :]
            prefix = gl.amd.cdna4.buffer_load(sb, so, cache=LC)
        if M == 4:
            rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
        z, history = _m1_convolve(X, CW, CS, m, h, slot, SC0, SC1, SC2)
        if M != 1:
            r = gl.arange(0, 128, gl.SliceLayout(1, ST))
            c = gl.arange(0, 32, gl.SliceLayout(0, ST))
            sb = S + slot * SS + h * 16384
            so = r[:, None] * 128 + c[None, :]
            if M == 2:
                prefix = gl.amd.cdna4.buffer_load(sb, so, cache=LC)
        if M == 1:
            extra_packet = gl.amd.cdna4.buffer_load(sb, so + 64, cache=LC)
        elif M == 2:
            extra_packet = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
        if M == 4:

            last_packet = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
            prefix = gl.amd.cdna4.buffer_load(sb, so + 64, cache=LC)
            prefix2 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=LC)
            first_packet = gl.amd.cdna4.buffer_load(sb, so, cache=LC)
        f = _m1_forget(X, FW, m, h, SW)
        if M == 4:
            decay_layout: gl.constexpr = gl.DistributedLinearLayout(
                [], [[0], [0], [64], [1], [2], [4]],
                [[8], [16], [32]], [], [128],
            )
            f = gl.convert_layout(f, decay_layout)
        if M != 4:
            prefix2 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=LC)
        shared = gl.allocate_shared_memory(gl.float32, (8, 128), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        partial_norm = gl.sum(gl.reshape(z * z, (4, 2, 64)), 2)
        shared.slice(7, 1).slice(0, 8, dim=1).store(gl.reshape(partial_norm, (1, 8)))
        shared.slice(0, 4).store(z)
        if M == 1:

            last_packet = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
        dt = gl.load(DT + h * 128 + gl.arange(0, 128, f.type.layout))
        if M != 4:
            rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
        decay = gl.exp2(LOWER * _m1_sigmoid(rate * (f + dt)) * 1.4426950408889634)
        if M == 4:
            shared.slice(4, 1).store(gl.reshape(decay, (1, 128)))
        else:
            shared.slice(4, 1).store(decay[None, :])
        if M == 1:
            q = gl.sum(shared.slice(0, 1).load(ST), 0)
        key = gl.sum(shared.slice(1, 1).load(ST), 0)
        value = gl.sum(shared.slice(2, 1).load(ST_T), 0)
        value = gl.convert_layout(value, gl.SliceLayout(1, ST))
        decay = gl.sum(shared.slice(4, 1).load(ST), 0)
        gate = gl.sum(shared.slice(3, 1).load(_OUTPUT), 0)
        gate = gl.convert_layout(gate, _VECTOR)
        nl: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [0, 1])
        qp = shared.slice(7, 1).slice(0, 2, dim=1).load(nl)
        kp = shared.slice(7, 1).slice(2, 2, dim=1).load(nl)
        qnorm = gl.rsqrt(gl.sum(gl.sum(qp, 1), 0) + 1e-6)
        knorm = gl.rsqrt(gl.sum(gl.sum(kp, 1), 0) + 1e-6)
        keys = _m1_split_vector(key, ST)
        if M == 1:
            queries = _m1_split_vector(q, ST)
        decays = _m1_split_vector(decay, ST)
        decayed = ()
        for p in gl.static_range(4):
            if p == (0 if M <= 2 else 2):
                packet = prefix
            elif p == 1:
                packet = prefix2
            elif M == 1 and p == 2:
                packet = extra_packet
            elif M == 2 and p == 3:
                packet = extra_packet
            elif M == 1 and p == 3:
                packet = last_packet
            elif M == 4 and p == 0:
                packet = first_packet
            elif M == 4 and p == 3:
                packet = last_packet
            else:
                packet = gl.amd.cdna4.buffer_load(sb, so + p * 32, cache=LC)
            decayed += (packet * decays[p][None, :],)
        P_ORDER: gl.constexpr = (0, 1, 2, 3) if M == 2 else (0, 2, 1, 3) if M == 4 else (3, 2, 1, 0)
        prediction = _m1_contract(decayed, keys, P_ORDER)
        if M >= 2:
            q = gl.sum(shared.slice(0, 1).load(ST), 0)
            queries = _m1_split_vector(q, ST)
        if M == 4:
            delta = gl.fma(-prediction, knorm, value) * (beta * knorm)
        else:
            delta = (value - prediction * knorm) * beta
            delta = delta * knorm
        if M == 2:
            projected = _m1_contract(decayed, queries, P_ORDER)
            key_query = gl.sum(key * q, 0)
            out = gl.fma(delta, key_query, projected) * qnorm * 128 ** (-0.5)
        UPDATE_ORDER: gl.constexpr = (0, 1, 2, 3) if M == 2 else (3, 2, 1, 0)
        for p in gl.static_range(4):
            updated = gl.fma(delta[:, None], keys[UPDATE_ORDER[p]][None, :], decayed[UPDATE_ORDER[p]])
            if M != 2:
                if p == 0:
                    partial = updated * queries[UPDATE_ORDER[p]][None, :]
                else:
                    partial = gl.fma(updated, queries[UPDATE_ORDER[p]][None, :], partial)
            gl.amd.cdna4.buffer_store(updated, sb, so + UPDATE_ORDER[p] * 32, cache=SC)
        if M != 2:
            out = gl.sum(partial, 1) * qnorm * 128 ** (-0.5)
        out = out.to(gl.bfloat16).to(gl.float32)
        out = gl.convert_layout(out, gl.SliceLayout(0, ST_T))
        shared.slice(6, 1).store(out[None, :])
        out = gl.sum(shared.slice(6, 1).load(_OUTPUT), 0)
        out = gl.convert_layout(out, _VECTOR)
        scale = gl.rsqrt(gl.sum(out * out, 0) / 128 + EPS)
        if M == 2:
            out = out * scale * nw * gate
        else:
            out = out * scale * (nw * gate)
        old1, old2, x, hb, ho, valid = history
        gl.amd.cdna4.buffer_store(old1, hb, ho, valid)
        gl.amd.cdna4.buffer_store(old2, hb, SC1 + ho, valid)
        gl.amd.cdna4.buffer_store(x, hb, 2 * SC1 + ho, valid)
        gl.store(O + (m * 6336 + h * 128 if M == 2 else mh * 128) + i, out)


@gluon.jit
def _m1_prefetch_output_weights(W, SW: gl.constexpr, pid, M: gl.constexpr,
                             CTAS: gl.constexpr, PREFETCH_EXTENT: gl.constexpr):

    STEP: gl.constexpr = 64
    LINES: gl.constexpr = 896 * 1536 // STEP
    CHUNK: gl.constexpr = gl.cdiv(LINES, CTAS // 8)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lane = gl.arange(0, PREFETCH_EXTENT, layout)
    logical = pid // 8 * CHUNK + lane
    element = ((pid + M * 12) % 8) * (896 * 1536) + logical * STEP
    row = element // 1536
    col = element % 1536 + (row & 1) * 32
    valid = (logical < LINES) & (lane < CHUNK)
    word = gl.load(W + row * SW + col, valid, other=0)
    word = word.to(gl.uint16, bitcast=True).to(gl.uint32)
    gl.inline_asm_elementwise("", constraints="=v,0", args=[word],
                              dtype=gl.uint32, is_pure=False, pack=1)


@gluon.jit
def _m1_recurrent_and_prefetch(X, CW, FW, CS, S, IDX, A, DT, NW, O, WO,
                            M: gl.constexpr, SW: gl.constexpr, SI: gl.constexpr,
                            SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                            SS: gl.constexpr, LOWER, EPS, OWN: gl.constexpr,
                            PREFETCH_CTAS: gl.constexpr, PREFETCH_EXTENT: gl.constexpr):

    pid = gl.program_id(0)
    if PREFETCH_CTAS == 0 or pid < M * 12:
        _m1_recurrent_head(X, CW, FW, CS, S, IDX, A, DT, NW, O, M, SW, SI,
                        SC0, SC1, SC2, SS, LOWER, EPS)
    else:
        _m1_prefetch_output_weights(WO, OWN, pid - M * 12, M, PREFETCH_CTAS, PREFETCH_EXTENT)


@gluon.jit
def _m1_output_m1(X, W, Y, N: gl.constexpr, SWN: gl.constexpr):
    BN: gl.constexpr = 4
    PACK: gl.constexpr = 8
    PARTITIONS: gl.constexpr = 64
    tile = gl.program_id(0).to(gl.uint32)
    tile = tile % 8 * (N // (BN * 8)) + tile // 8
    gl.assume(tile >= 0)
    gl.assume(tile < N // BN)
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [1, 1, 1], [0, 2, 1])
    al: gl.constexpr = gl.DotOperandLayout(0, layout, 0)
    bl: gl.constexpr = gl.DotOperandLayout(1, layout, 0)
    aq = gl.arange(0, PARTITIONS, gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ap = gl.arange(0, PACK, gl.SliceLayout(0, gl.SliceLayout(1, al)))
    bq = gl.arange(0, PARTITIONS, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    bp = gl.arange(0, PACK, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    bn = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    ao = aq[:, None, None] * PACK + ap[None, None, :]
    bo = bn[None, None, :] * SWN + bq[:, None, None] * PACK + bp[None, :, None]
    a0 = gl.load(X + ao)
    a1 = gl.load(X + ao + 512)
    a2 = gl.load(X + ao + 1024)
    acc = gl.zeros((PARTITIONS, 1, BN), gl.float32, layout)
    for step in gl.static_range(3):
        if step == 0:
            a = a0
        elif step == 1:
            a = a1
        else:
            a = a2
        b = gl.load(W + bo + step * 512)
        acc = gl.dot_fma(a, b, acc)
    acc = gl.reshape(acc, (2, 32, 1, BN))
    first: gl.constexpr = gl.DistributedLinearLayout(
        [[1, 0, 0, 0], [0, 0, 0, 1]],
        [[0, 1, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 0, 0, 2]],
        [], [], [2, 32, 1, BN],
    )
    acc = gl.sum(gl.convert_layout(acc, first), 0)
    acc = gl.reshape(acc, (2, 16, 1, BN))
    second: gl.constexpr = gl.BlockedLayout([2, 1, 1, 1], [1, 16, 1, 4], [1, 1, 1, 1], [1, 3, 2, 0])
    acc = gl.sum(gl.convert_layout(acc, second), 0)
    acc = gl.sum(acc, 0)
    ol: gl.constexpr = acc.type.layout
    cols = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, ol))
    gl.amd.cdna4.buffer_store(acc.to(gl.bfloat16), Y, cols[None, :])


def kda_layer_decode_m1(
    x, qkvg_weight, beta_forget_weight, output_weight, forget_weight,
    conv_weight, a_log, dt_bias, norm_weight, conv_state, state, state_indices,
    *, lower_bound=-5.0, norm_eps=1e-5, output_tensor=None,
):

    m = x.shape[0]
    assert m in (1, 2, 4)

    packed = torch.empty((m, 6336), dtype=torch.bfloat16, device=x.device)
    core = packed[:, 4608:6144]
    out = (
        torch.empty((m, 7168), dtype=torch.bfloat16, device=x.device)
        if output_tensor is None else output_tensor
    )
    assert out.shape == (m, 7168) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _m1_input_projections[393,](
        x, qkvg_weight, beta_forget_weight, packed, m,
        x.stride(0), qkvg_weight.stride(0), beta_forget_weight.stride(0),
        num_warps=1,
    )
    prefetch_ctas = 128
    prefetch_extent = triton.next_power_of_2(triton.cdiv(896 * 1536 // 64, prefetch_ctas // 8))
    _m1_recurrent_and_prefetch[m * 12 + prefetch_ctas,](
        packed, conv_weight, forget_weight, conv_state, state, state_indices,
        a_log, dt_bias, norm_weight, core, output_weight, m,
        forget_weight.stride(0), state_indices.stride(0),
        *conv_state.stride(), state.stride(0), lower_bound, norm_eps,
        output_weight.stride(0), prefetch_ctas, prefetch_extent,
        num_warps=8, enable_fp_fusion=False, waves_per_eu=2,
    )
    _m1_output_m1[1792,](
        core, output_weight, out, 7168, output_weight.stride(0), num_warps=1,
    )
    return out, conv_state, state


@gluon.jit
def _m128_stage_masked_panel(sa, sb, X, W, ao, bo, am, bm, offset):
    la = sa._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    lb = sb._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(la, X + offset, ao, am, 0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(lb, W + offset, bo, bm, 0)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m128_masked_dense_tile(X, W, Y, tile_m, tile_n,
                 M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                 SX: gl.constexpr, SW: gl.constexpr,
                 RM: gl.constexpr, RN: gl.constexpr,
                 BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                 NW: gl.constexpr, DEPTH: gl.constexpr, WT: gl.constexpr, SY: gl.constexpr):
    phase: gl.constexpr = BK // 8
    ca: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
    cb: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, NW], [0, 1])
    wm: gl.constexpr = 1 if NW == 1 or BM == 16 else 2
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [wm, NW // wm])
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    sa = ()
    sb = ()
    for i in gl.static_range(DEPTH):
        sa += (gl.allocate_shared_memory(gl.bfloat16, [BM, BK], gl.SwizzledSharedLayout(8, 2, phase, [1, 0])),)
        sb += (gl.allocate_shared_memory(gl.bfloat16, [BK, BN], gl.SwizzledSharedLayout(8, 2, phase, [0, 1])),)
    ar = gl.arange(0, BM, gl.SliceLayout(1, ca))
    ak = gl.arange(0, BK, gl.SliceLayout(0, ca))
    bk = gl.arange(0, BK, gl.SliceLayout(1, cb))
    bc = gl.arange(0, BN, gl.SliceLayout(0, cb))
    ao = ar[:, None] * SX + (ak[None, :] ^ (ar[:, None] // 2 % phase * 8))
    bo = bc[None, :] * SW + (bk[:, None] ^ (bc[None, :] // 2 % phase * 8))
    ao = gl.max_contiguous(gl.multiple_of(ao, [1, 8]), [1, 8])
    bo = gl.max_contiguous(gl.multiple_of(bo, [8, 1]), [8, 1])
    if M % RM == 0:
        am = ar[:, None] < RM
    else:
        am = (ar[:, None] < RM) & (tile_m * RM + ar[:, None] < M)
    if N % RN == 0:
        bm = bc[None, :] < RN
    else:
        bm = (bc[None, :] < RN) & (tile_n * RN + bc[None, :] < N)
    X += tile_m * RM * SX
    W += tile_n * RN * SW
    for i in gl.static_range(DEPTH - 1):
        _m128_stage_masked_panel(sa[i], sb[i], X, W, ao, bo, am, bm, i * BK)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    steady: gl.constexpr = (K // BK - DEPTH + 1) // DEPTH * DEPTH if M == 256 else 0
    for cycle in range(steady // DEPTH):
        for phase in gl.static_range(DEPTH):
            step = cycle * DEPTH + phase
            gl.amd.cdna4.async_copy.wait_group(DEPTH - 2)
            gl.barrier()
            _m128_stage_masked_panel(sa[(phase + DEPTH - 1) % DEPTH], sb[(phase + DEPTH - 1) % DEPTH], X, W, ao, bo, am, bm, (step + DEPTH - 1) * BK)
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(sa[phase], da)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(sb[phase], db)
            acc = gl.amd.cdna4.mfma(a, b, acc)
    for step in gl.static_range(steady, K // BK):
        if step < K // BK - DEPTH + 1:
            gl.amd.cdna4.async_copy.wait_group(DEPTH - 2)
            gl.barrier()
            _m128_stage_masked_panel(sa[(step + DEPTH - 1) % DEPTH], sb[(step + DEPTH - 1) % DEPTH], X, W, ao, bo, am, bm, (step + DEPTH - 1) * BK)
        else:
            gl.amd.cdna4.async_copy.wait_group(K // BK - step - 1)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(sa[step % DEPTH], da)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(sb[step % DEPTH], db)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    out_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [NW, 1], [1, 0])
    out = gl.convert_layout(acc.to(gl.bfloat16), out_layout)
    r = gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    c = gl.arange(0, BN, gl.SliceLayout(0, out_layout))
    offset = (tile_m * RM + r[:, None]) * SY + tile_n * RN + c[None, :]
    mask = (r[:, None] < RM) & (tile_m * RM + r[:, None] < M) & (c[None, :] < RN) & (tile_n * RN + c[None, :] < N)
    gl.amd.cdna4.buffer_store(out, Y, offset, mask, cache='.wt' if WT else '')


@gluon.jit
def _m128_stage_dense_panel(a_shared, b_shared, x_base, w_base, a_offsets, b_offsets):
    a_linear = a_shared._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    b_linear = b_shared._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_linear, x_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_linear, w_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m128_gemm_accumulate(
    X, W, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    SXM: gl.constexpr, SWN: gl.constexpr, BN: gl.constexpr, NW: gl.constexpr,
    tile_m, tile_n, BM: gl.constexpr, BK: gl.constexpr,
    ASYMMETRIC: gl.constexpr, DEPTH: gl.constexpr, LATE_REFILL: gl.constexpr,
):

    FRAGMENT: gl.constexpr = 64 if M == 128 and (N == 6144 or N == 7168) else 32
    STAGES: gl.constexpr = 3 if ASYMMETRIC else DEPTH
    PHASE: gl.constexpr = BK // 8
    PER_PHASE: gl.constexpr = 1 if M == 128 and N == 7168 else 2
    gl.static_assert(M % BM == 0 and N % BN == 0 and K % BK == 0)
    copy_a: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BK, BK // 8], [NW, 1], [1, 0])
    copy_b: gl.constexpr = gl.BlockedLayout([8, 1], [BK // 8, 512 // BK], [1, NW], [0, 1])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, NW // 2])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, PER_PHASE, PHASE, [1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, PER_PHASE, PHASE, [0, 1])
    a_slots = ()
    b_slots = ()
    for slot in gl.static_range(2 if ASYMMETRIC else STAGES):
        a_slots += (gl.allocate_shared_memory(gl.bfloat16, [BM, BK], shared_a),)
    for slot in gl.static_range(STAGES):
        b_slots += (gl.allocate_shared_memory(gl.bfloat16, [BK, BN], shared_b),)
    rows = gl.arange(0, BM, gl.SliceLayout(1, copy_a))
    cols = gl.arange(0, BN, gl.SliceLayout(0, copy_b))
    a_k = gl.arange(0, BK, gl.SliceLayout(0, copy_a))
    b_k = gl.arange(0, BK, gl.SliceLayout(1, copy_b))
    a_offsets = rows[:, None] * SXM + (a_k[None, :] ^ rows[:, None] // PER_PHASE % PHASE * 8)
    b_offsets = cols[None, :] * SWN + (b_k[:, None] ^ cols[None, :] // PER_PHASE % PHASE * 8)
    a_offsets = gl.max_contiguous(gl.multiple_of(a_offsets, [1, 8]), [1, 8])
    b_offsets = gl.max_contiguous(gl.multiple_of(b_offsets, [8, 1]), [8, 1])
    x_base = X + tile_m * BM * SXM
    w_base = W + tile_n * BN * SWN
    if ASYMMETRIC:
        a_linear = a_slots[0]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_linear, x_base, a_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        for slot in gl.static_range(2):
            b_linear = b_slots[slot]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_linear, w_base + slot * BK, b_offsets, cache_modifier='.cg'
            )
            gl.amd.cdna4.async_copy.commit_group()
        acc = gl.zeros((BM, BN), gl.float32, mma)
        for step in gl.static_range(K // BK):
            gl.amd.cdna4.async_copy.wait_group(1 if step < K // BK - 1 else 0)
            gl.barrier()
            b0 = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[step % 3].slice(0, FRAGMENT, 0), dot_b)
            if step + 1 < K // BK:
                a_spare = a_slots[(step + 1) % 2]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
                gl.amd.cdna4.async_copy.buffer_load_to_shared(a_spare, x_base + (step + 1) * BK, a_offsets)
                gl.amd.cdna4.async_copy.commit_group()
            a0 = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[step % 2].slice(0, FRAGMENT, 1), dot_a)
            acc = gl.amd.cdna4.mfma(a0, b0, acc)
            if (not LATE_REFILL or FRAGMENT == 64) and step + 2 < K // BK:
                b_spare = b_slots[(step + 2) % 3]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    b_spare, w_base + (step + 2) * BK, b_offsets, cache_modifier='.cg'
                )
                gl.amd.cdna4.async_copy.commit_group()
            for fragment in gl.static_range(1, BK // FRAGMENT):
                bf = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[step % 3].slice(fragment * FRAGMENT, FRAGMENT, 0), dot_b)
                af = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[step % 2].slice(fragment * FRAGMENT, FRAGMENT, 1), dot_a)
                acc = gl.amd.cdna4.mfma(af, bf, acc)
                if LATE_REFILL and FRAGMENT == 32 and fragment == 1 and step + 2 < K // BK:
                    b_spare = b_slots[(step + 2) % 3]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
                    gl.amd.cdna4.async_copy.buffer_load_to_shared(
                        b_spare, w_base + (step + 2) * BK, b_offsets, cache_modifier='.cg'
                    )
                    gl.amd.cdna4.async_copy.commit_group()
    else:
        for slot in gl.static_range(STAGES - 1):
            _m128_stage_dense_panel(a_slots[slot], b_slots[slot], x_base + slot * BK, w_base + slot * BK, a_offsets, b_offsets)
        acc = gl.zeros((BM, BN), gl.float32, mma)
        for step in gl.static_range(K // BK - STAGES + 1):
            gl.amd.cdna4.async_copy.wait_group(STAGES - 2)
            gl.barrier()
            future = (step + STAGES - 1) * BK
            _m128_stage_dense_panel(a_slots[(step + STAGES - 1) % STAGES], b_slots[(step + STAGES - 1) % STAGES], x_base + future, w_base + future, a_offsets, b_offsets)
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[step % STAGES], dot_a)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[step % STAGES], dot_b)
            acc = gl.amd.cdna4.mfma(a, b, acc)
        for tail in gl.static_range(STAGES - 1):
            gl.amd.cdna4.async_copy.wait_group(STAGES - 2 - tail)
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[(K // BK - STAGES + 1 + tail) % STAGES], dot_a)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[(K // BK - STAGES + 1 + tail) % STAGES], dot_b)
            acc = gl.amd.cdna4.mfma(a, b, acc)
    return acc


@gluon.jit
def _m128_dense_tile(
    X, W, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, SXM: gl.constexpr,
    SWN: gl.constexpr, BN: gl.constexpr, NW: gl.constexpr, tile_m,
    tile_n, WT: gl.constexpr, SY: gl.constexpr, DEPTH: gl.constexpr = 4,
    BM: gl.constexpr = 64,
):
    WIDE_K: gl.constexpr = (M == 128 or M == 256) and N == 6144
    BK: gl.constexpr = (256 if M == 128 else 128) if WIDE_K else 64
    acc = _m128_gemm_accumulate(
        X, W, M, N, K, SXM, SWN, BN, NW, tile_m, tile_n,
        BM, BK, ASYMMETRIC=WIDE_K, DEPTH=DEPTH, LATE_REFILL=M == 128,
    )
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
    result = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
    out_m = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, store_layout))
    out_n = tile_n * BN + gl.arange(0, BN, gl.SliceLayout(0, store_layout))
    gl.amd.cdna4.buffer_store(result, Y, out_m[:, None] * SY + out_n[None, :], cache='.wt' if WT else '')


@gluon.jit
def _m128_state_row_order(IDX, ORDER, M: gl.constexpr, SI: gl.constexpr,
                    NW: gl.constexpr, BLOCK: gl.constexpr, LOG: gl.constexpr):
    row = gl.arange(0, BLOCK, gl.BlockedLayout([BLOCK // 64], [64], [NW], [0]))
    slot = gl.load(IDX + row * SI, row < M, 0).to(gl.uint32)
    slot = gl.where(row < M, slot, 0xffffffff)
    key = (slot.to(gl.uint64) << 32) | row.to(gl.uint64)
    for outer in gl.static_range(1, LOG + 1):
        for inner in gl.static_range(outer - 1, -1, -1):
            other = gl.gather(key, row ^ (1 << inner), 0)
            ascending = (row & (1 << outer)) == 0
            lower = (row & (1 << inner)) == 0
            key = gl.where(ascending == lower, gl.minimum(key, other), gl.maximum(key, other))
    if IDX.dtype.element_ty == gl.int32:
        gl.store(ORDER + row, key.to(gl.int64), row < M)
    else:
        gl.store(ORDER + row, key.to(gl.int32), row < M)


@gluon.jit
def _m128_project_qkvg_beta(
    X, WQ, WB, Q, B, M: gl.constexpr, SX: gl.constexpr, SQ: gl.constexpr, SB: gl.constexpr,
    BN: gl.constexpr, NW: gl.constexpr, GROUP: gl.constexpr, BR: gl.constexpr, BC: gl.constexpr,
    BD: gl.constexpr, WT: gl.constexpr, QS: gl.constexpr, BS: gl.constexpr,
    IDX, ORDER, SI: gl.constexpr, SORT: gl.constexpr, OB: gl.constexpr, LOG: gl.constexpr,
    BM: gl.constexpr,
):
    qtiles: gl.constexpr = M // BM * (6144 // BN)
    pid = gl.program_id(0)
    if pid < qtiles:
        tile = pid % GROUP * (qtiles // GROUP) + pid // GROUP
        tile_m = tile // (6144 // BN)
        tile_n = tile % (6144 // BN)
        _m128_dense_tile(X, WQ, Q, M, 6144, 7168, SX, SQ, BN, NW,
                    tile_m, tile_n, WT, QS, 4, BM)
    elif pid < qtiles + gl.cdiv(M, 64) * gl.cdiv(144, BC):
        bpid = pid - qtiles
        tm = bpid // gl.cdiv(144, BC)
        tn = bpid % gl.cdiv(144, BC)
        if M == 128:
            _m128_dense_tile(X, WB, B, M, 144, 7168, SX, SB, BC, NW,
                        tm, tn, False, BS, BD)
        else:
            _m128_masked_dense_tile(X, WB, B, tm, tn, M, 144, 7168, SX, SB,
                               BR, BC, BR, BC, 64, NW, BD, False, BS)
    elif SORT:
        _m128_state_row_order(IDX, ORDER, M, SI, NW, OB, LOG)


@gluon.jit
def _m128_output_gemm(
    X, W, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, SXM: gl.constexpr,
    SWN: gl.constexpr, BN: gl.constexpr, NW: gl.constexpr,
    WT: gl.constexpr,
):
    BM: gl.constexpr = 64 if M == 128 else 128
    BK: gl.constexpr = 256 if M == 128 else 128
    GROUPS: gl.constexpr = 8 * (M // BM)
    gl.static_assert(M % BM == 0 and N % BN == 0 and K % BK == 0)
    pid = gl.program_id(0).to(gl.uint32)
    tile_m = (pid >> 3) & (M // BM - 1)
    tile_n = (pid & 7) * (N // BN // 8) + pid // GROUPS
    acc = _m128_gemm_accumulate(
        X, W, M, N, K, SXM, SWN, BN, NW, tile_m, tile_n,
        BM, BK, ASYMMETRIC=True, DEPTH=3, LATE_REFILL=False,
    )
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
    result = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
    out_m = gl.arange(0, BM, gl.SliceLayout(1, store_layout))
    out_n = gl.arange(0, BN, gl.SliceLayout(0, store_layout))
    y_base = Y + tile_m * BM * N + tile_n * BN
    out_offsets = out_m[:, None] * N + out_n[None, :]
    gl.amd.cdna4.buffer_store(result, y_base, out_offsets, cache='.wt' if WT else '')


@gluon.jit
def _m128_eight_parts(values):
    rows: gl.constexpr = values.shape[0]
    even, odd = gl.split(gl.reshape(values, (rows, 16, 4, 2)))
    e04, e26 = gl.split(gl.reshape(even, (rows, 16, 2, 2)))
    o15, o37 = gl.split(gl.reshape(odd, (rows, 16, 2, 2)))
    p0, p4 = gl.split(e04)
    p2, p6 = gl.split(e26)
    p1, p5 = gl.split(o15)
    p3, p7 = gl.split(o37)
    return (p0, p1, p2, p3, p4, p5, p6, p7)


@gluon.jit
def _m128_project_decay_head(
    FA, FW, m, h, SFA: gl.constexpr, SW: gl.constexpr, ROWS: gl.constexpr, OFFSET: gl.constexpr,
    NW: gl.constexpr, A, DT, lower,
):
    pl: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8] if ROWS == 64 else [4, 16], [NW, 1], [1, 0])
    n = gl.arange(0, ROWS, gl.SliceLayout(1, pl)) + OFFSET
    k = gl.arange(0, 128, gl.SliceLayout(0, pl))
    x = gl.amd.cdna4.buffer_load(FA + m * SFA, k)
    w = gl.amd.cdna4.buffer_load(FW + h * 128 * SW, n[:, None] * SW + k[None, :])
    w0, w1, w2, w3, w4, w5, w6, w7 = _m128_eight_parts(w.to(gl.float32))
    input_rows, _ = gl.broadcast(x[None, :].to(gl.float32), w)
    x0, x1, x2, x3, x4, x5, x6, x7 = _m128_eight_parts(input_rows)
    even = gl.fma(w0, x0, w2 * x2) + gl.fma(w4, x4, w6 * x6)
    odd = gl.fma(w1, x1, w3 * x3) + gl.fma(w5, x5, w7 * x7)
    projected = gl.sum(even + odd, 1).to(gl.bfloat16).to(gl.float32)
    projected = gl.convert_layout(projected, gl.SliceLayout(1, pl), assert_trivial=True)
    raw = projected + gl.load(DT + h * 128 + n)
    return gl.exp(lower * _m1_sigmoid(gl.exp(gl.load(A + h)) * raw))


@gluon.jit
def _m128_convolve_head(
    X, CW, CS, m, h, slot, H: gl.constexpr, SX: gl.constexpr, SC0: gl.constexpr, SC1: gl.constexpr,
    SC2: gl.constexpr, LAYOUT: gl.constexpr, PREP, NW: gl.constexpr,
    LOCAL_WEIGHT_BASE: gl.constexpr,
):
    cl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW // 2, 2], [1, 0])
    plane = gl.arange(0, 4, gl.SliceLayout(1, cl))
    channel = gl.arange(0, 128, gl.SliceLayout(0, cl))
    c = plane[:, None] * H * 128 + h * 128 + channel[None, :]
    valid = plane[:, None] < 3
    history_base = CS + slot * SC0 + h * 128 * SC2
    history_offsets = (plane[:, None] * H * 128 + channel[None, :]) * SC2
    old0 = gl.amd.cdna4.buffer_load(history_base, history_offsets, valid, 0)
    old1 = gl.amd.cdna4.buffer_load(history_base, SC1 + history_offsets, valid, 0)
    old2 = gl.amd.cdna4.buffer_load(history_base, 2 * SC1 + history_offsets, valid, 0)
    x = gl.load(X + m * SX + c, valid, 0)
    wl: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 64, 1], [NW // 2, 2, 1], [2, 1, 0])
    wp = gl.arange(0, 4, gl.SliceLayout(1, gl.SliceLayout(2, wl)))
    wc = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, wl)))
    taps = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    weight_c = wp[:, None] * H * 128 + h * 128 + wc[None, :]
    if LOCAL_WEIGHT_BASE:
        weight_offsets = (wp[:, None] * H * 128 + wc[None, :])[:, :, None] * 4 + taps[None, None, :]
        weights = gl.amd.cdna4.buffer_load(CW + h * 128 * 4, weight_offsets, wp[:, None, None] < 3, 0)
    else:
        weights = gl.load(CW + weight_c[:, :, None] * 4 + taps[None, None, :], weight_c[:, :, None] < 3 * H * 128, 0)
    even, odd = gl.split(gl.reshape(weights, (4, 128, 2, 2)))
    w0, w2 = gl.split(even)
    w1, w3 = gl.split(odd)
    w0 = gl.convert_layout(w0, cl)
    w1 = gl.convert_layout(w1, cl)
    w2 = gl.convert_layout(w2, cl)
    w3 = gl.convert_layout(w3, cl)
    z = old0.to(gl.float32) * w0 + old1.to(gl.float32) * w1 + old2.to(gl.float32) * w2 + x.to(gl.float32) * w3
    z = (z * _m1_sigmoid(z)).to(gl.bfloat16)
    changed0 = old0.to(gl.uint16, bitcast=True) != old1.to(gl.uint16, bitcast=True)
    changed1 = old1.to(gl.uint16, bitcast=True) != old2.to(gl.uint16, bitcast=True)
    changed2 = old2.to(gl.uint16, bitcast=True) != x.to(gl.uint16, bitcast=True)
    gl.amd.cdna4.buffer_store(old1, history_base, history_offsets, valid & changed0)
    gl.amd.cdna4.buffer_store(old2, history_base, SC1 + history_offsets, valid & changed1)
    gl.amd.cdna4.buffer_store(x, history_base, 2 * SC1 + history_offsets, valid & changed2)
    handoff = PREP.slice(0, 512)
    handoff.store(gl.reshape(z, (512,)))
    q = handoff.slice(0, 128).load(gl.SliceLayout(0, LAYOUT)).to(gl.float32)
    key = handoff.slice(128, 128).load(gl.SliceLayout(0, LAYOUT)).to(gl.float32)
    return (q, key)


@gluon.jit
def _m128_store_changed_vectors(updated, previous, base, offsets, CACHE: gl.constexpr):
    changed_bits = updated.to(gl.uint32, bitcast=True) ^ previous.to(gl.uint32, bitcast=True)
    rows: gl.constexpr = updated.shape[0]
    groups = gl.reshape(changed_bits, (rows, 8, 16))
    changed = gl.max(groups, 2) != 0
    mask, _ = gl.broadcast(changed[:, :, None], groups)
    gl.amd.cdna4.buffer_store(
        updated, base, offsets, gl.reshape(mask, (rows, 128)), cache=CACHE
    )


@gluon.jit
def _m128_advance_state_slab(previous, value, decay, key, beta):
    decayed = previous * decay[None, :]
    delta = (value - gl.sum(decayed * key[None, :], 1)) * beta
    return gl.fma(delta[:, None], key[None, :], decayed)


@gluon.jit
def _m128_kda_decode(
    X, Gate, FA, Beta, FW, CW, A, DT, W, CS, S, IDX, O, H: gl.constexpr, SX: gl.constexpr,
    SG: gl.constexpr, SFA: gl.constexpr, SB: gl.constexpr, SW: gl.constexpr, SC0: gl.constexpr,
    SC1: gl.constexpr, SC2: gl.constexpr, SS: gl.constexpr, SI: gl.constexpr, lower, eps,
    GROUP: gl.constexpr, LARGE_BATCH: gl.constexpr, NW: gl.constexpr,
    STATE_STORE: gl.constexpr, SO: gl.constexpr, ORDER, SORT: gl.constexpr,
    SLAB_ROWS: gl.constexpr, CONV_NARROW: gl.constexpr,
    PACK_SOURCE, PACK_OUTPUT, PACK_STRIDE: gl.constexpr,
    PACK_TILES: gl.constexpr, M: gl.constexpr,
):
    flat_tile = gl.program_id(1) * (H * GROUP) + gl.program_id(0)
    if PACK_TILES and flat_tile >= M * H:
        if flat_tile < M * H + PACK_TILES:
            _m128_pack_projection_matrix(
                PACK_SOURCE, PACK_OUTPUT, 7168, 1536, PACK_STRIDE,
                flat_tile - M * H, 8192, NW,
            )
    else:
        state_cache: gl.constexpr = '.cg' if STATE_STORE == '.cs' else ''
        layout: gl.constexpr = gl.BlockedLayout(
            [2, 4] if SLAB_ROWS == 64 else [1, 4], [4, 16], [NW, 1], [1, 0]
        )
        pid = gl.program_id(0)
        if not LARGE_BATCH:
            pid = pid.to(gl.uint32)
        m = gl.program_id(1) * GROUP + pid % GROUP
        h = pid // GROUP
        if SORT:
            record = gl.load(ORDER + m)
            if IDX.dtype.element_ty == gl.int32:
                m = record.to(gl.int32)
                slot = (record.to(gl.uint64) >> 32).to(gl.int32).to(gl.int64)
            else:
                m = record
                slot = gl.load(IDX + m * SI).to(gl.int64)
        else:
            slot = gl.load(IDX + m * SI).to(gl.int64)
        if LARGE_BATCH:
            h = (h & ~3) | ((h + slot.to(gl.int32)) & 3)
        v = gl.arange(0, 128, gl.SliceLayout(1, layout))
        k = gl.arange(0, 128, gl.SliceLayout(0, layout))
        if slot < 0:
            gl.store(O + m * SO + h * 128 + v, 0)
        else:
            prep = gl.allocate_shared_memory(gl.bfloat16, (512,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
            auxiliary = gl.allocate_shared_memory(gl.bfloat16, (128,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
            decay_shared = gl.allocate_shared_memory(gl.float32, (128,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
            if LARGE_BATCH:
                decay0 = _m128_project_decay_head(FA, FW, m, h, SFA, SW, 64, 0, NW, A, DT, lower)
                decay_shared.slice(0, 64).store(decay0)
                decay1 = _m128_project_decay_head(FA, FW, m, h, SFA, SW, 64, 64, NW, A, DT, lower)
                decay_shared.slice(64, 64).store(decay1)
            else:
                projected_decay = _m128_project_decay_head(FA, FW, m, h, SFA, SW, 128, 0, NW, A, DT, lower)
                decay_shared.store(projected_decay)
            if LARGE_BATCH:
                base = S + slot * SS + h * 128 * 128
            else:
                base = S + slot * SS
            slab_v = gl.arange(0, SLAB_ROWS, gl.SliceLayout(1, layout))
            if LARGE_BATCH:
                slab_offsets = slab_v[:, None] * 128 + k[None, :]
            else:
                slab_offsets = (h * 128 + slab_v[:, None]) * 128 + k[None, :]
            old0 = gl.amd.cdna4.buffer_load(base, slab_offsets, cache=state_cache)
            conv_slot = slot.to(gl.int32) if CONV_NARROW else slot
            q, key = _m128_convolve_head(X, CW, CS, m, h, conv_slot, H, SX, SC0, SC1, SC2, layout, prep, NW, SLAB_ROWS == 64)
            decay = decay_shared.load(gl.SliceLayout(0, layout))
            q = q * gl.rsqrt(gl.sum(q * q, 0) + 1e-06) * 128 ** (-0.5)
            key = key * gl.rsqrt(gl.sum(key * key, 0) + 1e-06)
            beta = _m1_sigmoid(gl.load(Beta + m * SB + h).to(gl.float32))
            norm_layout: gl.constexpr = gl.BlockedLayout([2], [64], [NW], [0])
            nv = gl.arange(0, 128, norm_layout)
            weight = gl.amd.cdna4.buffer_load(W, nv).to(gl.float32)
            gate = gl.amd.cdna4.buffer_load(Gate + m * SG + h * 128, nv).to(gl.float32)
            if SLAB_ROWS == 64:
                old1 = gl.amd.cdna4.buffer_load(base, slab_offsets + 64 * 128, cache=state_cache)
                value0 = prep.slice(256, 64).load(gl.SliceLayout(1, layout)).to(gl.float32)
                updated0 = _m128_advance_state_slab(old0, value0, decay, key, beta)
                _m128_store_changed_vectors(updated0, old0, base, slab_offsets, STATE_STORE)
                core0 = gl.sum(updated0 * q[None, :], 1)
                value1 = prep.slice(320, 64).load(gl.SliceLayout(1, layout)).to(gl.float32)
                updated1 = _m128_advance_state_slab(old1, value1, decay, key, beta)
                _m128_store_changed_vectors(updated1, old1, base, slab_offsets + 64 * 128, STATE_STORE)
                core1 = gl.sum(updated1 * q[None, :], 1)
                combined = gl.reshape(gl.permute(gl.join(core0, core1), (1, 0)), (128,))
                auxiliary.store(combined.to(gl.bfloat16))
            else:
                old1 = gl.amd.cdna4.buffer_load(base, slab_offsets + 32 * 128, cache=state_cache)
                if LARGE_BATCH:
                    old2 = gl.amd.cdna4.buffer_load(base, slab_offsets + 64 * 128, cache=state_cache)
                    old3 = gl.amd.cdna4.buffer_load(base, slab_offsets + 96 * 128, cache=state_cache)
                value0 = prep.slice(256, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
                updated0 = _m128_advance_state_slab(old0, value0, decay, key, beta)
                core0 = gl.sum(updated0 * q[None, :], 1)
                if LARGE_BATCH:
                    auxiliary.slice(0, 32).store(core0.to(gl.bfloat16))
                _m128_store_changed_vectors(updated0, old0, base, slab_offsets, STATE_STORE)
                if LARGE_BATCH:
                    value2 = prep.slice(320, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
                    updated2 = _m128_advance_state_slab(old2, value2, decay, key, beta)
                    core2 = gl.sum(updated2 * q[None, :], 1)
                    auxiliary.slice(64, 32).store(core2.to(gl.bfloat16))
                    _m128_store_changed_vectors(updated2, old2, base, slab_offsets + 64 * 128, STATE_STORE)
                    core2 = gl.inline_asm_elementwise('', '=v,0,~{memory}', [core2], dtype=gl.float32, is_pure=False, pack=1)
                    value1 = prep.slice(288, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
                    updated1 = _m128_advance_state_slab(old1, value1, decay, key, beta)
                    core1 = gl.sum(updated1 * q[None, :], 1)
                    auxiliary.slice(32, 32).store(core1.to(gl.bfloat16))
                    _m128_store_changed_vectors(updated1, old1, base, slab_offsets + 32 * 128, STATE_STORE)
                else:
                    value1 = prep.slice(288, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
                    updated1 = _m128_advance_state_slab(old1, value1, decay, key, beta)
                    core1 = gl.sum(updated1 * q[None, :], 1)
                    _m128_store_changed_vectors(updated1, old1, base, slab_offsets + 32 * 128, STATE_STORE)
                    pair0 = gl.reshape(gl.permute(gl.join(core0, core1), (1, 0)), (64,))
                    auxiliary.slice(0, 64).store(pair0.to(gl.bfloat16))
                    core1 = gl.inline_asm_elementwise('', '=v,0,~{memory}', [core1], dtype=gl.float32, is_pure=False, pack=1)
                    old2 = gl.amd.cdna4.buffer_load(base, slab_offsets + 64 * 128, cache=state_cache)
                    old3 = gl.amd.cdna4.buffer_load(base, slab_offsets + 96 * 128, cache=state_cache)
                    value2 = prep.slice(320, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
                    updated2 = _m128_advance_state_slab(old2, value2, decay, key, beta)
                    core2 = gl.sum(updated2 * q[None, :], 1)
                    _m128_store_changed_vectors(updated2, old2, base, slab_offsets + 64 * 128, STATE_STORE)
                    core2 = gl.inline_asm_elementwise('', '=v,0,~{memory}', [core2], dtype=gl.float32, is_pure=False, pack=1)
                value3 = prep.slice(352, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
                updated3 = _m128_advance_state_slab(old3, value3, decay, key, beta)
                core3 = gl.sum(updated3 * q[None, :], 1)
                if LARGE_BATCH:
                    auxiliary.slice(96, 32).store(core3.to(gl.bfloat16))
                _m128_store_changed_vectors(updated3, old3, base, slab_offsets + 96 * 128, STATE_STORE)
                if not LARGE_BATCH:
                    pair1 = gl.reshape(gl.permute(gl.join(core2, core3), (1, 0)), (64,))
                    auxiliary.slice(64, 64).store(pair1.to(gl.bfloat16))
            core = auxiliary.slice(0, 128).load(norm_layout).to(gl.float32)
            out = core * gl.rsqrt(gl.sum(core * core, 0) / 128 + eps) * weight * _m1_sigmoid(gate)
            gl.store(O + m * SO + h * 128 + nv, out)


@gluon.jit
def _m128_pack_projection_matrix(Source, Destination, M: gl.constexpr, K: gl.constexpr,
                            STRIDE: gl.constexpr, tile, BLOCK: gl.constexpr, NW: gl.constexpr = 4):
    gl.static_assert(M % 16 == 0 and K % 512 == 0 and BLOCK == 8192)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [2, NW // 2], [1, 0])
    row = tile // (K // 512) * 16 + gl.arange(0, 16, gl.SliceLayout(1, layout))
    col = tile % (K // 512) * 512 + gl.arange(0, 512, gl.SliceLayout(0, layout))
    value = gl.load(Source + row[:, None] * STRIDE + col[None, :])
    gl.store(Destination + row[:, None] * K + col[None, :], value)


@gluon.jit
def _m128_pack_projection_inputs(
    X, WQ, WB, PX, PQ, PB,
    M: gl.constexpr, SX: gl.constexpr, SQ: gl.constexpr, SB: gl.constexpr,
    NX: gl.constexpr, NQ: gl.constexpr, NB: gl.constexpr, BLOCK: gl.constexpr,
):
    tile = gl.program_id(0)
    if tile < NX:
        _m128_pack_projection_matrix(X, PX, M, 7168, SX, tile, BLOCK)
    elif tile < NX + NQ:
        _m128_pack_projection_matrix(WQ, PQ, 6144, 7168, SQ, tile - NX, BLOCK)
    else:
        _m128_pack_projection_matrix(WB, PB, 144, 7168, SB, tile - NX - NQ, BLOCK)


def _m128_prepare_projection_inputs(x, qkvg_weight, beta_forget_weight, output_weight):
    m = x.shape[0]
    matrices = (x, qkvg_weight, beta_forget_weight, output_weight)
    needs_pack = tuple(matrix.stride(0) % 8 != 0 for matrix in matrices)
    if any(needs_pack):
        packed = tuple(matrix.new_empty(matrix.shape) if needed else matrix
                       for matrix, needed in zip(matrices, needs_pack))
        block = 8192
        tiles = tuple(triton.cdiv(matrix.numel(), block) if needed else 0
                      for matrix, needed in zip(matrices, needs_pack))
        input_tiles = sum(tiles[:3])
        if input_tiles:
            _m128_pack_projection_inputs[(input_tiles,)](
                *matrices[:3], *packed[:3], m,
                *(matrix.stride(0) for matrix in matrices[:3]),
                *tiles[:3], block, num_warps=4,
            )
        source_output = output_weight if needs_pack[3] else None
        return (*packed, source_output)
    return x, qkvg_weight, beta_forget_weight, output_weight, None


def _m128_allocate_workspace(x):
    m = x.shape[0]
    arena = x.new_empty((m, 6400))
    qkvg = arena[:, :6144]
    beta_forget = arena[:, 6144:6288]
    core = qkvg[:, 4608:]
    return qkvg, beta_forget, core


def kda_layer_decode_m128(
    x, qkvg_weight, beta_forget_weight, output_weight, forget_weight,
    conv_weight, a_log, dt_bias, norm_weight, conv_state, state, state_indices,
    *, lower_bound=-5.0, norm_eps=1e-5, output_tensor=None,
):

    m = x.shape[0]
    x, qkvg_weight, beta_forget_weight, output_weight, source_output = _m128_prepare_projection_inputs(
        x, qkvg_weight, beta_forget_weight, output_weight,
    )
    group = 8
    qkvg, beta_forget, core = _m128_allocate_workspace(x)
    aligned = m in (128, 256)
    sort_rows = aligned
    order_dtype = torch.int64 if state_indices.dtype == torch.int32 else torch.int32
    row_order = torch.empty((m,), device=x.device, dtype=order_dtype)
    order_block = triton.next_power_of_2(m)
    order_log = order_block.bit_length() - 1
    tile_m = 64
    tile_n = 64
    waves = 4
    pid_groups = 16
    beta_columns = 16
    beta_depth = 5
    programs = m // tile_m * (6144 // tile_n) + m // 64 * triton.cdiv(144, beta_columns)
    _m128_project_qkvg_beta[(programs + int(sort_rows),)](
        x, qkvg_weight, beta_forget_weight, qkvg, beta_forget,
        m, x.stride(0), qkvg_weight.stride(0), beta_forget_weight.stride(0),
        tile_n, waves, pid_groups, 64, beta_columns, beta_depth, m != 128,
        qkvg.stride(0), beta_forget.stride(0),
        state_indices, row_order, state_indices.stride(0), sort_rows, order_block, order_log, tile_m,
        num_warps=waves, waves_per_eu=2,
    )
    conv_span = sum(
        (extent - 1) * stride
        for extent, stride in zip(conv_state.shape, conv_state.stride())
    )
    conv_narrow = (
        m >= 192
        and all(stride >= 0 for stride in conv_state.stride())
        and conv_span < 2**31
    )
    state_store = '.cs'
    pack_tiles = triton.cdiv(output_weight.numel(), 8192) if source_output is not None else 0
    copy_source = source_output if source_output is not None else output_weight
    _m128_kda_decode[(12 * group, triton.cdiv(12 * m + pack_tiles, 12 * group))](
        qkvg, qkvg[:, 4608:], beta_forget, beta_forget[:, 128:140],
        forget_weight, conv_weight, a_log, dt_bias, norm_weight,
        conv_state, state, state_indices, core,
        12, qkvg.stride(0), qkvg.stride(0), beta_forget.stride(0), beta_forget.stride(0), forget_weight.stride(0),
        *conv_state.stride(), state.stride(0), state_indices.stride(0),
        lower_bound, norm_eps, group, m >= 192, 8, state_store, core.stride(0), row_order, sort_rows, 64, conv_narrow,
        copy_source, output_weight, copy_source.stride(0), pack_tiles, m,
        num_warps=8, enable_fp_fusion=False, waves_per_eu=0,
    )
    out = x.new_empty((m, 7168)) if output_tensor is None else output_tensor
    tile_m, tile_n, waves = (64, 64, 4)
    _m128_output_gemm[(m // tile_m * (7168 // tile_n),)](
        core, output_weight, out, m, 7168, 1536,
        core.stride(0), output_weight.stride(0), tile_n, waves, True,
        num_warps=waves, waves_per_eu=2,
    )
    return out, conv_state, state


@gluon.jit
def _m1_256_forget(X, FW, m, h, SW: gl.constexpr, PACK: gl.constexpr):
    if PACK == 8:

        layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 8], [8, 8, 1], [1, 8, 1], [2, 0, 1])
        group = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
        row = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, layout)))
        inner = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
        k = group[:, None, None] * 8 + inner[None, None, :]
        w = gl.amd.cdna4.buffer_load(FW + h * 128 * SW, row[None, :, None] * SW + k)
        a = gl.amd.cdna4.buffer_load(X + m * 6336 + 6144, k)
        partial = gl.sum(w.to(gl.float32) * a.to(gl.float32), 2)
        return gl.sum(partial, 0).to(gl.bfloat16).to(gl.float32)
    else:


        if PACK == 1:
            layout: gl.constexpr = gl.BlockedLayout(
                [1, 1], [1, 64], [4, 2], [1, 0])
        elif PACK == 2:
            layout: gl.constexpr = gl.BlockedLayout(
                [1, 2], [1, 64], [8, 1], [1, 0])
        else:
            layout: gl.constexpr = gl.BlockedLayout(
                [1, 4], [2, 32], [8, 1], [1, 0])
        row = gl.arange(0, 128, gl.SliceLayout(1, layout))
        k = gl.arange(0, 128, gl.SliceLayout(0, layout))
        w = gl.load(FW + h * 128 * SW + row[:, None] * SW + k[None, :])
        a = gl.load(X + m * 6336 + 6144 + k)
        product = w.to(gl.float32) * a[None, :].to(gl.float32)
        return gl.sum(product, 1).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _m1_256_split_vector(v):

    packets = ()
    for p in gl.static_range(4):
        packets += (gl.amd.slice(v, [32], [p * 32]),)
    return packets


@gluon.jit
def _m1_256_recurrent_head(
    X, CW, FW, CS, S, IDX, A, DT, NW, O,
    SW: gl.constexpr, SI: gl.constexpr,
    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
    SS: gl.constexpr, LOWER, EPS, pid, STORE_CACHE: gl.constexpr, SMALL: gl.constexpr,
    LC: gl.constexpr, FORGET_PACK: gl.constexpr, GROUP: gl.constexpr,
):


    if not SMALL:
        ST: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [8, 1], [1, 0])
        ST_T: gl.constexpr = gl.BlockedLayout([4, 1], [8, 8], [1, 8], [0, 1])
    else:
        ST: gl.constexpr = gl.BlockedLayout([2, 4], [8, 8], [8, 1], [1, 0])
        ST_T: gl.constexpr = gl.BlockedLayout([4, 2], [8, 8], [1, 8], [0, 1])
    OUTPUT_PITCH: gl.constexpr = 1600 if SMALL else 6336

    m = pid // (12 * GROUP) * GROUP + pid % GROUP
    h = pid // GROUP % 12
    i = gl.arange(0, 128, _VECTOR)
    slot = gl.load(IDX + m * SI).to(gl.int64)
    if slot < 0:
        gl.store(O + m * OUTPUT_PITCH + h * 128 + i, 0)
    else:
        nw = gl.load(NW + i).to(gl.float32)
        beta = _m1_sigmoid(gl.load(X + m * 6336 + 6272 + h).to(gl.float32))
        rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
        r = gl.arange(0, 128, gl.SliceLayout(1, ST))
        c = gl.arange(0, 32, gl.SliceLayout(0, ST))
        sb = S + slot * SS + h * 16384
        so = r[:, None] * 128 + c[None, :]


        if not SMALL:
            state_packets = ()
            for p in gl.static_range(4):
                state_packets += (gl.amd.cdna4.buffer_load(
                    sb, so + p * 32, cache=LC),)
        else:
            full_c = gl.arange(0, 128, gl.SliceLayout(0, ST))
            state_tile = gl.amd.cdna4.buffer_load(
                sb, r[:, None] * 128 + full_c[None, :], cache=LC)
        z, history = _m1_convolve(X, CW, CS, m, h, slot, SC0, SC1, SC2)
        f = _m1_256_forget(X, FW, m, h, SW, FORGET_PACK)
        decay_layout: gl.constexpr = gl.DistributedLinearLayout(
            [], [[0], [0], [64], [1], [2], [4]],
            [[8], [16], [32]], [], [128],
        )
        f = gl.convert_layout(f, decay_layout)
        shared = gl.allocate_shared_memory(
            gl.float32, (8, 128), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        )
        partial_norm = gl.sum(gl.reshape(z * z, (4, 2, 64)), 2)
        shared.slice(7, 1).slice(0, 8, dim=1).store(gl.reshape(partial_norm, (1, 8)))
        shared.slice(0, 4).store(z)
        dt = gl.load(DT + h * 128 + gl.arange(0, 128, f.type.layout))
        decay = gl.exp2(LOWER * _m1_sigmoid(rate * (f + dt)) * 1.4426950408889634)
        shared.slice(4, 1).store(gl.reshape(decay, (1, 128)))
        key = gl.sum(shared.slice(1, 1).load(ST), 0)
        value = gl.sum(shared.slice(2, 1).load(ST_T), 0)
        value = gl.convert_layout(value, gl.SliceLayout(1, ST))
        decay = gl.sum(shared.slice(4, 1).load(ST), 0)
        gate = gl.sum(shared.slice(3, 1).load(_OUTPUT), 0)
        gate = gl.convert_layout(gate, _VECTOR)
        nl: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [0, 1])
        qp = shared.slice(7, 1).slice(0, 2, dim=1).load(nl)
        kp = shared.slice(7, 1).slice(2, 2, dim=1).load(nl)
        qnorm = gl.rsqrt(gl.sum(gl.sum(qp, 1), 0) + 1e-06)
        knorm = gl.rsqrt(gl.sum(gl.sum(kp, 1), 0) + 1e-06)
        keys = _m1_256_split_vector(key)
        decays = _m1_256_split_vector(decay)
        decayed = ()
        for p in gl.static_range(4):
            if not SMALL:
                packet = state_packets[p]
            else:
                packet = gl.amd.slice(state_tile, [128, 32], [0, p * 32])
            decayed += (packet * decays[p][None, :],)
        P_ORDER: gl.constexpr = (0, 2, 1, 3)
        prediction = _m1_contract(decayed, keys, P_ORDER)
        q = gl.sum(shared.slice(0, 1).load(ST), 0)
        queries = _m1_256_split_vector(q)
        delta = gl.fma(-prediction, knorm, value) * (beta * knorm)
        for p in gl.static_range(3, -1, -1):
            updated = gl.fma(delta[:, None], keys[p][None, :], decayed[p])
            if p == 3:
                partial = updated * queries[p][None, :]
            else:
                partial = gl.fma(updated, queries[p][None, :], partial)
            gl.amd.cdna4.buffer_store(
                updated, sb, so + p * 32,
                cache=STORE_CACHE,
            )
        out = gl.sum(partial, 1) * qnorm * 128 ** (-0.5)
        out = out.to(gl.bfloat16).to(gl.float32)
        out = gl.convert_layout(out, gl.SliceLayout(0, ST_T))
        shared.slice(6, 1).store(out[None, :])
        out = gl.sum(shared.slice(6, 1).load(_OUTPUT), 0)
        out = gl.convert_layout(out, _VECTOR)
        scale = gl.rsqrt(gl.sum(out * out, 0) / 128 + EPS)
        out = out * scale * (nw * gate)
        old1, old2, x, hb, ho, valid = history
        gl.amd.cdna4.buffer_store(old1, hb, ho, valid)
        gl.amd.cdna4.buffer_store(old2, hb, SC1 + ho, valid)
        gl.amd.cdna4.buffer_store(x, hb, 2 * SC1 + ho, valid)
        gl.store(O + m * OUTPUT_PITCH + h * 128 + i, out)


@gluon.jit
def _m1_256_prefetch_output_weights(W, SW: gl.constexpr, pid, SMALL: gl.constexpr):

    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    if SMALL:
        CHUNK: gl.constexpr = 512
        STEP: gl.constexpr = 128
        LINES_PER_BAND: gl.constexpr = 896 * 1536 // STEP
        lane = gl.arange(0, CHUNK, layout)
        logical = pid // 8 * CHUNK + lane
        line = pid % 8 * LINES_PER_BAND + logical
        valid = (logical < LINES_PER_BAND) & (lane < CHUNK)
        row = line // (1536 // STEP)
        col = line % (1536 // STEP) * (STEP // 2)
    else:
        CHUNK: gl.constexpr = 768
        lane = gl.arange(0, 1024, layout)
        logical = pid // 8 * CHUNK + lane

        row = (logical // 64 * 8 + pid % 8) * 32 + logical % 32
        col = logical // 32 % 2 * 32
        valid = (lane < CHUNK) & (row < 7168)
    ptr = W.to(gl.pointer_type(gl.int32)) + row * (SW // 2) + col
    word = gl.load(ptr, valid, other=0)
    gl.inline_asm_elementwise(
        '', constraints='=v,0', args=[word], dtype=gl.int32, is_pure=False, pack=1,
    )


@gluon.jit
def _m1_256_recurrent_batch(X, CW, FW, CS, S, IDX, A, DT, NW, O, WO,
                     M: gl.constexpr, SW: gl.constexpr, SI: gl.constexpr,
                     SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                     SS: gl.constexpr, LOWER, EPS, OWN: gl.constexpr,
                     FORGET_PACK: gl.constexpr):
    STORE_CACHE: gl.constexpr = '.wt' if M <= 8 else '.cs'
    LC: gl.constexpr = '' if M <= 8 else '.cg'
    GROUP: gl.constexpr = 4 if M > 8 and M % 4 == 0 else 1
    pid = gl.program_id(0)


    if M > 8:
        if pid < 24:
            _m1_256_prefetch_output_weights(WO, OWN, pid, False)
        elif pid < M * 12 + 24:
            _m1_256_recurrent_head(
                X, CW, FW, CS, S, IDX, A, DT, NW, O, SW, SI,
                SC0, SC1, SC2, SS, LOWER, EPS, pid - 24, STORE_CACHE, M <= 8, LC, FORGET_PACK, GROUP,
            )
    else:
        if (pid >= 0) & (pid < M * 12):
            _m1_256_recurrent_head(
                X, CW, FW, CS, S, IDX, A, DT, NW, O, SW, SI,
                SC0, SC1, SC2, SS, LOWER, EPS, pid, STORE_CACHE, M <= 8, LC, FORGET_PACK, GROUP,
            )
        elif M <= 8:
            _m1_256_prefetch_output_weights(WO, OWN, pid - M * 12, True)


@gluon.jit
def _m1_256_copy_projection_panel(
    activation_copy, weight_copy, X, W, ao, bo, rows,
    BLOCK: gl.constexpr, BK: gl.constexpr, M: gl.constexpr,
    ROW_COUNT: gl.constexpr, INPUT: gl.constexpr, PRIVATE: gl.constexpr,
    WEIGHT_MASK=None,
):

    if INPUT:


        X, W = X + BLOCK * BK, W + BLOCK * BK
    else:
        ao, bo = ao + BLOCK * BK, bo + BLOCK * BK
    if ROW_COUNT == M:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(activation_copy, X, ao)
    else:
        if PRIVATE:
            valid = rows[None, :, None] < ROW_COUNT
        else:
            valid = rows[:, None] < ROW_COUNT
        gl.amd.cdna4.async_copy.buffer_load_to_shared(activation_copy, X, ao, valid, 0)


    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        weight_copy, W, bo, mask=WEIGHT_MASK, cache_modifier='.cg')
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m1_256_ordered_mfma_ring(
    X, W, ao, bo, rows,
    M: gl.constexpr, ROW_COUNT: gl.constexpr, K: gl.constexpr,
    BK: gl.constexpr, DEPTH: gl.constexpr, MATRIX: gl.constexpr,
    INPUT: gl.constexpr, PRIVATE: gl.constexpr = False,
    WEIGHT_MASK=None,
):

    if PRIVATE:
        A_SHAPE: gl.constexpr = (2, M, BK)
        B_SHAPE: gl.constexpr = (2, BK, 16)
        ACC_SHAPE: gl.constexpr = (2, M, 16)
        A_ORDER: gl.constexpr = [2, 1, 0]
        B_ORDER: gl.constexpr = [1, 2, 0]
    else:
        A_SHAPE: gl.constexpr = (M, BK)
        B_SHAPE: gl.constexpr = (BK, 32)
        ACC_SHAPE: gl.constexpr = (M, 32)
        A_ORDER: gl.constexpr = [1, 0]
        B_ORDER: gl.constexpr = [0, 1]
    activation_ring, weight_ring = (), ()
    activation_copies, weight_copies = (), ()
    for slot in gl.static_range(DEPTH):
        activation_ring += (gl.allocate_shared_memory(
            gl.bfloat16, A_SHAPE, gl.SwizzledSharedLayout(8, 1, 8, A_ORDER)),)
        weight_ring += (gl.allocate_shared_memory(
            gl.bfloat16, B_SHAPE, gl.SwizzledSharedLayout(8, 1, 8, B_ORDER)),)
        activation_copies += (activation_ring[slot]._reinterpret(
            layout=gl.SwizzledSharedLayout(1, 1, 1, A_ORDER)),)
        weight_copies += (weight_ring[slot]._reinterpret(
            layout=gl.SwizzledSharedLayout(1, 1, 1, B_ORDER)),)
        _m1_256_copy_projection_panel(
            activation_copies[slot], weight_copies[slot], X, W, ao, bo, rows,
            slot, BK, M, ROW_COUNT, INPUT, PRIVATE, WEIGHT_MASK)
    acc = gl.zeros(ACC_SHAPE, gl.float32, MATRIX)
    for block in gl.static_range(K // BK):
        gl.amd.cdna4.async_copy.wait_group(min(DEPTH - 1, K // BK - block - 1))
        if not PRIVATE:
            gl.barrier()
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            activation_ring[block % DEPTH], gl.DotOperandLayout(0, MATRIX, 8))
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            weight_ring[block % DEPTH], gl.DotOperandLayout(1, MATRIX, 8))
        if PRIVATE:
            acc = gl.amd.cdna4.mfma(a, b, acc)
        if block + DEPTH < K // BK:

            gl.inline_asm_elementwise(
                's_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0',
                constraints='=v,~{memory}', args=[], dtype=gl.int32,
                is_pure=False, pack=1)
            if not PRIVATE:
                gl.barrier()
            _m1_256_copy_projection_panel(
                activation_copies[block % DEPTH], weight_copies[block % DEPTH],
                X, W, ao, bo, rows, block + DEPTH, BK, M, ROW_COUNT, INPUT, PRIVATE, WEIGHT_MASK)
        if not PRIVATE:
            acc = gl.amd.cdna4.mfma(a, b, acc)
    return acc


@gluon.jit
def _m1_256_project_cooperative(
    X, WQ, WB, Y,
    M: gl.constexpr, XM: gl.constexpr, WN: gl.constexpr, WBN: gl.constexpr,
    ROW_COUNT: gl.constexpr, INPUT: gl.constexpr,
):

    K: gl.constexpr = 7168 if INPUT else 1536
    PITCH: gl.constexpr = 6336 if INPUT else 7168
    BK: gl.constexpr = 128 if INPUT else 64
    BN: gl.constexpr = 32
    USEFUL: gl.constexpr = 25
    DEPTH: gl.constexpr = (5 if M <= 8 else 6) if INPUT else 8
    WAVES: gl.constexpr = 2
    BANDS: gl.constexpr = 8 if INPUT else 1
    pid = gl.program_id(0)
    if M > 8:

        pid = pid.to(gl.uint32)
    if INPUT:
        Q_TILES: gl.constexpr = triton.cdiv(6144, USEFUL)
        TILES: gl.constexpr = Q_TILES + triton.cdiv(144, 16)
        band = pid % BANDS
        tile = band * (TILES // BANDS) + gl.minimum(band, TILES % BANDS) + pid // BANDS
        stride = gl.where(tile < Q_TILES, WN, WBN)
        weight = gl.where(tile < Q_TILES, WQ, WB)
        width = gl.where(tile < Q_TILES, USEFUL, 16)
        first = gl.where(tile < Q_TILES, tile, tile - Q_TILES) * width
        remaining = gl.where(tile < Q_TILES, 6144, 144) - first
        valid_columns = gl.minimum(remaining, width)
        weight += first * stride
        output_first = gl.where(tile < Q_TILES, first, 6144 + first)
    else:
        tile = pid % BANDS * (7168 // BN // BANDS) + pid // BANDS
        stride = WN
        weight = WQ + tile * BN * WN
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, WAVES])
    load_a: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BK, BK // 8], [WAVES, 1], [1, 0])
    load_b: gl.constexpr = gl.BlockedLayout(
        [8, 1], [BK // 8, 512 // BK], [1, WAVES], [0, 1])
    rows = gl.arange(0, M, gl.SliceLayout(1, load_a))
    ak = gl.arange(0, BK, gl.SliceLayout(0, load_a))
    cols = gl.arange(0, BN, gl.SliceLayout(0, load_b))
    bk = gl.arange(0, BK, gl.SliceLayout(1, load_b))
    if INPUT and M <= 8:

        weight_cols = gl.minimum(cols, valid_columns - 1)
    else:
        weight_cols = cols

    ao = rows[:, None] * XM + (ak[None, :] ^ ((rows[:, None] % 8) * 8))
    bo = weight_cols[None, :] * stride + (bk[:, None] ^ ((cols[None, :] % 8) * 8))
    ao = gl.max_contiguous(gl.multiple_of(ao, [1, 8]), [1, 8])
    bo = gl.max_contiguous(gl.multiple_of(bo, [8, 1]), [8, 1])
    weight_mask = cols[None, :] < valid_columns if INPUT and M > 8 else None
    acc = _m1_256_ordered_mfma_ring(
        X, weight, ao, bo, rows, M, ROW_COUNT, K, BK, DEPTH, matrix, INPUT,
        WEIGHT_MASK=weight_mask)
    out_rows = gl.arange(0, M, gl.SliceLayout(1, matrix))
    out_cols = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, matrix))
    if INPUT:
        physical_cols = gl.arange(0, BN, gl.SliceLayout(0, matrix))
        out_cols = output_first + physical_cols
        gl.store(Y + out_rows[:, None] * PITCH + out_cols[None, :], acc.to(gl.bfloat16),
                 (out_rows[:, None] < ROW_COUNT) & (physical_cols[None, :] < valid_columns))
    elif ROW_COUNT == M:
        gl.store(Y + out_rows[:, None] * PITCH + out_cols[None, :], acc.to(gl.bfloat16))
    else:
        gl.store(Y + out_rows[:, None] * PITCH + out_cols[None, :], acc.to(gl.bfloat16),
                 out_rows[:, None] < ROW_COUNT)


@gluon.jit
def _m1_256_project_inputs_async(
    X, WQ, WB, Y,
    M: gl.constexpr, XM: gl.constexpr, WN: gl.constexpr, WBN: gl.constexpr,
    ROW_COUNT: gl.constexpr,
):
    _m1_256_project_cooperative(X, WQ, WB, Y, M, XM, WN, WBN, ROW_COUNT, True)


@gluon.jit
def _m1_256_project_output_async(
    X, W, Y, M: gl.constexpr, XM: gl.constexpr, WN: gl.constexpr,
    ROW_COUNT: gl.constexpr,
):
    _m1_256_project_cooperative(X, W, W, Y, M, XM, WN, WN, ROW_COUNT, False)


@gluon.jit
def _m1_256_project_output_private(
    X, W, Y, M: gl.constexpr, XM: gl.constexpr, WN: gl.constexpr,
    ROW_COUNT: gl.constexpr,
):

    BK: gl.constexpr = 64
    DEPTH: gl.constexpr = 12
    tile = gl.program_id(0) % 8 * 32 + gl.program_id(0) // 8
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[2, 1, 1])
    la: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 8, 8], [2, 1, 1], [2, 1, 0])
    lb: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 8, 8], [2, 1, 1], [1, 2, 0])
    ga = gl.arange(0, 2, gl.SliceLayout(1, gl.SliceLayout(2, la)))
    ma = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, la)))
    ka = gl.arange(0, BK, gl.SliceLayout(0, gl.SliceLayout(1, la)))
    gb = gl.arange(0, 2, gl.SliceLayout(1, gl.SliceLayout(2, lb)))
    kb = gl.arange(0, BK, gl.SliceLayout(0, gl.SliceLayout(2, lb)))
    nb = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, lb)))
    ao = ga[:, None, None] * 0 + ma[None, :, None] * XM + (
        ka[None, None, :] ^ (ma[None, :, None] * 8))
    weight_col = gl.minimum(gb[:, None, None] * 16 + nb[None, None, :], 27)
    bo = weight_col * WN + (
        kb[None, :, None] ^ ((nb[None, None, :] % 8) * 8))
    ao = gl.max_contiguous(gl.multiple_of(ao, [1, 1, 8]), [1, 1, 8])
    bo = gl.max_contiguous(gl.multiple_of(bo, [1, 8, 1]), [1, 8, 1])
    weight = W + tile * 28 * WN
    acc = _m1_256_ordered_mfma_ring(
        X, weight, ao, bo, ma, M, ROW_COUNT, 1536, BK, DEPTH, matrix,
        INPUT=False, PRIVATE=True)
    group = gl.arange(0, 2, gl.SliceLayout(1, gl.SliceLayout(2, matrix)))
    row = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, matrix)))
    col = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, matrix)))
    physical_col = group[:, None, None] * 16 + col[None, None, :]
    offset = row[None, :, None] * 7168 + tile * 28 + physical_col
    gl.store(Y + offset, acc.to(gl.bfloat16),
             (row[None, :, None] < ROW_COUNT) & (physical_col < 28))


@gluon.jit
def _m1_256_copy_to_aligned(SRC, DST, ROWS: gl.constexpr, COLS: gl.constexpr,
                     STRIDE: gl.constexpr):

    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
    row = gl.program_id(0) * 8 + gl.arange(0, 8, gl.SliceLayout(1, layout))
    col = gl.program_id(1) * 512 + gl.arange(0, 512, gl.SliceLayout(0, layout))
    valid = (row[:, None] < ROWS) & (col[None, :] < COLS)
    values = gl.load(SRC + row[:, None] * STRIDE + col[None, :], valid, 0)
    gl.store(DST + row[:, None] * COLS + col[None, :], values, valid)


def _m1_256_stage_aligned_matrix(tensor):

    if tensor.stride(0) % 8 == 0 and tensor.storage_offset() % 8 == 0:
        return tensor
    rows, cols = tensor.shape
    aligned = torch.empty((rows, cols), dtype=tensor.dtype, device=tensor.device)
    _m1_256_copy_to_aligned[(triton.cdiv(rows, 8), triton.cdiv(cols, 512))](
        tensor, aligned, rows, cols, tensor.stride(0), num_warps=4)
    return aligned


def _m1_256_forget_pack(weight):

    if weight.storage_offset() % 8:
        return 1
    for pack in (8, 4, 2):
        if weight.stride(0) % pack == 0:
            return pack
    return 1


def kda_layer_decode_m1_256(
    x, qkvg_weight, beta_forget_weight, output_weight, forget_weight,
    conv_weight, a_log, dt_bias, norm_weight, conv_state, state, state_indices,
    *, lower_bound=-5.0, norm_eps=1e-5, output_tensor=None,
):
    x = _m1_256_stage_aligned_matrix(x)
    qkvg_weight = _m1_256_stage_aligned_matrix(qkvg_weight)
    beta_forget_weight = _m1_256_stage_aligned_matrix(beta_forget_weight)
    output_weight = _m1_256_stage_aligned_matrix(output_weight)
    m = x.shape[0]
    out = (
        torch.empty((m, 7168), device=x.device, dtype=torch.bfloat16)
        if output_tensor is None else output_tensor
    )
    assert out.shape == (m, 7168) and out.dtype == torch.bfloat16
    assert out.device == x.device and out.is_contiguous()
    forget_pack = _m1_256_forget_pack(forget_weight)

    for start in range(0, m, 16):
        rows = min(m - start, 16)
        block_rows = max(8, triton.next_power_of_2(rows))
        small = rows <= 8
        core_pitch = 1600 if small else 6336
        scratch_pitch = 6336 + 1600 if small else 6336
        workspace = torch.empty(
            (rows * scratch_pitch,), dtype=torch.bfloat16, device=x.device,
        )
        packed = workspace[:rows * 6336].view(rows, 6336)
        if small:
            core = workspace[rows * 6336:].view(rows, core_pitch)
        else:


            core = packed[:, :1536]

        _m1_256_project_inputs_async[triton.cdiv(6144, 25) + triton.cdiv(144, 16),](
            x[start:start + rows], qkvg_weight, beta_forget_weight, packed,
            block_rows, x.stride(0), qkvg_weight.stride(0),
            beta_forget_weight.stride(0), rows,
            num_warps=2, waves_per_eu=1)
        helpers = 32 if small else 24
        _m1_256_recurrent_batch[rows * 12 + helpers,](
            packed, conv_weight, forget_weight, conv_state, state,
            state_indices[start:start + rows], a_log, dt_bias, norm_weight,
            core, output_weight, rows, forget_weight.stride(0), state_indices.stride(0),
            *conv_state.stride(), state.stride(0), lower_bound, norm_eps,
            output_weight.stride(0), forget_pack,
            num_warps=8, enable_fp_fusion=False, waves_per_eu=4 if small else 2)
        if small:
            _m1_256_project_output_private[7168 // 28,](
                core, output_weight, out[start:start + rows], block_rows, core_pitch,
                output_weight.stride(0), rows, num_warps=2)
        else:
            _m1_256_project_output_async[7168 // 32,](
                core, output_weight, out[start:start + rows], block_rows, core_pitch,
                output_weight.stride(0), rows, num_warps=2)
    return out, conv_state, state


@gluon.jit
def _m2_input_projections(
    X, WQ, WB, Y, M: gl.constexpr,
    XM: gl.constexpr, WN: gl.constexpr, WBN: gl.constexpr,
):

    CN: gl.constexpr = 16
    BK: gl.constexpr = 128
    AK_LANES: gl.constexpr = 64 if M == 1 else 32 if M == 2 else 16

    APACK: gl.constexpr = 2 if M == 1 else 4 if M == 2 else 8
    BK_LANES: gl.constexpr = 16
    WPACK: gl.constexpr = 8
    K: gl.constexpr = 7168
    N: gl.constexpr = 8192 if M == 2 else 6336
    pid = gl.program_id(0)
    if M == 4:

        shifted = pid - 9
        tile = gl.where(pid < 9, 384 + pid, shifted % 24 * 16 + shifted // 24)
    elif M == 1:

        shifted = pid - 9
        tile = gl.where(pid < 9, 384 + pid, shifted % 8 * 48 + shifted // 8)
    else:
        band = pid % 8
        tile = band * 49 + gl.minimum(band, 1) + pid // 8
    weight = gl.where(tile < 6144 // CN, WQ, WB)
    weight_stride = gl.where(tile < 6144 // CN, WN, WBN)
    weight_tile = gl.where(tile < 6144 // CN, tile, tile - 6144 // CN)
    weight += weight_tile * CN * weight_stride
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[1, 1],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    load_a: gl.constexpr = gl.BlockedLayout(
        [1, APACK], [64 // AK_LANES, AK_LANES], [1, 1], [1, 0],
    )
    load_b: gl.constexpr = gl.BlockedLayout(
        [WPACK, 1], [BK_LANES, 64 // BK_LANES], [1, 1], [0, 1],
    )
    rows = gl.arange(0, M, gl.SliceLayout(1, load_a))
    ak = gl.arange(0, BK, gl.SliceLayout(0, load_a))
    cols = gl.arange(0, CN, gl.SliceLayout(0, load_b))
    bk = gl.arange(0, BK, gl.SliceLayout(1, load_b))
    a_offsets = rows[:, None] * XM + ak[None, :]
    b_offsets = cols[None, :] * weight_stride + bk[:, None]
    acc = gl.zeros((M, CN), gl.float32, mma)
    if M == 1:


        LOOKAHEAD: gl.constexpr = 5
        allk = gl.arange(0, 8192, gl.SliceLayout(0, load_a))
        all_a = gl.amd.cdna4.buffer_load(
            X, rows[:, None] * XM + allk[None, :], allk[None, :] < K, 0,
        )
        weights = ()
        for block in gl.static_range(LOOKAHEAD):
            weights += (gl.amd.cdna4.buffer_load(
                weight + (block * BK // 1024) * 1024,
                b_offsets + block * BK % 1024, cache=".cg",
            ),)
        for block in gl.static_range(K // BK):
            a = gl.amd.slice(all_a, [M, BK], [0, block * BK])
            b = weights[0]
            weights = weights[1:]
            a = gl.convert_layout(a, dot_a)
            b = gl.convert_layout(b, dot_b)
            acc = gl.amd.cdna4.mfma(
                gl.amd.slice(a, [M, 64], [0, 0]),
                gl.amd.slice(b, [64, CN], [0, 0]), acc,
            )
            if block + LOOKAHEAD < K // BK:
                weights += (gl.amd.cdna4.buffer_load(
                    weight + ((block + LOOKAHEAD) * BK // 1024) * 1024,
                    b_offsets + (block + LOOKAHEAD) * BK % 1024, cache=".cg",
                ),)
            acc = gl.amd.cdna4.mfma(
                gl.amd.slice(a, [M, 64], [0, 64]),
                gl.amd.slice(b, [64, CN], [64, 0]), acc,
            )
    elif M == 2:


        LOOKAHEAD: gl.constexpr = 4
        activations = ()
        weights = ()
        for block in gl.static_range(LOOKAHEAD):
            activations += (gl.amd.cdna4.buffer_load(
                X, a_offsets + block * BK,
            ),)
            weights += (gl.amd.cdna4.buffer_load(
                weight + block * BK, b_offsets, cache=".cg",
            ),)
        for block in gl.static_range(K // BK):
            a = gl.convert_layout(activations[0], dot_a)
            b = gl.convert_layout(weights[0], dot_b)
            activations = activations[1:]
            weights = weights[1:]
            acc = gl.amd.cdna4.mfma(
                gl.amd.slice(a, [M, 64], [0, 0]),
                gl.amd.slice(b, [64, CN], [0, 0]), acc,
            )
            if block + LOOKAHEAD < K // BK:
                activations += (gl.amd.cdna4.buffer_load(
                    X, a_offsets + (block + LOOKAHEAD) * BK,
                ),)
                weights += (gl.amd.cdna4.buffer_load(
                    weight + (block + LOOKAHEAD) * BK, b_offsets, cache=".cg",
                ),)
            acc = gl.amd.cdna4.mfma(
                gl.amd.slice(a, [M, 64], [0, 64]),
                gl.amd.slice(b, [64, CN], [64, 0]), acc,
            )
    else:

        DEPTH: gl.constexpr = 4
        a_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 4, [1, 0])
        b_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 4, [0, 1])
        a_slots = ()
        b_slots = ()
        for stage in gl.static_range(DEPTH):
            a_slots += (gl.allocate_shared_memory(gl.bfloat16, (M, BK), a_shared),)
            b_slots += (gl.allocate_shared_memory(gl.bfloat16, (BK, CN), b_shared),)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_slots[stage], weight + stage * BK, b_offsets, cache_modifier=".cg",
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                a_slots[stage], X + stage * BK, a_offsets,
            )
            gl.amd.cdna4.async_copy.commit_group()
        for block in gl.static_range(K // BK):
            gl.amd.cdna4.async_copy.wait_group(min(DEPTH - 1, K // BK - 1 - block))
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[block % DEPTH], dot_a)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[block % DEPTH], dot_b)
            if block + DEPTH < K // BK:


                gl.barrier()
                gl.inline_asm_elementwise(
                    "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                    constraints="=v,~{memory}", args=[], dtype=gl.int32,
                    is_pure=False, pack=1,
                )
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    b_slots[block % DEPTH], weight + (block + DEPTH) * BK,
                    b_offsets, cache_modifier=".cg",
                )
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    a_slots[block % DEPTH], X + (block + DEPTH) * BK, a_offsets,
                )
                gl.amd.cdna4.async_copy.commit_group()
            acc = gl.amd.cdna4.mfma(a, b, acc)
    out_rows = gl.arange(0, M, gl.SliceLayout(1, mma))
    out_cols = tile * CN + gl.arange(0, CN, gl.SliceLayout(0, mma))
    gl.store(Y + out_rows[:, None] * N + out_cols[None, :], acc.to(gl.bfloat16))


@gluon.jit
def _m2_convolve(X, CW, CS, row, head, slot, SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr, PX: gl.constexpr):
    group = gl.arange(0, 4, gl.SliceLayout(1, _CONV))
    channel = gl.arange(0, 128, gl.SliceLayout(0, _CONV))
    offset = (group[:, None] * 12 + head) * 128 + channel[None, :]
    valid = group[:, None] < 3
    history = CS + slot * SC0
    old0 = gl.amd.cdna4.buffer_load(history, offset * SC2, valid, 0).to(gl.float32)
    old1 = gl.amd.cdna4.buffer_load(history, SC1 + offset * SC2, valid, 0)
    old2 = gl.amd.cdna4.buffer_load(history, 2 * SC1 + offset * SC2, valid, 0)
    x = gl.load(X + row * PX + offset)
    wl: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 64, 1], [4, 2, 1], [2, 1, 0])
    wo = gl.convert_layout(offset, gl.SliceLayout(2, wl), assert_trivial=True)
    wvalid = gl.convert_layout(valid, gl.SliceLayout(2, wl), assert_trivial=True)
    tap = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    weights = gl.amd.cdna4.buffer_load(CW, wo[:, :, None] * 4 + tap[None, None, :], wvalid[:, :, None], 0)
    even, odd = gl.split(gl.reshape(weights, (4, 128, 2, 2)))
    w0, w2 = gl.split(even)
    w1, w3 = gl.split(odd)
    w0 = gl.convert_layout(w0, _CONV)
    w1 = gl.convert_layout(w1, _CONV)
    w2 = gl.convert_layout(w2, _CONV)
    w3 = gl.convert_layout(w3, _CONV)
    z = old0 * w0 + old1.to(gl.float32) * w1 + old2.to(gl.float32) * w2 + x.to(gl.float32) * w3
    z = gl.where(group[:, None] == 3, x.to(gl.float32), z)
    sigmoid = _m1_sigmoid(z)
    z = gl.where(group[:, None] == 3, sigmoid, (z * sigmoid).to(gl.bfloat16).to(gl.float32))
    return z, (old1, old2, x, history, offset * SC2, valid)


@gluon.jit
def _m2_forget(X, FW, m, h, SW: gl.constexpr, PX: gl.constexpr):

    pl: gl.constexpr = gl.BlockedLayout([1, 1, 16], [8, 8, 1], [1, 8, 1], [2, 0, 1])
    split = gl.arange(0, 8, gl.SliceLayout(1, gl.SliceLayout(2, pl)))
    row = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, pl)))
    kk = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, pl)))
    wk = split[:, None, None] * 8 + kk[None, None, :] % 8 + kk[None, None, :] // 8 * 64
    w = gl.amd.cdna4.buffer_load(FW + h * 128 * SW, row[None, :, None] * SW + wk).to(gl.float32)
    a = gl.amd.cdna4.buffer_load(X + m * PX + 6144, wk).to(gl.float32)
    products = gl.reshape(w * a, (8, 128, 2, 4, 2))
    even, odd = gl.split(products)
    e0, e2 = gl.split(gl.reshape(even, (8, 128, 2, 2, 2)))
    o1, o3 = gl.split(gl.reshape(odd, (8, 128, 2, 2, 2)))
    pair_sums = gl.sum(e0 + e2, 3) + gl.sum(o1 + o3, 3)
    f = gl.sum(gl.sum(pair_sums, 2), 0)
    return f.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _m2_recurrent_head(X, CW, FW, CS, S, IDX, A, DT, NW, O,
                    M: gl.constexpr, SW: gl.constexpr, SI: gl.constexpr,
                    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                    SS: gl.constexpr, LOWER, EPS):

    PX: gl.constexpr = 8192 if M == 2 else 6336

    ST: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [8, 1], [1, 0])
    ST_T: gl.constexpr = gl.BlockedLayout([4, 1], [8, 8], [1, 8], [0, 1])

    LC: gl.constexpr = '.cg' if M <= 2 else ''


    SC: gl.constexpr = '.cs' if M == 2 else '.wt'
    pid = gl.program_id(0)
    if M == 2:
        m = pid % M
        h = pid // M
    else:
        m = pid // 12
        h = pid % 12
    mh = m * 12 + h
    i = gl.arange(0, 128, _VECTOR)
    slot = gl.load(IDX + m * SI).to(gl.int64)
    if slot < 0:
        gl.store(O + (m * PX + h * 128 if M == 2 else mh * 128) + i, 0)
    else:
        nw = gl.load(NW + i).to(gl.float32)
        beta = _m1_sigmoid(gl.load(X + m * PX + 6272 + h).to(gl.float32))
        if M == 1:
            r = gl.arange(0, 128, gl.SliceLayout(1, ST))
            c = gl.arange(0, 32, gl.SliceLayout(0, ST))
            sb = S + slot * SS + h * 16384
            so = r[:, None] * 128 + c[None, :]
            prefix = gl.amd.cdna4.buffer_load(sb, so, cache=LC)
        rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
        z, history = _m2_convolve(X, CW, CS, m, h, slot, SC0, SC1, SC2, PX)
        if M != 1:
            r = gl.arange(0, 128, gl.SliceLayout(1, ST))
            c = gl.arange(0, 32, gl.SliceLayout(0, ST))
            sb = S + slot * SS + h * 16384
            so = r[:, None] * 128 + c[None, :]
            prefix = gl.amd.cdna4.buffer_load(sb, so + (0 if M == 2 else 64), cache=LC)
        if M == 1:
            extra_packet = gl.amd.cdna4.buffer_load(sb, so + 64, cache=LC)
        elif M == 2:
            extra_packet = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
        if M == 4:

            packet0 = gl.amd.cdna4.buffer_load(sb, so, cache=LC)
            packet96 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
        if M == 1:
            packet96 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
        if M == 2:
            packet64 = gl.amd.cdna4.buffer_load(sb, so + 64, cache=LC)
        if M <= 2:

            forget_layout: gl.constexpr = gl.BlockedLayout(
                [1, 1, 16], [8, 8, 1], [1, 8, 1], [2, 0, 1],
            )
            dt_index = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, forget_layout)))
            dt_early = gl.load(DT + h * 128 + dt_index)
        f = _m2_forget(X, FW, m, h, SW, PX)
        if M == 4:
            decay_layout: gl.constexpr = gl.DistributedLinearLayout(
                [], [[0], [0], [64], [1], [2], [4]],
                [[8], [16], [32]], [], [128],
            )
            f = gl.convert_layout(f, decay_layout)
        prefix2 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=LC)
        shared = gl.allocate_shared_memory(gl.float32, (8, 128), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        partial_norm = gl.sum(gl.reshape(z * z, (4, 2, 64)), 2)
        shared.slice(7, 1).slice(0, 8, dim=1).store(gl.reshape(partial_norm, (1, 8)))
        shared.slice(0, 4).store(z)
        if M <= 2:
            dt = gl.convert_layout(dt_early, f.type.layout)
        else:
            dt = gl.load(DT + h * 128 + gl.arange(0, 128, f.type.layout))
        decay = gl.exp2(LOWER * _m1_sigmoid(rate * (f + dt)) * 1.4426950408889634)
        if M == 4:
            shared.slice(4, 1).store(gl.reshape(decay, (1, 128)))
        else:
            shared.slice(4, 1).store(decay[None, :])
        if M == 1:
            q = gl.sum(shared.slice(0, 1).load(ST), 0)
        key = gl.sum(shared.slice(1, 1).load(ST), 0)
        value = gl.sum(shared.slice(2, 1).load(ST_T), 0)
        value = gl.convert_layout(value, gl.SliceLayout(1, ST))
        decay = gl.sum(shared.slice(4, 1).load(ST), 0)
        gate = gl.sum(shared.slice(3, 1).load(_OUTPUT), 0)
        gate = gl.convert_layout(gate, _VECTOR)
        nl: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [0, 1])
        qp = shared.slice(7, 1).slice(0, 2, dim=1).load(nl)
        kp = shared.slice(7, 1).slice(2, 2, dim=1).load(nl)
        qnorm = gl.rsqrt(gl.sum(gl.sum(qp, 1), 0) + 1e-6)
        knorm = gl.rsqrt(gl.sum(gl.sum(kp, 1), 0) + 1e-6)
        keys = _m1_split_vector(key, ST)
        if M == 1:
            queries = _m1_split_vector(q, ST)
        decays = _m1_split_vector(decay, ST)
        decayed = ()
        for p in gl.static_range(4):
            if p == (0 if M <= 2 else 2):
                packet = prefix
            elif p == 1:
                packet = prefix2
            elif M == 4 and p == 0:
                packet = packet0
            elif M == 4 and p == 3:
                packet = packet96
            elif M == 2 and p == 2:
                packet = packet64
            elif M == 1 and p == 2:
                packet = extra_packet
            elif M == 2 and p == 3:
                packet = extra_packet
            elif M == 1 and p == 3:
                packet = packet96
            decayed += (packet * decays[p][None, :],)
        P_ORDER: gl.constexpr = (0, 1, 2, 3) if M == 2 else (0, 2, 1, 3) if M == 4 else (3, 2, 1, 0)
        prediction = _m1_contract(decayed, keys, P_ORDER)
        if M >= 2:
            q = gl.sum(shared.slice(0, 1).load(ST), 0)
            queries = _m1_split_vector(q, ST)
        if M == 4:
            delta = gl.fma(-prediction, knorm, value) * (beta * knorm)
        else:
            delta = (value - prediction * knorm) * beta
            delta = delta * knorm
        if M == 2:
            projected = _m1_contract(decayed, queries, P_ORDER)
            key_query = gl.sum(key * q, 0)
            out = gl.fma(delta, key_query, projected) * qnorm * 128 ** (-0.5)
        UPDATE_ORDER: gl.constexpr = (0, 1, 2, 3) if M == 2 else (3, 2, 1, 0)
        for p in gl.static_range(4):
            updated = gl.fma(delta[:, None], keys[UPDATE_ORDER[p]][None, :], decayed[UPDATE_ORDER[p]])
            if M != 2:
                if p == 0:
                    partial = updated * queries[UPDATE_ORDER[p]][None, :]
                else:
                    partial = gl.fma(updated, queries[UPDATE_ORDER[p]][None, :], partial)
            gl.amd.cdna4.buffer_store(updated, sb, so + UPDATE_ORDER[p] * 32, cache=SC)
        if M != 2:
            out = gl.sum(partial, 1) * qnorm * 128 ** (-0.5)
        out = out.to(gl.bfloat16).to(gl.float32)
        out = gl.convert_layout(out, gl.SliceLayout(0, ST_T))
        shared.slice(6, 1).store(out[None, :])
        out = gl.sum(shared.slice(6, 1).load(_OUTPUT), 0)
        out = gl.convert_layout(out, _VECTOR)
        scale = gl.rsqrt(gl.sum(out * out, 0) / 128 + EPS)
        if M == 2:
            out = out * scale * nw * gate
        else:
            out = out * scale * (nw * gate)
        old1, old2, x, hb, ho, valid = history
        gl.amd.cdna4.buffer_store(old1, hb, ho, valid)
        gl.amd.cdna4.buffer_store(old2, hb, SC1 + ho, valid)
        gl.amd.cdna4.buffer_store(x, hb, 2 * SC1 + ho, valid)
        gl.store(O + (m * PX + h * 128 if M == 2 else mh * 128) + i, out)


@gluon.jit
def _m2_prefetch_output_weights(W, SW: gl.constexpr, pid, M: gl.constexpr):

    if M == 4:

        STEP: gl.constexpr = 64
        COLS: gl.constexpr = 32
        pl: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [8, 1], [1, 0])
        rr = gl.arange(0, 64, gl.SliceLayout(1, pl))
        cc = gl.arange(0, COLS, gl.SliceLayout(0, pl))
        band = (pid + M * 12) % 8
        row = band * 896 + (pid // 8) * 56 + rr
        col = cc * (STEP // 2)
        ptr = W.to(gl.pointer_type(gl.int32)) + row[:, None] * (SW // 2) + col[None, :]
        word = gl.load(ptr, (rr[:, None] < 56) & (cc[None, :] < 1536 // STEP), other=0)
        gl.inline_asm_elementwise("", constraints="=v,0", args=[word],
                                  dtype=gl.int32, is_pure=False, pack=1)
    elif M == 2:

        pl: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [8, 1], [1, 0])
        rr = gl.arange(0, 64, gl.SliceLayout(1, pl))
        cc = gl.arange(0, 32, gl.SliceLayout(0, pl))
        local_row = rr // 2 * 36 + pid // 8 * 2 + rr % 2
        band = (pid + M * 12) % 8
        row = band * 896 + local_row
        ptr = W.to(gl.pointer_type(gl.int32)) + row[:, None] * (SW // 2) + cc[None, :] * 32
        word = gl.load(ptr, (local_row[:, None] < 896) & (cc[None, :] < 24), other=0)
        gl.inline_asm_elementwise("", constraints="=v,0", args=[word],
                                  dtype=gl.int32, is_pure=False, pack=1)
    else:
        STEP: gl.constexpr = 32

        CTAS: gl.constexpr = 144
        LINES: gl.constexpr = triton.cdiv(896 * 1536, STEP)
        CHUNK: gl.constexpr = triton.cdiv(LINES, CTAS // 8)
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
        lane = gl.arange(0, 4096, layout)
        logical = pid // 8 * CHUNK + lane
        band = (pid + M * 12) % 8
        element = band * (896 * 1536) + logical * STEP
        valid = (logical < LINES) & (lane < CHUNK)
        row = element // 1536
        col = element % 1536 // 2
        ptr = W.to(gl.pointer_type(gl.int32)) + row * (SW // 2) + col
        word = gl.load(ptr, valid, other=0)
        gl.inline_asm_elementwise("", constraints="=v,0", args=[word],
                                  dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _m2_recurrent_and_prefetch(X, CW, FW, CS, S, IDX, A, DT, NW, O, WO,
                            M: gl.constexpr, SW: gl.constexpr, SI: gl.constexpr,
                            SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                            SS: gl.constexpr, LOWER, EPS, OWN: gl.constexpr):

    pid = gl.program_id(0)
    if pid < M * 12:
        _m2_recurrent_head(X, CW, FW, CS, S, IDX, A, DT, NW, O, M, SW, SI,
                        SC0, SC1, SC2, SS, LOWER, EPS)
    else:
        _m2_prefetch_output_weights(WO, OWN, pid - M * 12, M)


@gluon.jit
def _m2_output_folded(X, W, Y, M: gl.constexpr, N: gl.constexpr, SX: gl.constexpr, SW: gl.constexpr):
    AN: gl.constexpr = 32
    BK: gl.constexpr = 128
    ROWS: gl.constexpr = 4
    ml: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, 4])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 4], [0, 1])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 4], [1, 0])
    ar = gl.arange(0, 16, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    br = gl.arange(0, AN * 4, gl.SliceLayout(0, bl))
    pid = gl.program_id(0).to(gl.uint32)

    tile = pid % 8 * (N // AN // 8) + (
        N // AN // 8 - 1 - pid // 8 if M == 2 else pid // 8
    )
    ao = ar[:, None] // 4 % M * SX + ar[:, None] % 4 * 8 + ak[None, :] // 8 * 32 + ak[None, :] % 8
    bo = (tile * AN + br[None, :] // 4) * SW + br[None, :] % 4 * 8 + bk[:, None] // 8 * 32 + bk[:, None] % 8
    acc = gl.zeros((16, AN * 4), gl.float32, ml)
    if M == 2:
        a0 = gl.amd.cdna4.buffer_load(X, ao)
        a1 = gl.amd.cdna4.buffer_load(X + 512, ao)
        a2 = gl.amd.cdna4.buffer_load(X + 1024, ao)
        b0 = gl.amd.cdna4.buffer_load(W, bo)
        b1 = gl.amd.cdna4.buffer_load(W + 512, bo)
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a0, gl.DotOperandLayout(0, ml, 8)),
            gl.convert_layout(b0, gl.DotOperandLayout(1, ml, 8)), acc,
        )
        b2 = gl.amd.cdna4.buffer_load(W + 1024, bo)
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a1, gl.DotOperandLayout(0, ml, 8)),
            gl.convert_layout(b1, gl.DotOperandLayout(1, ml, 8)), acc,
        )
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a2, gl.DotOperandLayout(0, ml, 8)),
            gl.convert_layout(b2, gl.DotOperandLayout(1, ml, 8)), acc,
        )
    else:

        a0 = gl.load(X + ao)
        a1 = gl.load(X + 512 + ao)
        a2 = gl.load(X + 1024 + ao)
        for step in gl.static_range(3):
            if step == 0:
                a = a0
            elif step == 1:
                a = a1
            else:
                a = a2
            b = gl.amd.cdna4.buffer_load(W + step * 512, bo)
            acc = gl.amd.cdna4.mfma(
                gl.convert_layout(a, gl.DotOperandLayout(0, ml, 8)),
                gl.convert_layout(b, gl.DotOperandLayout(1, ml, 8)), acc,
            )
    value = gl.reshape(gl.permute(acc, (1, 0)), (AN * 4, ROWS, 2, 2))
    even, odd = gl.split(value)
    v0, v2 = gl.split(even)
    v1, v3 = gl.split(odd)
    coord = gl.arange(0, AN * 4, gl.SliceLayout(1, v0.type.layout))
    low = gl.where(coord[:, None] & 1 == 0, v0, v1)
    high = gl.where(coord[:, None] & 1 == 0, v2, v3)
    diagonal = gl.where(coord[:, None] & 2 == 0, low, high)
    result = gl.permute(gl.sum(gl.reshape(diagonal, (AN, 4, ROWS)), 1), (1, 0))
    ol: gl.constexpr = result.type.layout
    om = gl.arange(0, ROWS, gl.SliceLayout(1, ol))
    on = tile * AN + gl.arange(0, AN, gl.SliceLayout(0, ol))
    gl.store(Y + om[:, None] * N + on[None, :], result, om[:, None] < M)


def kda_layer_decode_m2(
    x, qkvg_weight, beta_forget_weight, output_weight, forget_weight,
    conv_weight, a_log, dt_bias, norm_weight, conv_state, state, state_indices,
    *, lower_bound=-5.0, norm_eps=1e-5, output_tensor=None,
):

    m = x.shape[0]
    assert m in (1, 2, 4)


    packed = torch.empty(
        (m, 8192), dtype=torch.bfloat16, device=x.device,
    )
    core = packed[:, 4608:6144]
    out = (
        torch.empty((m, 7168), dtype=torch.bfloat16, device=x.device)
        if output_tensor is None else output_tensor
    )
    assert out.shape == (m, 7168) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _m2_input_projections[393,](x, qkvg_weight, beta_forget_weight, packed, m,
        x.stride(0), qkvg_weight.stride(0), beta_forget_weight.stride(0),
        num_warps=1, waves_per_eu=0)
    prefetch_ctas = 144
    _m2_recurrent_and_prefetch[m * 12 + prefetch_ctas,](packed, conv_weight, forget_weight, conv_state, state, state_indices,
        a_log, dt_bias, norm_weight, core, output_weight, m, forget_weight.stride(0), state_indices.stride(0),
        *conv_state.stride(), state.stride(0), lower_bound, norm_eps, output_weight.stride(0),
        num_warps=8, enable_fp_fusion=False, waves_per_eu=2)
    _m2_output_folded[224,](core, output_weight, out, m, 7168, 8192, output_weight.stride(0), num_warps=4)
    return out, conv_state, state


@gluon.jit
def _m256_fallback_gemm(X, W, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                   SXM: gl.constexpr, SXK: gl.constexpr, SWN: gl.constexpr,
                   SWK: gl.constexpr, SY: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    rows = gl.program_id(1) * 16 + gl.arange(0, 16, gl.SliceLayout(1, layout))
    cols = gl.program_id(0) * 64 + gl.arange(0, 64, gl.SliceLayout(1, layout))
    ks = gl.arange(0, 64, gl.SliceLayout(0, layout))
    acc = gl.zeros((16, 64), gl.float32, mma)

    for block in range(gl.cdiv(K, 64)):
        kk = block * 64 + ks
        a = gl.load(X + rows[:, None] * SXM + kk[None, :] * SXK,
                    (rows[:, None] < M) & (kk[None, :] < K), 0)
        b = gl.load(W + cols[:, None] * SWN + kk[None, :] * SWK,
                    (cols[:, None] < N) & (kk[None, :] < K), 0)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b = gl.convert_layout(gl.permute(b, (1, 0)), gl.DotOperandLayout(1, mma, 8))
        acc = gl.amd.cdna4.mfma(a, b, acc)
    row = gl.program_id(1) * 16 + gl.arange(0, 16, gl.SliceLayout(1, mma))
    col = gl.program_id(0) * 64 + gl.arange(0, 64, gl.SliceLayout(0, mma))
    gl.store(Y + row[:, None] * SY + col[None, :], acc.to(gl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


def _m256_fallback_linear(x, weight, output_tensor=None):
    m, k = x.shape
    n = weight.shape[0]
    out = x.new_empty((m, n)) if output_tensor is None else output_tensor
    _m256_fallback_gemm[(triton.cdiv(n, 64), triton.cdiv(m, 16))](
        x, weight, out, m, n, k, *x.stride(), *weight.stride(), out.stride(0), num_warps=4)
    return out


@gluon.jit
def _m256_masked_dense_tile(X, W, Y, tile_m, tile_n, M: gl.constexpr, N: gl.constexpr,
                       K: gl.constexpr, SX: gl.constexpr, SW: gl.constexpr,
                       RM: gl.constexpr, RN: gl.constexpr, BM: gl.constexpr,
                       BN: gl.constexpr, BK: gl.constexpr, NW: gl.constexpr,
                       DEPTH: gl.constexpr, WT: gl.constexpr, SY: gl.constexpr):
    phase: gl.constexpr = BK // 8
    ca: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
    cb: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, NW], [0, 1])
    wm: gl.constexpr = 1 if NW == 1 or BM == 16 else 2
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [wm, NW // wm])
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    sa = ()
    sb = ()
    for i in gl.static_range(DEPTH):
        sa += (gl.allocate_shared_memory(gl.bfloat16, [BM, BK], gl.SwizzledSharedLayout(8, 2, phase, [1, 0])),)
        sb += (gl.allocate_shared_memory(gl.bfloat16, [BK, BN], gl.SwizzledSharedLayout(8, 2, phase, [0, 1])),)
    ar = gl.arange(0, BM, gl.SliceLayout(1, ca))
    ak = gl.arange(0, BK, gl.SliceLayout(0, ca))
    bk = gl.arange(0, BK, gl.SliceLayout(1, cb))
    bc = gl.arange(0, BN, gl.SliceLayout(0, cb))
    ao = ar[:, None] * SX + (ak[None, :] ^ (ar[:, None] // 2 % phase * 8))
    bo = bc[None, :] * SW + (bk[:, None] ^ (bc[None, :] // 2 % phase * 8))
    ao = gl.max_contiguous(gl.multiple_of(ao, [1, 8]), [1, 8])
    bo = gl.max_contiguous(gl.multiple_of(bo, [8, 1]), [8, 1])
    if M % RM == 0:
        am = ar[:, None] < RM
    else:
        am = (ar[:, None] < RM) & (tile_m * RM + ar[:, None] < M)
    if N % RN == 0:
        bm = bc[None, :] < RN
    else:
        bm = (bc[None, :] < RN) & (tile_n * RN + bc[None, :] < N)
    X += tile_m * RM * SX
    W += tile_n * RN * SW
    for i in gl.static_range(DEPTH - 1):
        _m128_stage_masked_panel(sa[i], sb[i], X, W, ao, bo, am, bm, i * BK)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for step in gl.static_range(K // BK):
        if step < K // BK - DEPTH + 1:
            gl.amd.cdna4.async_copy.wait_group(DEPTH - 2)
            gl.barrier()
            _m128_stage_masked_panel(sa[(step + DEPTH - 1) % DEPTH],
                sb[(step + DEPTH - 1) % DEPTH],
                X,
                W,
                ao,
                bo,
                am,
                bm,
                (step + DEPTH - 1) * BK)
        else:
            gl.amd.cdna4.async_copy.wait_group(K // BK - step - 1)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(sa[step % DEPTH], da)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(sb[step % DEPTH], db)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    out_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [NW, 1], [1, 0])
    out = gl.convert_layout(acc.to(gl.bfloat16), out_layout)
    r = gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    c = gl.arange(0, BN, gl.SliceLayout(0, out_layout))
    offset = (tile_m * RM + r[:, None]) * SY + tile_n * RN + c[None, :]
    mask = (r[:, None] < RM) & (tile_m * RM + r[:, None] < M) & (c[None, :] < RN) & (tile_n * RN + c[None, :] < N)
    gl.amd.cdna4.buffer_store(out, Y, offset, mask, cache='.wt' if WT else '')


@gluon.jit
def _m256_stage_dense_panel(a_shared, b_shared, x_base, w_base, a_offsets, b_offsets, CACHE: gl.constexpr):
    a_linear = a_shared._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    b_linear = b_shared._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_linear, x_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_linear, w_base, b_offsets, cache_modifier=CACHE)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m256_consume_dense_panel(a_shared, b_shared, acc, DA: gl.constexpr,
                         DB: gl.constexpr, BK: gl.constexpr, FRAGMENTS: gl.constexpr):
    if FRAGMENTS:
        for k in gl.static_range(0, BK, 32):
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_shared.slice(k, 32, 0), DB)
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_shared.slice(k, 32, 1), DA)
            acc = gl.amd.cdna4.mfma(a, b, acc)
    else:
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_shared, DA)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_shared, DB)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    return acc


@gluon.jit
def _m256_dense_tile(X, W, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                SXM: gl.constexpr, SWN: gl.constexpr, BN: gl.constexpr,
                NW: gl.constexpr, tile_m, tile_n, WT: gl.constexpr,
                SY: gl.constexpr, DEPTH: gl.constexpr = 4,
                BM: gl.constexpr = 64, WM: gl.constexpr = 2, BK: gl.constexpr = 64):
    gl.static_assert(M % BM == 0 and N % BN == 0 and (K % BK == 0))
    copy_a: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BK, BK // 8], [NW, 1], [1, 0])
    copy_b: gl.constexpr = gl.BlockedLayout([8, 1], [BK // 8, 512 // BK], [1, NW], [0, 1])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [WM, NW // WM])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 2, BK // 8, [1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 2, BK // 8, [0, 1])
    ASYMMETRIC: gl.constexpr = M == 256 and BK == 128 and N >= 6144
    a_slots = ()
    b_slots = ()
    for slot in gl.static_range(DEPTH):
        if not ASYMMETRIC or slot < 2:
            a_slots += (gl.allocate_shared_memory(gl.bfloat16, [BM, BK], shared_a),)
        b_slots += (gl.allocate_shared_memory(gl.bfloat16, [BK, BN], shared_b),)
    rows = gl.arange(0, BM, gl.SliceLayout(1, copy_a))
    cols = gl.arange(0, BN, gl.SliceLayout(0, copy_b))
    a_k = gl.arange(0, BK, gl.SliceLayout(0, copy_a))
    b_k = gl.arange(0, BK, gl.SliceLayout(1, copy_b))
    a_offsets = rows[:, None] * SXM + (a_k[None, :] ^ rows[:, None] // 2 % (BK // 8) * 8)
    b_offsets = cols[None, :] * SWN + (b_k[:, None] ^ cols[None, :] // 2 % (BK // 8) * 8)
    a_offsets = gl.max_contiguous(gl.multiple_of(a_offsets, [1, 8]), [1, 8])
    b_offsets = gl.max_contiguous(gl.multiple_of(b_offsets, [8, 1]), [8, 1])
    x_base = X + tile_m * BM * SXM
    w_base = W + tile_n * BN * SWN
    if ASYMMETRIC:

        _m256_stage_dense_panel(a_slots[0], b_slots[0], x_base, w_base, a_offsets, b_offsets, '.cs')
        b_next = b_slots[1]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_next, w_base + BK, b_offsets, cache_modifier='.cs')
        gl.amd.cdna4.async_copy.commit_group()
        acc = gl.zeros((BM, BN), gl.float32, mma)
        for step in gl.static_range(K // BK):
            gl.amd.cdna4.async_copy.wait_group(1 if step < K // BK - 1 else 0)
            gl.barrier()
            for fragment in gl.static_range(0, BK, 32):
                b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[step % 3].slice(fragment, 32, 0), dot_b)
                a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[step % 2].slice(fragment, 32, 1), dot_a)
                acc = gl.amd.cdna4.mfma(a, b, acc)
                if fragment == 32:
                    if step < K // BK - 1:
                        a_spare = a_slots[(step + 1) % 2]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
                        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_spare, x_base + (step + 1) * BK, a_offsets)
                        gl.amd.cdna4.async_copy.commit_group()
                    if step < K // BK - 2:
                        b_spare = b_slots[(step + 2) % 3]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
                        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_spare,
                            w_base + (step + 2) * BK,
                            b_offsets,
                            cache_modifier='.cs')
                        gl.amd.cdna4.async_copy.commit_group()
    else:
        for slot in gl.static_range(DEPTH - 1):
            _m256_stage_dense_panel(a_slots[slot],
                b_slots[slot],
                x_base + slot * BK,
                w_base + slot * BK,
                a_offsets,
                b_offsets,
                '.cs' if N >= 6144 else '')
        acc = gl.zeros((BM, BN), gl.float32, mma)
        for step in gl.static_range(K // BK - DEPTH + 1):
            gl.amd.cdna4.async_copy.wait_group(DEPTH - 2)
            gl.barrier()
            future = (step + DEPTH - 1) * BK
            _m256_stage_dense_panel(a_slots[(step + DEPTH - 1) % DEPTH],
                b_slots[(step + DEPTH - 1) % DEPTH],
                x_base + future,
                w_base + future,
                a_offsets,
                b_offsets,
                '.cs' if N >= 6144 else '')
            acc = _m256_consume_dense_panel(a_slots[step % DEPTH],
                b_slots[step % DEPTH],
                acc,
                dot_a,
                dot_b,
                BK,
                BK == 128)
        for tail in gl.static_range(DEPTH - 1):
            gl.amd.cdna4.async_copy.wait_group(DEPTH - 2 - tail)
            acc = _m256_consume_dense_panel(a_slots[(K // BK - DEPTH + 1 + tail) % DEPTH],
                b_slots[(K // BK - DEPTH + 1 + tail) % DEPTH],
                acc,
                dot_a,
                dot_b,
                BK,
                BK == 128)
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
    result = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
    out_m = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, store_layout))
    out_n = tile_n * BN + gl.arange(0, BN, gl.SliceLayout(0, store_layout))
    gl.amd.cdna4.buffer_store(result, Y, out_m[:, None] * SY + out_n[None, :],
                             cache='.cs' if M == 256 and N == 7168 else '.wt' if WT else '')


@gluon.jit
def _m256_state_row_order(IDX, ORDER, M: gl.constexpr, SI: gl.constexpr,
                     NW: gl.constexpr, BLOCK: gl.constexpr, LOG: gl.constexpr):
    row = gl.arange(0, BLOCK, gl.BlockedLayout([BLOCK // 64], [64], [NW], [0]))
    slot = gl.load(IDX + row * SI, row < M, 0).to(gl.uint32)
    slot = gl.where(row < M, slot, 0xffffffff)
    key = (slot.to(gl.uint64) << 32) | row.to(gl.uint64)
    for outer in gl.static_range(1, LOG + 1):
        for inner in gl.static_range(outer - 1, -1, -1):
            other = gl.gather(key, row ^ (1 << inner), 0)
            ascending = (row & (1 << outer)) == 0
            lower = (row & (1 << inner)) == 0
            take_min = ascending == lower
            key = gl.where(take_min, gl.minimum(key, other), gl.maximum(key, other))
    gl.store(ORDER + row, key, row < M)


@gluon.jit
def _m256_project_qkvg_beta(
    X, WQ, WB, Q, B, M: gl.constexpr, SX: gl.constexpr, SQ: gl.constexpr, SB: gl.constexpr,
    BN: gl.constexpr, NW: gl.constexpr, GROUP: gl.constexpr, BR: gl.constexpr, BC: gl.constexpr,
    BD: gl.constexpr, WT: gl.constexpr, QS: gl.constexpr, BS: gl.constexpr,
    IDX, ORDER, SI: gl.constexpr, SORT: gl.constexpr, OB: gl.constexpr, LOG: gl.constexpr,
    BM: gl.constexpr = 64, WM: gl.constexpr = 2, QDEPTH: gl.constexpr = 4, QK: gl.constexpr = 64,
):
    qtiles: gl.constexpr = M // BM * (6144 // BN)
    pid = gl.program_id(0)
    if pid < qtiles:
        tile = pid % GROUP * (qtiles // GROUP) + pid // GROUP
        tile_m = tile // (6144 // BN)
        tile_n = tile % (6144 // BN)
        _m256_dense_tile(X, WQ, Q, M, 6144, 7168, SX, SQ, BN, NW, tile_m, tile_n, WT, QS, QDEPTH, BM, WM, QK)
    elif pid < qtiles + gl.cdiv(M, 64) * gl.cdiv(144, BC):
        bpid = pid - qtiles
        tm = bpid // gl.cdiv(144, BC)
        tn = bpid % gl.cdiv(144, BC)
        if BC == 16:
            _m256_dense_tile(X, WB, B, M, 144, 7168, SX, SB, BC, NW, tm, tn, False, BS, BD)
        else:
            _m256_masked_dense_tile(X, WB, B, tm, tn, M, 144, 7168, SX, SB, BR, BC, BR, BC, 64, NW, BD, False, BS)
    elif SORT:
        _m256_state_row_order(IDX, ORDER, M, SI, NW, OB, LOG)


@gluon.jit
def _m256_project_ragged(X, WQ, WB, Q, B, M: gl.constexpr, SX: gl.constexpr,
                    SQ: gl.constexpr, SB: gl.constexpr, QS: gl.constexpr, BS: gl.constexpr,
                    IDX, ORDER, SI: gl.constexpr, SORT: gl.constexpr, OB: gl.constexpr, LOG: gl.constexpr):
    qtiles: gl.constexpr = gl.cdiv(M, 64) * 96
    pid = gl.program_id(0)
    if pid < qtiles:
        tile = pid % 16 * (qtiles // 16) + pid // 16
        tm = tile // 96
        tn = tile % 96
        _m256_masked_dense_tile(X, WQ, Q, tm, tn, M, 6144, 7168, SX, SQ, 64, 64, 64, 64, 64, 4, 4, M > 128, QS)
    elif pid < qtiles + gl.cdiv(M, 64) * 9:
        tile = pid - qtiles
        _m256_masked_dense_tile(X,
            WB,
            B,
            tile // 9,
            tile % 9,
            M,
            144,
            7168,
            SX,
            SB,
            64,
            16,
            64,
            16,
            64,
            4,
            5,
            False,
            BS)
    elif SORT:
        _m256_state_row_order(IDX, ORDER, M, SI, 4, OB, LOG)


@gluon.jit
def _m256_output_ragged(X, W, Y, M: gl.constexpr, SX: gl.constexpr, SW: gl.constexpr, SY: gl.constexpr):
    tile = gl.program_id(0)
    _m256_masked_dense_tile(X, W, Y, tile // 112, tile % 112, M, 7168, 1536, SX, SW, 64, 64, 64, 64, 64, 4, 4, True, SY)


@gluon.jit
def _m256_stage_output_a(a_slots, X, offsets, PANEL: gl.constexpr):

    for segment in gl.static_range(3):
        linear = a_slots[PANEL % 2 * 3 + segment]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(linear, X + PANEL * 192, offsets + segment * 64)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m256_stage_output_b(b_slots, W, offsets, PANEL: gl.constexpr):
    for segment in gl.static_range(3):
        linear = b_slots[PANEL % 2 * 3 + segment]._reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(linear,
            W + PANEL * 192,
            offsets + segment * 64,
            cache_modifier='.cs')
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m256_output_m128(X, W, Y, SX: gl.constexpr, SW: gl.constexpr, SY: gl.constexpr):
    pid = gl.program_id(0).to(gl.uint32)
    tile_m = pid >> 3 & 1
    tile_n = (pid & 7) * 14 + pid // 16
    copy_a: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    copy_b: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [2, 2])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    a_slots = ()
    b_slots = ()
    for segment in gl.static_range(6):
        a_slots += (gl.allocate_shared_memory(gl.bfloat16, [64, 64], gl.SwizzledSharedLayout(8, 2, 8, [1, 0])),)
    for segment in gl.static_range(6):
        b_slots += (gl.allocate_shared_memory(gl.bfloat16, [64, 64], gl.SwizzledSharedLayout(8, 2, 8, [0, 1])),)
    rows = gl.arange(0, 64, gl.SliceLayout(1, copy_a))
    cols = gl.arange(0, 64, gl.SliceLayout(0, copy_b))
    ak = gl.arange(0, 64, gl.SliceLayout(0, copy_a))
    bk = gl.arange(0, 64, gl.SliceLayout(1, copy_b))
    ao = rows[:, None] * SX + (ak[None, :] ^ rows[:, None] // 2 % 8 * 8)
    bo = cols[None, :] * SW + (bk[:, None] ^ cols[None, :] // 2 % 8 * 8)
    ao = gl.max_contiguous(gl.multiple_of(ao, [1, 8]), [1, 8])
    bo = gl.max_contiguous(gl.multiple_of(bo, [8, 1]), [8, 1])
    X += tile_m * 64 * SX
    W += tile_n * 64 * SW
    _m256_stage_output_a(a_slots, X, ao, 0)
    _m256_stage_output_b(b_slots, W, bo, 0)
    _m256_stage_output_b(b_slots, W, bo, 1)
    acc = gl.zeros((64, 64), gl.float32, mma)
    for panel in gl.static_range(8):
        gl.amd.cdna4.async_copy.wait_group(1 if panel < 7 else 0)
        gl.barrier()
        for segment in gl.static_range(3):
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[panel % 2 * 3 + segment], dot_b)
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[panel % 2 * 3 + segment], dot_a)
            acc = gl.amd.cdna4.mfma(a, b, acc)
            if segment == 0 and panel < 7:

                _m256_stage_output_a(a_slots, X, ao, panel + 1)
        if panel < 7:


            acc = gl.inline_asm_elementwise('', '=v,0,~{memory}', [acc], dtype=gl.float32, is_pure=False, pack=1)
            gl.inline_asm_elementwise('s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0',
                '=v,~{memory}',
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1)
            gl.barrier()
            if panel < 6:
                _m256_stage_output_b(b_slots, W, bo, panel + 2)
    result = gl.convert_layout(acc.to(gl.bfloat16), copy_a)
    out_m = gl.arange(0, 64, gl.SliceLayout(1, copy_a))
    out_n = gl.arange(0, 64, gl.SliceLayout(0, copy_a))
    y_base = Y + tile_m * 64 * SY + tile_n * 64
    gl.amd.cdna4.buffer_store(result, y_base, out_m[:, None] * SY + out_n[None, :], cache='.wt')


@gluon.jit
def _m256_output_m256(X, W, Y, M: gl.constexpr, SX: gl.constexpr, SW: gl.constexpr, SY: gl.constexpr):
    pid = gl.program_id(0).to(gl.uint32)
    tile_m = pid >> 3 & 1
    tile_n = (pid & 7) * 14 + pid // 16
    _m256_dense_tile(X, W, Y, M, 7168, 1536, SX, SW, 64, 4, tile_m, tile_n, True, SY, 3, 128, 2, 128)


@gluon.jit
def _m256_bf16_pair_sum(w0, w1, x0, x1, NW: gl.constexpr):
    rows: gl.constexpr = w0.shape[0]
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [16, 4, 1], [1, NW, 1], [0, 1, 2])
    weights = gl.permute(gl.join(w0, w1), (1, 0, 2))
    values = gl.permute(gl.join(x0, x1), (1, 2, 0))
    weights = gl.convert_layout(weights, gl.DotOperandLayout(0, layout, 0))
    values = gl.convert_layout(values, gl.DotOperandLayout(1, layout, 0))
    dots = gl.dot_fma(weights, values, gl.zeros((16, rows, 1), gl.float32, layout))
    return gl.permute(gl.reshape(dots, (16, rows)), (1, 0))


@gluon.jit
def _m256_project_forget_head(FA, FW, m, h, SFA: gl.constexpr, SW: gl.constexpr,
                         ROWS: gl.constexpr, OFFSET: gl.constexpr, NW: gl.constexpr):
    pl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NW, 1], [1, 0])
    n = gl.arange(0, ROWS, gl.SliceLayout(1, pl)) + OFFSET
    k = gl.arange(0, 128, gl.SliceLayout(0, pl))
    x = gl.amd.cdna4.buffer_load(FA + m * SFA, k)
    w = gl.amd.cdna4.buffer_load(FW + h * 128 * SW, n[:, None] * SW + k[None, :])
    w0, w1, w2, w3, w4, w5, w6, w7 = _m128_eight_parts(w)
    x0, x1, x2, x3, x4, x5, x6, x7 = _m128_eight_parts(x[None, :])
    even = _m256_bf16_pair_sum(w0, w2, x0, x2, NW) + _m256_bf16_pair_sum(w4, w6, x4, x6, NW)
    odd = _m256_bf16_pair_sum(w1, w3, x1, x3, NW) + _m256_bf16_pair_sum(w5, w7, x5, x7, NW)
    partial = even + odd
    return gl.sum(partial, 1).to(gl.bfloat16)


@gluon.jit
def _m256_convolve_head(X, CW, CS, m, h, slot, H: gl.constexpr, SX: gl.constexpr,
                    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                    LAYOUT: gl.constexpr, PREP, NW: gl.constexpr):
    cl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW // 2, 2], [1, 0])
    plane = gl.arange(0, 4, gl.SliceLayout(1, cl))
    channel = gl.arange(0, 128, gl.SliceLayout(0, cl))
    c = plane[:, None] * H * 128 + h * 128 + channel[None, :]
    valid = plane[:, None] < 3
    history_base = CS + slot * SC0 + h * 128 * SC2
    history_offsets = (plane[:, None] * H * 128 + channel[None, :]) * SC2
    old0 = gl.amd.cdna4.buffer_load(history_base, history_offsets, valid, 0)
    old1 = gl.amd.cdna4.buffer_load(history_base, SC1 + history_offsets, valid, 0)
    old2 = gl.amd.cdna4.buffer_load(history_base, 2 * SC1 + history_offsets, valid, 0)
    x = gl.load(X + m * SX + c, valid, 0)
    wl: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 64, 1], [NW // 2, 2, 1], [2, 1, 0])
    wp = gl.arange(0, 4, gl.SliceLayout(1, gl.SliceLayout(2, wl)))
    wc = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, wl)))
    taps = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    weight_c = wp[:, None] * H * 128 + h * 128 + wc[None, :]
    weights = gl.load(CW + weight_c[:, :, None] * 4 + taps[None, None, :], weight_c[:, :, None] < 3 * H * 128, 0)
    even, odd = gl.split(gl.reshape(weights, (4, 128, 2, 2)))
    w0, w2 = gl.split(even)
    w1, w3 = gl.split(odd)
    w0 = gl.convert_layout(w0, cl)
    w1 = gl.convert_layout(w1, cl)
    w2 = gl.convert_layout(w2, cl)
    w3 = gl.convert_layout(w3, cl)
    z = old0.to(gl.float32) * w0 + old1.to(gl.float32) * w1 + old2.to(gl.float32) * w2 + x.to(gl.float32) * w3
    z = (z * _m1_sigmoid(z)).to(gl.bfloat16)
    changed0 = old0.to(gl.uint16, bitcast=True) != old1.to(gl.uint16, bitcast=True)
    changed1 = old1.to(gl.uint16, bitcast=True) != old2.to(gl.uint16, bitcast=True)
    changed2 = old2.to(gl.uint16, bitcast=True) != x.to(gl.uint16, bitcast=True)
    gl.amd.cdna4.buffer_store(old1, history_base, history_offsets, valid & changed0)
    gl.amd.cdna4.buffer_store(old2, history_base, SC1 + history_offsets, valid & changed1)
    gl.amd.cdna4.buffer_store(x, history_base, 2 * SC1 + history_offsets, valid & changed2)
    handoff = PREP.slice(0, 512)
    handoff.store(gl.reshape(z, (512,)))
    q = handoff.slice(0, 128).load(gl.SliceLayout(0, LAYOUT)).to(gl.float32)
    key = handoff.slice(128, 128).load(gl.SliceLayout(0, LAYOUT)).to(gl.float32)
    return (q, key)


@gluon.jit
def _m256_or_reduce(a, b):
    return a | b


@gluon.jit
def _m256_store_changed_vectors(updated, previous, base, offsets, WT: gl.constexpr,
                            LINE_VECTORS: gl.constexpr, STREAM: gl.constexpr = False):

    changed_bits = updated.to(gl.uint32, bitcast=True) ^ previous.to(gl.uint32, bitcast=True)
    rows: gl.constexpr = updated.shape[0]
    even, odd = gl.split(gl.reshape(changed_bits, (rows, 32, 2, 2)))
    b0, b2 = gl.split(even)
    b1, b3 = gl.split(odd)
    diff = gl.inline_asm_elementwise('v_or3_b32 $0, $1, $2, $3\nv_or_b32 $0, $0, $4',
        constraints='=&v,v,v,v,v',
        args=[b0, b1, b2, b3],
        dtype=gl.uint32,
        is_pure=True,
        pack=1)
    if LINE_VECTORS > 1:
        grouped = gl.reshape(diff, (rows, 32 // LINE_VECTORS, LINE_VECTORS))
        line_diff = gl.reduce(grouped, 2, _m256_or_reduce)
        line_mask, _ = gl.broadcast(line_diff[:, :, None], grouped)
        diff = gl.convert_layout(gl.reshape(line_mask, (rows, 32)), b0.type.layout, assert_trivial=True)
    mask, _ = gl.broadcast((diff != 0)[:, :, None, None], gl.reshape(changed_bits, (rows, 32, 2, 2)))
    store_mask = gl.reshape(mask, (rows, 128))
    gl.amd.cdna4.buffer_store(updated, base, offsets, store_mask, cache='.cs' if STREAM else '.wt' if WT else '')


@gluon.jit
def _m256_kda_decode(
    X, Gate, FA, Beta, FW, CW, A, DT, W, CS, S, IDX, O, H: gl.constexpr, SX: gl.constexpr,
    SG: gl.constexpr, SFA: gl.constexpr, SB: gl.constexpr, SW: gl.constexpr, SC0: gl.constexpr,
    SC1: gl.constexpr, SC2: gl.constexpr, SS: gl.constexpr, SI: gl.constexpr, lower, eps,
    GROUP: gl.constexpr, LARGE_BATCH: gl.constexpr, NW: gl.constexpr,
    WT: gl.constexpr, SO: gl.constexpr, ORDER, SORT: gl.constexpr,
    STATE_LINE_VECTORS: gl.constexpr, STATE_STREAM: gl.constexpr,
):
    state_cache: gl.constexpr = '.cg'
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [NW, 1], [1, 0])
    pid = gl.program_id(0)
    if not LARGE_BATCH:
        pid = pid.to(gl.uint32)
    m = gl.program_id(1) * GROUP + pid % GROUP
    h = pid // GROUP
    if LARGE_BATCH:
        h = h ^ (gl.program_id(1) & 3)
    else:
        group_id = gl.program_id(1)
        h = h ^ ((group_id ^ (group_id >> 2)) & 3)
    if SORT:
        record = gl.load(ORDER + m)
        m = record.to(gl.int32)
        if ORDER.dtype.element_ty == gl.int64:
            slot = (record.to(gl.uint64) >> 32).to(gl.int32).to(gl.int64)
        else:
            slot = gl.load(IDX + m * SI).to(gl.int64)
    else:
        slot = gl.load(IDX + m * SI).to(gl.int64)
    v = gl.arange(0, 128, gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, gl.SliceLayout(0, layout))
    if slot < 0:
        gl.store(O + m * SO + h * 128 + v, 0)
    else:
        prep = gl.allocate_shared_memory(gl.bfloat16, (512,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
        auxiliary = gl.allocate_shared_memory(gl.bfloat16, (256,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
        if not LARGE_BATCH:
            projection = gl.allocate_shared_memory(gl.bfloat16, (128,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
        if LARGE_BATCH:
            for offset in gl.static_range(0, 128, 32):
                projected_part = _m256_project_forget_head(FA, FW, m, h, SFA, SW, 32, offset, NW)
                auxiliary.slice(offset, 32).store(projected_part)
        else:
            projected = _m256_project_forget_head(FA, FW, m, h, SFA, SW, 128, 0, NW)
            projection = projection.reshape((32, 4)).permute((1, 0)).reshape((128,))
            projection.store(projected)
        if LARGE_BATCH:
            base = S + slot * SS + h * 128 * 128
        else:
            base = S + slot * SS
        slab_v = gl.arange(0, 32, gl.SliceLayout(1, layout))
        if LARGE_BATCH:
            slab_offsets = slab_v[:, None] * 128 + k[None, :]
        else:
            slab_offsets = (h * 128 + slab_v[:, None]) * 128 + k[None, :]
        row0: gl.constexpr = 96 if LARGE_BATCH else 0
        row1: gl.constexpr = 32
        row2: gl.constexpr = 64
        row3: gl.constexpr = 0 if LARGE_BATCH else 96
        old0 = gl.amd.cdna4.buffer_load(base, slab_offsets + row0 * 128, cache=state_cache)
        if LARGE_BATCH:
            old1 = gl.amd.cdna4.buffer_load(base, slab_offsets + row1 * 128, cache=state_cache)
        q, key = _m256_convolve_head(X, CW, CS, m, h, slot, H, SX, SC0, SC1, SC2, layout, prep, NW)
        if not LARGE_BATCH:
            old1 = gl.amd.cdna4.buffer_load(base, slab_offsets + row1 * 128, cache=state_cache)
        decay_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
        dk = gl.arange(0, 128, decay_layout)
        if LARGE_BATCH:
            projected = auxiliary.slice(0, 128).load(decay_layout).to(gl.float32)
        else:
            projected = projection.load(decay_layout).to(gl.float32)
        raw = projected + gl.load(DT + h * 128 + dk)
        decay = gl.exp(lower * _m1_sigmoid(gl.exp(gl.load(A + h)) * raw))
        decay = gl.convert_layout(decay, gl.SliceLayout(0, layout))
        q = q * gl.rsqrt(gl.sum(q * q, 0) + 1e-06) * 128 ** (-0.5)
        key = key * gl.rsqrt(gl.sum(key * key, 0) + 1e-06)
        beta = _m1_sigmoid(gl.load(Beta + m * SB + h).to(gl.float32))
        norm_layout: gl.constexpr = gl.BlockedLayout([2], [64], [NW], [0])
        nv = gl.arange(0, 128, norm_layout)
        weight = gl.amd.cdna4.buffer_load(W, nv).to(gl.float32)
        gate = gl.amd.cdna4.buffer_load(Gate + m * SG + h * 128, nv).to(gl.float32)
        value0 = prep.slice(256 + row0, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
        updated0 = old0 * decay[None, :]
        delta0 = (value0 - gl.sum(updated0 * key[None, :], 1)) * beta
        updated0 = gl.fma(delta0[:, None], key[None, :], updated0)
        if LARGE_BATCH:
            _m256_store_changed_vectors(updated0,
                old0,
                base,
                slab_offsets + row0 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM)
        core0 = gl.sum(updated0 * q[None, :], 1)
        if not LARGE_BATCH:
            _m256_store_changed_vectors(updated0,
                old0,
                base,
                slab_offsets + row0 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM)
        if LARGE_BATCH:
            auxiliary.slice(row0, 32).store(core0.to(gl.bfloat16))
        value1 = prep.slice(256 + row1, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
        updated1 = old1 * decay[None, :]
        delta1 = (value1 - gl.sum(updated1 * key[None, :], 1)) * beta
        updated1 = gl.fma(delta1[:, None], key[None, :], updated1)
        if LARGE_BATCH:
            _m256_store_changed_vectors(updated1,
                old1,
                base,
                slab_offsets + row1 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM)
        core1 = gl.sum(updated1 * q[None, :], 1)
        if not LARGE_BATCH:
            _m256_store_changed_vectors(updated1,
                old1,
                base,
                slab_offsets + row1 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM)
        if LARGE_BATCH:
            auxiliary.slice(row1, 32).store(core1.to(gl.bfloat16))
        if not LARGE_BATCH:
            pair0 = gl.reshape(gl.permute(gl.join(core0, core1), (1, 0)), (64,))
            auxiliary.slice(0, 64).store(pair0.to(gl.bfloat16))
        core1 = gl.inline_asm_elementwise('', '=v,0,~{memory}', [core1], dtype=gl.float32, is_pure=False, pack=1)

        old2 = gl.amd.cdna4.buffer_load(base, slab_offsets + row2 * 128, cache=state_cache)
        old3 = gl.amd.cdna4.buffer_load(base, slab_offsets + row3 * 128, cache=state_cache)
        value2 = prep.slice(256 + row2, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
        updated2 = old2 * decay[None, :]
        delta2 = (value2 - gl.sum(updated2 * key[None, :], 1)) * beta
        updated2 = gl.fma(delta2[:, None], key[None, :], updated2)
        if LARGE_BATCH:
            _m256_store_changed_vectors(updated2,
                old2,
                base,
                slab_offsets + row2 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM)
        core2 = gl.sum(updated2 * q[None, :], 1)
        if not LARGE_BATCH:
            _m256_store_changed_vectors(updated2,
                old2,
                base,
                slab_offsets + row2 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM)
        if LARGE_BATCH:
            auxiliary.slice(row2, 32).store(core2.to(gl.bfloat16))
        if not LARGE_BATCH:
            core2 = gl.inline_asm_elementwise('',
                '=v,0,~{memory}',
                [core2],
                dtype=gl.float32,
                is_pure=False,
                pack=1)
        value3 = prep.slice(256 + row3, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
        updated3 = old3 * decay[None, :]
        delta3 = (value3 - gl.sum(updated3 * key[None, :], 1)) * beta
        updated3 = gl.fma(delta3[:, None], key[None, :], updated3)
        if LARGE_BATCH:
            _m256_store_changed_vectors(updated3,
                old3,
                base,
                slab_offsets + row3 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM)
        core3 = gl.sum(updated3 * q[None, :], 1)
        if not LARGE_BATCH:
            _m256_store_changed_vectors(updated3,
                old3,
                base,
                slab_offsets + row3 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM)
        if LARGE_BATCH:
            auxiliary.slice(row3, 32).store(core3.to(gl.bfloat16))
        if not LARGE_BATCH:
            pair1 = gl.reshape(gl.permute(gl.join(core2, core3), (1, 0)), (64,))
            auxiliary.slice(64, 64).store(pair1.to(gl.bfloat16))
        core = auxiliary.slice(0, 128).load(norm_layout).to(gl.float32)
        out = core * gl.rsqrt(gl.sum(core * core, 0) / 128 + eps) * weight * _m1_sigmoid(gate)
        gl.store(O + m * SO + h * 128 + nv, out)


def kda_layer_decode_m256(
    x, qkvg_weight, beta_forget_weight, output_weight, forget_weight,
    conv_weight, a_log, dt_bias, norm_weight, conv_state, state, state_indices,
    *, lower_bound=-5.0, norm_eps=1e-5, output_tensor=None,
):

    m = x.shape[0]
    group = 8
    q_write_through = m not in (128, 256)
    qkvg = x.new_empty((m, 6144))
    beta_forget = x.new_empty((m, 144))
    core = x.new_empty((m, 1536))
    aligned = (m in (128, 256) and x.stride(0) % 8 == 0
               and qkvg_weight.stride(0) % 8 == 0
               and beta_forget_weight.stride(0) % 8 == 0
               and output_weight.stride(0) % 8 == 0)
    ragged_aligned = (m >= 64 and x.stride(0) % 8 == 0
                      and qkvg_weight.stride(0) % 8 == 0
                      and beta_forget_weight.stride(0) % 8 == 0
                      and output_weight.stride(0) % 8 == 0)
    sort_rows = 128 <= m <= 512 and ragged_aligned
    order_dtype = torch.int64 if state_indices.dtype == torch.int32 else torch.int32
    row_order = torch.empty((m,), device=x.device, dtype=order_dtype) if sort_rows else None
    order_block = triton.next_power_of_2(m)
    order_log = order_block.bit_length() - 1
    if aligned:
        tile_n = 64
        tile_m = 128
        waves = 4
        pid_groups = 16
        beta_columns = 32
        beta_depth = 4
        programs = m // tile_m * (6144 // tile_n) + m // 64 * triton.cdiv(144, beta_columns)
        _m256_project_qkvg_beta[(programs + int(sort_rows),)](
            x, qkvg_weight, beta_forget_weight, qkvg, beta_forget,
            m, x.stride(0), qkvg_weight.stride(0), beta_forget_weight.stride(0),
            tile_n, waves, pid_groups, 64, beta_columns, beta_depth, q_write_through,
            qkvg.stride(0), beta_forget.stride(0),
            state_indices, row_order, state_indices.stride(0), sort_rows, order_block, order_log,
            tile_m, 2, 3, 128, num_warps=waves, waves_per_eu=2)
    elif ragged_aligned:
        _m256_project_ragged[(triton.cdiv(m, 64) * 105 + int(sort_rows),)](
            x, qkvg_weight, beta_forget_weight, qkvg, beta_forget, m,
            x.stride(0), qkvg_weight.stride(0), beta_forget_weight.stride(0),
            qkvg.stride(0), beta_forget.stride(0),
            state_indices, row_order, state_indices.stride(0), sort_rows, order_block, order_log,
            num_warps=4, waves_per_eu=2)
    else:
        _m256_fallback_linear(x, qkvg_weight, output_tensor=qkvg)
        _m256_fallback_linear(x, beta_forget_weight, output_tensor=beta_forget)
    decode = _m256_kda_decode
    decode[(12 * group, m // group)](
        qkvg, qkvg[:, 4608:], beta_forget, beta_forget[:, 128:140],
        forget_weight, conv_weight, a_log, dt_bias, norm_weight,
        conv_state, state, state_indices, core,
        12, qkvg.stride(0), qkvg.stride(0), beta_forget.stride(0), beta_forget.stride(0), forget_weight.stride(0),
        *conv_state.stride(), state.stride(0), state_indices.stride(0),
        lower_bound, norm_eps, group, m >= 192, 8, True, core.stride(0), row_order, sort_rows,
        4, m >= 128, num_warps=8, enable_fp_fusion=False)
    out = x.new_empty((m, 7168)) if output_tensor is None else output_tensor
    if aligned and m == 256:
        _m256_output_m256[(m // 128 * 112,)](core, output_weight, out, m, core.stride(0), output_weight.stride(0), out.stride(0), num_warps=4, waves_per_eu=2)
    elif aligned:
        _m256_output_m128[(m // 64 * 112,)](core, output_weight, out, core.stride(0), output_weight.stride(0), out.stride(0), num_warps=4, waves_per_eu=2)
    elif ragged_aligned:
        _m256_output_ragged[(triton.cdiv(m, 64) * 112,)](core, output_weight, out, m, core.stride(0), output_weight.stride(0), out.stride(0), num_warps=4)
    else:
        _m256_fallback_linear(core, output_weight, output_tensor=out)
    return out, conv_state, state


@gluon.jit
def _m32_convolve(
    X, CW, CS, row, head, slot,
    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr, NW: gl.constexpr,
):
    conv_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, NW // 4], [1, 0])
    group = gl.arange(0, 4, gl.SliceLayout(1, conv_layout))
    channel = gl.arange(0, 128, gl.SliceLayout(0, conv_layout))
    offset = (group[:, None] * 12 + head) * 128 + channel[None, :]
    valid = group[:, None] < 3
    history = CS + slot * SC0
    old0 = gl.amd.cdna4.buffer_load(history, offset * SC2, valid, 0).to(gl.float32)
    old1 = gl.amd.cdna4.buffer_load(history, SC1 + offset * SC2, valid, 0)
    old2 = gl.amd.cdna4.buffer_load(history, 2 * SC1 + offset * SC2, valid, 0)
    x = gl.load(X + row * 6336 + offset)
    wl: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 64, 1], [4, NW // 4, 1], [2, 1, 0])
    wo = gl.convert_layout(offset, gl.SliceLayout(2, wl), assert_trivial=True)
    wvalid = gl.convert_layout(valid, gl.SliceLayout(2, wl), assert_trivial=True)
    tap = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    weights = gl.amd.cdna4.buffer_load(CW, wo[:, :, None] * 4 + tap[None, None, :], wvalid[:, :, None], 0)
    even, odd = gl.split(gl.reshape(weights, (4, 128, 2, 2)))
    w0, w2 = gl.split(even)
    w1, w3 = gl.split(odd)
    w0 = gl.convert_layout(w0, conv_layout)
    w1 = gl.convert_layout(w1, conv_layout)
    w2 = gl.convert_layout(w2, conv_layout)
    w3 = gl.convert_layout(w3, conv_layout)
    z = old0 * w0 + old1.to(gl.float32) * w1 + old2.to(gl.float32) * w2 + x.to(gl.float32) * w3
    z = gl.where(group[:, None] == 3, x.to(gl.float32), z)
    sigmoid = _m1_sigmoid(z)
    z = gl.where(group[:, None] == 3, sigmoid, (z * sigmoid).to(gl.bfloat16).to(gl.float32))
    return z, (old1, old2, x, history, offset * SC2, valid)


@gluon.jit
def _m32_publish_decay(X, FW, A, DT, shared, m, h, LOWER,
                   SW: gl.constexpr, NW: gl.constexpr, STAGED: gl.constexpr):

    width: gl.constexpr = min(8, SW & -SW)
    lanes_k: gl.constexpr = min(64, 128 // width)
    waves_k: gl.constexpr = 128 // (width * lanes_k)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, width], [64 // lanes_k, lanes_k], [NW // waves_k, waves_k], [1, 0])
    rows = gl.arange(0, 32, gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, gl.SliceLayout(0, layout))
    a = gl.amd.cdna4.buffer_load(X + m * 6336 + 6144, k).to(gl.float32)
    if not STAGED:
        rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
    current = gl.amd.cdna4.buffer_load(
        FW + h * 128 * SW, rows[:, None] * SW + k[None, :]).to(gl.float32)
    paired: gl.constexpr = SW % 8 == 0
    if STAGED:
        staging = gl.allocate_shared_memory(
            gl.bfloat16, (128,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
    if not STAGED:
        contiguous: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
        channels = gl.arange(0, 64 if paired else 32, contiguous)
        bias = gl.load(DT + h * 128 + channels)
    for tile in gl.static_range(4):
        with gl.amd.warp_pipeline_stage("forget_decay"):
            if tile < 3:
                following = gl.amd.cdna4.buffer_load(
                    FW + (h * 128 + (tile + 1) * 32) * SW,
                    rows[:, None] * SW + k[None, :]).to(gl.float32)
            if not STAGED:
                if paired and tile == 1:
                    next_bias = gl.load(DT + h * 128 + 64 + channels)
                elif not paired and tile < 3:
                    next_bias = gl.load(DT + h * 128 + (tile + 1) * 32 + channels)
            f = gl.sum(current * a[None, :], 1).to(gl.bfloat16)
            if STAGED:
                staging.slice(tile * 32, 32).store(f)
            elif paired:
                if tile % 2 == 0:
                    first_half = f
                else:
                    joined = gl.reshape(gl.permute(gl.join(first_half, f), (1, 0)), (64,))
                    pair = gl.convert_layout(joined, contiguous).to(gl.float32)
                    decay = gl.exp2(LOWER * _m1_sigmoid(rate * (pair + bias)) * 1.4426950408889634)
                    shared.slice(4, 1).slice((tile - 1) * 32, 64, dim=1).store(gl.reshape(decay, (1, 64)))
            else:
                panel = gl.convert_layout(f.to(gl.float32), contiguous)
                decay = gl.exp2(LOWER * _m1_sigmoid(rate * (panel + bias)) * 1.4426950408889634)
                shared.slice(4, 1).slice(tile * 32, 32, dim=1).store(gl.reshape(decay, (1, 32)))
            if tile < 3:
                current = following
                if not STAGED and ((paired and tile == 1) or not paired):
                    bias = next_bias
    if STAGED:
        rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
        vl: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
        channel = gl.arange(0, 128, vl)
        dt = gl.load(DT + h * 128 + channel)
        f = staging.load(vl).to(gl.float32)
        decay = gl.exp2(LOWER * _m1_sigmoid(rate * (f + dt)) * 1.4426950408889634)
        shared.slice(4, 1).store(gl.reshape(decay, (1, 128)))


@gluon.jit
def _m32_recurrent_head(
    X, CW, FW, CS, S, IDX, A, DT, NORM, O,
    SW: gl.constexpr, SI: gl.constexpr,
    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
    SS: gl.constexpr, LOWER, EPS,
    NW: gl.constexpr = 8,
    M: gl.constexpr = 32, GROUP: gl.constexpr = 1,
    CORE_STRIDE: gl.constexpr = 1536,
    NARROW_SLOT: gl.constexpr = False,
):

    st: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [NW, 1], [1, 0])
    st_t: gl.constexpr = gl.BlockedLayout([4, 1], [8, 8], [1, NW], [0, 1])
    output_layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [4, NW // 4], [1, 0])
    vector_layout: gl.constexpr = gl.BlockedLayout([2], [64], [NW], [0])
    pid = gl.program_id(0)
    group = pid // (12 * GROUP)
    remainder = pid % (12 * GROUP)
    if M % GROUP == 0:
        group_size = GROUP
    else:
        group_size = gl.minimum(GROUP, M - group * GROUP)
    m = group * GROUP + remainder % group_size
    h = remainder // group_size
    if M > 32:

        h = (h * 5) % 12
    i = gl.arange(0, 128, vector_layout)
    slot = gl.load(IDX + m * SI).to(gl.int64)
    if slot < 0:
        gl.store(O + m * CORE_STRIDE + h * 128 + i, 0)
    else:
        if NARROW_SLOT:
            storage_slot = slot.to(gl.int32)
        else:
            storage_slot = slot
        r = gl.arange(0, 128, gl.SliceLayout(1, st))
        c = gl.arange(0, 32, gl.SliceLayout(0, st))
        sb = S + storage_slot * SS + h * 16384
        so = r[:, None] * 128 + c[None, :]
        shared = gl.allocate_shared_memory(
            gl.float32, (8, 128), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        if M <= 32:
            with gl.amd.warp_pipeline_stage("state_prefetch"):
                state_k3 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=".cg")
                state_k2 = gl.amd.cdna4.buffer_load(sb, so + 64, cache=".cg")
                state_k1 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=".cg")
                state_k0 = gl.amd.cdna4.buffer_load(sb, so, cache=".cg")
        else:
            with gl.amd.warp_pipeline_stage("state_prefetch"):
                state_k2 = gl.amd.cdna4.buffer_load(sb, so + 64, cache=".cg")
                state_k0 = gl.amd.cdna4.buffer_load(sb, so, cache=".cg")
                state_k1 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=".cg")
                state_k3 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=".cg")
        _m32_publish_decay(X, FW, A, DT, shared, m, h, LOWER, SW, NW, M <= 32)
        z, history = _m32_convolve(X, CW, CS, m, h, storage_slot, SC0, SC1, SC2, NW)
        partial_norm = gl.sum(gl.reshape(z * z, (4, 2, 64)), 2)
        shared.slice(7, 1).slice(0, 8, dim=1).store(gl.reshape(partial_norm, (1, 8)))
        shared.slice(0, 4).store(z)
        key = gl.sum(shared.slice(1, 1).load(st), 0)
        value = gl.sum(shared.slice(2, 1).load(st_t), 0)
        value = gl.convert_layout(value, gl.SliceLayout(1, st))
        decay = gl.sum(shared.slice(4, 1).load(st), 0)
        nl: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [NW, 1], [0, 1])
        qp = shared.slice(7, 1).slice(0, 2, dim=1).load(nl)
        kp = shared.slice(7, 1).slice(2, 2, dim=1).load(nl)
        qnorm = gl.rsqrt(gl.sum(gl.sum(qp, 1), 0) + 1e-6)
        knorm = gl.rsqrt(gl.sum(gl.sum(kp, 1), 0) + 1e-6)
        keys = _m1_split_vector(key, st)
        decays = _m1_split_vector(decay, st)
        decayed = ()
        for p in gl.static_range(4):
            if p == 2:
                packet = state_k2
            elif p == 1:
                packet = state_k1
            elif p == 0:
                packet = state_k0
            else:
                packet = state_k3
            decayed += (packet * decays[p][None, :],)
        prediction = _m1_contract(decayed, keys, (0, 2, 1, 3))
        q = gl.sum(shared.slice(0, 1).load(st), 0)
        queries = _m1_split_vector(q, st)
        beta = _m1_sigmoid(gl.load(X + m * 6336 + 6272 + h).to(gl.float32))
        delta = gl.fma(-prediction, knorm, value) * (beta * knorm)
        update_order: gl.constexpr = (3, 2, 1, 0)
        updates = ()
        for p in gl.static_range(4):
            updated = gl.fma(delta[:, None], keys[update_order[p]][None, :], decayed[update_order[p]])
            if p == 0:
                partial = updated * queries[update_order[p]][None, :]
            else:
                partial = gl.fma(updated, queries[update_order[p]][None, :], partial)
            updates += (updated,)
        out = gl.sum(partial, 1) * qnorm * 128 ** (-0.5)
        retire: gl.constexpr = (3, 1, 2, 0)
        for p in gl.static_range(4):
            if M > 32:
                gl.amd.cdna4.buffer_store(updates[3 - retire[p]], sb, so + retire[p] * 32, cache=".cs")
            else:
                gl.amd.cdna4.buffer_store(updates[p], sb, so + update_order[p] * 32, cache=".cs")
        out = out.to(gl.bfloat16).to(gl.float32)
        out = gl.convert_layout(out, gl.SliceLayout(0, st_t))
        shared.slice(6, 1).store(out[None, :])
        out = gl.sum(shared.slice(6, 1).load(output_layout), 0)
        out = gl.convert_layout(out, vector_layout)
        gate = gl.sum(shared.slice(3, 1).load(output_layout), 0)
        gate = gl.convert_layout(gate, vector_layout)
        nw = gl.load(NORM + i).to(gl.float32)
        scale = gl.rsqrt(gl.sum(out * out, 0) / 128 + EPS)
        out = out * scale * (nw * gate)
        old1, old2, x, hb, ho, valid = history
        gl.amd.cdna4.buffer_store(old1, hb, ho, valid)
        gl.amd.cdna4.buffer_store(old2, hb, SC1 + ho, valid)
        gl.amd.cdna4.buffer_store(x, hb, 2 * SC1 + ho, valid)
        gl.store(O + m * CORE_STRIDE + h * 128 + i, out)


@gluon.jit
def _m32_project_stream(
    X, weight, ao, bo, amask, acc, sa, sb,
    K: gl.constexpr, BK: gl.constexpr, MMA: gl.constexpr,
    AD: gl.constexpr, BD: gl.constexpr, FRAG: gl.constexpr,
    COUPLED: gl.constexpr, FIXED_WEIGHT_BASE: gl.constexpr = False,
):

    aa = ()
    bb = ()
    for block in gl.static_range(AD):
        aa += (gl.amd.cdna4.buffer_load(X + block * BK, ao, amask, 0),)
    for block in gl.static_range(BD):
        if FIXED_WEIGHT_BASE:
            bb += (gl.amd.cdna4.buffer_load(weight, bo + block * BK, cache=".cg"),)
        else:
            bb += (gl.amd.cdna4.buffer_load(weight + block * BK, bo, cache=".cg"),)
    for block in gl.static_range(K // BK):
        a = aa[0]
        b = bb[0]
        if not COUPLED:
            if block + BD < K // BK:
                if FIXED_WEIGHT_BASE:
                    bb = bb[1:] + (gl.amd.cdna4.buffer_load(weight, bo + (block + BD) * BK, cache=".cg"),)
                else:
                    bb = bb[1:] + (gl.amd.cdna4.buffer_load(weight + (block + BD) * BK, bo, cache=".cg"),)
            else:
                bb = bb[1:]
        sa.store(a)
        sb.store(b)
        if COUPLED:
            if block + BD < K // BK:
                if FIXED_WEIGHT_BASE:
                    bb = bb[1:] + (gl.amd.cdna4.buffer_load(weight, bo + (block + BD) * BK, cache=".cg"),)
                else:
                    bb = bb[1:] + (gl.amd.cdna4.buffer_load(weight + (block + BD) * BK, bo, cache=".cg"),)
            else:
                bb = bb[1:]
        else:
            if block + AD < K // BK:
                aa = aa[1:] + (gl.amd.cdna4.buffer_load(X + (block + AD) * BK, ao, amask, 0),)
            else:
                aa = aa[1:]
        for fragment in gl.static_range(BK // FRAG):
            a = sa.slice(fragment * FRAG, FRAG, dim=1).load(gl.DotOperandLayout(0, MMA, 8))
            b = sb.slice(fragment * FRAG, FRAG, dim=0).load(gl.DotOperandLayout(1, MMA, 8))
            if COUPLED and fragment == 0:
                if block + AD < K // BK:
                    aa = aa[1:] + (gl.amd.cdna4.buffer_load(X + (block + AD) * BK, ao, amask, 0),)
                else:
                    aa = aa[1:]
            acc = gl.amd.cdna4.mfma(a, b, acc)
    return acc


@gluon.jit
def _m32_projection(
    X, WQ, WB, Y, M: gl.constexpr, K: gl.constexpr,
    XM: gl.constexpr, WN: gl.constexpr, WBN: gl.constexpr, YS: gl.constexpr,
    INPUT: gl.constexpr, BM: gl.constexpr, NW: gl.constexpr, BK: gl.constexpr,
):
    BN: gl.constexpr = 32
    MT: gl.constexpr = triton.cdiv(M, BM)
    pid = gl.program_id(0)
    if M <= 32:
        pid = pid.to(gl.uint32)
    tile_n = pid // MT
    tile_m = pid % MT
    if M == 64:
        tile_n = (tile_n * (149 if INPUT else 11)) % (197 if INPUT else 224)
    if INPUT:
        auxiliary = tile_n >= 6144 // BN
        weight = gl.where(auxiliary, WB, WQ)
        stride = gl.where(auxiliary, WBN, WN)
        first_n = gl.where(auxiliary, tile_n * BN - 6144, tile_n * BN)
    else:
        weight = WQ
        stride = WN
        first_n = tile_n * BN
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[BM // 16, NW // (BM // 16)],
    )
    la: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BK, BK // 8], [NW, 1], [1, 0])
    lb: gl.constexpr = gl.BlockedLayout([8, 1], [BK // 8, 512 // BK], [1, NW], [0, 1])
    am = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, la))
    ak = gl.arange(0, BK, gl.SliceLayout(0, la))
    bk = gl.arange(0, BK, gl.SliceLayout(1, lb))
    bn = first_n + gl.arange(0, BN, gl.SliceLayout(0, lb))
    if INPUT:
        bn = gl.where(auxiliary & (bn >= 144), 0, bn)
    ao = am[:, None] * XM + ak[None, :]
    bo = bn[None, :] * stride + bk[:, None]
    amask = am[:, None] < M
    acc = gl.zeros((BM, BN), gl.float32, mma)
    if INPUT and BM == 32:
        shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
        shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
    else:
        shared_a: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
            gl.DotOperandLayout(0, mma, 8), (BM, BK), gl.bfloat16)
        shared_b: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
            gl.DotOperandLayout(1, mma, 8), (BK, BN), gl.bfloat16)
    sa = gl.allocate_shared_memory(gl.bfloat16, (BM, BK), shared_a)
    sb = gl.allocate_shared_memory(gl.bfloat16, (BK, BN), shared_b)
    if not INPUT and BM == 64:
        acc = _m32_project_stream(X, weight, ao, bo, amask, acc, sa, sb, K, BK, mma,
                              1, 1, 256, False)
    else:
        depth_a: gl.constexpr = (6 if INPUT else 3) if BM == 32 else 2
        depth_b: gl.constexpr = 3 if BM == 32 and not INPUT else 1
        acc = _m32_project_stream(X, weight, ao, bo, amask, acc, sa, sb, K, BK, mma,
                              depth_a, depth_b,
                              512 if BM == 32 and INPUT else 128,
                              BM == 64 and INPUT, BM == 32 and not INPUT)
    om = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = tile_n * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(Y + om[:, None] * YS + on[None, :], acc.to(gl.bfloat16),
             (om[:, None] < M) & (on[None, :] < YS))


def kda_layer_decode_m32(
    x, qkvg_weight, beta_forget_weight, output_weight, forget_weight,
    conv_weight, a_log, dt_bias, norm_weight, conv_state, state, state_indices,
    *, lower_bound=-5.0, norm_eps=1e-5, output_tensor=None,
):

    m = x.shape[0]
    packed = torch.empty((m, 6336), device=x.device, dtype=torch.bfloat16)
    core = packed[:, :1536]
    out = (torch.empty((m, 7168), device=x.device, dtype=torch.bfloat16)
           if output_tensor is None else output_tensor)
    assert (out.shape == (m, 7168) and out.dtype == torch.bfloat16
            and out.device == x.device and out.is_contiguous())
    small_batch = m <= 32
    rows = 32
    waves = 4
    input_panel = 512
    output_panel = 128
    _m32_projection[(197 * triton.cdiv(m, rows),)](
        x, qkvg_weight, beta_forget_weight, packed, m, 7168,
        x.stride(0), qkvg_weight.stride(0), beta_forget_weight.stride(0), 6336,
        True, BM=rows, NW=waves, BK=input_panel, num_warps=waves,
    )
    _m32_recurrent_head[(m * 12,)](
        packed, conv_weight, forget_weight, conv_state, state, state_indices,
        a_log, dt_bias, norm_weight, core, forget_weight.stride(0), state_indices.stride(0),
        *conv_state.stride(), state.stride(0), lower_bound, norm_eps,
        NW=8, M=m, GROUP=2, CORE_STRIDE=core.stride(0),
        NARROW_SLOT=(
            m > 32
            and state.shape[0] * state.stride(0) < 2**31
            and conv_state.shape[0] * conv_state.stride(0) < 2**31
        ),
        num_warps=8, enable_fp_fusion=False, waves_per_eu=5,
    )
    _m32_projection[(224 * triton.cdiv(m, rows),)](
        core, output_weight, output_weight, out, m, 1536,
        core.stride(0), output_weight.stride(0), output_weight.stride(0), 7168,
        False, BM=rows, NW=waves, BK=output_panel, num_warps=waves,
    )
    return out, conv_state, state


@gluon.jit
def _m4_input_tile_m4(X, weight, Y, tile, XM: gl.constexpr, WS: gl.constexpr,
                   TEMPORAL: gl.constexpr):

    M: gl.constexpr = 4
    CN: gl.constexpr = 16
    BK: gl.constexpr = 128
    K: gl.constexpr = 7168
    WEIGHT_WINDOW: gl.constexpr = 5
    ACTIVATION_WINDOW: gl.constexpr = 20
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[1, 1],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    load_a: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    load_b: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    rows = gl.arange(0, M, gl.SliceLayout(1, load_a))
    ak = gl.arange(0, BK, gl.SliceLayout(0, load_a))
    cols = gl.arange(0, CN, gl.SliceLayout(0, load_b))
    bk = gl.arange(0, BK, gl.SliceLayout(1, load_b))
    ao = rows[:, None] * XM + ak[None, :]
    bo = cols[None, :] * WS + bk[:, None]
    acc = gl.zeros((M, CN), gl.float32, mma)
    inputs = ()
    weights = ()
    for block in gl.static_range(WEIGHT_WINDOW):
        for a in gl.static_range(4):
            inputs += (gl.amd.cdna4.buffer_load(X + (4 * block + a) * BK, ao),)
        weights += (gl.amd.cdna4.buffer_load(
            weight + block * BK, bo,
            cache="" if TEMPORAL or block < 2 else ".cg",
        ),)
    for block in gl.static_range(K // BK):
        a, b = inputs[0], weights[0]
        inputs, weights = inputs[1:], weights[1:]
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a, dot_a), gl.convert_layout(b, dot_b), acc,
        )
        if block + ACTIVATION_WINDOW < K // BK:
            inputs += (gl.amd.cdna4.buffer_load(
                X + (block + ACTIVATION_WINDOW) * BK, ao,
            ),)
        if block + WEIGHT_WINDOW < K // BK:
            weights += (gl.amd.cdna4.buffer_load(
                weight + (block + WEIGHT_WINDOW) * BK, bo, cache="" if TEMPORAL else ".cg",
            ),)
    out_rows = gl.arange(0, M, gl.SliceLayout(1, mma))
    out_cols = tile * CN + gl.arange(0, CN, gl.SliceLayout(0, mma))
    gl.store(Y + out_rows[:, None] * 6336 + out_cols[None, :], acc.to(gl.bfloat16))


@gluon.jit
def _m4_input_projections(
    X, WQ, WB, Y, M: gl.constexpr,
    XM: gl.constexpr, WN: gl.constexpr, WBN: gl.constexpr,
):

    CN: gl.constexpr = 16
    BK: gl.constexpr = 128
    WEIGHT_WINDOW: gl.constexpr = 5
    ACTIVATIONS_PER_WEIGHT: gl.constexpr = 6 if M == 2 else 4
    ACTIVATION_WINDOW: gl.constexpr = WEIGHT_WINDOW * ACTIVATIONS_PER_WEIGHT
    K: gl.constexpr = 7168
    TILES: gl.constexpr = 393
    pid = gl.program_id(0)


    if M == 1:
        tile = (pid * 49) % TILES
    elif M == 4:
        tile = (pid * 49 + 16) % TILES
    else:
        band = pid % 8
        tile = band * (TILES // 8) + gl.minimum(band, TILES % 8) + pid // 8
    is_q = tile < 6144 // CN
    if M == 4:
        if is_q:
            _m4_input_tile_m4(X, WQ + tile * CN * WN, Y, tile, XM, WN, False)
        else:
            _m4_input_tile_m4(X, WB + (tile - 384) * CN * WBN, Y, tile, XM, WBN, True)
    else:
        weight = gl.where(is_q, WQ, WB)
        weight_stride = gl.where(is_q, WN, WBN)
        weight_tile = gl.where(is_q, tile, tile - 6144 // CN)
        weight += weight_tile * CN * weight_stride
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 32], transposed=True,
            warps_per_cta=[1, 1],
        )
        dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
        dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
        if M == 1:
            load_a: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [1, 1], [1, 0])
        else:
            load_a: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
        load_b: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
        rows = gl.arange(0, M, gl.SliceLayout(1, load_a))
        ak = gl.arange(0, BK, gl.SliceLayout(0, load_a))
        cols = gl.arange(0, CN, gl.SliceLayout(0, load_b))
        bk = gl.arange(0, BK, gl.SliceLayout(1, load_b))
        ao = rows[:, None] * XM + ak[None, :]
        bo = cols[None, :] * weight_stride + bk[:, None]
        acc = gl.zeros((M, CN), gl.float32, mma)
        inputs = ()
        weights = ()
        for block in gl.static_range(WEIGHT_WINDOW):


            if M == 2:
                weights += (gl.amd.cdna4.buffer_load(weight + block * BK, bo, cache=".cg"),)
            for a in gl.static_range(ACTIVATIONS_PER_WEIGHT):
                inputs += (gl.amd.cdna4.buffer_load(
                    X + (ACTIVATIONS_PER_WEIGHT * block + a) * BK, ao,
                ),)
            if M != 2:
                weights += (gl.amd.cdna4.buffer_load(weight + block * BK, bo, cache=".cg"),)
        for block in gl.static_range(K // BK):
            a, b = inputs[0], weights[0]
            inputs, weights = inputs[1:], weights[1:]
            acc = gl.amd.cdna4.mfma(
                gl.convert_layout(a, dot_a), gl.convert_layout(b, dot_b), acc,
            )
            if block + ACTIVATION_WINDOW < K // BK:
                inputs += (gl.amd.cdna4.buffer_load(
                    X + (block + ACTIVATION_WINDOW) * BK, ao,
                ),)
            if block + WEIGHT_WINDOW < K // BK:
                weights += (gl.amd.cdna4.buffer_load(
                    weight + (block + WEIGHT_WINDOW) * BK, bo, cache=".cg",
                ),)
        out_rows = gl.arange(0, M, gl.SliceLayout(1, mma))
        out_cols = tile * CN + gl.arange(0, CN, gl.SliceLayout(0, mma))
        gl.store(Y + out_rows[:, None] * 6336 + out_cols[None, :], acc.to(gl.bfloat16))


@gluon.jit
def _m4_recurrent_head(X, CW, FW, CS, S, IDX, A, DT, NW, O,
                    M: gl.constexpr, SW: gl.constexpr, SI: gl.constexpr,
                    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                    SS: gl.constexpr, LOWER, EPS):

    ST: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [8, 1], [1, 0])
    ST_T: gl.constexpr = gl.BlockedLayout([4, 1], [8, 8], [1, 8], [0, 1])
    LC: gl.constexpr = '.cg' if M <= 2 else ''
    SC: gl.constexpr = '.cs' if M <= 2 else '.wt'
    pid = gl.program_id(0)
    if M == 2:
        m = pid % M
        h = pid // M
    else:
        m = pid // 12
        h = pid % 12
    mh = m * 12 + h
    i = gl.arange(0, 128, _VECTOR)
    slot = gl.load(IDX + m * SI).to(gl.int64)
    if slot < 0:
        gl.store(O + (m * 6336 + h * 128 if M == 2 else mh * 128) + i, 0)
    else:
        nw = gl.load(NW + i).to(gl.float32)
        beta = _m1_sigmoid(gl.load(X + m * 6336 + 6272 + h).to(gl.float32))
        if M == 4:

            r = gl.arange(0, 128, gl.SliceLayout(1, ST))
            c = gl.arange(0, 32, gl.SliceLayout(0, ST))
            sb = S + slot * SS + h * 16384
            so = r[:, None] * 128 + c[None, :]
            rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
            z, history = _m1_convolve(X, CW, CS, m, h, slot, SC0, SC1, SC2)
            packet3 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
            packet2 = gl.amd.cdna4.buffer_load(sb, so + 64, cache=LC)
            packet1 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=LC)
            packet0 = gl.amd.cdna4.buffer_load(sb, so, cache=LC)
        else:
            if M == 1:
                r = gl.arange(0, 128, gl.SliceLayout(1, ST))
                c = gl.arange(0, 32, gl.SliceLayout(0, ST))
                sb = S + slot * SS + h * 16384
                so = r[:, None] * 128 + c[None, :]
                prefix = gl.amd.cdna4.buffer_load(sb, so, cache=LC)
            z, history = _m1_convolve(X, CW, CS, m, h, slot, SC0, SC1, SC2)
            if M == 2:
                r = gl.arange(0, 128, gl.SliceLayout(1, ST))
                c = gl.arange(0, 32, gl.SliceLayout(0, ST))
                sb = S + slot * SS + h * 16384
                so = r[:, None] * 128 + c[None, :]
                prefix = gl.amd.cdna4.buffer_load(sb, so, cache=LC)
            if M == 1:
                extra_packet = gl.amd.cdna4.buffer_load(sb, so + 64, cache=LC)
            elif M == 2:
                extra_packet = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
        f = _m1_forget(X, FW, m, h, SW)
        if M == 4:
            decay_layout: gl.constexpr = gl.DistributedLinearLayout(
                [], [[0], [0], [64], [1], [2], [4]],
                [[8], [16], [32]], [], [128],
            )
            f = gl.convert_layout(f, decay_layout)
        if M != 4:
            prefix2 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=LC)
        shared = gl.allocate_shared_memory(gl.float32, (8, 128), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        partial_norm = gl.sum(gl.reshape(z * z, (4, 2, 64)), 2)
        shared.slice(7, 1).slice(0, 8, dim=1).store(gl.reshape(partial_norm, (1, 8)))
        shared.slice(0, 4).store(z)
        if M == 1:

            packet3 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=LC)
        dt = gl.load(DT + h * 128 + gl.arange(0, 128, f.type.layout))
        if M != 4:
            rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
        decay = gl.exp2(LOWER * _m1_sigmoid(rate * (f + dt)) * 1.4426950408889634)
        if M == 4:
            shared.slice(4, 1).store(gl.reshape(decay, (1, 128)))
        else:
            shared.slice(4, 1).store(decay[None, :])
        if M == 1:
            q = gl.sum(shared.slice(0, 1).load(ST), 0)
        key = gl.sum(shared.slice(1, 1).load(ST), 0)
        value = gl.sum(shared.slice(2, 1).load(ST_T), 0)
        value = gl.convert_layout(value, gl.SliceLayout(1, ST))
        decay = gl.sum(shared.slice(4, 1).load(ST), 0)
        gate = gl.sum(shared.slice(3, 1).load(_OUTPUT), 0)
        gate = gl.convert_layout(gate, _VECTOR)
        nl: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [0, 1])
        qp = shared.slice(7, 1).slice(0, 2, dim=1).load(nl)
        kp = shared.slice(7, 1).slice(2, 2, dim=1).load(nl)
        qnorm = gl.rsqrt(gl.sum(gl.sum(qp, 1), 0) + 1e-6)
        knorm = gl.rsqrt(gl.sum(gl.sum(kp, 1), 0) + 1e-6)
        keys = _m1_split_vector(key, ST)
        if M == 1:
            queries = _m1_split_vector(q, ST)
        decays = _m1_split_vector(decay, ST)
        decayed = ()
        for p in gl.static_range(4):
            if M == 4:
                packets = (packet0, packet1, packet2, packet3)
                packet = packets[p]
            elif p == 0:
                packet = prefix
            elif p == 1:
                packet = prefix2
            elif M == 1 and p == 2:
                packet = extra_packet
            elif M == 2 and p == 3:
                packet = extra_packet
            elif M == 1:
                packet = packet3
            else:
                packet = gl.amd.cdna4.buffer_load(sb, so + p * 32, cache=LC)
            decayed += (packet * decays[p][None, :],)
        P_ORDER: gl.constexpr = (0, 1, 2, 3) if M == 2 else (0, 2, 1, 3) if M == 4 else (3, 2, 1, 0)
        prediction = _m1_contract(decayed, keys, P_ORDER)
        if M >= 2:
            q = gl.sum(shared.slice(0, 1).load(ST), 0)
            queries = _m1_split_vector(q, ST)
        if M == 4:
            delta = gl.fma(-prediction, knorm, value) * (beta * knorm)
        else:
            delta = (value - prediction * knorm) * beta
            delta = delta * knorm
        if M == 2:
            projected = _m1_contract(decayed, queries, P_ORDER)
            key_query = gl.sum(key * q, 0)
            out = gl.fma(delta, key_query, projected) * qnorm * 128 ** (-0.5)
        UPDATE_ORDER: gl.constexpr = (0, 1, 2, 3) if M == 2 else (3, 2, 1, 0)
        for p in gl.static_range(4):
            updated = gl.fma(delta[:, None], keys[UPDATE_ORDER[p]][None, :], decayed[UPDATE_ORDER[p]])
            if M != 2:
                if p == 0:
                    partial = updated * queries[UPDATE_ORDER[p]][None, :]
                else:
                    partial = gl.fma(updated, queries[UPDATE_ORDER[p]][None, :], partial)
            gl.amd.cdna4.buffer_store(updated, sb, so + UPDATE_ORDER[p] * 32, cache=SC)
        if M != 2:
            out = gl.sum(partial, 1) * qnorm * 128 ** (-0.5)
        out = out.to(gl.bfloat16).to(gl.float32)
        if M == 1:
            out = gl.convert_layout(out, _VECTOR)
        else:
            out = gl.convert_layout(out, gl.SliceLayout(0, ST_T))
            shared.slice(6, 1).store(out[None, :])
            out = gl.sum(shared.slice(6, 1).load(_OUTPUT), 0)
            out = gl.convert_layout(out, _VECTOR)
        scale = gl.rsqrt(gl.sum(out * out, 0) / 128 + EPS)
        if M == 2:
            out = out * scale * nw * gate
        else:
            out = out * scale * (nw * gate)
        old1, old2, x, hb, ho, valid = history
        gl.amd.cdna4.buffer_store(old1, hb, ho, valid)
        gl.amd.cdna4.buffer_store(old2, hb, SC1 + ho, valid)
        gl.amd.cdna4.buffer_store(x, hb, 2 * SC1 + ho, valid)
        gl.store(O + (m * 6336 + h * 128 if M == 2 else mh * 128) + i, out)


@gluon.jit
def _m4_recurrent_and_prefetch(X, CW, FW, CS, S, IDX, A, DT, NW, O, WO,
                            M: gl.constexpr, SW: gl.constexpr, SI: gl.constexpr,
                            SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                            SS: gl.constexpr, LOWER, EPS, OWN: gl.constexpr,
                            PREFETCH_CTAS: gl.constexpr, PREFETCH_EXTENT: gl.constexpr):

    pid = gl.program_id(0)
    if PREFETCH_CTAS == 0 or pid < M * 12:
        _m4_recurrent_head(X, CW, FW, CS, S, IDX, A, DT, NW, O, M, SW, SI,
                        SC0, SC1, SC2, SS, LOWER, EPS)
    else:
        _m1_prefetch_output_weights(WO, OWN, pid - M * 12, M, PREFETCH_CTAS, PREFETCH_EXTENT)


@gluon.jit
def _m4_output_folded(X, W, Y, M: gl.constexpr, N: gl.constexpr, SX: gl.constexpr, SW: gl.constexpr):
    AN: gl.constexpr = 32
    BK: gl.constexpr = 128

    ROWS: gl.constexpr = M
    ml: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, 4])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 4], [0, 1])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 4], [1, 0])
    ar = gl.arange(0, 4 * ROWS, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    br = gl.arange(0, AN * 4, gl.SliceLayout(0, bl))
    pid = gl.program_id(0).to(gl.uint32)
    tile = pid % 8 * (N // AN // 8) + pid // 8
    ao = ar[:, None] // 4 % M * SX + ar[:, None] % 4 * 8 + ak[None, :] // 8 * 32 + ak[None, :] % 8
    bo = (tile * AN + br[None, :] // 4) * SW + br[None, :] % 4 * 8 + bk[:, None] // 8 * 32 + bk[:, None] % 8
    acc = gl.zeros((4 * ROWS, AN * 4), gl.float32, ml)
    if M == 2:
        a0 = gl.load(X + ao)
        a1 = gl.load(X + 512 + ao)
        a2 = gl.load(X + 1024 + ao)
        b0 = gl.load(W + bo)
        b1 = gl.load(W + 512 + bo)
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a0, gl.DotOperandLayout(0, ml, 8)),
            gl.convert_layout(b0, gl.DotOperandLayout(1, ml, 8)), acc,
        )
        b2 = gl.load(W + 1024 + bo)
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a1, gl.DotOperandLayout(0, ml, 8)),
            gl.convert_layout(b1, gl.DotOperandLayout(1, ml, 8)), acc,
        )
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a2, gl.DotOperandLayout(0, ml, 8)),
            gl.convert_layout(b2, gl.DotOperandLayout(1, ml, 8)), acc,
        )
    else:
        for step in loop_range(3, loop_unroll_factor=1):
            a = gl.load(X + step * 512 + ao)
            b = gl.amd.cdna4.buffer_load(W + step * 512, bo, cache=".cg")
            acc = gl.amd.cdna4.mfma(
                gl.convert_layout(a, gl.DotOperandLayout(0, ml, 8)),
                gl.convert_layout(b, gl.DotOperandLayout(1, ml, 8)), acc,
            )
    value = gl.reshape(gl.permute(acc, (1, 0)), (AN * 4, ROWS, 2, 2))
    even, odd = gl.split(value)
    v0, v2 = gl.split(even)
    v1, v3 = gl.split(odd)
    coord = gl.arange(0, AN * 4, gl.SliceLayout(1, v0.type.layout))
    low = gl.where(coord[:, None] & 1 == 0, v0, v1)
    high = gl.where(coord[:, None] & 1 == 0, v2, v3)
    diagonal = gl.where(coord[:, None] & 2 == 0, low, high)
    result = gl.permute(gl.sum(gl.reshape(diagonal, (AN, 4, ROWS)), 1), (1, 0))
    ol: gl.constexpr = result.type.layout
    om = gl.arange(0, ROWS, gl.SliceLayout(1, ol))
    on = tile * AN + gl.arange(0, AN, gl.SliceLayout(0, ol))
    gl.store(Y + om[:, None] * N + on[None, :], result, om[:, None] < M)


def kda_layer_decode_m4(
    x, qkvg_weight, beta_forget_weight, output_weight, forget_weight,
    conv_weight, a_log, dt_bias, norm_weight, conv_state, state, state_indices,
    *, lower_bound=-5.0, norm_eps=1e-5, output_tensor=None,
):

    m = x.shape[0]
    assert m in (1, 2, 4)
    workspace = torch.empty((m * (6336 + 1536),), dtype=torch.bfloat16, device=x.device)
    packed = workspace.as_strided((m, 6336), (6336, 1))
    core = workspace.as_strided((m, 1536), (1536, 1), storage_offset=m * 6336)
    out = (
        torch.empty((m, 7168), dtype=torch.bfloat16, device=x.device)
        if output_tensor is None else output_tensor
    )
    assert out.shape == (m, 7168) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _m4_input_projections[393,](x, qkvg_weight, beta_forget_weight, packed, m,
        x.stride(0), qkvg_weight.stride(0), beta_forget_weight.stride(0),
        num_warps=1)
    prefetch_ctas = 0
    prefetch_extent = 1
    _m4_recurrent_and_prefetch[m * 12 + prefetch_ctas,](packed, conv_weight, forget_weight, conv_state, state, state_indices,
        a_log, dt_bias, norm_weight, core, output_weight, m, forget_weight.stride(0), state_indices.stride(0),
        *conv_state.stride(), state.stride(0), lower_bound, norm_eps, output_weight.stride(0), prefetch_ctas, prefetch_extent,
        num_warps=8, enable_fp_fusion=False, waves_per_eu=2)
    _m4_output_folded[224,](core, output_weight, out, m, 7168, 1536, output_weight.stride(0), num_warps=4)
    return out, conv_state, state


@gluon.jit
def _m64_recurrent_head(
    X, CW, FW, CS, S, IDX, A, DT, NORM, O,
    SW: gl.constexpr, SI: gl.constexpr,
    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
    SS: gl.constexpr, LOWER, EPS,
    NW: gl.constexpr = 8,
    M: gl.constexpr = 32, GROUP: gl.constexpr = 1,
    CORE_STRIDE: gl.constexpr = 1536,
    NARROW_SLOT: gl.constexpr = False,
):

    st: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [NW, 1], [1, 0])
    st_t: gl.constexpr = gl.BlockedLayout([4, 1], [8, 8], [1, NW], [0, 1])
    output_layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [4, NW // 4], [1, 0])
    vector_layout: gl.constexpr = gl.BlockedLayout([2], [64], [NW], [0])
    pid = gl.program_id(0)
    group = pid // (12 * GROUP)
    remainder = pid % (12 * GROUP)
    if M % GROUP == 0:
        group_size = GROUP
    else:
        group_size = gl.minimum(GROUP, M - group * GROUP)
    m = group * GROUP + remainder % group_size
    h = remainder // group_size
    i = gl.arange(0, 128, vector_layout)
    slot = gl.load(IDX + m * SI).to(gl.int64)
    if slot < 0:
        gl.store(O + m * CORE_STRIDE + h * 128 + i, 0)
    else:


        if NARROW_SLOT:
            storage_slot = slot.to(gl.int32)
        else:
            storage_slot = slot
        r = gl.arange(0, 128, gl.SliceLayout(1, st))
        c = gl.arange(0, 32, gl.SliceLayout(0, st))
        sb = S + storage_slot * SS + h * 16384
        so = r[:, None] * 128 + c[None, :]
        shared = gl.allocate_shared_memory(
            gl.float32, (8, 128), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        if M <= 32:
            with gl.amd.warp_pipeline_stage("state_prefetch"):
                state_k3 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=".cg")
                state_k2 = gl.amd.cdna4.buffer_load(sb, so + 64, cache=".cg")
                state_k1 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=".cg")
                state_k0 = gl.amd.cdna4.buffer_load(sb, so, cache=".cg")
        else:
            with gl.amd.warp_pipeline_stage("state_prefetch"):
                state_k2 = gl.amd.cdna4.buffer_load(sb, so + 64, cache=".cg")
                state_k0 = gl.amd.cdna4.buffer_load(sb, so, cache=".cg")
                state_k1 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=".cg")
                state_k3 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=".cg")
        _m32_publish_decay(X, FW, A, DT, shared, m, h, LOWER, SW, NW, M <= 32)
        z, history = _m32_convolve(X, CW, CS, m, h, storage_slot, SC0, SC1, SC2, NW)
        partial_norm = gl.sum(gl.reshape(z * z, (4, 2, 64)), 2)
        shared.slice(7, 1).slice(0, 8, dim=1).store(gl.reshape(partial_norm, (1, 8)))
        shared.slice(0, 4).store(z)
        key = gl.sum(shared.slice(1, 1).load(st), 0)
        value = gl.sum(shared.slice(2, 1).load(st_t), 0)
        value = gl.convert_layout(value, gl.SliceLayout(1, st))
        decay = gl.sum(shared.slice(4, 1).load(st), 0)
        nl: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [NW, 1], [0, 1])
        qp = shared.slice(7, 1).slice(0, 2, dim=1).load(nl)
        kp = shared.slice(7, 1).slice(2, 2, dim=1).load(nl)
        qnorm = gl.rsqrt(gl.sum(gl.sum(qp, 1), 0) + 1e-6)
        knorm = gl.rsqrt(gl.sum(gl.sum(kp, 1), 0) + 1e-6)
        keys = _m1_split_vector(key, st)
        decays = _m1_split_vector(decay, st)
        decayed = ()
        for p in gl.static_range(4):
            if p == 2:
                packet = state_k2
            elif p == 1:
                packet = state_k1
            elif p == 0:
                packet = state_k0
            else:
                packet = state_k3
            decayed += (packet * decays[p][None, :],)
        prediction = _m1_contract(decayed, keys, (0, 2, 1, 3))
        q = gl.sum(shared.slice(0, 1).load(st), 0)
        queries = _m1_split_vector(q, st)
        beta = _m1_sigmoid(gl.load(X + m * 6336 + 6272 + h).to(gl.float32))
        delta = gl.fma(-prediction, knorm, value) * (beta * knorm)
        update_order: gl.constexpr = (3, 2, 1, 0)
        updates = ()
        for p in gl.static_range(4):
            updated = gl.fma(delta[:, None], keys[update_order[p]][None, :], decayed[update_order[p]])
            if p == 0:
                partial = updated * queries[update_order[p]][None, :]
            else:
                partial = gl.fma(updated, queries[update_order[p]][None, :], partial)
            updates += (updated,)
        out = gl.sum(partial, 1) * qnorm * 128 ** (-0.5)
        retire: gl.constexpr = (3, 1, 2, 0)
        for p in gl.static_range(4):
            if M > 32:
                gl.amd.cdna4.buffer_store(updates[3 - retire[p]], sb, so + retire[p] * 32, cache=".cs")
            else:
                gl.amd.cdna4.buffer_store(updates[p], sb, so + update_order[p] * 32, cache=".cs")
        out = out.to(gl.bfloat16).to(gl.float32)
        out = gl.convert_layout(out, gl.SliceLayout(0, st_t))
        shared.slice(6, 1).store(out[None, :])
        out = gl.sum(shared.slice(6, 1).load(output_layout), 0)
        out = gl.convert_layout(out, vector_layout)
        gate = gl.sum(shared.slice(3, 1).load(output_layout), 0)
        gate = gl.convert_layout(gate, vector_layout)
        nw = gl.load(NORM + i).to(gl.float32)
        scale = gl.rsqrt(gl.sum(out * out, 0) / 128 + EPS)
        out = out * scale * (nw * gate)
        old1, old2, x, hb, ho, valid = history
        gl.amd.cdna4.buffer_store(old1, hb, ho, valid)
        gl.amd.cdna4.buffer_store(old2, hb, SC1 + ho, valid)
        gl.amd.cdna4.buffer_store(x, hb, 2 * SC1 + ho, valid)
        gl.store(O + m * CORE_STRIDE + h * 128 + i, out)


def kda_layer_decode_m64(
    x, qkvg_weight, beta_forget_weight, output_weight, forget_weight,
    conv_weight, a_log, dt_bias, norm_weight, conv_state, state, state_indices,
    *, lower_bound=-5.0, norm_eps=1e-5, output_tensor=None,
):

    m = x.shape[0]
    packed = torch.empty((m, 6336), device=x.device, dtype=torch.bfloat16)


    core = torch.empty((m, 1536), device=x.device, dtype=torch.bfloat16)
    out = (torch.empty((m, 7168), device=x.device, dtype=torch.bfloat16)
           if output_tensor is None else output_tensor)
    assert (out.shape == (m, 7168) and out.dtype == torch.bfloat16
            and out.device == x.device and out.is_contiguous())
    small_batch = m <= 32
    rows = 64
    waves = 8
    input_panel = 256
    output_panel = 256
    _m32_projection[(197 * triton.cdiv(m, rows),)](
        x, qkvg_weight, beta_forget_weight, packed, m, 7168,
        x.stride(0), qkvg_weight.stride(0), beta_forget_weight.stride(0), 6336,
        True, BM=rows, NW=waves, BK=input_panel, num_warps=waves,
    )
    _m64_recurrent_head[(m * 12,)](
        packed, conv_weight, forget_weight, conv_state, state, state_indices,
        a_log, dt_bias, norm_weight, core, forget_weight.stride(0), state_indices.stride(0),
        *conv_state.stride(), state.stride(0), lower_bound, norm_eps,
        NW=8, M=m, GROUP=2, CORE_STRIDE=core.stride(0),
        NARROW_SLOT=(
            m > 32
            and state.shape[0] * state.stride(0) < 2**31
            and conv_state.shape[0] * conv_state.stride(0) < 2**31
        ),
        num_warps=8, enable_fp_fusion=False, waves_per_eu=1,
    )
    _m32_projection[(224 * triton.cdiv(m, rows),)](
        core, output_weight, output_weight, out, m, 1536,
        core.stride(0), output_weight.stride(0), output_weight.stride(0), 7168,
        False, BM=rows, NW=waves, BK=output_panel, num_warps=waves,
    )
    return out, conv_state, state
