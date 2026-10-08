"""Generated Kimi-K3 KDA prefill kernel for gfx950.

Source: OpenAI-Partners/artemis-kernel-integrations PR 17,
commit 35b249f7a551278946a81b7da1d58c286c41fb8f.
"""

# ruff: noqa
# fmt: off

from typing import NamedTuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _sigmoid(x):
    return 1.0 / (1.0 + gl.exp(-x))


@gluon.jit
def _load_prepared(Prepared, t, h, index, H: gl.constexpr, PART: gl.constexpr):
    base = Prepared + (t * 3 * H + h) * 128
    return gl.amd.cdna4.buffer_load(base, PART * H * 128 + index)


@gluon.jit
def _commit_history(X, CS, INITIAL, seq, start, end, slot, h, part,
                    H: gl.constexpr, SX: gl.constexpr,
                    SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr):
    c = (part * H + h) * 128 + gl.arange(0, 128, gl.BlockedLayout([1], [64], [1], [0]))
    initial = gl.load(INITIAL + seq)
    slot = slot.to(gl.int64)
    h0 = gl.load(CS + slot * SC0 + (end - start) * SC1 + c * SC2,
                 (end - start < 3) & initial, 0)
    h1 = gl.load(CS + slot * SC0 + (end - start + 1) * SC1 + c * SC2,
                 (end - start < 2) & initial, 0)
    x0 = gl.load(X + (end - 3) * SX + c, end - 3 >= start, 0)
    x1 = gl.load(X + (end - 2) * SX + c, end - 2 >= start, 0)
    x2 = gl.load(X + (end - 1) * SX + c)
    gl.store(CS + slot * SC0 + c * SC2, gl.where(end - 3 >= start, x0, h0))
    gl.store(CS + slot * SC0 + SC1 + c * SC2, gl.where(end - 2 >= start, x1, h1))
    gl.store(CS + slot * SC0 + 2 * SC1 + c * SC2, x2)


@gluon.jit
def _chunk_bounds(CU, seq, CHUNK: gl.constexpr):
    chunk = gl.program_id(2)
    first = gl.load(CU + seq)
    end = gl.load(CU + seq + 1)
    chunk_id = first // CHUNK + seq + chunk
    start = first + chunk * CHUNK
    return first, start, end, chunk_id, chunk


@gluon.jit
def _store_transition(state, Base, row_block, ROWS: gl.constexpr):
    packed = gl.reshape(gl.permute(gl.reshape(state, (ROWS // 16, 4, 4, 128)), (0, 3, 2, 1)), (ROWS * 128,))
    flat: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    packed = gl.convert_layout(packed, flat)
    offset = (row_block % (128 // ROWS)) * ROWS * 128 + gl.arange(0, ROWS * 128, flat)
    gl.amd.cdna4.buffer_store(packed, Base, offset)


@gluon.jit
def _load_transition(Base, Flags, MFMA: gl.constexpr, SPARSE: gl.constexpr):
    flat: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1], [2], [2048], [4096], [8192]],
        lane_bases=[[16], [32], [64], [128], [4], [8]],
        warp_bases=[[256], [512], [1024]], block_bases=[], shape=[16384])
    offset = gl.arange(0, 16384, flat)
    if SPARSE:
        present = gl.load(Flags + offset // 4096) == 0
        packed = gl.amd.cdna4.buffer_load(Base, offset, present, 0)
    else:
        packed = gl.amd.cdna4.buffer_load(Base, offset)
    matrix = gl.reshape(gl.permute(gl.reshape(packed, (8, 128, 4, 4)), (0, 3, 2, 1)), (128, 128))
    return gl.convert_layout(matrix, gl.DotOperandLayout(1, MFMA, 1), assert_trivial=True)


@gluon.jit
def _project_decay(FA, W, A, DT, Decay, lower, M: gl.constexpr, H: gl.constexpr,
                   SFA: gl.constexpr, SW: gl.constexpr, BM: gl.constexpr, tile_m, tile_n):
    BN: gl.constexpr = 64
    la: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    lb: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
    lm: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    m = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, la))
    k = gl.arange(0, 128, gl.SliceLayout(0, la))
    a = gl.load(FA + m[:, None] * SFA + k[None, :], (M % BM == 0) | (m[:, None] < M), 0)
    k2 = gl.arange(0, 128, gl.SliceLayout(1, lb))
    n = tile_n * BN + gl.arange(0, BN, gl.SliceLayout(0, lb))
    b = gl.load(W + n[None, :] * SW + k2[:, None])
    a = gl.convert_layout(a, gl.DotOperandLayout(0, lm, 8))
    b = gl.convert_layout(b, gl.DotOperandLayout(1, lm, 8))
    acc = gl.amd.cdna4.mfma(a, b, gl.zeros((BM, BN), gl.float32, lm))
    mi = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, lm))
    ni = tile_n * BN + gl.arange(0, BN, gl.SliceLayout(0, lm))
    dt = gl.load(DT + ni)
    if BM == 64 and M <= 4096:
        rate = gl.exp(gl.load(A + tile_n // 2))
        raw = acc.to(gl.bfloat16).to(gl.float32) + dt[None, :]
        decay = gl.exp(lower * _sigmoid(rate * raw))
    else:
        rate = gl.exp(gl.load(A + ni // 128))
        raw = acc.to(gl.bfloat16).to(gl.float32) + dt[None, :]
        decay = gl.exp(lower * _sigmoid(rate[None, :] * raw))
    gl.store(Decay + mi[:, None] * (H * 128) + ni[None, :], decay,
             (M % BM == 0) | (mi[:, None] < M))


@gluon.jit
def _conv_prepare(X, CW, CS, IDX, CU, INITIAL, Beta, Prepared, Betas,
                  H: gl.constexpr, NS: gl.constexpr, SX: gl.constexpr, SB: gl.constexpr, SI: gl.constexpr,
                  SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr, M: gl.constexpr, BT: gl.constexpr, tile_t, part_head):
    layout: gl.constexpr = gl.BlockedLayout([BT // 4, 2 if BT == 32 else 1], [1, 64], [4, 1], [1, 0])
    tile_start = tile_t * BT
    t = tile_start + gl.arange(0, BT, gl.SliceLayout(1, layout))
    part = part_head // H
    h = part_head % H
    c = part_head * 128 + gl.arange(0, 128, gl.SliceLayout(0, layout))
    tile_seq = 0
    for s in gl.static_range(1, NS):
        tile_seq = gl.where(tile_start >= gl.load(CU + s), s, tile_seq)
    tile_first = gl.load(CU + tile_seq)
    tile_end = gl.load(CU + tile_seq + 1)
    tile_slot = gl.load(IDX + tile_seq * SI)
    interior = (tile_start >= tile_first + 3) & (tile_start + BT <= tile_end) & (tile_slot >= 0)
    weight_layout: gl.constexpr = gl.BlockedLayout([2 if BT == 32 else 1, 4], [64, 1], [1, 4], [1, 0])
    wc = part_head * 128 + gl.arange(0, 128, gl.SliceLayout(1, weight_layout))
    taps = gl.arange(0, 4, gl.SliceLayout(0, weight_layout))
    weights = gl.load(CW + wc[:, None] * 4 + taps[None, :])
    z = gl.full((BT, 128), 0, gl.float32, layout)
    if interior:
        if BT == 32:
            window_layout: gl.constexpr = gl.BlockedLayout([1, 16, 2], [1, 1, 64], [4, 1, 1], [2, 1, 0])
            wave = gl.arange(0, 4, gl.SliceLayout(1, gl.SliceLayout(2, window_layout)))
            window_row = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, window_layout)))
            channel = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, window_layout)))
            window = gl.load(
                X + (tile_start + wave[:, None, None] * 8 + window_row[None, :, None] - 3) * SX
                + part_head * 128 + channel[None, None, :],
                window_row[None, :, None] < 11, 0)
        for j in gl.static_range(4):
            if BT == 32:
                gather_row = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(2, window_layout)))
                index = gather_row[None, :, None] + j + gl.zeros((4, 8, 128), gl.int32, window_layout)
                x = gl.reshape(gl.gather(window, index, 1), (32, 128))
                x = gl.convert_layout(x, layout, assert_trivial=True).to(gl.float32)
            else:
                x = gl.load(X + (t[:, None] - 3 + j) * SX + c[None, :]).to(gl.float32)
            tap = gl.full((128, 1), j, gl.int32, weight_layout)
            w = gl.sum(gl.gather(weights, tap, 1), 1)
            w = gl.convert_layout(w, gl.SliceLayout(0, layout), assert_trivial=True)
            z = z + x * w[None, :]
    else:
        seq = gl.full((BT,), 0, gl.int32, gl.SliceLayout(1, layout))
        for s in gl.static_range(1, NS):
            seq = gl.where(t >= gl.load(CU + s), s, seq)
        first = gl.load(CU + seq)
        slot = gl.load(IDX + seq * SI).to(gl.int64)
        initial = gl.load(INITIAL + seq)
        valid = (slot >= 0) & (t < M)
        for j in gl.static_range(4):
            source_t = t - 3 + j
            from_input = source_t >= first
            source_row = gl.where(from_input, X + source_t * SX,
                                  CS + slot * SC0 + (source_t - first + 3) * SC1)
            source_stride = gl.where(from_input, 1, SC2)
            x = gl.load(source_row[:, None] + c[None, :] * source_stride[:, None],
                        (valid & (from_input | initial))[:, None], 0).to(gl.float32)
            tap = gl.full((128, 1), j, gl.int32, weight_layout)
            w = gl.sum(gl.gather(weights, tap, 1), 1)
            w = gl.convert_layout(w, gl.SliceLayout(0, layout), assert_trivial=True)
            z = z + x * w[None, :]
    z = (z * _sigmoid(z)).to(gl.bfloat16).to(gl.float32)
    if BT == 32 and (
        (M <= 4096 and NS > 1) or (M == 4096 and NS == 1) or (M == 8192 and NS == 8)
    ):
        if part < 2:
            norm_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
            normalized = gl.convert_layout(z, norm_layout)
            normalized = normalized * gl.rsqrt(gl.sum(normalized * normalized, 1) + 1e-6)[:, None]
            nt = tile_start + gl.arange(0, BT, gl.SliceLayout(1, norm_layout))
            nc = part_head * 128 + gl.arange(0, 128, gl.SliceLayout(0, norm_layout))
            gl.store(Prepared + nt[:, None] * (3 * H * 128) + nc[None, :], normalized, nt[:, None] < M)
        else:
            gl.store(Prepared + t[:, None] * (3 * H * 128) + c[None, :], z, t[:, None] < M)
    else:
        if part < 2:
            z = z * gl.rsqrt(gl.sum(z * z, 1) + 1e-6)[:, None]
        gl.store(Prepared + t[:, None] * (3 * H * 128) + c[None, :], z, t[:, None] < M)
    if part == 2:
        beta_t = tile_start + gl.arange(0, BT, gl.BlockedLayout([1], [64], [4], [0]))
        beta = _sigmoid(gl.load(Beta + beta_t * SB + h, beta_t < M, 0).to(gl.float32))
        gl.store(Betas + beta_t * H + h, beta, beta_t < M)


@gluon.jit
def _fused_prepare(X, CW, CS, IDX, CU, INITIAL, Beta, Prepared, Betas,
                   FA, W, A, DT, Decay, lower,
                   H: gl.constexpr, NS: gl.constexpr, SX: gl.constexpr, SB: gl.constexpr, SI: gl.constexpr,
                   SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                   M: gl.constexpr, SFA: gl.constexpr, SW: gl.constexpr,
                   PROJECT_ROWS: gl.constexpr):
    tile = gl.program_id(0)
    role = tile % 4
    tile //= 4
    if role == 0:
        _project_decay(FA, W, A, DT, Decay, lower, M, H, SFA, SW, PROJECT_ROWS,
                       tile // (H * 2), tile % (H * 2))
    else:
        tile = tile * 3 + role - 1
        if PROJECT_ROWS == 32:
            if tile >= gl.cdiv(M, 16) * H * 3:
                return
        CONV_ROWS: gl.constexpr = 32 if PROJECT_ROWS == 64 else 16
        _conv_prepare(X, CW, CS, IDX, CU, INITIAL, Beta, Prepared, Betas,
                      H, NS, SX, SB, SI, SC0, SC1, SC2, M, CONV_ROWS,
                      tile // (H * 3), tile % (H * 3))


@gluon.jit
def _store_update_factor(matrix, Base):
    packed = gl.permute(gl.reshape(matrix, (4, 4, 128)), (2, 1, 0))
    packed = gl.reshape(packed, (2048,))
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    packed = gl.convert_layout(packed, layout)
    gl.amd.cdna4.buffer_store(packed, Base, gl.arange(0, 2048, layout))


@gluon.jit
def _load_recurrent_factor(Base, MFMA: gl.constexpr, KIND: gl.constexpr):
    flat: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1], [2], [256], [512], [1024]],
        lane_bases=[[16], [32], [64], [128], [4], [8]],
        warp_bases=[], block_bases=[], shape=[2048])
    packed = gl.amd.cdna4.buffer_load(Base, gl.arange(0, 2048, flat))
    if KIND == 0:
        matrix = gl.reshape(gl.permute(gl.reshape(packed, (8, 16, 4, 4)), (0, 3, 2, 1)), (128, 16))
    else:
        matrix = gl.reshape(gl.permute(gl.reshape(packed, (128, 4, 4)), (2, 1, 0)), (16, 128))
    return gl.convert_layout(matrix, gl.DotOperandLayout(1, MFMA, 1), assert_trivial=True)


@gluon.jit
def _load_output_factor(Base, MFMA: gl.constexpr, KIND: gl.constexpr):
    if KIND == 0:
        flat: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1], [2], [4], [512], [1024]],
            lane_bases=[[32], [64], [128], [256], [8], [16]],
            warp_bases=[], block_bases=[], shape=[2048])
        packed = gl.amd.cdna4.buffer_load(Base, gl.arange(0, 2048, flat))
        matrix = gl.reshape(gl.permute(gl.reshape(packed, (4, 16, 32)), (0, 2, 1)), (128, 16))
    else:
        flat: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1], [2]],
            lane_bases=[[16], [32], [64], [128], [4], [8]],
            warp_bases=[], block_bases=[], shape=[256])
        packed = gl.amd.cdna4.buffer_load(Base, gl.arange(0, 256, flat))
        matrix = gl.permute(gl.reshape(packed, (16, 16)), (1, 0))
    return gl.convert_layout(matrix, gl.DotOperandLayout(1, MFMA, 8 if KIND == 0 else 4), assert_trivial=True)


@gluon.jit
def _update_response(response, product, k0, q0, d0, beta, row, li, i,
                     base, output_base):
    layout: gl.constexpr = response.type.layout
    product = product * d0
    prediction_offset = (li // 16) * 256 + i * 16 + (li % 4) * 4 + li % 16 // 4
    query_offset = (li // 32) * 512 + i * 32 + li % 32
    gl.amd.cdna4.buffer_store(product * k0, base, prediction_offset)
    gl.amd.cdna4.buffer_store((product * q0).to(gl.bfloat16), output_base, query_offset)
    k = gl.convert_layout(k0, gl.SliceLayout(0, layout))
    q = gl.convert_layout(q0, gl.SliceLayout(0, layout))
    d = gl.convert_layout(d0, gl.SliceLayout(0, layout))
    response = response * d[None, :]
    delta = ((row == i).to(gl.float32) - gl.sum(response * k[None, :], 1)) * beta
    response = response + delta[:, None] * k[None, :]
    projected = gl.sum(response * q[None, :], 1)
    gl.amd.cdna4.buffer_store(projected.to(gl.bfloat16), output_base, 2048 + i * 16 + row)
    return response, product


@gluon.jit
def _response_factor_body(Prepared, Decay, Betas, Factors, OutputFactors,
                          h, group_id, start, end,
                          H: gl.constexpr, BULK: gl.constexpr,
                          UNROLL: gl.constexpr, FULL: gl.constexpr, PREFETCH: gl.constexpr):
    base = Factors + (group_id * H + h) * 4224
    output_base = OutputFactors + (group_id * H + h) * 2304
    if BULK:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [1, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [16, 4], [1, 1], [1, 0])
    row = gl.arange(0, 16, gl.SliceLayout(1, layout))
    li = gl.arange(0, 128, gl.BlockedLayout([2], [64], [1], [0]))
    response = gl.zeros((16, 128), gl.float32, layout)
    product = gl.full((128,), 1, gl.float32, li.type.layout)
    if BULK:
        preload: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [1, 1], [1, 0])
        pt = start + gl.arange(0, 16, gl.SliceLayout(1, preload))
        pk = gl.arange(0, 128, gl.SliceLayout(0, preload))
        all_k = gl.load(Prepared + (pt[:, None] * 3 * H + H + h) * 128 + pk[None, :], (FULL | (pt[:, None] < end)), 0)
        all_q = gl.load(Prepared + (pt[:, None] * 3 * H + h) * 128 + pk[None, :], (FULL | (pt[:, None] < end)), 0)
        all_d = gl.load(Decay + (pt[:, None] * H + h) * 128 + pk[None, :], (FULL | (pt[:, None] < end)), 1)
        beta_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        beta_row = gl.arange(0, 16, beta_layout)
        all_beta = gl.load(Betas + (start + beta_row) * H + h, FULL | (start + beta_row < end), 0)
        for i in gl.static_range(16):
            index = gl.full((1, 128), i, gl.int32, preload)
            k0 = gl.sum(gl.gather(all_k, index, 0).to(gl.float32), 0)
            q0 = gl.sum(gl.gather(all_q, index, 0).to(gl.float32), 0) * (128 ** -0.5)
            d0 = gl.sum(gl.gather(all_d, index, 0), 0)
            k0 = gl.convert_layout(k0, li.type.layout, assert_trivial=True)
            q0 = gl.convert_layout(q0, li.type.layout, assert_trivial=True)
            d0 = gl.convert_layout(d0, li.type.layout, assert_trivial=True)
            beta_index = gl.full((1,), i, gl.int32, beta_layout)
            beta = gl.sum(gl.gather(all_beta, beta_index, 0), 0)
            response, product = _update_response(
                response, product, k0, q0, d0, beta, row, li, i, base, output_base)
    else:
        if not PREFETCH:
            beta_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
            beta_row = gl.arange(0, 16, beta_layout)
            all_beta = gl.load(Betas + (start + beta_row) * H + h,
                               FULL | (start + beta_row < end), 0)
        if PREFETCH:
            next_k = gl.amd.cdna4.buffer_load(
                Prepared + (start * 3 * H + h) * 128, H * 128 + li)
            next_q = gl.amd.cdna4.buffer_load(
                Prepared + (start * 3 * H + h) * 128, li)
            next_d = gl.amd.cdna4.buffer_load(
                Decay + (start * H + h) * 128, li)
            next_beta = gl.load(Betas + start * H + h)
        for base_i in range(0, 16, UNROLL):
            for step_i in gl.static_range(UNROLL):
                i = base_i + step_i
                t = start + i
                if PREFETCH:
                    k0 = next_k.to(gl.float32)
                    q0 = next_q.to(gl.float32) * (128 ** -0.5)
                    d0 = next_d
                    beta = next_beta
                    next_t = start + gl.minimum(i + 1, 15)
                    next_k = gl.amd.cdna4.buffer_load(
                        Prepared + (next_t * 3 * H + h) * 128,
                        H * 128 + li, (FULL | (next_t < end)), 0)
                    next_q = gl.amd.cdna4.buffer_load(
                        Prepared + (next_t * 3 * H + h) * 128,
                        li, (FULL | (next_t < end)), 0)
                    next_d = gl.amd.cdna4.buffer_load(
                        Decay + (next_t * H + h) * 128,
                        li, (FULL | (next_t < end)), 1)
                    next_beta = gl.load(Betas + next_t * H + h, (FULL | (next_t < end)), 0)
                else:
                    k0 = gl.amd.cdna4.buffer_load(Prepared + (t * 3 * H + h) * 128, H * 128 + li, (FULL | (t < end)), 0).to(gl.float32)
                    q0 = gl.amd.cdna4.buffer_load(Prepared + (t * 3 * H + h) * 128, li, (FULL | (t < end)), 0).to(gl.float32) * (128 ** -0.5)
                    d0 = gl.amd.cdna4.buffer_load(Decay + (t * H + h) * 128, li, (FULL | (t < end)), 1)
                    beta_index = gl.full((1,), i, gl.int32, beta_layout)
                    beta = gl.sum(gl.gather(all_beta, beta_index, 0), 0)
                response, product = _update_response(
                    response, product, k0, q0, d0, beta, row, li, i, base, output_base)
    _store_update_factor(response, base + 2048)
    gl.amd.cdna4.buffer_store(product, base, 4096 + li)


@gluon.jit
def _response_factors(Prepared, Decay, Betas, Factors, OutputFactors, IDX, CU,
                      H: gl.constexpr, SI: gl.constexpr, NS: gl.constexpr, BULK: gl.constexpr, UNROLL: gl.constexpr,
                      PREFETCH: gl.constexpr):
    h = gl.program_id(0)
    group_id = gl.program_id(1)
    seq = 0
    for candidate in gl.static_range(1, NS):
        boundary = gl.load(CU + candidate) // 16 + candidate
        seq = gl.where(group_id >= boundary, candidate, seq)
    first = gl.load(CU + seq)
    block = group_id - (first // 16 + seq)
    end = gl.load(CU + seq + 1)
    start = first + block * 16
    slot = gl.load(IDX + seq * SI)
    if (start >= end) | (slot < 0):
        return
    if start + 16 <= end:
        _response_factor_body(Prepared, Decay, Betas, Factors, OutputFactors,
                              h, group_id, start, end, H, BULK, UNROLL, True, PREFETCH)
    else:
        _response_factor_body(Prepared, Decay, Betas, Factors, OutputFactors,
                              h, group_id, start, end, H, BULK, UNROLL, False, PREFETCH)


@gluon.jit
def _serial_step(Prepared, Decay, Betas, Output, state, q_next, key_next,
                 decay_next, beta_next, value_next, t, stop, h, row, load_index,
                 H: gl.constexpr, BASIS: gl.constexpr):
    layout: gl.constexpr = state.type.layout
    q_raw = q_next
    key_raw = key_next
    decay_raw = decay_next
    beta = beta_next
    if not BASIS:
        value = value_next.to(gl.float32)
    next_t = gl.minimum(t + 1, stop - 1)
    q_next = _load_prepared(Prepared, next_t, h, load_index, H, 0)
    key_next = _load_prepared(Prepared, next_t, h, load_index, H, 1)
    decay_next = gl.amd.cdna4.buffer_load(Decay + (next_t * H + h) * 128, load_index)
    beta_next = gl.load(Betas + next_t * H + h)
    if not BASIS:
        value_next = _load_prepared(Prepared, next_t, h, row, H, 2)
    q = gl.convert_layout(q_raw.to(gl.float32) * (128 ** -0.5), gl.SliceLayout(0, layout))
    key = gl.convert_layout(key_raw.to(gl.float32), gl.SliceLayout(0, layout))
    decay = gl.convert_layout(decay_raw, gl.SliceLayout(0, layout))
    state = state * decay[None, :]
    if BASIS:
        delta = -gl.sum(state * key[None, :], 1) * beta
    else:
        delta = (value - gl.sum(state * key[None, :], 1)) * beta
    state = state + delta[:, None] * key[None, :]
    projected = gl.sum(state * q[None, :], 1)
    gl.amd.cdna4.buffer_store(projected.to(Output.dtype.element_ty), Output + (t * H + h) * 128, row)
    return state, q_next, key_next, decay_next, beta_next, value_next


@gluon.jit
def _serial_affine_body(Prepared, Decay, Betas, Maps, Output, S, INITIAL,
                        seq, start, end, chunk_id, h, slot, first_chunk, row_block,
                        H: gl.constexpr, SS: gl.constexpr, CHUNK: gl.constexpr,
                        BASIS: gl.constexpr):
    ROWS: gl.constexpr = 16
    UNROLL: gl.constexpr = 2
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 1], [1, 0])
    row = (row_block % (128 // ROWS)) * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    key_index = gl.arange(0, 128, gl.SliceLayout(0, layout))
    load_index = gl.arange(0, 128, gl.BlockedLayout([2], [64], [1], [0]))
    if BASIS:
        state = (row[:, None] == key_index[None, :]).to(gl.float32)
    else:
        state = gl.zeros((ROWS, 128), gl.float32, layout)
        if first_chunk:
            initial = gl.load(INITIAL + seq)
            state = gl.load(S + slot.to(gl.int64) * SS + h * 16384
                            + row[:, None] * 128 + key_index[None, :], initial, 0)
    stop = gl.minimum(start + CHUNK, end)
    q_next = _load_prepared(Prepared, start, h, load_index, H, 0)
    key_next = _load_prepared(Prepared, start, h, load_index, H, 1)
    decay_next = gl.amd.cdna4.buffer_load(Decay + (start * H + h) * 128, load_index)
    beta_next = gl.load(Betas + start * H + h)
    if not BASIS:
        value_next = _load_prepared(Prepared, start, h, row, H, 2)
    else:
        value_next = 0
    full_stop = start + (stop - start) // UNROLL * UNROLL
    for block_t in range(start, full_stop, UNROLL):
        for step in gl.static_range(UNROLL):
            state, q_next, key_next, decay_next, beta_next, value_next = _serial_step(
                Prepared, Decay, Betas, Output, state, q_next, key_next,
                decay_next, beta_next, value_next, block_t + step, stop, h,
                row, load_index, H, BASIS)
    for t in range(full_stop, stop):
        state, q_next, key_next, decay_next, beta_next, value_next = _serial_step(
            Prepared, Decay, Betas, Output, state, q_next, key_next,
            decay_next, beta_next, value_next, t, stop, h, row, load_index, H, BASIS)
    map_base = Maps + (chunk_id * H + h) * 32768
    if not BASIS:
        map_base += 16384
    if BASIS:
        _store_transition(state, map_base, row_block, ROWS)
    else:
        gl.amd.cdna4.buffer_store(state, map_base, row[:, None] * 128 + key_index[None, :])


@gluon.jit
def _serial_affine_chunks(Prepared, Decay, Betas, Maps, Coeff, Local, S, INITIAL, IDX, CU,
                          X, CS, H: gl.constexpr, SI: gl.constexpr, SS: gl.constexpr,
                          SX: gl.constexpr, SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                          CHUNK: gl.constexpr):
    ROWS: gl.constexpr = 16
    seq_head = gl.program_id(1)
    row_block = gl.program_id(0)
    seq = seq_head // H
    h = seq_head % H
    first, start, end, chunk_id, chunk = _chunk_bounds(CU, seq, CHUNK)
    slot = gl.load(IDX + seq * SI)
    if (start >= end) | (slot < 0):
        return
    if row_block < 128 // ROWS:
        if chunk == 0:
            if row_block < 3:
                _commit_history(X, CS, INITIAL, seq, start, end, slot, h, row_block,
                                H, SX, SC0, SC1, SC2)
        else:
            _serial_affine_body(Prepared, Decay, Betas, Maps, Coeff, S, INITIAL,
                         seq, start, end, chunk_id, h, slot, chunk == 0, row_block,
                         H, SS, CHUNK, True)
    else:
        _serial_affine_body(Prepared, Decay, Betas, Maps, Local, S, INITIAL,
                     seq, start, end, chunk_id, h, slot, chunk == 0, row_block,
                     H, SS, CHUNK, False)


@gluon.jit
def _blocked_response_step(Prepared, Factors, OutputFactors, CoeffFlags, Output, state,
                           t, end, group_id, h, row_block,
                           H: gl.constexpr, BASIS: gl.constexpr, INITIAL_KIND: gl.constexpr,
                           OUTPUT_STRIDE: gl.constexpr, BOUNDED: gl.constexpr,
                           SPARSE_COEFF: gl.constexpr, LATE_UPDATE: gl.constexpr,
                           MIRROR_PREDICTION: gl.constexpr, EARLY_DECAY: gl.constexpr,
                           MIRROR_UPDATE: gl.constexpr):
    lm: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[1, 1])
    r = (row_block % 4) * 32 + gl.arange(0, 32, gl.SliceLayout(1, lm))
    k = gl.arange(0, 128, gl.SliceLayout(0, lm))
    out_t = gl.arange(0, 16, gl.SliceLayout(0, lm))
    base = Factors + (group_id * H + h) * 4224
    output_base = OutputFactors + (group_id * H + h) * 2304
    if INITIAL_KIND == 1:
        offset = (r[:, None] // 16) * 256 + out_t[None, :] * 16 + (r[:, None] % 4) * 4 + r[:, None] % 16 // 4
        prediction = gl.amd.cdna4.buffer_load(base, offset)
    elif INITIAL_KIND == 2:
        prediction = gl.zeros((32, 16), gl.float32, lm)
    else:
        p = _load_recurrent_factor(base, lm, 0)
        if MIRROR_PREDICTION:
            prediction_layout: gl.constexpr = gl.amd.AMDMFMALayout(
                version=4, instr_shape=[16, 16, 4], transposed=False,
                warps_per_cta=[1, 1])
            prediction_a = gl.convert_layout(
                gl.permute(p, (1, 0)), gl.DotOperandLayout(0, prediction_layout, 1))
            prediction_b = gl.convert_layout(
                gl.permute(state, (1, 0)), gl.DotOperandLayout(1, prediction_layout, 1))
            prediction = gl.amd.cdna4.mfma(
                prediction_a, prediction_b,
                gl.zeros((16, 32), gl.float32, prediction_layout))
            prediction = gl.convert_layout(gl.permute(prediction, (1, 0)), lm)
        else:
            state_a = gl.convert_layout(state, gl.DotOperandLayout(0, lm, 1))
            prediction = gl.amd.cdna4.mfma(state_a, p, gl.zeros((32, 16), gl.float32, lm))
    if BASIS:
        residual = -prediction
    else:
        value_base = Prepared + (t * 3 * H + 2 * H + h) * 128
        value_offset = out_t[None, :] * 3 * H * 128 + r[:, None]
        if BOUNDED and t + 16 <= end:
            v = gl.amd.cdna4.buffer_load(value_base, value_offset).to(gl.float32)
        else:
            v = gl.amd.cdna4.buffer_load(
                value_base, value_offset, t + out_t[None, :] < end, 0).to(gl.float32)
        residual = v - prediction
    if EARLY_DECAY and INITIAL_KIND != 2:
        d = gl.amd.cdna4.buffer_load(base, 4096 + k)
    output_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, 1])
    query_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
    if INITIAL_KIND == 1:
        offset = (r[:, None] // 32) * 512 + out_t[None, :] * 32 + r[:, None] % 32
        projected = gl.amd.cdna4.buffer_load(output_base, offset).to(gl.float32)
        projected = gl.convert_layout(projected, output_layout, assert_trivial=True)
    elif INITIAL_KIND == 2:
        projected = gl.zeros((32, 16), gl.float32, output_layout)
    else:
        q_output = _load_output_factor(output_base, query_layout, 0)
        state_output = gl.convert_layout(state.to(gl.bfloat16), gl.DotOperandLayout(0, query_layout, 8))
        projected = gl.amd.cdna4.mfma(state_output, q_output, gl.zeros((32, 16), gl.float32, query_layout))
        projected = gl.convert_layout(projected, output_layout, assert_trivial=True)
    residual_a = gl.convert_layout(residual, gl.DotOperandLayout(0, lm, 1))
    c_output = _load_output_factor(output_base + 2048, output_layout, 1)
    if not LATE_UPDATE:
        e = _load_recurrent_factor(base + 2048, lm, 1)
    residual_output = gl.convert_layout(residual.to(gl.bfloat16), gl.DotOperandLayout(0, output_layout, 4))
    projected = gl.amd.cdna4.mfma(residual_output, c_output, projected)
    projected = gl.convert_layout(projected, lm, assert_trivial=True)
    if SPARSE_COEFF and BASIS:
        rounded = projected.to(gl.bfloat16)
        magnitude = rounded.to(gl.int16, bitcast=True) & 0x7fff
        present = gl.max(gl.max(magnitude, 1), 0) != 0
        gl.store(CoeffFlags + (group_id * H + h) * 4 + row_block, present.to(gl.int32))
        if present:
            gl.amd.cdna4.buffer_store(
                rounded, Output + t * OUTPUT_STRIDE + h * 128,
                out_t[None, :] * OUTPUT_STRIDE + r[:, None], t + out_t[None, :] < end)
    else:
        if BOUNDED:
            projection_base = Output + t * OUTPUT_STRIDE + h * 128
            output_offset = out_t[None, :] * OUTPUT_STRIDE + r[:, None]
            if t + 16 <= end:
                gl.amd.cdna4.buffer_store(
                    projected.to(Output.dtype.element_ty), projection_base, output_offset)
            else:
                gl.amd.cdna4.buffer_store(
                    projected.to(Output.dtype.element_ty), projection_base, output_offset,
                    t + out_t[None, :] < end)
        else:
            gl.store(Output + (t + out_t[None, :]) * OUTPUT_STRIDE + h * 128 + r[:, None], projected,
                      t + out_t[None, :] < end)
    if INITIAL_KIND == 2:
        decayed = gl.zeros((32, 128), gl.float32, lm)
    else:
        if not EARLY_DECAY:
            d = gl.amd.cdna4.buffer_load(base, 4096 + k)
        decayed = state * d[None, :]
    if LATE_UPDATE:
        e = _load_recurrent_factor(base + 2048, lm, 1)
    if MIRROR_UPDATE:
        update_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 4], transposed=False,
            warps_per_cta=[1, 1])
        update_a = gl.convert_layout(
            gl.permute(e, (1, 0)), gl.DotOperandLayout(0, update_layout, 1))
        update_b = gl.convert_layout(
            gl.permute(residual, (1, 0)), gl.DotOperandLayout(1, update_layout, 1))
        update_c = gl.convert_layout(gl.permute(decayed, (1, 0)), update_layout)
        updated = gl.amd.cdna4.mfma(update_a, update_b, update_c)
        return gl.convert_layout(gl.permute(updated, (1, 0)), lm)
    else:
        return gl.amd.cdna4.mfma(residual_a, e, decayed)


@gluon.jit
def _blocked_affine_body(Prepared, Factors, OutputFactors, Maps, ZeroFlags, CoeffFlags, Output, S, INITIAL,
                         seq, first, start, end, chunk_id, h, slot, first_chunk, row_block,
                         H: gl.constexpr, SS: gl.constexpr, CHUNK: gl.constexpr,
                         BASIS: gl.constexpr, OUTPUT_STRIDE: gl.constexpr,
                         ZERO_SKIP: gl.constexpr, SPARSE: gl.constexpr,
                         ROUNDOFF_SKIP: gl.constexpr, BOUNDED: gl.constexpr,
                         SPARSE_COEFF: gl.constexpr, LATE_UPDATE: gl.constexpr,
                         MIRROR_PREDICTION: gl.constexpr, EARLY_DECAY: gl.constexpr,
                         Incoming, PUBLISH: gl.constexpr, MIRROR_UPDATE: gl.constexpr = False):
    gl.static_assert(not ROUNDOFF_SKIP or (ZERO_SKIP and not SPARSE))
    lm: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[1, 1])
    r = (row_block % 4) * 32 + gl.arange(0, 32, gl.SliceLayout(1, lm))
    k = gl.arange(0, 128, gl.SliceLayout(0, lm))
    if BASIS:
        state = (r[:, None] == k[None, :]).to(gl.float32)
    else:
        state = gl.zeros((32, 128), gl.float32, lm)
        if first_chunk:
            initial = gl.load(INITIAL + seq)
            state = gl.load(S + slot.to(gl.int64) * SS + h * 16384 + r[:, None] * 128 + k[None, :], initial, 0)
    stop = gl.minimum(start + CHUNK, end)
    group_base = first // 16 + seq
    loop_start = start
    if BASIS:
        state = _blocked_response_step(Prepared, Factors, OutputFactors, CoeffFlags, Output, state,
                                        start, end, group_base + (start - first) // 16, h, row_block,
                                        H, BASIS, 1, OUTPUT_STRIDE, BOUNDED, SPARSE_COEFF, LATE_UPDATE, MIRROR_PREDICTION, EARLY_DECAY, MIRROR_UPDATE)
        loop_start += 16
    elif not first_chunk:
        state = _blocked_response_step(Prepared, Factors, OutputFactors, CoeffFlags, Output, state,
                                        start, end, group_base + (start - first) // 16, h, row_block,
                                        H, BASIS, 2, OUTPUT_STRIDE, BOUNDED, SPARSE_COEFF, LATE_UPDATE, MIRROR_PREDICTION, EARLY_DECAY, MIRROR_UPDATE)
        loop_start += 16
    for t in range(loop_start, stop, 16):
        group_id = group_base + (t - first) // 16
        state = _blocked_response_step(Prepared, Factors, OutputFactors, CoeffFlags, Output, state,
                                        t, end, group_id, h, row_block, H, BASIS, 0, OUTPUT_STRIDE, BOUNDED, SPARSE_COEFF, LATE_UPDATE, MIRROR_PREDICTION, EARLY_DECAY, MIRROR_UPDATE)
    map_base = Maps + (chunk_id * H + h) * 32768
    if not BASIS:
        map_base += 16384
    if BASIS:
        if not SPARSE:
            _store_transition(state, map_base, row_block, 32)
        if ROUNDOFF_SKIP:
            exponent = (state.to(gl.int32, bitcast=True) >> 23) & 255
            maximum = gl.max(gl.max(exponent, 1), 0)
            gl.store(ZeroFlags + (chunk_id * H + h) * 8 + row_block, maximum)
        elif ZERO_SKIP:
            magnitude = state.to(gl.int32, bitcast=True) & 0x7fffffff
            is_zero = gl.max(gl.max(magnitude, 1), 0) == 0
            gl.store(ZeroFlags + (chunk_id * H + h) * 8 + row_block, is_zero.to(gl.int32))
            if SPARSE:
                if not is_zero:
                    _store_transition(state, map_base, row_block, 32)
    else:
        gl.amd.cdna4.buffer_store(state, map_base, r[:, None] * 128 + k[None, :])
        if PUBLISH:


            if start + CHUNK < end:
                gl.amd.cdna4.buffer_store(
                    state.to(gl.bfloat16),
                    Incoming + ((chunk_id + 1) * H + h) * 16384,
                    (k[None, :] // 8) * 1024 + r[:, None] * 8 + k[None, :] % 8)
        if ROUNDOFF_SKIP:
            exponent = (state.to(gl.int32, bitcast=True) >> 23) & 255
            minimum = gl.min(gl.min(exponent, 1), 0)
            maximum = gl.max(gl.max(exponent, 1), 0)
            gl.store(ZeroFlags + (chunk_id * H + h) * 8 + row_block, minimum | (maximum << 8))
        elif ZERO_SKIP:
            if SPARSE_COEFF:
                exponent = (state.to(gl.int32, bitcast=True) >> 23) & 255
                certificate = gl.max(gl.max(exponent, 1), 0)
            else:
                nonfinite = (state.to(gl.int32, bitcast=True) & 0x7f800000) == 0x7f800000
                certificate = (gl.max(gl.max(nonfinite.to(gl.int32), 1), 0) == 0).to(gl.int32)
            gl.store(ZeroFlags + (chunk_id * H + h) * 8 + row_block, certificate)


@gluon.jit
def _blocked_affine_chunks(Prepared, Factors, OutputFactors, Maps, ZeroFlags, CoeffFlags, Coeff, Local, S, INITIAL, IDX, CU,
                           X, CS, H: gl.constexpr, SI: gl.constexpr, SS: gl.constexpr,
                           SX: gl.constexpr, SC0: gl.constexpr, SC1: gl.constexpr, SC2: gl.constexpr,
                           CHUNK: gl.constexpr, LOCAL_STRIDE: gl.constexpr,
                           ZERO_SKIP: gl.constexpr, SPARSE: gl.constexpr,
                           ROUNDOFF_SKIP: gl.constexpr, BOUNDED: gl.constexpr,
                           SPARSE_COEFF: gl.constexpr, LATE_UPDATE: gl.constexpr,
                           MIRROR_PREDICTION: gl.constexpr, EARLY_DECAY: gl.constexpr,
                           Incoming, PUBLISH: gl.constexpr, MIRROR_UPDATE: gl.constexpr = False):
    gl.static_assert(not ROUNDOFF_SKIP or (ZERO_SKIP and not SPARSE))
    seq_head = gl.program_id(0)
    row_block = gl.program_id(1)
    seq = seq_head // H
    h = seq_head % H
    first, start, end, chunk_id, chunk = _chunk_bounds(CU, seq, CHUNK)
    slot = gl.load(IDX + seq * SI)
    if (start >= end) | (slot < 0):
        return
    if row_block < 4:
        if chunk == 0:
            if row_block < 3:
                _commit_history(X, CS, INITIAL, seq, start, end, slot, h, row_block,
                                H, SX, SC0, SC1, SC2)
        else:
            _blocked_affine_body(Prepared, Factors, OutputFactors, Maps, ZeroFlags, CoeffFlags, Coeff, S, INITIAL,
                                 seq, first, start, end, chunk_id, h, slot, chunk == 0, row_block,
                                 H, SS, CHUNK, True, H * 128, ZERO_SKIP, SPARSE, ROUNDOFF_SKIP, BOUNDED, SPARSE_COEFF, LATE_UPDATE, MIRROR_PREDICTION, EARLY_DECAY, Incoming, PUBLISH, MIRROR_UPDATE)
    else:
        _blocked_affine_body(Prepared, Factors, OutputFactors, Maps, ZeroFlags, CoeffFlags, Local, S, INITIAL,
                             seq, first, start, end, chunk_id, h, slot, chunk == 0, row_block,
                             H, SS, CHUNK, False, LOCAL_STRIDE, ZERO_SKIP, SPARSE, ROUNDOFF_SKIP, BOUNDED, SPARSE_COEFF, LATE_UPDATE, MIRROR_PREDICTION, EARLY_DECAY, Incoming, PUBLISH, MIRROR_UPDATE)


@gluon.jit
def _chunk_prefix(Maps, Incoming, ZeroFlags, S, IDX, CU, H: gl.constexpr, SI: gl.constexpr, SS: gl.constexpr,
                  CHUNK: gl.constexpr, BM: gl.constexpr, PACKED_INCOMING: gl.constexpr,
                  ZERO_SKIP: gl.constexpr, SPARSE: gl.constexpr,
                  FLAG_CHUNKS: gl.constexpr, ROUNDOFF_SKIP: gl.constexpr,
                  IncomingFlags, CERTIFY_OUTPUT: gl.constexpr,
                  COPY_INITIAL: gl.constexpr = False,
                  PROPAGATE_BOUND: gl.constexpr = False,
                  OMIT_FINAL_EXP: gl.constexpr = False):
    gl.static_assert(not ROUNDOFF_SKIP or (ZERO_SKIP and not SPARSE and FLAG_CHUNKS > 0))
    gl.static_assert(not CERTIFY_OUTPUT or (ZERO_SKIP and FLAG_CHUNKS > 0 and not ROUNDOFF_SKIP))
    seq = gl.program_id(0) // H
    h = gl.program_id(0) % H
    start = gl.load(CU + seq)
    end = gl.load(CU + seq + 1)
    slot = gl.load(IDX + seq * SI).to(gl.int64)
    if (end <= start) | (slot < 0):
        return
    lm: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[1, 8])
    r = gl.program_id(1) * BM + gl.arange(0, BM, gl.SliceLayout(1, lm))
    c = gl.arange(0, 128, gl.SliceLayout(0, lm))
    chunk_base = start // CHUNK + seq
    first_base = (chunk_base * H + h) * 32768 + 16384
    row_offsets = r[:, None] * 128 + c[None, :]
    state = gl.amd.cdna4.buffer_load(Maps + first_base, row_offsets)
    finite = True
    finite_known = True
    bf16_finite = False
    chunks = gl.cdiv(end - start, CHUNK)
    if FLAG_CHUNKS > 0:
        flag_layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0])
        panel_layout: gl.constexpr = gl.SliceLayout(1, flag_layout)
        flag_chunk = gl.arange(0, FLAG_CHUNKS, panel_layout)
        fragment = gl.arange(0, 4, gl.SliceLayout(0, flag_layout))
        flag_base = ZeroFlags + ((chunk_base + flag_chunk) * H + h) * 8
        transition_flags = gl.load(flag_base[:, None] + fragment[None, :],
                                   ((flag_chunk > 0) & (flag_chunk < chunks))[:, None], 1)
        if ROUNDOFF_SKIP:
            transition_panel = gl.max(transition_flags, 1)
        else:
            zero_panel = gl.min(transition_flags, 1)
        finite_panel = gl.load(flag_base + 4 + (gl.program_id(1) * BM) // 32,
                              flag_chunk < chunks, 1)
        first_index = gl.full((1,), 0, gl.int32, panel_layout)
        if ROUNDOFF_SKIP:
            state_exponent = gl.sum(gl.gather(finite_panel, first_index, 0), 0) >> 8
        else:
            first_certificate = gl.sum(gl.gather(finite_panel, first_index, 0), 0)
            if CERTIFY_OUTPUT:
                finite = first_certificate < 255
                bf16_finite = first_certificate < 254
            else:
                finite = first_certificate != 0
    elif ZERO_SKIP:
        finite = gl.load(ZeroFlags + (chunk_base * H + h) * 8 + 4 + (gl.program_id(1) * BM) // 32) != 0
    first_serial = 1
    if COPY_INITIAL:
        gl.static_assert(ROUNDOFF_SKIP and PACKED_INCOMING and not CERTIFY_OUTPUT)
        previous_index = gl.maximum(flag_chunk - 1, 0)
        previous_max = gl.gather(finite_panel, previous_index, 0) >> 8
        panel_min = finite_panel & 255
        panel_max = finite_panel >> 8
        certified = ((previous_max < 255) & (transition_panel < 255)
                     & (panel_min > 1) & (panel_max < 255)
                     & (previous_max + transition_panel < panel_min + 93))
        first_serial = gl.min(gl.where(
            (flag_chunk > 0) & (flag_chunk < chunks) & ~certified,
            flag_chunk, chunks), 0)

        copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 4], [1, 8, 8], [8, 1, 1], [2, 1, 0])
        copy_chunk = gl.arange(0, 8, gl.SliceLayout(1, gl.SliceLayout(2, copy_layout)))
        copy_row = gl.program_id(1) * BM + gl.arange(
            0, BM, gl.SliceLayout(0, gl.SliceLayout(2, copy_layout)))
        copy_col = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, copy_layout)))
        for copy_start in range(1, first_serial, 8):
            source_chunk = chunk_base + copy_start + copy_chunk - 1
            source = Maps + (source_chunk[:, None, None] * H + h) * 32768 + 16384
            copied = gl.load(source + copy_row[None, :, None] * 128 + copy_col[None, None, :],
                             (copy_start + copy_chunk < first_serial)[:, None, None], 0)
            destination = Incoming + ((source_chunk[:, None, None] + 1) * H + h) * 16384
            gl.store(destination + (copy_col[None, None, :] // 8) * 1024
                     + copy_row[None, :, None] * 8 + copy_col[None, None, :] % 8,
                     copied.to(Incoming.dtype.element_ty),
                     (copy_start + copy_chunk < first_serial)[:, None, None])
        if first_serial > 1:
            state = gl.amd.cdna4.buffer_load(
                Maps + ((chunk_base + first_serial - 1) * H + h) * 32768 + 16384,
                row_offsets)
            index = gl.full((1,), first_serial - 1, gl.int32, panel_layout)
            state_exponent = gl.sum(gl.gather(finite_panel, index, 0), 0) >> 8
    for chunk in range(first_serial, chunks):
        base = ((chunk_base + chunk) * H + h) * 32768
        if not ZERO_SKIP:
            transition = _load_transition(Maps + base, ZeroFlags, lm, False)
        additive = gl.amd.cdna4.buffer_load(Maps + base + 16384, row_offsets)
        if CERTIFY_OUTPUT:
            gl.store(IncomingFlags + ((chunk_base + chunk) * H + h) * (128 // BM) + gl.program_id(1),
                     (finite_known & bf16_finite).to(gl.int32))
        if PACKED_INCOMING:
            incoming_base = ((chunk_base + chunk) * H + h) * 16384
            gl.amd.cdna4.buffer_store(state.to(Incoming.dtype.element_ty),
                                       Incoming + incoming_base,
                                       (c[None, :] // 8) * 1024 + r[:, None] * 8 + c[None, :] % 8)
        else:
            gl.amd.cdna4.buffer_store(state, Maps + base + 16384, row_offsets)
        if ROUNDOFF_SKIP:
            panel_index = gl.full((1,), chunk, gl.int32, panel_layout)
            transition_exponent = gl.sum(gl.gather(transition_panel, panel_index, 0), 0)
            additive_flags = gl.sum(gl.gather(finite_panel, panel_index, 0), 0)
            additive_min = additive_flags & 255
            additive_max = additive_flags >> 8


            skip = ((state_exponent < 255) & (transition_exponent < 255)
                    & (additive_min > 1) & (additive_max < 255)
                    & (state_exponent + transition_exponent < additive_min + (96 if CHUNK == 96 else 93)))
            if skip:
                state = additive
                state_exponent = additive_max
            else:
                transition = _load_transition(Maps + base, ZeroFlags, lm, False)
                state_operand = gl.convert_layout(state, gl.DotOperandLayout(0, lm, 1))
                state = gl.amd.cdna4.mfma(state_operand, transition, additive)
                if PROPAGATE_BOUND:
                    all_finite = ((state_exponent < 255) & (transition_exponent < 255)
                                  & (additive_max < 255))
                    product_bound = state_exponent + transition_exponent - 117
                    state_exponent = gl.where(all_finite,
                        gl.minimum(255, gl.maximum(product_bound, additive_max + 2)), 255)
                else:
                    if not OMIT_FINAL_EXP or chunk + 1 < chunks:
                        exponent = (state.to(gl.int32, bitcast=True) >> 23) & 255
                        state_exponent = gl.max(gl.max(exponent, 1), 0)
        elif ZERO_SKIP:
            flags = ZeroFlags + ((chunk_base + chunk) * H + h) * 8
            if FLAG_CHUNKS > 0:
                panel_index = gl.full((1,), chunk, gl.int32, panel_layout)
                zero = gl.sum(gl.gather(zero_panel, panel_index, 0), 0)
            else:
                fragment = gl.arange(0, 4, gl.BlockedLayout([1], [64], [8], [0]))
                zero = gl.min(gl.load(flags + fragment), 0)
            skip = False
            if zero != 0:
                if not finite_known:
                    nonfinite = (state.to(gl.int32, bitcast=True) & 0x7f800000) == 0x7f800000
                    finite = gl.max(gl.max(nonfinite.to(gl.int32), 1), 0) == 0
                skip = finite
            if skip:
                state = additive
                if FLAG_CHUNKS > 0:
                    additive_certificate = gl.sum(gl.gather(finite_panel, panel_index, 0), 0)
                    if CERTIFY_OUTPUT:
                        finite = additive_certificate < 255
                        bf16_finite = additive_certificate < 254
                    else:
                        finite = additive_certificate != 0
                else:
                    finite = gl.load(flags + 4 + (gl.program_id(1) * BM) // 32) != 0
                finite_known = True
            else:
                transition = _load_transition(Maps + base, flags, lm, SPARSE)
                state_operand = gl.convert_layout(state, gl.DotOperandLayout(0, lm, 1))
                state = gl.amd.cdna4.mfma(state_operand, transition, additive)
                finite_known = False
        else:
            state_operand = gl.convert_layout(state, gl.DotOperandLayout(0, lm, 1))
            state = gl.amd.cdna4.mfma(state_operand, transition, additive)
    gl.amd.cdna4.buffer_store(state, S + slot * SS + h * 16384, row_offsets)


@gluon.jit
def _published_prefix(Maps, Incoming, Flags, S, IDX, CU,
                      H: gl.constexpr, SI: gl.constexpr, SS: gl.constexpr,
                      CHUNK: gl.constexpr, BM: gl.constexpr,
                      FLAG_CHUNKS: gl.constexpr, LOW_YIELD_DENSE: gl.constexpr,
                      BITMAP_SCAN: gl.constexpr = True):

    seq = gl.program_id(0) // H
    h = gl.program_id(0) % H
    first = gl.load(CU + seq)
    end = gl.load(CU + seq + 1)
    slot = gl.load(IDX + seq * SI).to(gl.int64)
    if (end <= first) | (slot < 0):
        return
    lm: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 4], transposed=True,
        warps_per_cta=[1, 8])
    r = gl.program_id(1) * BM + gl.arange(0, BM, gl.SliceLayout(1, lm))
    c = gl.arange(0, 128, gl.SliceLayout(0, lm))
    offsets = r[:, None] * 128 + c[None, :]
    packed_offsets = (c[None, :] // 8) * 1024 + r[:, None] * 8 + c[None, :] % 8
    chunk_base = first // CHUNK + seq
    chunks = gl.cdiv(end - first, CHUNK)
    flags_layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0])
    panel_layout: gl.constexpr = gl.SliceLayout(1, flags_layout)
    fc = gl.arange(0, FLAG_CHUNKS, panel_layout)
    fragment = gl.arange(0, 4, gl.SliceLayout(0, flags_layout))
    flag_base = Flags + ((chunk_base + fc) * H + h) * 8
    transition_flags = gl.load(flag_base[:, None] + fragment[None, :],
                              ((fc > 0) & (fc < chunks))[:, None], 255)
    transition_panel = gl.max(transition_flags, 1)
    additive_panel = gl.load(flag_base + 4 + (gl.program_id(1) * BM) // 32,
                             fc < chunks, 255)
    minimum_panel = additive_panel & 255
    maximum_panel = additive_panel >> 8
    previous_max = gl.gather(maximum_panel, gl.maximum(fc - 1, 0), 0)


    certified = ((previous_max < 255) & (transition_panel < 255)
                 & (minimum_panel > 1) & (maximum_panel < 255)
                 & (previous_max + transition_panel < minimum_panel + 96))
    if BITMAP_SCAN:
        gl.static_assert(FLAG_CHUNKS <= 32)
        failure_bits = gl.sum(gl.where(
            (fc > 0) & (fc < chunks) & ~certified,
            gl.full((FLAG_CHUNKS,), 0x80000000, gl.uint32, panel_layout) >> fc,
            0), 0)
    dense = False
    if LOW_YIELD_DENSE:
        successes = gl.sum(((fc > 0) & (fc < chunks) & certified).to(gl.int32), 0)
        dense = successes * 4 < chunks - 1
    if dense:
        state = gl.amd.cdna4.buffer_load(
            Maps + (chunk_base * H + h) * 32768 + 16384, offsets)
        for chunk in range(1, chunks):
            base = ((chunk_base + chunk) * H + h) * 32768
            additive = gl.amd.cdna4.buffer_load(Maps + base + 16384, offsets)
            transition = _load_transition(Maps + base, Flags, lm, False)
            state_operand = gl.convert_layout(state, gl.DotOperandLayout(0, lm, 1))
            state = gl.amd.cdna4.mfma(state_operand, transition, additive)
            if chunk + 1 < chunks:
                gl.amd.cdna4.buffer_store(
                    state.to(gl.bfloat16),
                    Incoming + ((chunk_base + chunk + 1) * H + h) * 16384,
                    packed_offsets)
    else:
        if BITMAP_SCAN:
            chunk = gl.minimum(gl.extra.libdevice.clz(failure_bits.to(gl.int32)), chunks)
        else:
            chunk = gl.min(gl.where(
                (fc > 0) & (fc < chunks) & ~certified, fc, chunks), 0)
        state = gl.amd.cdna4.buffer_load(
            Maps + ((chunk_base + chunk - 1) * H + h) * 32768 + 16384, offsets)
        index = gl.full((1,), chunk - 1, gl.int32, panel_layout)
        state_exponent = gl.sum(gl.gather(maximum_panel, index, 0), 0)
        while chunk < chunks:
            index = gl.full((1,), chunk, gl.int32, panel_layout)
            te = gl.sum(gl.gather(transition_panel, index, 0), 0)
            amin = gl.sum(gl.gather(minimum_panel, index, 0), 0)
            amax = gl.sum(gl.gather(maximum_panel, index, 0), 0)
            skip = ((state_exponent < 255) & (te < 255) & (amin > 1)
                    & (amax < 255) & (state_exponent + te < amin + 96))
            if skip:


                if BITMAP_SCAN:
                    remaining_bits = failure_bits & (0x7fffffff >> chunk)
                    next_chunk = gl.minimum(
                        gl.extra.libdevice.clz(remaining_bits.to(gl.int32)), chunks)
                else:
                    next_chunk = gl.min(gl.where(
                        (fc > chunk) & (fc < chunks) & ~certified, fc, chunks), 0)
                state = gl.amd.cdna4.buffer_load(
                    Maps + ((chunk_base + next_chunk - 1) * H + h) * 32768 + 16384,
                    offsets)
                index = gl.full((1,), next_chunk - 1, gl.int32, panel_layout)
                state_exponent = gl.sum(gl.gather(maximum_panel, index, 0), 0)
                chunk = next_chunk
            else:
                base = ((chunk_base + chunk) * H + h) * 32768
                additive = gl.amd.cdna4.buffer_load(Maps + base + 16384, offsets)
                transition = _load_transition(Maps + base, Flags, lm, False)
                state_operand = gl.convert_layout(state, gl.DotOperandLayout(0, lm, 1))
                state = gl.amd.cdna4.mfma(state_operand, transition, additive)
                if chunk + 1 < chunks:
                    gl.amd.cdna4.buffer_store(
                        state.to(gl.bfloat16),
                        Incoming + ((chunk_base + chunk + 1) * H + h) * 16384,
                        packed_offsets)
                    if CHUNK == 96:

                        finite = (state_exponent < 255) & (te < 255) & (amax < 255)
                        state_exponent = gl.where(finite, gl.minimum(255,
                            gl.maximum(state_exponent + te - 117, amax + 2)), 255)
                    else:
                        exponent = (state.to(gl.int32, bitcast=True) >> 23) & 255
                        state_exponent = gl.max(gl.max(exponent, 1), 0)
                chunk += 1
    gl.amd.cdna4.buffer_store(state, S + slot * SS + h * 16384, offsets)


@gluon.jit
def _load_output_state(Maps, Incoming, chunk_id, h, H: gl.constexpr, MFMA: gl.constexpr, PACKED_INCOMING: gl.constexpr):
    if PACKED_INCOMING:
        operand: gl.constexpr = gl.DotOperandLayout(1, MFMA, 8)
        native: gl.constexpr = gl.to_linear_layout(operand, [128, 128])
        bk = gl.arange(0, 128, gl.SliceLayout(1, native))
        bc = gl.arange(0, 128, gl.SliceLayout(0, native))
        offset = (bk[:, None] // 8) * 1024 + bc[None, :] * 8 + bk[:, None] % 8
        offset = gl.convert_layout(offset, operand, assert_trivial=True)
        b = gl.amd.cdna4.buffer_load(Incoming + (chunk_id * H + h) * 16384, offset)
    else:
        coalesced: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
        bk = gl.arange(0, 128, gl.SliceLayout(1, coalesced))
        bc = gl.arange(0, 128, gl.SliceLayout(0, coalesced))
        base = (chunk_id * H + h) * 32768 + 16384
        b = gl.load(Maps + base + bc[None, :] * 128 + bk[:, None]).to(gl.bfloat16)
    return gl.convert_layout(b, gl.DotOperandLayout(1, MFMA, 8))


@gluon.jit
def _output_panel(Coeff, CoeffFlags, Local, Maps, Incoming, Gate, W, Out,
                    start, end, chunk_id, chunk, h, slot, eps,
                    H: gl.constexpr, SG: gl.constexpr, LOCAL_STRIDE: gl.constexpr,
                    PACKED_INCOMING: gl.constexpr, FULL: gl.constexpr, BM: gl.constexpr,
                    DIRECT: gl.constexpr, SPARSE_COEFF: gl.constexpr, group_id,
                    IncomingFlags, PREFIX_TILES: gl.constexpr):
    gl.static_assert(not SPARSE_COEFF or DIRECT)
    if DIRECT:
        la: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        lm: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
        r = gl.arange(0, BM, gl.SliceLayout(1, la))
        c = gl.arange(0, 128, gl.SliceLayout(0, la))
        valid = FULL | (start + r[:, None] < end)
        acc = gl.zeros((BM, 128), gl.float32, la)
        if slot >= 0:
            acc = gl.amd.cdna4.buffer_load(
                Local + start * LOCAL_STRIDE + h * 128,
                r[:, None] * LOCAL_STRIDE + c[None, :], valid, 0)
        core = acc.to(gl.bfloat16)
        if (slot >= 0) & (chunk != 0):
            skip_correction = False
            if SPARSE_COEFF:
                flag_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
                packet_flag = gl.arange(0, BM // 16 * 4, flag_layout)
                packet = packet_flag // 4
                present_flags = gl.load(
                    CoeffFlags + ((group_id + packet) * H + h) * 4 + packet_flag % 4,
                    FULL | (start + packet * 16 < end), 0)
                if gl.max(present_flags, 0) == 0:
                    prefix_tile = gl.arange(0, PREFIX_TILES, flag_layout)
                    incoming_finite = gl.load(IncomingFlags + (chunk_id * H + h) * PREFIX_TILES + prefix_tile)
                    skip_correction = gl.min(incoming_finite, 0) != 0
            if not skip_correction:
                if SPARSE_COEFF:
                    present = gl.load(CoeffFlags + ((group_id + r[:, None] // 16) * H + h) * 4 + c[None, :] // 32,
                                      valid, 0) != 0
                else:
                    present = True
                a = gl.amd.cdna4.buffer_load(
                    Coeff + (start * H + h) * 128,
                    r[:, None] * H * 128 + c[None, :], valid & present, 0)
                a = gl.convert_layout(a, gl.DotOperandLayout(0, lm, 8))
                b = _load_output_state(Maps, Incoming, chunk_id, h, H, lm, PACKED_INCOMING)
                acc_mfma = gl.convert_layout(acc, lm)
                acc_mfma = gl.amd.cdna4.mfma(a, b, acc_mfma)
                core = gl.convert_layout(acc_mfma.to(gl.bfloat16), la)
        core = core.to(gl.float32)
    else:
        lm: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
        la: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        acc = gl.zeros((BM, 128), gl.float32, lm)
        if slot >= 0:
            lr = gl.arange(0, BM, gl.SliceLayout(1, la))
            lc = gl.arange(0, 128, gl.SliceLayout(0, la))
            local = gl.amd.cdna4.buffer_load(
                Local + start * LOCAL_STRIDE + h * 128,
                lr[:, None] * LOCAL_STRIDE + lc[None, :], start + lr[:, None] < end, 0)
            acc = gl.convert_layout(local, lm)
        first_chunk = chunk == 0
        if (slot >= 0) & ~first_chunk:
            ar = start + gl.arange(0, BM, gl.SliceLayout(1, la))
            ak = gl.arange(0, 128, gl.SliceLayout(0, la))
            a = gl.load(Coeff + (ar[:, None] * H + h) * 128 + ak[None, :], ar[:, None] < end, 0)
            a = gl.convert_layout(a, gl.DotOperandLayout(0, lm, 8))
            b = _load_output_state(Maps, Incoming, chunk_id, h, H, lm, PACKED_INCOMING)
            acc = gl.amd.cdna4.mfma(a, b, acc)
        core = gl.convert_layout(acc.to(gl.bfloat16), la).to(gl.float32)
        r = start + gl.arange(0, BM, gl.SliceLayout(1, la))
        c = gl.arange(0, 128, gl.SliceLayout(0, la))
    inv = gl.rsqrt(gl.sum(core * core, 1) / 128 + eps)
    weight = gl.load(W + c).to(gl.float32)
    gate = gl.amd.cdna4.buffer_load(
        Gate + start * SG + h * 128,
        (r[:, None] if DIRECT else r[:, None] - start) * SG + c[None, :],
        (valid if DIRECT else r[:, None] < end), 0).to(gl.float32)
    out = core * inv[:, None] * weight[None, :] * _sigmoid(gate)
    gl.amd.cdna4.buffer_store(
        out.to(Out.dtype.element_ty), Out + (start * H + h) * 128,
        (r[:, None] if DIRECT else r[:, None] - start) * H * 128 + c[None, :],
        (valid if DIRECT else r[:, None] < end))


@gluon.jit
def _normalize_output(Coeff, CoeffFlags, Local, Maps, Incoming, Gate, W, Out, IDX, CU, eps,
                      H: gl.constexpr, SI: gl.constexpr, SG: gl.constexpr,
                      CHUNK: gl.constexpr, BM: gl.constexpr, PACKED_INCOMING: gl.constexpr,
                      LOCAL_STRIDE: gl.constexpr, DIRECT: gl.constexpr,
                      HEAD_GROUP: gl.constexpr, SPARSE_COEFF: gl.constexpr,
                      IncomingFlags, PREFIX_TILES: gl.constexpr):
    if HEAD_GROUP > 1:
        panels: gl.constexpr = gl.cdiv(CHUNK, BM)
        linear = gl.program_id(1) * panels + gl.program_id(0)
        seq_head = linear // (panels * HEAD_GROUP) * HEAD_GROUP + linear % HEAD_GROUP
        panel = linear % (panels * HEAD_GROUP) // HEAD_GROUP
        seq = seq_head // H
        h = seq_head % H
        chunk = gl.program_id(2)
        first = gl.load(CU + seq)
        end = gl.load(CU + seq + 1)
        chunk_id = first // CHUNK + seq + chunk
        start = first + chunk * CHUNK + panel * BM
    else:
        seq = gl.program_id(1) // H
        first, start, end, chunk_id, chunk = _chunk_bounds(CU, seq, CHUNK)
        h = gl.program_id(1) % H
        start += gl.program_id(0) * BM
    slot = gl.load(IDX + seq * SI)
    if start >= end:
        return
    group_id = 0
    if SPARSE_COEFF:
        first = gl.load(CU + seq)
        group_id = first // 16 + seq + (start - first) // 16
    if DIRECT and start + BM <= end:
        _output_panel(Coeff, CoeffFlags, Local, Maps, Incoming, Gate, W, Out,
                      start, end, chunk_id, chunk, h, slot, eps,
                      H, SG, LOCAL_STRIDE, PACKED_INCOMING, True, BM, DIRECT, SPARSE_COEFF, group_id,
                      IncomingFlags, PREFIX_TILES)
    else:
        _output_panel(Coeff, CoeffFlags, Local, Maps, Incoming, Gate, W, Out,
                      start, end, chunk_id, chunk, h, slot, eps,
                      H, SG, LOCAL_STRIDE, PACKED_INCOMING, False, BM, DIRECT, SPARSE_COEFF, group_id,
                      IncomingFlags, PREFIX_TILES)


class _LaunchPolicy(NamedTuple):

    chunk: int
    prefix_rows: int
    project_rows: int
    output_rows: int
    direct_output: bool
    head_group: int
    blocked: bool
    zero_skip: bool
    sparse_maps: bool
    roundoff_skip: bool
    sparse_coeff: bool
    flag_chunks: int
    factor_unroll: int
    factor_prefetch: bool
    bulk_factors: bool
    bounded_recurrence: bool
    late_update_factor: bool
    mirror_prediction: bool
    early_decay: bool


def _launch_policy(m, sequences, heads):
    blocked = m > 257
    if not blocked:
        chunk = 64
    elif m <= 2048:
        chunk = 96
    elif m <= 4096:
        chunk = 192
    elif m <= 6144:
        chunk = 320
    else:
        chunk = 320 if sequences >= 8 else 384
    prefix_rows = 8 if sequences == 1 else 16
    if blocked and 4 <= sequences < 8:
        prefix_rows = 32
    if m == 4096 and sequences == 4:
        prefix_rows = 16
    sparse_maps = blocked and sequences >= 8
    roundoff_skip = (m, sequences) in ((1024, 1), (4096, 1), (4096, 4), (6144, 2))
    zero_skip = (roundoff_skip or sparse_maps or (m > 6144 and sequences == 1)
                 or (m == 4096 and sequences == 4))
    short_factors = m <= 1024 and sequences == 1
    output_rows = 32 if chunk == 96 or m == 4096 or (m, sequences) == (8192, 1) else 64
    head_group = 4 if blocked and sequences > 1 and heads % 4 == 0 else 1
    if m == 4096 and sequences == 1 and heads % 12 == 0:
        head_group = 12
    if (m, sequences) == (8192, 1) and heads % 4 == 0:
        head_group = 4
    return _LaunchPolicy(
        chunk=chunk,
        prefix_rows=prefix_rows,
        project_rows=64 if m >= 2048 and m % 64 == 0 else 32,
        output_rows=output_rows,
        direct_output=output_rows == 32 or (m == 8192 and sequences == 1),
        head_group=head_group,
        blocked=blocked,
        zero_skip=zero_skip,
        sparse_maps=sparse_maps,
        roundoff_skip=roundoff_skip,
        sparse_coeff=m == 8192 and sequences == 1,
        flag_chunks=triton.next_power_of_2(triton.cdiv(m, chunk)) if zero_skip and not sparse_maps else 0,
        factor_unroll=8 if short_factors else (16 if sequences >= 8 else 4),
        factor_prefetch=short_factors or (1024 < m <= 2048 and sequences == 2),
        bulk_factors=sequences >= 8,
        mirror_prediction=(m, sequences) in (
            (2048, 2), (4096, 1), (4096, 4), (6144, 2), (8192, 1), (8192, 8)
        ),
        late_update_factor=(m, sequences) in ((2048, 2), (4096, 1)),
        early_decay=(m, sequences) == (1024, 1),
        bounded_recurrence=(m, sequences) in (
            (2048, 2), (4096, 1), (4096, 4), (6144, 2), (8192, 1), (8192, 8)
        ),
    )


def fused_kda_prefill(
    qkv, gate, forget_a, beta, forget_weight, conv_weight, a_log, dt_bias,
    norm_weight, conv_state, state, state_indices, cu_seqlens, has_initial_state,
    *, lower_bound=-5.0, norm_eps=1e-5,
):

    m, heads, sequences = qkv.shape[0], state.shape[1], state_indices.numel()
    policy = _launch_policy(m, sequences, heads)
    chunk = policy.chunk
    max_chunks = triton.cdiv(m, chunk)
    map_slots = max_chunks + sequences - 1
    decay_elements = m * heads * 128
    if policy.blocked:
        decay_elements = max(decay_elements, map_slots * heads * 32768)
    prepared = torch.empty((m, 3 * heads * 128), dtype=torch.bfloat16, device=qkv.device)
    decay = torch.empty((decay_elements,), dtype=torch.float32, device=qkv.device)
    betas = torch.empty((m, heads), dtype=torch.float32, device=qkv.device)
    prepare_roles = 4 if policy.project_rows == 64 else 8
    _fused_prepare[(triton.cdiv(m, 32) * heads * prepare_roles,)](
        qkv, conv_weight, conv_state, state_indices, cu_seqlens, has_initial_state,
        beta, prepared, betas, forget_a, forget_weight, a_log, dt_bias, decay, lower_bound,
        heads, sequences, qkv.stride(0), beta.stride(0), state_indices.stride(0),
        *conv_state.stride(), m, forget_a.stride(0), forget_weight.stride(0), policy.project_rows,
        num_warps=4, enable_fp_fusion=False)


    if policy.blocked:
        maps = decay[:map_slots * heads * 32768].view(map_slots, heads, 256, 128)
    else:
        maps = torch.empty((map_slots, heads, 256, 128), dtype=torch.float32, device=qkv.device)
    if policy.roundoff_skip and chunk == 96:
        zero_flags = betas.view(torch.int32).view(-1)[:map_slots * heads * 8].view(map_slots, heads, 8)
    else:
        zero_flags = torch.empty((map_slots, heads, 8), dtype=torch.int32, device=qkv.device) if policy.zero_skip else None
    factor_slots = triton.cdiv(m, 16) + sequences - 1
    coeff_flags = torch.empty((factor_slots, heads, 4), dtype=torch.int32, device=qkv.device) if policy.sparse_coeff else None
    incoming_flags = (
        torch.empty((map_slots, heads, 128 // policy.prefix_rows), dtype=torch.int32, device=qkv.device)
        if policy.sparse_coeff else None
    )
    coeff = torch.empty((m, heads * 128), dtype=torch.bfloat16, device=qkv.device)
    if policy.blocked:
        local = prepared.view(torch.float32)[:, :heads * 128]
        out = coeff
        factor_elements = max(factor_slots * heads * 4224, map_slots * heads * 8192)
        factors = torch.empty((factor_elements,), dtype=torch.float32, device=qkv.device)
        incoming = (
            torch.empty((map_slots, heads, 16384), dtype=torch.bfloat16, device=qkv.device)
            if policy.roundoff_skip else factors.view(torch.bfloat16)
        )
        output_factors = torch.empty((factor_slots, heads, 2304), dtype=torch.bfloat16, device=qkv.device)
        _response_factors[(heads, factor_slots)](
            prepared, decay, betas, factors, output_factors, state_indices, cu_seqlens,
            heads, state_indices.stride(0), sequences, policy.bulk_factors,
            policy.factor_unroll, policy.factor_prefetch,
            num_warps=1, enable_fp_fusion=True)
        recurrence_grid = (sequences * heads, 8, max_chunks)
        _blocked_affine_chunks[recurrence_grid](
            prepared, factors, output_factors, maps, zero_flags, coeff_flags, coeff, local, state, has_initial_state, state_indices, cu_seqlens,
            qkv, conv_state, heads, state_indices.stride(0), state.stride(0),
            qkv.stride(0), *conv_state.stride(), chunk, local.stride(0),
            policy.zero_skip, policy.sparse_maps, policy.roundoff_skip, policy.bounded_recurrence, policy.sparse_coeff, policy.late_update_factor, policy.mirror_prediction, policy.early_decay, incoming, policy.roundoff_skip,
            MIRROR_UPDATE=(m, sequences) == (4096, 1),
            num_warps=1, enable_fp_fusion=True)
    else:
        incoming = maps
        local = torch.empty((m, heads * 128), dtype=torch.float32, device=qkv.device)
        out = prepared.view(-1)[:m * heads * 128].view(m, heads * 128)
        _serial_affine_chunks[(16, sequences * heads, max_chunks)](
            prepared, decay, betas, maps, coeff, local, state, has_initial_state, state_indices, cu_seqlens,
            qkv, conv_state, heads, state_indices.stride(0), state.stride(0),
            qkv.stride(0), *conv_state.stride(), chunk,
            num_warps=1, enable_fp_fusion=True)
    if policy.roundoff_skip:
        _published_prefix[(sequences * heads, 128 // policy.prefix_rows)](
            maps, incoming, zero_flags, state, state_indices, cu_seqlens,
            heads, state_indices.stride(0), state.stride(0), chunk,
            policy.prefix_rows, policy.flag_chunks,
            LOW_YIELD_DENSE=(m, sequences) in ((1024, 1), (6144, 2)),
            BITMAP_SCAN=(m, sequences) in ((4096, 1), (6144, 2)),
            num_warps=8)
    else:
        _chunk_prefix[(sequences * heads, 128 // policy.prefix_rows)](
            maps, incoming, zero_flags, state, state_indices, cu_seqlens,
            heads, state_indices.stride(0), state.stride(0), chunk, policy.prefix_rows, policy.blocked, policy.zero_skip, policy.sparse_maps,
            policy.flag_chunks, policy.roundoff_skip, incoming_flags, policy.sparse_coeff,
            COPY_INITIAL=(m, sequences) == (4096, 1),
            PROPAGATE_BOUND=(m, sequences) in ((1024, 1), (4096, 1)),
            OMIT_FINAL_EXP=(m, sequences) == (4096, 4),
            num_warps=8,
            waves_per_eu=6 if (m, sequences) == (2048, 2) else 0)
    output_rows = policy.output_rows
    _normalize_output[(triton.cdiv(chunk, output_rows), sequences * heads, max_chunks)](
        coeff, coeff_flags, local, maps, incoming, gate, norm_weight, out, state_indices, cu_seqlens, norm_eps,
        heads, state_indices.stride(0), gate.stride(0), chunk, output_rows, policy.blocked, local.stride(0),
        policy.direct_output, policy.head_group, policy.sparse_coeff,
        incoming_flags, 128 // policy.prefix_rows, num_warps=4)
    return out, conv_state, state
