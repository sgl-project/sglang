"""Generated Kimi-K3 causal attention/output-gate kernel for gfx950.

Source: OpenAI-Partners/artemis-kernel-integrations PR 17,
commit 35b249f7a551278946a81b7da1d58c286c41fb8f.
"""

# ruff: noqa
# fmt: off

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _prefix_tile(
    Q, K, V, G, Indptr, Out, scale, row_start,
    H: gl.constexpr, DQ: gl.constexpr, DV: gl.constexpr,
    BQ: gl.constexpr, BV: gl.constexpr,
):
    BM: gl.constexpr = 64
    sequence = gl.program_id(0)
    head = gl.program_id(1)
    start = gl.load(Indptr + sequence).to(gl.int32)
    length = (gl.load(Indptr + sequence + 1) - start).to(gl.int32)

    warps_m: gl.constexpr = min(gl.num_warps(), BM // 16)
    qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[warps_m, gl.num_warps() // warps_m],
    )
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True,
        warps_per_cta=[warps_m, gl.num_warps() // warps_m],
    )
    q_load: gl.constexpr = gl.to_linear_layout(
        gl.DotOperandLayout(0, qk_layout, 8), [BM, 128]
    )
    k_load: gl.constexpr = gl.BlockedLayout(
        [8, 1], [16, 4], [1, gl.num_warps()], [0, 1]
    )
    v_load: gl.constexpr = gl.BlockedLayout(
        [1, 8], [4, 16], [gl.num_warps(), 1], [1, 0]
    )

    qr = row_start + gl.arange(0, BM, gl.SliceLayout(1, q_load))
    qd = gl.arange(0, 128, gl.SliceLayout(0, q_load))
    q = gl.load(
        Q + (start + qr[:, None]) * H * DQ + head * DQ + qd[None, :],
        (qr[:, None] < length) & (qd[None, :] < DQ), 0,
    )
    q = gl.convert_layout(q, gl.DotOperandLayout(0, qk_layout, 8))
    tail_layout: gl.constexpr = gl.to_linear_layout(
        gl.DotOperandLayout(0, qk_layout, 8), [BM, 64]
    )
    tail_rows = row_start + gl.arange(0, BM, gl.SliceLayout(1, tail_layout))
    tail_dims = 128 + gl.arange(0, 64, gl.SliceLayout(0, tail_layout))
    q_tail = gl.load(
        Q + (start + tail_rows[:, None]) * H * DQ + head * DQ + tail_dims[None, :],
        (tail_rows[:, None] < length) & (tail_dims[None, :] < DQ), 0,
    )
    q_tail = gl.convert_layout(q_tail, gl.DotOperandLayout(0, qk_layout, 8))
    kd = gl.arange(0, BQ, gl.SliceLayout(1, k_load))
    kn = gl.arange(0, 16, gl.SliceLayout(0, k_load))
    vn = gl.arange(0, 16, gl.SliceLayout(1, v_load))
    vd = gl.arange(0, BV, gl.SliceLayout(0, v_load))
    rows = row_start + gl.arange(0, BM, gl.SliceLayout(1, qk_layout))
    cols = gl.arange(0, 16, gl.SliceLayout(0, qk_layout))

    k_smem = gl.allocate_shared_memory(
        gl.bfloat16, [BQ, 16], gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
    )
    v_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, BV],
        gl.SwizzledSharedLayout(8, 1, 8, order=[1, 0]),
    )
    maximum = gl.full((BM,), -float("inf"), gl.float32, gl.SliceLayout(1, qk_layout))
    denominator = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, qk_layout))
    accumulator = gl.full((BM, BV), 0, gl.float32, pv_layout)
    k_base = K + start * H * DQ + head * DQ
    v_base = V + start * H * DV + head * DV
    k_offsets = kn[None, :] * H * DQ + kd[:, None]
    v_offsets = vn[:, None] * H * DV + vd[None, :]
    k_values = gl.amd.cdna4.buffer_load(
        k_base, k_offsets,
        mask=(kn[None, :] < length) & (kd[:, None] < DQ), other=0,
    )
    v_values = gl.amd.cdna4.buffer_load(
        v_base, v_offsets,
        mask=(vn[:, None] < length) & (vd[None, :] < DV), other=0,
    )
    prefix_end = gl.minimum(row_start, length) // 16
    causal_end = gl.cdiv(gl.minimum(row_start + BM, length), 16)
    for stage in gl.static_range(2):
        begin = 0 if stage == 0 else prefix_end
        end = prefix_end if stage == 0 else causal_end
        for block in range(begin, end):
            k_smem.store(k_values)
            v_smem.store(v_values)
            next_column = (block + 1) * 16
            k_values = gl.amd.cdna4.buffer_load(
                k_base, next_column * H * DQ + k_offsets,
                mask=(next_column + kn[None, :] < length) & (kd[:, None] < DQ), other=0,
            )
            v_values = gl.amd.cdna4.buffer_load(
                v_base, next_column * H * DV + v_offsets,
                mask=(next_column + vn[:, None] < length) & (vd[None, :] < DV), other=0,
            )
            k = k_smem.slice(0, 128, dim=0).load(gl.DotOperandLayout(1, qk_layout, 8))
            k_tail = k_smem.slice(128, 64, dim=0).load(gl.DotOperandLayout(1, qk_layout, 8))
            v = v_smem.load(gl.DotOperandLayout(1, pv_layout, 4))
            scores = gl.amd.cdna4.mfma(q, k, gl.full((BM, 16), 0, gl.float32, qk_layout))
            scores = gl.amd.cdna4.mfma(q_tail, k_tail, scores) * scale
            if stage == 1:
                columns = block * 16 + cols
                scores = gl.where(
                    (columns[None, :] <= rows[:, None]) & (columns[None, :] < length),
                    scores, -float("inf"),
                )
            next_maximum = gl.maximum(maximum, gl.max(scores, 1))
            alpha = gl.exp(maximum - next_maximum)
            probabilities = gl.exp(scores - next_maximum[:, None])

            p = gl.convert_layout(
                probabilities.to(gl.bfloat16), gl.DotOperandLayout(0, pv_layout, 4)
            )
            a = gl.convert_layout(alpha, gl.SliceLayout(1, pv_layout))
            partial = gl.amd.cdna4.mfma(p, v, gl.full((BM, BV), 0, gl.float32, pv_layout))
            updated_accumulator = accumulator * a[:, None] + partial
            if stage == 1:
                active = rows // 16 >= block
                active_pv = gl.convert_layout(active, gl.SliceLayout(1, pv_layout))
                accumulator = gl.where(active_pv[:, None], updated_accumulator, accumulator)
                denominator = gl.where(
                    active, denominator * alpha + gl.sum(probabilities, 1), denominator
                )
                maximum = gl.where(active, next_maximum, maximum)
            else:
                accumulator = updated_accumulator
                denominator = denominator * alpha + gl.sum(probabilities, 1)
                maximum = next_maximum

    denom = gl.convert_layout(denominator, gl.SliceLayout(1, pv_layout))
    reciprocal = 1.0 / denom

    attention = (accumulator * reciprocal[:, None]).to(gl.bfloat16)
    output_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [16, 4], [gl.num_warps(), 1], [1, 0]
    )
    attention = gl.convert_layout(attention, output_layout).to(gl.float32)
    orows = row_start + gl.arange(0, BM, gl.SliceLayout(1, output_layout))
    odims = gl.arange(0, BV, gl.SliceLayout(0, output_layout))
    offsets = (start + orows[:, None]) * H * DV + head * DV + odims[None, :]
    valid = (orows[:, None] < length) & (odims[None, :] < DV)
    gate = gl.load(G + offsets, valid, 0).to(gl.float32)
    sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    gl.store(Out + offsets, (attention * sigmoid).to(gl.bfloat16), valid)


@gluon.jit
def _native_long_tile(
    Q, K, V, G, Indptr, Out, scale, row_start,
    H: gl.constexpr, DQ: gl.constexpr, DV: gl.constexpr,
    BV: gl.constexpr,
):
    sequence = gl.program_id(0)
    head = gl.program_id(1)
    start = gl.load(Indptr + sequence).to(gl.int32)
    length = (gl.load(Indptr + sequence + 1) - start).to(gl.int32)
    BM: gl.constexpr = 128
    matrix_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=True,
        warps_per_cta=[4, 1],
    )
    q_load: gl.constexpr = gl.to_linear_layout(
        gl.DotOperandLayout(0, matrix_layout, 8), [BM, 128]
    )
    qt_load: gl.constexpr = gl.to_linear_layout(
        gl.DotOperandLayout(0, matrix_layout, 8), [BM, 64]
    )
    k_load: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
    kt_load: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
    v_load: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    qr = row_start + gl.arange(0, BM, gl.SliceLayout(1, q_load))
    qd = gl.arange(0, 128, gl.SliceLayout(0, q_load))
    qtr = row_start + gl.arange(0, BM, gl.SliceLayout(1, qt_load))
    qtd = 128 + gl.arange(0, 64, gl.SliceLayout(0, qt_load))
    q_base = Q + start * H * DQ + head * DQ
    q = gl.amd.cdna4.buffer_load(
        q_base, qr[:, None] * H * DQ + qd[None, :],
        mask=(qr[:, None] < length) & (qd[None, :] < DQ), other=0,
    )
    qt = gl.amd.cdna4.buffer_load(
        q_base, qtr[:, None] * H * DQ + qtd[None, :],
        mask=(qtr[:, None] < length) & (qtd[None, :] < DQ), other=0,
    )
    q = gl.convert_layout(q, gl.DotOperandLayout(0, matrix_layout, 8))
    qt = gl.convert_layout(qt, gl.DotOperandLayout(0, matrix_layout, 8))
    kd = gl.arange(0, 128, gl.SliceLayout(1, k_load))
    kn = gl.arange(0, 32, gl.SliceLayout(0, k_load))
    ktd = 128 + gl.arange(0, 64, gl.SliceLayout(1, kt_load))
    ktn = gl.arange(0, 32, gl.SliceLayout(0, kt_load))
    vn = gl.arange(0, 32, gl.SliceLayout(1, v_load))
    vd = gl.arange(0, BV, gl.SliceLayout(0, v_load))
    k_smem = gl.allocate_shared_memory(
        gl.bfloat16, [128, 32], gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
    )

    kt_storage = gl.allocate_shared_memory(
        gl.bfloat16, [128, 32], gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
    )
    kt_smem = kt_storage.slice(0, 64, dim=0)
    v_smem = gl.allocate_shared_memory(
        gl.bfloat16, [32, BV], gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    )
    rows = row_start + gl.arange(0, BM, gl.SliceLayout(1, matrix_layout))
    cols = gl.arange(0, 16, gl.SliceLayout(0, matrix_layout))
    maximum = gl.full((BM,), -float("inf"), gl.float32, gl.SliceLayout(1, matrix_layout))
    denominator = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, matrix_layout))
    accumulator = gl.full((BM, BV), 0, gl.float32, matrix_layout)
    k_base = K + start * H * DQ + head * DQ
    v_base = V + start * H * DV + head * DV
    k_values = gl.amd.cdna4.buffer_load(
        k_base, kn[None, :] * H * DQ + kd[:, None],
        mask=(kn[None, :] < length) & (kd[:, None] < DQ), other=0,
    )
    kt_values = gl.amd.cdna4.buffer_load(
        k_base, ktn[None, :] * H * DQ + ktd[:, None],
        mask=(ktn[None, :] < length) & (ktd[:, None] < DQ), other=0,
    )
    v_values = gl.amd.cdna4.buffer_load(
        v_base, vn[:, None] * H * DV + vd[None, :],
        mask=(vn[:, None] < length) & (vd[None, :] < DV), other=0,
    )
    prefix_end = gl.minimum(row_start, length) // 32
    causal_end = gl.cdiv(gl.minimum(row_start + BM, length), 32)
    original_end = gl.cdiv(gl.minimum(row_start + BM, length), 16)
    for stage in gl.static_range(2):
        begin = 0 if stage == 0 else prefix_end
        end = prefix_end if stage == 0 else causal_end
        for group in range(begin, end):
            k_smem.store(k_values)
            kt_smem.store(kt_values)
            v_smem.store(v_values)
            next_column = (group + 1) * 32
            k_values = gl.amd.cdna4.buffer_load(
                k_base, (next_column + kn[None, :]) * H * DQ + kd[:, None],
                mask=(next_column + kn[None, :] < length) & (kd[:, None] < DQ), other=0,
            )
            k = k_smem.load(gl.DotOperandLayout(1, matrix_layout, 8))
            kt = kt_smem.load(gl.DotOperandLayout(1, matrix_layout, 8))
            scores = gl.amd.cdna4.mfma(q, k, gl.full((BM, 32), 0, gl.float32, matrix_layout))
            scores = gl.amd.cdna4.mfma(qt, kt, scores) * scale
            kt_values = gl.amd.cdna4.buffer_load(
                k_base, (next_column + ktn[None, :]) * H * DQ + ktd[:, None],
                mask=(next_column + ktn[None, :] < length) & (ktd[:, None] < DQ), other=0,
            )
            v_values = gl.amd.cdna4.buffer_load(
                v_base, (next_column + vn[:, None]) * H * DV + vd[None, :],
                mask=(next_column + vn[:, None] < length) & (vd[None, :] < DV), other=0,
            )
            scores = gl.convert_layout(scores, gl.to_linear_layout(matrix_layout, [BM, 32]))
            first, second = scores.reshape((BM, 2, 16)).permute((0, 2, 1)).split()
            for subtile in gl.static_range(2):
                block = group * 2 + subtile
                if stage == 0 or block < original_end:
                    score = first if subtile == 0 else second
                    score = gl.convert_layout(score, matrix_layout)
                    if stage == 1:
                        columns = block * 16 + cols
                        score = gl.where(
                            (columns[None, :] <= rows[:, None]) & (columns[None, :] < length),
                            score, -float("inf"),
                        )
                    next_maximum = gl.maximum(maximum, gl.max(score, 1))
                    alpha = gl.exp(maximum - next_maximum)
                    probabilities = gl.exp(score - next_maximum[:, None])
                    p = gl.convert_layout(
                        probabilities.to(gl.bfloat16), gl.DotOperandLayout(0, matrix_layout, 4)
                    )
                    v = v_smem.slice(subtile * 16, 16, dim=0).load(gl.DotOperandLayout(1, matrix_layout, 4))
                    updated = gl.amd.cdna4.mfma(p, v, accumulator * alpha[:, None])
                    if stage == 1:
                        active = rows // 16 >= block
                        accumulator = gl.where(active[:, None], updated, accumulator)
                        denominator = gl.where(
                            active, denominator * alpha + gl.sum(probabilities, 1), denominator
                        )
                        maximum = gl.where(active, next_maximum, maximum)
                    else:
                        accumulator = updated
                        denominator = denominator * alpha + gl.sum(probabilities, 1)
                        maximum = next_maximum
    attention = (accumulator * (1.0 / denominator)[:, None]).to(gl.bfloat16)
    out_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    attention = gl.convert_layout(attention, out_layout).to(gl.float32)
    out_rows = row_start + gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    out_dims = gl.arange(0, BV, gl.SliceLayout(0, out_layout))
    offsets = (start + out_rows[:, None]) * H * DV + head * DV + out_dims[None, :]
    valid = (out_rows[:, None] < length) & (out_dims[None, :] < DV)
    gate = gl.load(G + offsets, valid, 0).to(gl.float32)
    sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    gl.store(Out + offsets, (attention * sigmoid).to(gl.bfloat16), valid)


@gluon.jit
def _compact_causal_gate(
    Q, K, V, G, Indptr, Out, scale,
    H: gl.constexpr, DQ: gl.constexpr, DV: gl.constexpr,
    BV: gl.constexpr, BM: gl.constexpr,
    GROUP: gl.constexpr, TAIL_VEC: gl.constexpr,
    OUT_VEC: gl.constexpr, OUT_M: gl.constexpr,
    COMPACT: gl.constexpr, SEQUENCES: gl.constexpr,
):
    head = gl.program_id(1)
    tile = gl.num_programs(2) - 1 - gl.program_id(2)
    if COMPACT:

        start = 0
        length = 0
        row_start = 0
        tile_base = 0
        for sequence in gl.static_range(SEQUENCES):
            sequence_start = gl.load(Indptr + sequence).to(gl.int32)
            sequence_end = gl.load(Indptr + sequence + 1).to(gl.int32)
            sequence_length = sequence_end - sequence_start
            next_base = tile_base + gl.cdiv(sequence_length, BM)
            owns_tile = (tile >= tile_base) & (tile < next_base)
            start = gl.where(owns_tile, sequence_start, start)
            length = gl.where(owns_tile, sequence_length, length)
            row_start = gl.where(owns_tile, (tile - tile_base) * BM, row_start)
            tile_base = next_base
    else:
        sequence = gl.program_id(0)
        row_start = tile * BM
        start = gl.load(Indptr + sequence).to(gl.int32)
        length = (gl.load(Indptr + sequence + 1) - start).to(gl.int32)


    if row_start < length:
        warps_m: gl.constexpr = min(gl.num_warps(), BM // 16)
        qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 32], transposed=True,
            warps_per_cta=[warps_m, gl.num_warps() // warps_m],
        )
        pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 16], transposed=True,
            warps_per_cta=[warps_m, gl.num_warps() // warps_m],
        )
        q_load: gl.constexpr = gl.to_linear_layout(
            gl.DotOperandLayout(0, qk_layout, 8), [BM, 128]
        )
        k_load: gl.constexpr = gl.BlockedLayout(
            [8, 1], [16, 4], [1, gl.num_warps()], [0, 1]
        )
        v_load: gl.constexpr = gl.BlockedLayout(
            [1, 8], [4, 16], [gl.num_warps(), 1], [1, 0]
        )
        qr = row_start + gl.arange(0, BM, gl.SliceLayout(1, q_load))
        qd = gl.arange(0, 128, gl.SliceLayout(0, q_load))
        q = gl.load(
            Q + (start + qr[:, None]) * H * DQ + head * DQ + qd[None, :],
            (qr[:, None] < length) & (qd[None, :] < DQ), 0,
        )
        q = gl.convert_layout(q, gl.DotOperandLayout(0, qk_layout, 8))
        tail_layout: gl.constexpr = gl.to_linear_layout(
            gl.DotOperandLayout(0, qk_layout, 8), [BM, 64]
        )
        tail_rows = row_start + gl.arange(0, BM, gl.SliceLayout(1, tail_layout))
        tail_dims = 128 + gl.arange(0, 64, gl.SliceLayout(0, tail_layout))
        q_tail = gl.load(
            Q + (start + tail_rows[:, None]) * H * DQ + head * DQ + tail_dims[None, :],
            (tail_rows[:, None] < length) & (tail_dims[None, :] < DQ), 0,
        )
        q_tail = gl.convert_layout(q_tail, gl.DotOperandLayout(0, qk_layout, 8))
        kt_load: gl.constexpr = gl.BlockedLayout(
            [TAIL_VEC, 1], [16, 4], [1, gl.num_warps()], [0, 1]
        )
        kd = gl.arange(0, 128, gl.SliceLayout(1, k_load))
        ktd = 128 + gl.arange(0, 64, gl.SliceLayout(1, kt_load))
        ktn = gl.arange(0, GROUP, gl.SliceLayout(0, kt_load))
        kn = gl.arange(0, GROUP, gl.SliceLayout(0, k_load))
        vn = gl.arange(0, GROUP, gl.SliceLayout(1, v_load))
        vd = gl.arange(0, BV, gl.SliceLayout(0, v_load))
        rows = row_start + gl.arange(0, BM, gl.SliceLayout(1, qk_layout))
        cols = gl.arange(0, 16, gl.SliceLayout(0, qk_layout))
        k_smem = gl.allocate_shared_memory(
            gl.bfloat16, [128, GROUP], gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
        )
        kt_smem = gl.allocate_shared_memory(
            gl.bfloat16, [64, GROUP], gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
        )
        v_smem = gl.allocate_shared_memory(
            gl.bfloat16, [GROUP, BV], gl.SwizzledSharedLayout(8, 1, 8, order=[1, 0])
        )
        maximum = gl.full((BM,), -float("inf"), gl.float32, gl.SliceLayout(1, qk_layout))
        denominator = gl.full((BM,), 0, gl.float32, gl.SliceLayout(1, qk_layout))
        accumulator = gl.full((BM, BV), 0, gl.float32, pv_layout)
        k_base = K + start * H * DQ + head * DQ
        v_base = V + start * H * DV + head * DV
        k_offsets = gl.minimum(kn[None, :], length - 1) * H * DQ + kd[:, None]
        kt_offsets = gl.minimum(ktn[None, :], length - 1) * H * DQ + ktd[:, None]
        v_offsets = vn[:, None] * H * DV + vd[None, :]
        k_values = gl.amd.cdna4.buffer_load(k_base, k_offsets, mask=kd[:, None] < DQ, other=0)
        if BM == 32:
            v_values = gl.amd.cdna4.buffer_load(
                v_base, v_offsets,
                mask=(vn[:, None] < length) & (vd[None, :] < DV), other=0,
            )
        kt_values = gl.amd.cdna4.buffer_load(k_base, kt_offsets, mask=ktd[:, None] < DQ, other=0)
        if BM != 32:
            v_values = gl.amd.cdna4.buffer_load(
                v_base, v_offsets,
                mask=(vn[:, None] < length) & (vd[None, :] < DV), other=0,
            )
        prefix_end = gl.minimum(row_start, length) // GROUP
        causal_end = gl.cdiv(gl.minimum(row_start + BM, length), GROUP)
        original_end = gl.cdiv(gl.minimum(row_start + BM, length), 16)
        for stage in gl.static_range(2):
            begin = 0 if stage == 0 else prefix_end
            end = prefix_end if stage == 0 else causal_end
            if stage == 1 and BM == 32:
                end = begin + 1
            for group in range(begin, end):
                k_smem.store(k_values)
                kt_smem.store(kt_values)
                v_smem.store(v_values)
                if stage == 0 or BM != 32:
                    next_column = (group + 1) * GROUP
                    k_values = gl.amd.cdna4.buffer_load(
                        k_base, gl.minimum(next_column + kn[None, :], length - 1) * H * DQ + kd[:, None],
                        mask=kd[:, None] < DQ, other=0,
                    )
                    if BM == 32:
                        v_values = gl.amd.cdna4.buffer_load(
                            v_base, next_column * H * DV + v_offsets,
                            mask=(next_column + vn[:, None] < length) & (vd[None, :] < DV), other=0,
                        )
                    kt_values = gl.amd.cdna4.buffer_load(
                        k_base, gl.minimum(next_column + ktn[None, :], length - 1) * H * DQ + ktd[:, None],
                        mask=ktd[:, None] < DQ, other=0,
                    )
                    if BM != 32:
                        v_values = gl.amd.cdna4.buffer_load(
                            v_base, next_column * H * DV + v_offsets,
                            mask=(next_column + vn[:, None] < length) & (vd[None, :] < DV), other=0,
                        )
                if BM == 32:
                    k0 = k_smem.slice(0, 16, dim=1).load(gl.DotOperandLayout(1, qk_layout, 8))
                    k1 = k_smem.slice(16, 16, dim=1).load(gl.DotOperandLayout(1, qk_layout, 8))
                    kt0 = kt_smem.slice(0, 16, dim=1).load(gl.DotOperandLayout(1, qk_layout, 8))
                    kt1 = kt_smem.slice(16, 16, dim=1).load(gl.DotOperandLayout(1, qk_layout, 8))
                    score0 = gl.amd.cdna4.mfma(q, k0, gl.full((BM, 16), 0, gl.float32, qk_layout))
                    score1 = gl.amd.cdna4.mfma(q, k1, gl.full((BM, 16), 0, gl.float32, qk_layout))
                    score0 = gl.amd.cdna4.mfma(q_tail, kt0, score0) * scale
                    score1 = gl.amd.cdna4.mfma(q_tail, kt1, score1) * scale

                for subtile in gl.static_range(GROUP // 16):
                    block = group * (GROUP // 16) + subtile
                    if stage == 0 or block < original_end:
                        if BM == 32:
                            v = v_smem.slice(subtile * 16, 16, dim=0).load(gl.DotOperandLayout(1, pv_layout, 4))
                            scores = score0 if subtile == 0 else score1
                        else:
                            k = k_smem.slice(subtile * 16, 16, dim=1).load(gl.DotOperandLayout(1, qk_layout, 8))
                            k_tail = kt_smem.slice(subtile * 16, 16, dim=1).load(gl.DotOperandLayout(1, qk_layout, 8))
                            v = v_smem.slice(subtile * 16, 16, dim=0).load(gl.DotOperandLayout(1, pv_layout, 4))
                            scores = gl.amd.cdna4.mfma(q, k, gl.full((BM, 16), 0, gl.float32, qk_layout))
                            scores = gl.amd.cdna4.mfma(q_tail, k_tail, scores) * scale
                        if stage == 1:
                            columns = block * 16 + cols
                            scores = gl.where(
                                (columns[None, :] <= rows[:, None]) & (columns[None, :] < length),
                                scores, -float("inf"),
                            )
                        next_maximum = gl.maximum(maximum, gl.max(scores, 1))
                        alpha = gl.exp(maximum - next_maximum)
                        probabilities = gl.exp(scores - next_maximum[:, None])

                        p = gl.convert_layout(
                            probabilities.to(gl.bfloat16), gl.DotOperandLayout(0, pv_layout, 4)
                        )
                        a = gl.convert_layout(alpha, gl.SliceLayout(1, pv_layout))
                        updated_accumulator = gl.amd.cdna4.mfma(p, v, accumulator * a[:, None])
                        if stage == 1:
                            active = rows // 16 >= block
                            active_pv = gl.convert_layout(active, gl.SliceLayout(1, pv_layout))
                            accumulator = gl.where(active_pv[:, None], updated_accumulator, accumulator)
                            denominator = gl.where(
                                active, denominator * alpha + gl.sum(probabilities, 1), denominator
                            )
                            maximum = gl.where(active, next_maximum, maximum)
                        else:
                            accumulator = updated_accumulator
                            denominator = denominator * alpha + gl.sum(probabilities, 1)
                            maximum = next_maximum

        denom = gl.convert_layout(denominator, gl.SliceLayout(1, pv_layout))
        reciprocal = 1.0 / denom

        attention = (accumulator * reciprocal[:, None]).to(gl.bfloat16)
        output_layout: gl.constexpr = gl.BlockedLayout(
            [1, OUT_VEC], [OUT_M, 64 // OUT_M], [gl.num_warps(), 1], [1, 0]
        )
        attention = gl.convert_layout(attention, output_layout).to(gl.float32)
        orows = row_start + gl.arange(0, BM, gl.SliceLayout(1, output_layout))
        odims = gl.arange(0, BV, gl.SliceLayout(0, output_layout))
        offsets = (start + orows[:, None]) * H * DV + head * DV + odims[None, :]
        valid = (orows[:, None] < length) & (odims[None, :] < DV)
        gate = gl.load(G + offsets, valid, 0).to(gl.float32)
        sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        gl.store(Out + offsets, (attention * sigmoid).to(gl.bfloat16), valid)


@gluon.jit
def _mixed_causal_gate(
    Q, K, V, G, Indptr, Out, scale,
    H: gl.constexpr, DQ: gl.constexpr, DV: gl.constexpr,
    BQ: gl.constexpr, BV: gl.constexpr,
    LONG_TILES: gl.constexpr, SPLIT: gl.constexpr,
):

    tile = gl.program_id(2)
    if tile < LONG_TILES:
        row_start = SPLIT + (LONG_TILES - 1 - tile) * 128
        _native_long_tile(
            Q, K, V, G, Indptr, Out, scale, row_start, H, DQ, DV, BV,
        )
    else:
        row_start = (LONG_TILES + SPLIT // 64 - 1 - tile) * 64
        _prefix_tile(
            Q, K, V, G, Indptr, Out, scale, row_start, H, DQ, DV, BQ, BV,
        )


def kimi_causal_attention_gate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    qo_indptr: torch.Tensor,
    *,
    scale: float,
    max_query_len: int,
) -> torch.Tensor:

    m, h, dq = query.shape
    dv = value.shape[-1]
    output = torch.empty((m, h * dv), device=query.device, dtype=query.dtype)
    sequence_count = qo_indptr.numel() - 1
    if sequence_count == 1 and max_query_len > 4096:
        split = 512
        long_tiles = triton.cdiv(max_query_len - split, 128)
        _mixed_causal_gate[(sequence_count, h, long_tiles + split // 64)](
            query, key, value, gate, qo_indptr, output, scale,
            H=h, DQ=dq, DV=dv, BQ=triton.next_power_of_2(dq),
            BV=triton.next_power_of_2(dv), LONG_TILES=long_tiles, SPLIT=split,
            num_warps=4, num_stages=1, waves_per_eu=2,
        )
    else:
        block_m = 32 if m <= 2048 else 64
        group = 32 if m <= 2048 or sequence_count == 1 else 16
        direct_tiles = sequence_count * triton.cdiv(max_query_len, block_m)

        compact_tiles = triton.cdiv(m, block_m) + sequence_count - 1
        compact_grid = 4 * direct_tiles > 7 * compact_tiles
        grid = (
            1 if compact_grid else sequence_count,
            h,
            compact_tiles if compact_grid else triton.cdiv(max_query_len, block_m),
        )
        _compact_causal_gate[grid](
            query, key, value, gate, qo_indptr, output, scale,
            H=h, DQ=dq, DV=dv, BV=triton.next_power_of_2(dv),
            BM=block_m, GROUP=group,
            TAIL_VEC=4 if block_m == 32 or sequence_count > 4 else 8,
            COMPACT=compact_grid, SEQUENCES=sequence_count,
            OUT_VEC=4 if group == 32 else 8,
            OUT_M=8 if block_m == 32 else 16,
            num_warps=4, num_stages=1,
            waves_per_eu=1 if block_m == 32 else (2 if group == 32 else 3),
        )
    return output
