"""Generated Kimi-K3 MLA value projection and gate kernels for gfx950.

Source: OpenAI-Partners/artemis-kernel-integrations PR 17,
commit 35b249f7a551278946a81b7da1d58c286c41fb8f.
"""

# ruff: noqa
# fmt: off

"""Selected mla vc output gate schedules and shared helpers."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.runtime.jit import constexpr_function


@gluon.jit
def _m1_rounded_sigmoid(gate):

    return (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _m1_single_row_gate(
    X, W, G, Y,
    H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
):

    head = gl.program_id(0)
    col_start = gl.program_id(2) * 8
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[1, 1],
    )
    la: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    lb: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    vr = gl.arange(0, 16, gl.SliceLayout(1, la))
    ka = gl.arange(0, 256, gl.SliceLayout(0, la))
    kb = gl.arange(0, 256, gl.SliceLayout(1, lb))
    vc = gl.arange(0, 16, gl.SliceLayout(0, lb))
    axk = (ka[None, :] // 8 * 2 + vr[:, None] % 2) * 8 + ka[None, :] % 8
    bwk = (kb[:, None] // 8 * 2 + vc[None, :] % 2) * 8 + kb[:, None] % 8
    cols = col_start + vc // 2
    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [1, 1], [1, 0])
    rr = gl.arange(0, 8, gl.SliceLayout(1, out_layout))
    cc = col_start + gl.arange(0, 8, gl.SliceLayout(0, out_layout))
    offsets = rr[:, None] * H * N + cc[None, :]
    valid = rr[:, None] < 1
    gate = gl.amd.cdna4.buffer_load(G + head * N, offsets, valid, 0).to(gl.float32)
    sigmoid = _m1_rounded_sigmoid(gate)
    a = gl.load(X + head * K + axk)
    b = gl.load(W + head * WH + bwk * WK + cols[None, :] * WN)
    acc = gl.amd.cdna4.mfma(a, b, gl.zeros((16, 16), gl.float32, mma))
    ir = gl.arange(0, 16, gl.SliceLayout(1, mma))
    ic = gl.arange(0, 16, gl.SliceLayout(0, mma))
    diagonal = gl.where(ir[:, None] % 2 == ic[None, :] % 2, acc, 0.0)
    partial = gl.sum(diagonal.reshape((16, 8, 2)), 2)
    value = gl.sum(partial.reshape((8, 2, 8)), 1)
    value = gl.convert_layout(value, out_layout).to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store((value * sigmoid).to(gl.bfloat16), Y + head * N, offsets, valid)


@gluon.jit
def _m1_general_gate(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
):
    head = gl.program_id(0).to(gl.int64)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[1, 1],
    )
    la: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    lb: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    rows = gl.program_id(1) * 16 + gl.arange(0, 16, gl.SliceLayout(1, la))
    cols = gl.program_id(2) * 16 + gl.arange(0, 16, gl.SliceLayout(0, lb))
    rows = rows.to(gl.int64)
    cols = cols.to(gl.int64)
    ka = gl.arange(0, 32, gl.SliceLayout(0, la))
    kb = gl.arange(0, 32, gl.SliceLayout(1, lb))
    acc = gl.zeros((16, 16), gl.float32, mma)
    for start in range(gl.cdiv(K, 32)):
        ak = start * 32 + ka
        bk = (start * 32 + kb).to(gl.int64)
        a = gl.load(X + rows[:, None] * H * K + head * K + ak[None, :],
                    (rows[:, None] < M) & (ak[None, :] < K), 0)
        b = gl.load(W + head * WH + bk[:, None] * WK + cols[None, :] * WN,
                    (bk[:, None] < K) & (cols[None, :] < N), 0)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    rr = gl.program_id(1) * 16 + gl.arange(0, 16, gl.SliceLayout(1, mma))
    cc = gl.program_id(2) * 16 + gl.arange(0, 16, gl.SliceLayout(0, mma))
    rr = rr.to(gl.int64)
    cc = cc.to(gl.int64)
    offset = rr[:, None] * H * N + head * N + cc[None, :]
    mask = (rr[:, None] < M) & (cc[None, :] < N)
    gate = gl.load(G + offset, mask, 0).to(gl.float32)
    sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    reconstructed = acc.to(gl.bfloat16).to(gl.float32)
    gl.store(Y + offset, (reconstructed * sigmoid).to(gl.bfloat16), mask)


def mla_vc_output_gate_m1(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:

    assert latent.ndim == weight.ndim == 3
    m, h, k = latent.shape
    assert m > 0 and h > 0 and k > 0 and weight.shape[:2] == (h, k)
    n = weight.shape[2]
    assert n > 0 and gate.shape == (m, h * n)
    assert latent.dtype is weight.dtype is gate.dtype is torch.bfloat16
    assert latent.device == weight.device == gate.device
    assert latent.is_contiguous() and gate.is_contiguous()
    assert all(stride > 0 for stride in weight.stride())
    wh, wk, wn = weight.stride()
    output = torch.empty((m, h * n), device=latent.device, dtype=latent.dtype)


    offset_bound = max(
        latent.storage_offset() + max(m, 8) * h * k,
        gate.storage_offset() + max(m, 8) * h * n,
        weight.storage_offset() + (h - 1) * wh + (k - 1) * wk + (n - 1) * wn + 1,
        wh, wk, wn,
    )
    if k == 512 and n % 8 == 0 and m in (1, 2, 4, 8, 16) and offset_bound < 2**29:
        _m1_single_row_gate[(h, 1, n // 8)](
            latent, weight, gate, output, h, k, n, wh, wk, wn,
            num_warps=1, enable_fp_fusion=False,
        )
    else:
        _m1_general_gate[(h, triton.cdiv(m, 16), triton.cdiv(n, 16))](
            latent, weight, gate, output, m, h, k, n, wh, wk, wn,
            num_warps=1, enable_fp_fusion=False,
        )
    return output


@gluon.jit
def _m1024_8192_rounded_sigmoid(gate):

    exponential = gl.exp2(gate.to(gl.float32) * -1.4426950408889634)
    return (1.0 / (1.0 + exponential)).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _m1024_8192_stage_operands(a_reg, b_reg, a_shared, b_shared,
                    dot_a: gl.constexpr, dot_b: gl.constexpr):

    a_shared.store(a_reg)
    b_shared.store(b_reg)
    return a_shared.load(dot_a), b_shared.load(dot_b)


@gluon.jit
def _m1024_8192_store_gated_panel(value, gate, output, offsets, valid=None,
                       layout: gl.constexpr = None):

    sigmoid = _m1024_8192_rounded_sigmoid(gate)
    if layout is not None:
        value = gl.convert_layout(value, layout)
    result = (value.to(gl.float32) * sigmoid).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(result, output, offsets, valid, cache=".wt")


@gluon.jit
def _m1024_8192_reconstruct_and_gate_64_body(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
    WIDE: gl.constexpr, FULL_ROWS: gl.constexpr,
):

    BM: gl.constexpr = 64
    BN: gl.constexpr = 128 if WIDE else 64
    BK: gl.constexpr = 64 if WIDE else 128
    NUM_WARPS: gl.constexpr = 8 if WIDE else 4
    head = gl.program_id(0)
    row_tile = gl.program_id(1)
    col_tile = gl.program_id(2)
    if not WIDE:
        head = head.to(gl.uint32)
        row_tile = row_tile.to(gl.uint32)
        col_tile = col_tile.to(gl.uint32)
    a_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BK, BK // 8], [NUM_WARPS, 1], [1, 0]
    )
    b_layout: gl.constexpr = gl.BlockedLayout(
        [8, 1], [BK // 8, 512 // BK], [1, NUM_WARPS], [0, 1]
    )
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32],
        transposed=True, warps_per_cta=[2, NUM_WARPS // 2],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    local_rows = gl.arange(0, BM, gl.SliceLayout(1, a_layout))
    local_cols = gl.arange(0, BN, gl.SliceLayout(0, b_layout))
    a_k = gl.arange(0, BK, gl.SliceLayout(0, a_layout))
    b_k = gl.arange(0, BK, gl.SliceLayout(1, b_layout))
    rows = row_tile * BM + local_rows
    cols = col_tile * BN + local_cols
    if not WIDE:
        a_base = X + (row_tile * BM * H + head) * K
    else:
        a_base = X + head * K + row_tile * BM * H * K
    b_base = W + head * WH + col_tile * BN * WN
    a_offsets = local_rows[:, None] * (H * K) + a_k[None, :]
    b_offsets = b_k[:, None] * WK + local_cols[None, :] * WN
    a_valid = FULL_ROWS | (rows[:, None] < M)
    b_valid = (N % BN == 0) | (cols[None, :] < N)
    out_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BN, BN // 8], [NUM_WARPS, 1], [1, 0]
    )
    epilogue_rows: gl.constexpr = 32 if WIDE else 64
    out_rows = row_tile * BM + gl.arange(0, epilogue_rows, gl.SliceLayout(1, out_layout))
    out_cols = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, out_layout))
    offsets = out_rows[:, None] * (H * N) + head * N + out_cols[None, :]
    out_valid = (FULL_ROWS | (out_rows[:, None] < M)) & (
        (N % BN == 0) | (out_cols[None, :] < N)
    )
    if not WIDE:
        a_shared_layout: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
            dot_a, (BM, BK), X.dtype.element_ty
        )
        b_shared_layout: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
            dot_b, (BK, BN), W.dtype.element_ty
        )
    else:
        a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, [1, 0])
        b_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, [0, 1])
    a_shared = gl.allocate_shared_memory(X.dtype.element_ty, (BM, BK), a_shared_layout)
    b_shared = gl.allocate_shared_memory(W.dtype.element_ty, (BK, BN), b_shared_layout)
    if not WIDE:
        a_reg = gl.load(a_base + a_offsets, a_valid & (a_k[None, :] < K), other=0)
    else:
        a_reg = gl.amd.cdna4.buffer_load(
            a_base, a_offsets, a_valid & (a_k[None, :] < K), other=0,
        )
    acc = gl.zeros((BM, BN), gl.float32, mma_layout)
    for start in gl.static_range(gl.cdiv(K, BK) - WIDE):
        b_reg = gl.amd.cdna4.buffer_load(
            b_base + start * BK * WK * (not WIDE),
            b_offsets + start * BK * WK * WIDE,
            b_valid & ((K % BK == 0) | (b_k[:, None] + start * BK < K)), other=0,
        )
        a, b = _m1024_8192_stage_operands(a_reg, b_reg, a_shared, b_shared, dot_a, dot_b)
        if start + 1 < gl.cdiv(K, BK):
            if not WIDE:
                a_reg = gl.load(
                    a_base + a_offsets + (start + 1) * BK,
                    a_valid & ((K % BK == 0) | (a_k[None, :] + (start + 1) * BK < K)),
                    other=0,
                )
            else:
                a_reg = gl.amd.cdna4.buffer_load(
                    a_base, a_offsets + (start + 1) * BK,
                    a_valid & ((K % BK == 0) | (a_k[None, :] + (start + 1) * BK < K)),
                    other=0,
                )
        if not WIDE and start == gl.cdiv(K, BK) - 2:
            gate = gl.load(G + offsets, out_valid, other=0).to(gl.float32)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    if WIDE:
        last: gl.constexpr = (gl.cdiv(K, BK) - 1) * BK
        b_reg = gl.amd.cdna4.buffer_load(
            b_base, b_offsets + last * WK,
            b_valid & ((K % BK == 0) | (b_k[:, None] + last < K)), other=0,
        )
        a, b = _m1024_8192_stage_operands(a_reg, b_reg, a_shared, b_shared, dot_a, dot_b)
        gate = gl.amd.cdna4.buffer_load(G, offsets, out_valid, other=0)
        acc = gl.amd.cdna4.mfma(a, b, acc)
        value = gl.convert_layout(acc.to(gl.bfloat16), out_layout)
        panel0 = gl.amd.slice(value, (32, BN), (0, 0))
        panel1 = gl.amd.slice(value, (32, BN), (32, 0))
        offsets1 = offsets + 32 * H * N
        out_valid1 = (FULL_ROWS | (out_rows[:, None] + 32 < M)) & (
            (N % BN == 0) | (out_cols[None, :] < N)
        )
        gate1 = gl.amd.cdna4.buffer_load(G, offsets1, out_valid1, other=0)
        _m1024_8192_store_gated_panel(panel0, gate, Y, offsets, out_valid)
        _m1024_8192_store_gated_panel(panel1, gate1, Y, offsets1, out_valid1)
    else:
        sigmoid = _m1024_8192_rounded_sigmoid(gate)
        reconstructed = gl.convert_layout(acc.to(gl.bfloat16), out_layout).to(gl.float32)
        output = (reconstructed * sigmoid).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(output, Y, offsets, out_valid, cache=".wt")


@gluon.jit
def _m1024_8192_reconstruct_and_gate_64(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
    WIDE: gl.constexpr,
):

    if M % 64 == 0:
        _m1024_8192_reconstruct_and_gate_64_body(X, W, G, Y, M, H, K, N, WH, WK, WN, WIDE, True)
    elif gl.program_id(1) * 64 + 64 <= M:
        _m1024_8192_reconstruct_and_gate_64_body(X, W, G, Y, M, H, K, N, WH, WK, WN, WIDE, True)
    else:
        _m1024_8192_reconstruct_and_gate_64_body(X, W, G, Y, M, H, K, N, WH, WK, WN, WIDE, False)


@gluon.jit
def _m1024_8192_reconstruct_and_gate_shared_b(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
):

    BUFFER_GATE: gl.constexpr = M == 8192 and WN == K
    COMPACT: gl.constexpr = M == 6144
    SECOND_ROWS: gl.constexpr = 32 if COMPACT else 64
    PHASE: gl.constexpr = 2 if COMPACT else 1
    if COMPACT:
        head = gl.program_id(0) // 2
        row_start = (gl.program_id(1) * 2 + gl.program_id(0) % 2) * 96
    else:
        head = gl.program_id(1)
        row_start = (gl.program_id(2) * 2 + gl.program_id(0)) * 128
    a_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[2, 2],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    rows0 = gl.arange(0, 64, gl.SliceLayout(1, a_layout))
    rows1 = gl.arange(0, SECOND_ROWS, gl.SliceLayout(1, a_layout))
    a_k = gl.arange(0, 64, gl.SliceLayout(0, a_layout))
    b_k = gl.arange(0, 64, gl.SliceLayout(1, b_layout))
    cols = gl.arange(0, 128, gl.SliceLayout(0, b_layout))
    a_base = X + (row_start * H + head) * K
    b_base = W + head * WH
    a_offsets0 = rows0[:, None] * H * K + a_k[None, :]
    a_offsets1 = (rows1[:, None] + 64) * H * K + a_k[None, :]
    b_offsets = b_k[:, None] * WK + cols[None, :] * WN
    valid0 = (COMPACT or M % 256 == 0) | (row_start + rows0[:, None] < M)
    valid1 = (COMPACT or M % 256 == 0) | (row_start + rows1[:, None] + 64 < M)
    a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, PHASE, 8, [1, 0])
    b_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, PHASE, 8, [0, 1])
    a_shared0 = gl.allocate_shared_memory(gl.bfloat16, (64, 64), a_shared_layout)
    a_shared1 = gl.allocate_shared_memory(gl.bfloat16, (SECOND_ROWS, 64), a_shared_layout)
    b_shared = gl.allocate_shared_memory(gl.bfloat16, (64, 128), b_shared_layout)
    out_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [8, 8] if COMPACT else [4, 16], [4, 1], [1, 0]
    )
    out_rows0 = gl.arange(0, 32, gl.SliceLayout(1, out_layout))
    if not COMPACT:
        out_rows1 = 32 + gl.arange(0, 32, gl.SliceLayout(1, out_layout))
    out_cols = gl.arange(0, 128, gl.SliceLayout(0, out_layout))
    offsets0 = (
        out_rows0[:, None] * H * 128 + out_cols[None, :]
        + (row_start * H + head) * 128
    )
    if COMPACT:
        offsets1 = offsets0 + 32 * H * 128
        offsets2 = offsets0 + 64 * H * 128
    else:
        offsets1 = (
            out_rows1[:, None] * H * 128 + out_cols[None, :]
            + (row_start * H + head) * 128
        )
    acc0 = gl.zeros((64, 128), gl.float32, mma_layout)
    acc1 = gl.zeros((SECOND_ROWS, 128), gl.float32, mma_layout)
    a_reg0 = gl.amd.cdna4.buffer_load(
        a_base, a_offsets0, valid0 & ((K % 64 == 0) | (a_k[None, :] < K)), other=0,
    )
    a_reg1 = gl.amd.cdna4.buffer_load(
        a_base, a_offsets1, valid1 & ((K % 64 == 0) | (a_k[None, :] < K)), other=0,
    )
    for start in gl.static_range(gl.cdiv(K, 64)):
        b_reg = gl.amd.cdna4.buffer_load(
            b_base, b_offsets + start * 64 * WK,
            (K % 64 == 0) | (b_k[:, None] + start * 64 < K), other=0,
        )
        a_shared0.store(a_reg0)
        a_shared1.store(a_reg1)

        b_shared.store(b_reg)
        a0 = a_shared0.load(dot_a)
        a1 = a_shared1.load(dot_a)
        b = b_shared.load(dot_b)
        acc0 = gl.amd.cdna4.mfma(a0, b, acc0)
        if not COMPACT and start == gl.cdiv(K, 64) - 1:
            packed0 = acc0.to(gl.bfloat16)
        if start + 1 < gl.cdiv(K, 64):
            if COMPACT:
                a_reg0 = gl.amd.cdna4.buffer_load(
                    a_base, a_offsets0 + (start + 1) * 64,
                    (K % 64 == 0) | (a_k[None, :] + (start + 1) * 64 < K), other=0,
                )
            else:
                a_reg0 = gl.amd.cdna4.buffer_load(
                    a_base + (start + 1) * 64, a_offsets0,
                    valid0 & ((K % 64 == 0) | (a_k[None, :] + (start + 1) * 64 < K)), other=0,
                )
        if COMPACT and start == gl.cdiv(K, 64) - 1:
            gate0 = gl.amd.cdna4.buffer_load(G, offsets0)
        acc1 = gl.amd.cdna4.mfma(a1, b, acc1)
        if start + 1 < gl.cdiv(K, 64):
            if COMPACT:
                a_reg1 = gl.amd.cdna4.buffer_load(
                    a_base, a_offsets1 + (start + 1) * 64,
                    (K % 64 == 0) | (a_k[None, :] + (start + 1) * 64 < K), other=0,
                )
            else:
                a_reg1 = gl.amd.cdna4.buffer_load(
                    a_base + (start + 1) * 64, a_offsets1,
                    valid1 & ((K % 64 == 0) | (a_k[None, :] + (start + 1) * 64 < K)), other=0,
                )
    if COMPACT:
        value = gl.convert_layout(acc0.to(gl.bfloat16), out_layout)
        value0 = gl.amd.slice(value, (32, 128), (0, 0))
        value1 = gl.amd.slice(value, (32, 128), (32, 0))
        value2 = gl.convert_layout(acc1.to(gl.bfloat16), out_layout)
        gate1 = gl.amd.cdna4.buffer_load(G, offsets1)
        _m1024_8192_store_gated_panel(value0, gate0, Y, offsets0)
        gate2 = gl.amd.cdna4.buffer_load(G, offsets2)
        _m1024_8192_store_gated_panel(value1, gate1, Y, offsets1)
        _m1024_8192_store_gated_panel(value2, gate2, Y, offsets2)
    else:
        offsets2 = offsets0 + 64 * H * 128
        offsets3 = offsets0 + 96 * H * 128
        out_valid0 = (M % 256 == 0) | (row_start + out_rows0[:, None] < M)
        out_valid1 = (M % 256 == 0) | (row_start + out_rows1[:, None] < M)
        out_valid2 = (M % 256 == 0) | (row_start + out_rows0[:, None] + 64 < M)
        out_valid3 = (M % 256 == 0) | (row_start + out_rows0[:, None] + 96 < M)
        if BUFFER_GATE:
            gate0 = gl.amd.cdna4.buffer_load(G, offsets0, out_valid0, other=0)
        else:
            gate0 = gl.load(G + offsets0, out_valid0, other=0)
        value0 = gl.convert_layout(packed0, out_layout)
        value1 = gl.convert_layout(acc1.to(gl.bfloat16), out_layout)
        quarter0, quarter1 = gl.split(gl.permute(gl.reshape(value0, (2, 32, 128)), (1, 2, 0)))
        quarter2, quarter3 = gl.split(gl.permute(gl.reshape(value1, (2, 32, 128)), (1, 2, 0)))
        if BUFFER_GATE:
            gate1 = gl.amd.cdna4.buffer_load(G, offsets1, out_valid1, other=0)
        else:
            gate1 = gl.load(G + offsets1, out_valid1, other=0)
        _m1024_8192_store_gated_panel(quarter0, gate0, Y, offsets0, out_valid0, out_layout)
        if BUFFER_GATE:
            gate2 = gl.amd.cdna4.buffer_load(G, offsets2, out_valid2, other=0)
        else:
            gate2 = gl.load(G + offsets2, out_valid2, other=0)
        _m1024_8192_store_gated_panel(quarter1, gate1, Y, offsets1, out_valid1, out_layout)
        if BUFFER_GATE:
            gate3 = gl.amd.cdna4.buffer_load(G, offsets3, out_valid3, other=0)
        else:
            gate3 = gl.load(G + offsets3, out_valid3, other=0)
        _m1024_8192_store_gated_panel(quarter2, gate2, Y, offsets2, out_valid2, out_layout)
        _m1024_8192_store_gated_panel(quarter3, gate3, Y, offsets3, out_valid3, out_layout)


@gluon.jit
def _m1024_8192_reconstruct_and_gate_small(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
):

    BN: gl.constexpr = 128
    STAGES: gl.constexpr = 2
    SLOTS: gl.constexpr = 3
    NUM_WARPS: gl.constexpr = 8
    grouped_head = gl.program_id(0).to(gl.uint32)
    head = grouped_head // 4
    row_tile = gl.program_id(1).to(gl.uint32) * 4 + grouped_head % 4
    col_tile = gl.program_id(2).to(gl.uint32)
    a_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[1, 0], [2, 0], [32, 0]], block_bases=[], shape=[64, 64],
    )
    b_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2], [0, 4]], block_bases=[], shape=[64, 128],
    )
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    a_shared_layout: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_a, (64, 64), X.dtype.element_ty
    )
    b_shared_layout: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_b, (64, BN), W.dtype.element_ty
    )
    a_slots = ()
    b_slots = ()
    for slot in gl.static_range(SLOTS):
        a_slots += (gl.allocate_shared_memory(X.dtype.element_ty, (64, 64), a_shared_layout),)
        b_slots += (gl.allocate_shared_memory(W.dtype.element_ty, (64, BN), b_shared_layout),)
    rows = row_tile * 64 + gl.arange(0, 64, gl.SliceLayout(1, a_layout))
    cols = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, b_layout))
    a_k = gl.arange(0, 64, gl.SliceLayout(0, a_layout))
    b_k = gl.arange(0, 64, gl.SliceLayout(1, b_layout))
    a_base = X + head * K
    b_base = W + head * WH
    a_offsets = rows[:, None] * H * K + a_k[None, :]
    b_offsets = b_k[:, None] * WK + cols[None, :] * WN
    a_valid = (M % 64 == 0) | (rows[:, None] < M)
    b_valid = (N % BN == 0) | (cols[None, :] < N)
    out_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BN, BN // 8], [NUM_WARPS, 1], [1, 0]
    )
    out_rows = row_tile * 64 + gl.arange(0, 32, gl.SliceLayout(1, out_layout))
    out_cols = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, out_layout))
    offsets = out_rows[:, None] * H * N + head * N + out_cols[None, :]
    out_valid = ((M % 64 == 0) | (out_rows[:, None] < M)) & (
        (N % BN == 0) | (out_cols[None, :] < N)
    )
    for start in gl.static_range(STAGES):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a_slots[start], a_base, a_offsets + start * 64,
            a_valid & ((K % 64 == 0) | (a_k[None, :] + start * 64 < K)), other=0,
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b_slots[start], b_base, b_offsets + start * 64 * WK,
            b_valid & ((K % 64 == 0) | (b_k[:, None] + start * 64 < K)), other=0,
        )
        gl.amd.cdna4.async_copy.commit_group()
    acc = gl.zeros((64, BN), gl.float32, mma_layout)
    for start in gl.static_range(gl.cdiv(K, 64)):
        gl.amd.cdna4.async_copy.wait_group(min(STAGES - 1, gl.cdiv(K, 64) - 1 - start))
        gl.barrier()
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[start % SLOTS], dot_a)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[start % SLOTS], dot_b)
        if start + STAGES < gl.cdiv(K, 64):
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                a_slots[(start + STAGES) % SLOTS], a_base, a_offsets + (start + STAGES) * 64,
                a_valid & ((K % 64 == 0) | (a_k[None, :] + (start + STAGES) * 64 < K)), other=0,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_slots[(start + STAGES) % SLOTS], b_base, b_offsets + (start + STAGES) * 64 * WK,
                b_valid & ((K % 64 == 0) | (b_k[:, None] + (start + STAGES) * 64 < K)), other=0,
            )
            gl.amd.cdna4.async_copy.commit_group()
        if start == gl.cdiv(K, 64) - 2:
            gate = gl.amd.cdna4.buffer_load(G, offsets, out_valid, other=0)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    value = gl.convert_layout(acc.to(gl.bfloat16), out_layout)
    panel0 = gl.amd.slice(value, (32, BN), (0, 0))
    panel1 = gl.amd.slice(value, (32, BN), (32, 0))
    offsets1 = offsets + 32 * H * N
    out_valid1 = ((M % 64 == 0) | (out_rows[:, None] + 32 < M)) & (
        (N % BN == 0) | (out_cols[None, :] < N)
    )
    gate1 = gl.amd.cdna4.buffer_load(G, offsets1, out_valid1, other=0)
    result0 = (panel0.to(gl.float32) * _m1024_8192_rounded_sigmoid(gate)).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(result0, Y, offsets, out_valid, cache=".wt")
    result1 = (panel1.to(gl.float32) * _m1024_8192_rounded_sigmoid(gate1)).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(result1, Y, offsets1, out_valid1, cache=".wt")


@gluon.jit
def _m1024_8192_reconstruct_and_gate_ring(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
    WIDE: gl.constexpr,
):

    BN: gl.constexpr = 128 if WIDE else 64
    STAGES: gl.constexpr = 2 if WIDE else 3
    NUM_WARPS: gl.constexpr = 8 if WIDE else 4


    GATE_LEAD: gl.constexpr = 2 if M == 4096 and WN == K else 1
    head = gl.program_id(0).to(gl.uint32)
    row_tile = gl.program_id(1).to(gl.uint32)
    col_tile = gl.program_id(2).to(gl.uint32)
    if WIDE:
        a_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0]],
            warp_bases=[[1, 0], [2, 0], [32, 0]], block_bases=[], shape=[64, 64],
        )
        b_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[[0, 1], [0, 2], [0, 4]], block_bases=[], shape=[64, 128],
        )
    else:
        a_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [32, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0]],
            warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[64, 64],
        )
        b_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 32]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
            warp_bases=[[0, 1], [0, 2]], block_bases=[], shape=[64, 64],
        )
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    a_shared_layout: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_a, (64, 64), X.dtype.element_ty
    )
    b_shared_layout: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_b, (64, BN), W.dtype.element_ty
    )
    a_slots = ()
    b_slots = ()
    for slot in gl.static_range(STAGES):
        a_slots += (gl.allocate_shared_memory(X.dtype.element_ty, (64, 64), a_shared_layout),)
        b_slots += (gl.allocate_shared_memory(W.dtype.element_ty, (64, BN), b_shared_layout),)
    rows = row_tile * 64 + gl.arange(0, 64, gl.SliceLayout(1, a_layout))
    cols = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, b_layout))
    a_k = gl.arange(0, 64, gl.SliceLayout(0, a_layout))
    b_k = gl.arange(0, 64, gl.SliceLayout(1, b_layout))
    if not WIDE:
        a_base = X + (row_tile * 64 * H + head) * K
        b_base = W + head * WH + col_tile * BN * WN
        a_offsets = gl.arange(0, 64, gl.SliceLayout(1, a_layout))[:, None] * H * K + a_k[None, :]
        b_offsets = b_k[:, None] * WK + gl.arange(0, BN, gl.SliceLayout(0, b_layout))[None, :] * WN
    else:
        a_base = X + head * K
        b_base = W + head * WH
        a_offsets = rows[:, None] * H * K + a_k[None, :]
        b_offsets = b_k[:, None] * WK + cols[None, :] * WN
    a_valid = (M % 64 == 0) | (rows[:, None] < M)
    b_valid = (N % BN == 0) | (cols[None, :] < N)
    out_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BN, BN // 8], [NUM_WARPS, 1], [1, 0]
    )
    OUT_ROWS: gl.constexpr = 32
    out_rows = row_tile * 64 + gl.arange(0, OUT_ROWS, gl.SliceLayout(1, out_layout))
    out_cols = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, out_layout))
    offsets = out_rows[:, None] * H * N + head * N + out_cols[None, :]
    out_valid = ((M % 64 == 0) | (out_rows[:, None] < M)) & (
        (N % BN == 0) | (out_cols[None, :] < N)
    )
    for start in gl.static_range(STAGES):
        if not WIDE:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_slots[start], b_base, b_offsets + start * 64 * WK,
                b_valid & ((K % 64 == 0) | (b_k[:, None] + start * 64 < K)), other=0,
            )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a_slots[start], a_base, a_offsets + start * 64,
            a_valid & ((K % 64 == 0) | (a_k[None, :] + start * 64 < K)), other=0,
        )
        if WIDE:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_slots[start], b_base, b_offsets + start * 64 * WK,
                b_valid & ((K % 64 == 0) | (b_k[:, None] + start * 64 < K)), other=0,
            )
        gl.amd.cdna4.async_copy.commit_group()
    acc = gl.zeros((64, BN), gl.float32, mma_layout)
    for start in gl.static_range(gl.cdiv(K, 64)):
        gl.amd.cdna4.async_copy.wait_group(min(STAGES - 1, gl.cdiv(K, 64) - 1 - start))
        gl.barrier()
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_slots[start % STAGES], dot_a)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_slots[start % STAGES], dot_b)
        if start + STAGES < gl.cdiv(K, 64):
            gl.barrier()
            if not WIDE:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    b_slots[start % STAGES], b_base, b_offsets + (start + STAGES) * 64 * WK,
                    b_valid & ((K % 64 == 0) | (b_k[:, None] + (start + STAGES) * 64 < K)), other=0,
                )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                a_slots[start % STAGES], a_base, a_offsets + (start + STAGES) * 64,
                a_valid & ((K % 64 == 0) | (a_k[None, :] + (start + STAGES) * 64 < K)), other=0,
            )
            if WIDE:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    b_slots[start % STAGES], b_base, b_offsets + (start + STAGES) * 64 * WK,
                    b_valid & ((K % 64 == 0) | (b_k[:, None] + (start + STAGES) * 64 < K)), other=0,
                )
            gl.amd.cdna4.async_copy.commit_group()
        if start == gl.cdiv(K, 64) - GATE_LEAD:
            gate = gl.load(G + offsets, out_valid, other=0)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    value = gl.convert_layout(acc.to(gl.bfloat16), out_layout)
    panel0 = gl.amd.slice(value, (32, BN), (0, 0))
    panel1 = gl.amd.slice(value, (32, BN), (32, 0))
    offsets1 = offsets + 32 * H * N
    out_valid1 = ((M % 64 == 0) | (out_rows[:, None] + 32 < M)) & (
        (N % BN == 0) | (out_cols[None, :] < N)
    )
    gate1 = gl.amd.cdna4.buffer_load(G, offsets1, out_valid1, other=0)
    result0 = (panel0.to(gl.float32) * _m1024_8192_rounded_sigmoid(gate)).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(result0, Y, offsets, out_valid, cache=".wt")
    result1 = (panel1.to(gl.float32) * _m1024_8192_rounded_sigmoid(gate1)).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(result1, Y, offsets1, out_valid1, cache=".wt")


def mla_vc_output_gate_m1024_8192(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:

    m, h, k = latent.shape
    n = weight.shape[2]
    output = torch.empty((m, h * n), dtype=latent.dtype, device=latent.device)
    if m == 1024 and weight.stride(2) == k:
        _m1024_8192_reconstruct_and_gate_small[(h * 4, triton.cdiv(m, 256), triton.cdiv(n, 128))](
            latent, weight, gate, output, m, h, k, n, *weight.stride(),
            num_warps=8, enable_fp_fusion=False,
        )
        return output
    if m >= 6144 and n == 128:
        grid = (h * 2, m // 192) if m == 6144 else (2, h, triton.cdiv(m, 256))
        _m1024_8192_reconstruct_and_gate_shared_b[grid](
            latent, weight, gate, output, m, h, k, *weight.stride(),
            num_warps=4, enable_fp_fusion=False,
        )
        return output
    if m in (1024, 4095, 4096) or (m == 2048 and weight.stride(2) == k):
        wide = m != 2048
        bn = 128 if wide else 64
        _m1024_8192_reconstruct_and_gate_ring[(h, triton.cdiv(m, 64), triton.cdiv(n, bn))](
            latent, weight, gate, output, m, h, k, n, *weight.stride(), wide,
            num_warps=8 if wide else 4, enable_fp_fusion=False,
        )
        return output
    wide = m > 2048
    bn = 128 if wide else 64
    _m1024_8192_reconstruct_and_gate_64[(h, triton.cdiv(m, 64), triton.cdiv(n, bn))](
        latent, weight, gate, output, m, h, k, n, *weight.stride(), wide,
        num_warps=8 if wide else 4, enable_fp_fusion=False,
    )
    return output


@gluon.jit
def _m128_allocate_stages(dtype: gl.constexpr, count: gl.constexpr):

    layout: gl.constexpr = gl.SharedLinearLayout([
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
        [1, 8], [2, 16], [4, 32], [8, 0], [16, 0], [0, 64],
    ])

    weight_layout: gl.constexpr = gl.SharedLinearLayout([
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
        [1, 0], [2, 8], [4, 16], [8, 32], [16, 0], [0, 64],
    ])

    a0 = gl.allocate_shared_memory(dtype, [32, 128], layout)
    a1 = gl.allocate_shared_memory(dtype, [32, 128], layout)
    b0 = gl.allocate_shared_memory(dtype, [32, 128], weight_layout).permute((1, 0))
    b1 = gl.allocate_shared_memory(dtype, [32, 128], weight_layout).permute((1, 0))
    stages = ((a0, b0), (a1, b1))
    for stage in gl.static_range(2, count):
        a = gl.allocate_shared_memory(dtype, [32, 128], layout)
        b = gl.allocate_shared_memory(dtype, [32, 128], weight_layout).permute((1, 0))
        stages += ((a, b),)
    return stages


@gluon.jit
def _m128_prefetch(a_shared, b_shared, x_base, w_base, a_offsets, b_offsets):
    a_flat = a_shared._reinterpret(
        shape=[64, 64], layout=gl.SwizzledSharedLayout(8, 1, 1, [1, 0])
    )
    b_flat = b_shared._reinterpret(
        shape=[64, 64], layout=gl.SwizzledSharedLayout(8, 1, 1, [0, 1])
    )

    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_flat, x_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_flat, w_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m128_vc_gate_m128(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
):

    pid = gl.program_id(0).to(gl.uint32)
    gl.assume(pid < H * (M // 32) * (N // 32))
    head = pid // ((M // 32) * (N // 32))
    tile_m = pid // (N // 32) % (M // 32)
    tile_n = pid % (N // 32)
    row_head = tile_m * 32 * H + head

    a_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[2, 2],
    )
    a_dot: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    b_dot: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)

    out_rows = gl.arange(0, 32, gl.SliceLayout(1, mma_layout))
    out_cols = tile_n * 32 + gl.arange(0, 32, gl.SliceLayout(0, mma_layout))
    out_offsets = out_rows[:, None] * H * N + row_head * N + out_cols[None, :]

    out_bytes = (out_offsets * 2).to(gl.uint32)
    gate_ptrs = (G.to(gl.pointer_type(gl.uint8)) + out_bytes).to(G.dtype)
    gate = gl.load(gate_ptrs).to(gl.float32)

    rows = gl.arange(0, 64, gl.SliceLayout(1, a_layout))
    ak = gl.arange(0, 64, gl.SliceLayout(0, a_layout))
    bk = gl.arange(0, 64, gl.SliceLayout(1, b_layout))
    cols = gl.arange(0, 64, gl.SliceLayout(0, b_layout))
    x_base = X + row_head * K
    w_base = W + (head * WH + tile_n * 32 * WN)
    a_offsets = (
        (rows[:, None] % 32) * H * K
        + (ak[None, :] ^ ((rows[:, None] % 8) * 8))
        + (rows[:, None] // 32) * 64
    )
    b_offsets = (
        ((bk[:, None] ^ ((cols[None, :] // 2 % 8) * 8))
         + (cols[None, :] // 32) * 64) * WK
        + (cols[None, :] % 32) * WN
    )
    a_offsets = gl.max_contiguous(gl.multiple_of(a_offsets, [1, 8]), [1, 8])
    b_offsets = gl.max_contiguous(gl.multiple_of(b_offsets, [8, 1]), [8, 1])

    stages = _m128_allocate_stages(X.dtype.element_ty, 3)
    for stage in gl.static_range(3):
        a_flat = stages[stage][0]._reinterpret(
            shape=[64, 64], layout=gl.SwizzledSharedLayout(8, 1, 1, [1, 0])
        )
        b_flat = stages[stage][1]._reinterpret(
            shape=[64, 64], layout=gl.SwizzledSharedLayout(8, 1, 1, [0, 1])
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_flat, x_base + stage * 128, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_flat, w_base + stage * 128 * WK, b_offsets)
        gl.amd.cdna4.async_copy.commit_group()

    sigmoid = (1.0 / (1.0 + gl.exp2(-gate * 1.4426950408889634))).to(gl.bfloat16).to(gl.float32)
    acc = gl.full((32, 32), 0.0, gl.float32, mma_layout)
    for block in gl.static_range(K // 128):
        gl.amd.cdna4.async_copy.wait_group(2 if block == 0 else (0 if block == 3 else 1))
        if block == 1:

            gl.barrier()
            _m128_prefetch(stages[0][0], stages[0][1],
                      x_base + 384, w_base + 384 * WK, a_offsets, b_offsets)
        a_stage, b_stage = stages[block % 3]
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_stage, b_dot)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_stage, a_dot)
        acc = gl.amd.cdna4.mfma(a, b, acc)

    acc = acc.to(gl.bfloat16).to(gl.float32)
    output_ptrs = (Y.to(gl.pointer_type(gl.uint8)) + out_bytes).to(Y.dtype)

    gl.store(output_ptrs, (acc * sigmoid).to(gl.bfloat16), cache_modifier=".cs")


def mla_vc_output_gate_m128(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:

    m, h, k = latent.shape
    n = weight.shape[2]
    output = torch.empty((m, h * n), dtype=latent.dtype, device=latent.device)
    _m128_vc_gate_m128[(h * (m // 32) * (n // 32),)](
        latent, weight, gate, output, m, h, k, n, *weight.stride(),
        num_warps=4, enable_fp_fusion=False, waves_per_eu=2,
    )
    return output


@gluon.jit
def _m16_vc_gate(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
    FOLD: gl.constexpr, WIDE: gl.constexpr, LARGE_OFFSETS: gl.constexpr,
):
    tile_m: gl.constexpr = 16 // FOLD
    block_n: gl.constexpr = 32 if WIDE else 16
    tile_n: gl.constexpr = block_n // FOLD
    block_k: gl.constexpr = 512 // FOLD
    rebase: gl.constexpr = M >= tile_m
    early_gate: gl.constexpr = (M >= 4) and (M <= 16)
    early_sigmoid: gl.constexpr = M == 16
    buffer_operands: gl.constexpr = M < 8 and not LARGE_OFFSETS


    if WIDE:
        row_base = gl.program_id(0) * tile_m
        head = gl.program_id(1)
    else:
        head = gl.program_id(0)
        row_base = gl.program_id(1) * tile_m
    col_base = gl.program_id(2) * tile_n
    if LARGE_OFFSETS:
        head = head.to(gl.int64)
        row_base = row_base.to(gl.int64)
        col_base = col_base.to(gl.int64)
    if rebase:
        X += head * K + row_base * H * K
        W += head * WH + col_base * WN
        G += row_base * H * N + head * N + col_base
        Y += row_base * H * N + head * N + col_base

    a_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [0, 1])
    b_layout: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 1], [1, 0])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[1, 1],
    )
    result_layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, 1], [1, 0])
    output_layout: gl.constexpr = gl.SliceLayout(
        1, gl.BlockedLayout(
            [4 // FOLD, 1, 1], [4, FOLD, 16 // FOLD], [1, 1, 1], [1, 2, 0],
        ),
    )

    if WIDE:
        if FOLD == 2:
            gate_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[1, 0]],
                lane_bases=[[0, 8], [0, 1], [0, 2], [0, 4], [2, 0], [4, 0]],
                warp_bases=[], block_bases=[], shape=[8, 16],
            )
        else:
            gate_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[],
                lane_bases=[[0, 4], [0, 0], [0, 1], [0, 2], [1, 0], [2, 0]],
                warp_bases=[], block_bases=[], shape=[4, 8],
            )
    else:
        gate_layout: gl.constexpr = output_layout

    ar = gl.arange(0, 16, gl.SliceLayout(1, a_layout))
    ak = gl.arange(0, block_k, gl.SliceLayout(0, a_layout))
    bk = gl.arange(0, block_k, gl.SliceLayout(1, b_layout))
    bc = gl.arange(0, block_n, gl.SliceLayout(0, b_layout))
    out_r = gl.arange(0, tile_m, gl.SliceLayout(1, gate_layout))
    out_c = gl.arange(0, tile_n, gl.SliceLayout(0, gate_layout))
    if LARGE_OFFSETS:
        ar = ar.to(gl.int64)
        ak = ak.to(gl.int64)
        bk = bk.to(gl.int64)
        bc = bc.to(gl.int64)
        out_r = out_r.to(gl.int64)
        out_c = out_c.to(gl.int64)
    rows = row_base + ar // FOLD
    cols = col_base + bc // FOLD
    output_mask = gl.full((tile_m, tile_n), True, gl.int1, gate_layout)
    if M % tile_m != 0:
        output_mask = output_mask & (row_base + out_r[:, None] < M)
    if N % tile_n != 0:
        output_mask = output_mask & (col_base + out_c[None, :] < N)
    offsets = out_r[:, None] * H * N + out_c[None, :]
    if not rebase:
        offsets += row_base * H * N + head * N + col_base
    if early_gate:
        if LARGE_OFFSETS:
            gate = gl.load(G + offsets, output_mask, 0).to(gl.float32)
        else:
            gate = gl.amd.cdna4.buffer_load(G, offsets, output_mask, 0).to(gl.float32)
    if early_sigmoid:
        sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)

    zero = gl.zeros((16, block_n), gl.float32, mma_layout)
    acc = zero
    for start in range(gl.cdiv(K, block_k * FOLD)):
        ka = start * block_k + ak[None, :]
        kb = start * block_k + bk[:, None]

        a_k = (ar[:, None] % FOLD) * 8 + (ka // 8) * (FOLD * 8) + ka % 8
        b_k = (bc[None, :] % FOLD) * 8 + (kb // 8) * (FOLD * 8) + kb % 8
        if rebase:
            a_offsets = (ar[:, None] // FOLD) * H * K + a_k
            b_offsets = b_k * WK + (bc[None, :] // FOLD) * WN
            a_base = X
            b_base = W
        else:
            a_offsets = rows[:, None] * H * K + a_k
            b_offsets = b_k * WK + cols[None, :] * WN
            a_base = X + head * K
            b_base = W + head * WH
        a_mask = a_k < K
        b_mask = b_k < K
        if M % tile_m != 0:
            a_mask = a_mask & (rows[:, None] < M)
        if N % tile_n != 0:
            b_mask = b_mask & (cols[None, :] < N)
        if buffer_operands:
            a = gl.amd.cdna4.buffer_load(a_base, a_offsets, a_mask, 0)
            b = gl.amd.cdna4.buffer_load(b_base, b_offsets, b_mask, 0)
        else:
            a = gl.load(a_base + a_offsets, a_mask, 0)
            b = gl.load(b_base + b_offsets, b_mask, 0)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma_layout, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma_layout, 8))
        acc += gl.amd.cdna4.mfma(a, b, zero)

    acc = gl.convert_layout(acc, result_layout)
    rr = gl.arange(0, 16, gl.SliceLayout(1, result_layout))
    cc = gl.arange(0, block_n, gl.SliceLayout(0, result_layout))
    diagonal = rr[:, None] % FOLD == cc[None, :] % FOLD
    partials = gl.reshape(gl.where(diagonal, acc, 0.0), (tile_m, FOLD, tile_n, FOLD))
    values = gl.sum(gl.sum(partials, 1), 2)
    values = gl.convert_layout(values, output_layout, assert_trivial=True)
    if not early_gate:
        if LARGE_OFFSETS:
            gate = gl.load(G + offsets, output_mask, 0).to(gl.float32)
        else:
            gate = gl.amd.cdna4.buffer_load(G, offsets, output_mask, 0).to(gl.float32)
    if not early_sigmoid:
        if M == 4:


            sigmoid = (1.0 / (1.0 + gl.exp2(-gate * 1.4426950408889634))).to(gl.bfloat16).to(gl.float32)
        else:
            sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)


    if M == 16:
        reconstructed = values.to(gl.bfloat16)
        reconstructed = gl.convert_layout(reconstructed, gate_layout).to(gl.float32)
    else:
        reconstructed = values.to(gl.bfloat16).to(gl.float32)
        reconstructed = gl.convert_layout(reconstructed, gate_layout)
    result = (reconstructed * sigmoid).to(gl.bfloat16)
    if LARGE_OFFSETS:
        gl.store(Y + offsets, result, output_mask)
    else:
        gl.amd.cdna4.buffer_store(result, Y, offsets, output_mask)


@gluon.jit
def _m16_vc_gate_packed(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
):
    BLOCK_K: gl.constexpr = 32
    pid = gl.program_id(0).to(gl.uint32)
    head = pid % H
    row_base = ((pid // H) % (M // 8)) * 8
    col_base = (pid // (H * (M // 8))) * 16
    X += row_base * H * K + head * K
    W += head * WH + col_base * WN
    G += row_base * H * N + head * N + col_base
    Y += row_base * H * N + head * N + col_base

    a_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [0, 1])
    b_layout: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 1], [1, 0])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[1, 1],
    )
    result_layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, 1], [1, 0])
    gate_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1]],
        lane_bases=[[1, 0], [0, 2], [0, 4], [0, 8], [2, 0], [4, 0]],
        warp_bases=[], block_bases=[], shape=[8, 16],
    )

    fold_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1, 0], [0, 0, 1]],
        lane_bases=[[1, 0, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [2, 0, 0], [4, 0, 0]],
        warp_bases=[], block_bases=[], shape=[8, 16, 2],
    )
    out_r = gl.arange(0, 8, gl.SliceLayout(1, gate_layout))
    out_c = gl.arange(0, 16, gl.SliceLayout(0, gate_layout))
    offsets = out_r[:, None] * H * N + out_c[None, :]
    gate = gl.amd.cdna4.buffer_load(G, offsets).to(gl.float32)
    sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)

    ar = gl.arange(0, 16, gl.SliceLayout(1, a_layout))
    ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, a_layout))
    bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, b_layout))
    bc = gl.arange(0, 32, gl.SliceLayout(0, b_layout))
    acc = gl.zeros((16, 32), gl.float32, mma_layout)
    for start in gl.static_range(K // (2 * BLOCK_K)):
        ka = start * BLOCK_K + ak[None, :]
        kb = start * BLOCK_K + bk[:, None]
        a_k = (ar[:, None] % 2) * 8 + (ka // 8) * 16 + ka % 8
        b_k = (bc[None, :] % 2) * 8 + (kb // 8) * 16 + kb % 8
        a = gl.load(X + (ar[:, None] // 2) * H * K + a_k)
        b = gl.load(W + b_k * WK + ((bc[None, :] % 16) // 2 * 2 + bc[None, :] // 16) * WN)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma_layout, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma_layout, 8))
        acc = gl.amd.cdna4.mfma(a, b, acc)

    acc = gl.convert_layout(acc, result_layout)

    row_pairs = gl.permute(gl.reshape(acc, (8, 2, 32)), (0, 2, 1))
    even_rows, odd_rows = gl.split(row_pairs)
    cc = gl.arange(0, 32, gl.SliceLayout(0, even_rows.type.layout))
    partials = gl.where(cc[None, :] % 2 == 0, even_rows, odd_rows)
    partials = gl.reshape(partials, (8, 2, 8, 2))
    partials = gl.reshape(gl.permute(partials, (0, 2, 1, 3)), (8, 16, 2))
    partials = gl.convert_layout(partials, fold_layout)
    values = gl.sum(partials, 2)
    values = gl.convert_layout(values, gate_layout, assert_trivial=True)
    reconstructed = values.to(gl.bfloat16).to(gl.float32)
    result = (reconstructed * sigmoid).to(gl.bfloat16)

    gl.amd.cdna4.buffer_store(result, Y, offsets, cache=".wt")


def mla_vc_output_gate_m16(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor,
) -> torch.Tensor:

    m, h, k = latent.shape
    n = weight.shape[2]
    wh, wk, wn = weight.stride()
    output = torch.empty((m, h * n), dtype=latent.dtype, device=latent.device)
    large_offsets = max(
        m * h * k, m * h * n, (h - 1) * wh + (k - 1) * wk + (n - 1) * wn + 1,
        wh, wk, wn,
    ) >= (1 << 30)
    if m in (16, 32, 64, 128) and n % 16 == 0 and k % 64 == 0 and not large_offsets:
        _m16_vc_gate_packed[(m // 8 * h * (n // 16),)](
            latent, weight, gate, output, m, h, k, n, wh, wk, wn,
            num_warps=1, enable_fp_fusion=False,
        )
        return output
    fold = 2
    wide = m in (8, 16)
    tile_m = 16 // fold
    tile_n = 32 // fold
    grid = (triton.cdiv(m, tile_m), h, triton.cdiv(n, tile_n))
    _m16_vc_gate[grid](
        latent, weight, gate, output, m, h, k, n, wh, wk, wn,
        fold, wide, large_offsets, num_warps=1, enable_fp_fusion=False,
    )
    return output


@gluon.jit
def _m2_rounded_sigmoid(gate):
    gate = gate.to(gl.float32)
    return (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _m2_select_diagonal(
    acc, VM: gl.constexpr, VN: gl.constexpr,
    F: gl.constexpr, TRANS: gl.constexpr,
):


    parts = gl.reshape(acc, (VM // F, F, VN // F, F))
    if not TRANS:
        parts = gl.permute(parts, (0, 2, 3, 1))
    if F == 4:
        if TRANS:
            parts = gl.reshape(parts, (VM // F, F, VN // F, 2, 2))
        else:
            parts = gl.reshape(parts, (VM // F, VN // F, F, 2, 2))
        p0, p1 = gl.split(parts)
        p00, p10 = gl.split(p0)
        p01, p11 = gl.split(p1)
        if TRANS:
            selector = gl.arange(0, F, gl.SliceLayout(0, gl.SliceLayout(2, p00.type.layout)))
            selector = selector[None, :, None]
        else:
            selector = gl.arange(0, F, gl.SliceLayout(0, gl.SliceLayout(0, p00.type.layout)))
            selector = gl.expand_dims(gl.expand_dims(selector, 0), 0)
        selected = gl.where(
            selector == 0, p00,
            gl.where(selector == 1, p01, gl.where(selector == 2, p10, p11)),
        )
    else:
        p0, p1 = gl.split(parts)
        if TRANS:
            selector = gl.arange(0, F, gl.SliceLayout(0, gl.SliceLayout(2, p0.type.layout)))
            selector = selector[None, :, None]
        else:
            selector = gl.arange(0, F, gl.SliceLayout(0, gl.SliceLayout(0, p0.type.layout)))
            selector = gl.expand_dims(gl.expand_dims(selector, 0), 0)
        selected = gl.where(selector == 0, p0, p1)
    return gl.sum(selected, 1 if TRANS else 2)


@gluon.jit
def _m2_folded_selected_gate(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
    F: gl.constexpr, BK: gl.constexpr, VM: gl.constexpr, VN: gl.constexpr,
    TRANS: gl.constexpr, COMPACT: gl.constexpr, EARLY_SIGMOID: gl.constexpr,
    WIDE_WEIGHT: gl.constexpr,
):
    PACKET: gl.constexpr = 8
    head = gl.program_id(0)
    TM: gl.constexpr = VM // F
    TN: gl.constexpr = VN // F
    row_start = gl.program_id(1) * TM
    col_start = gl.program_id(2) * TN
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=TRANS, warps_per_cta=[1, 1],
    )
    al: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bl: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    zero = gl.full((VM, VN), 0, gl.float32, mma)
    template = _m2_select_diagonal(zero, VM, VN, F, TRANS)
    if COMPACT:
        if M == 2:


            ol: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[],
                lane_bases=[[0, 4], [0, 0], [0, 1], [0, 2], [0, 0], [0, 0]],
                warp_bases=[], block_bases=[], shape=[1, 8],
            )
        else:

            ol: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[],
                lane_bases=[[1, 0], [0, 1], [0, 2], [0, 4], [2, 0], [4, 0]],
                warp_bases=[], block_bases=[], shape=[8, 8],
            )
    else:
        ol: gl.constexpr = template.type.layout
    rr = gl.arange(0, TM, gl.SliceLayout(1, ol))
    cc = gl.arange(0, TN, gl.SliceLayout(0, ol))
    X += head * K + row_start * H * K
    if WIDE_WEIGHT:

        W += head.to(gl.int64) * WH + col_start.to(gl.int64) * WN
    else:
        W += head * WH + col_start * WN
    G += head * N + row_start * H * N + col_start
    Y += head * N + row_start * H * N + col_start
    offsets = rr[:, None] * H * N + cc[None, :]
    mask = ((rr[:, None] + row_start < M) | (M % TM == 0)) & (
        (cc[None, :] + col_start < N) | (N % TN == 0)
    )
    gate = gl.amd.cdna4.buffer_load(G, offsets, mask, 0).to(gl.float32)
    if EARLY_SIGMOID:
        sigmoid = _m2_rounded_sigmoid(gate)
    ar = gl.arange(0, VM, gl.SliceLayout(1, al))
    aq = gl.arange(0, BK, gl.SliceLayout(0, al))
    bq = gl.arange(0, BK, gl.SliceLayout(1, bl))
    bc = gl.arange(0, VN, gl.SliceLayout(0, bl))
    rows = ar // F
    cols = bc // F

    ak = (aq[None, :] // PACKET * F + ar[:, None] % F) * PACKET + aq[None, :] % PACKET
    bk = (bq[:, None] // PACKET * F + bc[None, :] % F) * PACKET + bq[:, None] % PACKET
    a_offsets = rows[:, None] * H * K + ak
    if WIDE_WEIGHT:
        b_offsets = bk.to(gl.int64) * WK + cols[None, :].to(gl.int64) * WN
    else:
        b_offsets = bk * WK + cols[None, :] * WN
    a_mask = (rows[:, None] + row_start < M) | (M % TM == 0)
    b_mask = (cols[None, :] + col_start < N) | (N % TN == 0)
    if K != BK * F:
        a_mask = a_mask & (ak < K)
        b_mask = b_mask & (bk < K)
    a = gl.load(X + a_offsets, a_mask, 0)
    b = gl.load(W + b_offsets, b_mask, 0)
    acc = gl.amd.cdna4.mfma(a, b, zero)
    reconstructed = _m2_select_diagonal(acc, VM, VN, F, TRANS)
    reconstructed = gl.convert_layout(reconstructed, ol)
    if not EARLY_SIGMOID:
        sigmoid = _m2_rounded_sigmoid(gate)

    reconstructed = reconstructed.to(gl.bfloat16).to(gl.float32)
    result = (reconstructed * sigmoid).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(result, Y, offsets, mask)


def mla_vc_output_gate_m2(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:

    m, h, k = latent.shape
    n = weight.shape[2]
    wh, wk, wn = weight.stride()
    output = torch.empty((m, h * n), device=latent.device, dtype=latent.dtype)


    weight_span = (k - 1) * wk + (n - 1) * wn
    buffer_safe = max(wk, wn, (h - 1) * wh, weight_span) < (1 << 30) - 2
    fold = 4
    virtual_m = 4
    virtual_n = 32
    tile_m, tile_n = virtual_m // fold, virtual_n // fold
    block_k = max(32, triton.next_power_of_2(triton.cdiv(k, fold)))
    _m2_folded_selected_gate[(h, triton.cdiv(m, tile_m), triton.cdiv(n, tile_n))](
        latent, weight, gate, output, m, h, k, n, wh, wk, wn,
        fold, block_k, virtual_m, virtual_n, m == 4, m == 2 or m > 8,
        m != 4, not buffer_safe,
        num_warps=1, enable_fp_fusion=False,
    )
    return output


@constexpr_function
def _m256_copy_layout(shared, waves, neighboring_columns=False):

    wave_bits = waves.bit_length() - 1
    bases = shared.offset_bases
    return gl.DistributedLinearLayout(
        reg_bases=([bases[i] for i in (0, 1, 2, 9)] if neighboring_columns
                   else bases[:3] + bases[9 + wave_bits :]),
        lane_bases=bases[3:9],
        warp_bases=bases[10:12] if neighboring_columns else bases[9 : 9 + wave_bits],
        block_bases=[],
        shape=shared.shape,
    )


@gluon.jit
def _m256_stage_panel(first, second, K_OFFSET: gl.constexpr):

    first_shared, first_ptr, first_offsets = first
    second_shared, second_ptr, second_offsets = second
    async_copy.buffer_load_to_shared(
        first_shared, first_ptr + K_OFFSET, first_offsets,
    )
    async_copy.buffer_load_to_shared(
        second_shared, second_ptr + K_OFFSET, second_offsets,
    )
    async_copy.commit_group()


@gluon.jit
def _m256_consume_panel(
    a_shared, b_shared, acc,
    AD: gl.constexpr, BD: gl.constexpr,
):
    b = async_copy.load_shared_relaxed(b_shared, BD)
    a = async_copy.load_shared_relaxed(a_shared, AD)
    return gl.amd.cdna4.mfma(a, b, acc)


@gluon.jit
def _m256_fused_staged(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WN: gl.constexpr,
    BN: gl.constexpr,
):
    BM: gl.constexpr = 32
    BK: gl.constexpr = 128
    WAVES: gl.constexpr = gl.num_warps()
    WIDE: gl.constexpr = BN == 64
    gl.static_assert(BN == 32 or BN == 64)
    gl.static_assert(WAVES == BN // 8 and K == 4 * BK)
    gl.static_assert(M % BM == 0 and N % BN == 0)
    if WIDE:
        col_tile = gl.program_id(0).to(gl.uint32)
        head = gl.program_id(1).to(gl.uint32)
        row_tile = gl.program_id(2).to(gl.uint32)
    else:
        pid = gl.program_id(0).to(gl.uint32)
        col_tile = pid % (N // BN)
        row_tile = (pid // (N // BN)) % (M // BM)
        head = pid // ((N // BN) * (M // BM))

    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[2, WAVES // 2],
    )
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    sa: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(ad, [BM, BK], gl.bfloat16)
    sb: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(bd, [BK, BN], gl.bfloat16)
    ca: gl.constexpr = _m256_copy_layout(sa, WAVES)
    cb: gl.constexpr = _m256_copy_layout(sb, WAVES, neighboring_columns=not WIDE)

    rm = gl.arange(0, BM, gl.SliceLayout(1, mma))
    rn = gl.arange(0, BN, gl.SliceLayout(0, mma))


    if not WIDE:
        head_col = head * N + col_tile * BN
    row_head = row_tile * BM * H + head
    out_base = row_head * N + col_tile * BN
    out_offsets = rm[:, None] * H * N + rn[None, :]

    gate = gl.amd.cdna4.buffer_load(G + out_base, out_offsets).to(gl.float32)

    am = gl.arange(0, BM, gl.SliceLayout(1, ca))
    ak = gl.arange(0, BK, gl.SliceLayout(0, ca))
    bk = gl.arange(0, BK, gl.SliceLayout(1, cb))
    bn = gl.arange(0, BN, gl.SliceLayout(0, cb))
    a_offsets = am[:, None] * H * K + ak[None, :]
    b_offsets = bk[:, None] + bn[None, :] * WN
    if WIDE:
        X += row_head * K
        W += head * WH + col_tile * BN * WN
    else:
        a_offsets += row_head * K

        if WH == N * WN:
            W += head_col * WN
        else:
            W += head * WH + col_tile * BN * WN
    a0 = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], sa)
    b0 = gl.allocate_shared_memory(gl.bfloat16, [BK, BN], sb)
    _m256_stage_panel((b0, W, b_offsets), (a0, X, a_offsets), 0)

    exponential = gl.exp2(-gate * 1.4426950408889634)

    sigmoid = (1.0 / (1.0 + exponential)).to(gl.bfloat16).to(gl.float32)
    a1 = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], sa)
    b1 = gl.allocate_shared_memory(gl.bfloat16, [BK, BN], sb)

    a2 = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], sa)
    b2 = gl.allocate_shared_memory(gl.bfloat16, [BK, BN], sb)
    if not WIDE:
        a3 = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], sa)
        b3 = gl.allocate_shared_memory(gl.bfloat16, [BK, BN], sb)
    if WIDE:
        _m256_stage_panel((b1, W, b_offsets), (a1, X, a_offsets), BK)
    else:
        _m256_stage_panel((a1, X, a_offsets), (b1, W, b_offsets), BK)

    acc = gl.zeros((BM, BN), gl.float32, mma)
    async_copy.wait_group(1)
    acc = _m256_consume_panel(a0, b0, acc, ad, bd)
    _m256_stage_panel((a2, X, a_offsets), (b2, W, b_offsets), 2 * BK)
    async_copy.wait_group(1)
    if WIDE:


        a0._keep_alive()
        b0._keep_alive()
        a3 = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], sa)
        b3 = gl.allocate_shared_memory(gl.bfloat16, [BK, BN], sb)
    acc = _m256_consume_panel(a1, b1, acc, ad, bd)
    _m256_stage_panel((a3, X, a_offsets), (b3, W, b_offsets), 3 * BK)
    async_copy.wait_group(1)
    acc = _m256_consume_panel(a2, b2, acc, ad, bd)
    async_copy.wait_group(0)
    acc = _m256_consume_panel(a3, b3, acc, ad, bd)


    reconstructed = acc.to(gl.bfloat16).to(gl.float32)
    result = (reconstructed * sigmoid).to(gl.bfloat16)
    gl.store(Y + out_base + out_offsets, result)


@gluon.jit
def _m256_fused_general(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
):

    BM: gl.constexpr = 16
    BN: gl.constexpr = 32
    BK: gl.constexpr = 32
    pid = gl.program_id(0)
    col_tile = pid % gl.cdiv(N, BN)
    row_tile = (pid // gl.cdiv(N, BN)) % gl.cdiv(M, BM)
    head = pid // (gl.cdiv(M, BM) * gl.cdiv(N, BN))
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [2, 2])
    la: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
    if WK == 1:
        lb: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 4], [0, 1])
    else:
        lb: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)

    head = head.to(gl.int64)
    am = row_tile.to(gl.int64) * BM + gl.arange(0, BM, gl.SliceLayout(1, la))
    ak = gl.arange(0, BK, gl.SliceLayout(0, la)).to(gl.int64)
    bk = gl.arange(0, BK, gl.SliceLayout(1, lb)).to(gl.int64)
    bn = col_tile.to(gl.int64) * BN + gl.arange(0, BN, gl.SliceLayout(0, lb))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for kk in range(gl.cdiv(K, BK)):
        a_k = kk * BK + ak
        b_k = kk * BK + bk
        a = gl.load(X + am[:, None] * H * K + head * K + a_k[None, :],
                    (am[:, None] < M) & (a_k[None, :] < K), other=0)
        b = gl.load(W + head * WH + b_k[:, None] * WK + bn[None, :] * WN,
                    (b_k[:, None] < K) & (bn[None, :] < N), other=0)
        a = gl.convert_layout(a, ad)
        b = gl.convert_layout(b, bd)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    rm = row_tile.to(gl.int64) * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    rn = col_tile.to(gl.int64) * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    offsets = rm[:, None] * H * N + head * N + rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    gate = gl.load(G + offsets, mask, other=0).to(gl.float32)
    sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    value = acc.to(gl.bfloat16).to(gl.float32)
    gl.store(Y + offsets, (value * sigmoid).to(gl.bfloat16), mask)


def mla_vc_output_gate_m256(latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:

    assert latent.ndim == weight.ndim == 3
    m, h, k = latent.shape
    assert m > 0 and h > 0 and k > 0 and weight.shape[:2] == (h, k)
    n = weight.shape[2]
    assert n > 0 and gate.shape == (m, h * n)
    assert latent.dtype is weight.dtype is gate.dtype is torch.bfloat16
    assert latent.device == weight.device == gate.device
    assert latent.is_contiguous() and gate.is_contiguous()
    assert all(s > 0 for s in weight.stride())
    output = latent.new_empty((m, h * n))
    wh, wk, wn = weight.stride()
    aligned_weights = wk == 1 and wn % 8 == 0 and wh % 8 == 0
    aligned_storage = latent.storage_offset() % 8 == 0 and weight.storage_offset() % 8 == 0
    weight_span = (h - 1) * wh + (k - 1) * wk + (n - 1) * wn + 1
    buffer_safe = max(m * h * k, m * h * n, weight_span) < 2**30
    if (m in (128, 256) and k == 512 and n == 128
            and aligned_weights and aligned_storage and buffer_safe):
        bn, waves = (64, 8)
        grid = (n // bn, h, m // 32)
        _m256_fused_staged[grid](
            latent, weight, gate, output, m, h, k, n, wh, wn,
            BN=bn, num_warps=waves, enable_fp_fusion=False,
        )
    else:
        _m256_fused_general[(triton.cdiv(m, 16) * h * triton.cdiv(n, 32)),](
            latent, weight, gate, output, m, h, k, n, wh, wk, wn,
            num_warps=4, enable_fp_fusion=False,
        )
    return output


@gluon.jit
def _m32_rounded_gate_product(accumulator, sigmoid):

    reconstructed = accumulator.to(gl.bfloat16).to(gl.float32)
    return (reconstructed * sigmoid).to(gl.bfloat16)


@gluon.jit
def _m32_flat_m32_reconstruct_and_gate(
    latent, weight, gate, output,
    M: gl.constexpr, H: gl.constexpr, N: gl.constexpr,
    WEIGHT_HEAD_STRIDE: gl.constexpr, WEIGHT_N_STRIDE: gl.constexpr,
):
    BM: gl.constexpr = 16
    BN: gl.constexpr = 16
    K: gl.constexpr = 512
    BK: gl.constexpr = 256
    tile = gl.program_id(0).to(gl.uint32)

    gl.assume(tile < H * (M // BM) * (N // BN))
    head = tile // ((M // BM) * (N // BN))
    row_start = (tile // (N // BN) % (M // BM)) * BM
    col_start = (tile % (N // BN)) * BN
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1],
    )

    flat_layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
    flat_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], shape=[BM * BK], order=[0]
    )
    latent_first = gl.allocate_shared_memory(gl.bfloat16, [BM * BK], flat_shared_layout)
    weight_first = gl.allocate_shared_memory(gl.bfloat16, [BM * BK], flat_shared_layout)
    latent_second = gl.allocate_shared_memory(gl.bfloat16, [BM * BK], flat_shared_layout)
    weight_second = gl.allocate_shared_memory(gl.bfloat16, [BM * BK], flat_shared_layout)
    row_head = row_start * H + head
    rows = gl.arange(0, BM, gl.SliceLayout(1, mma_layout))
    cols = col_start + gl.arange(0, BN, gl.SliceLayout(0, mma_layout))
    offsets = (rows[:, None] * H + row_head) * N + cols[None, :]
    gate_value = gl.load(gate + offsets).to(gl.float32)
    index = gl.arange(0, BM * BK, flat_layout)
    vector = index // BK
    reduction = index % BK
    latent_offsets = (vector * H + row_head) * K + reduction
    latent_base = latent
    weight_offsets = reduction + vector * WEIGHT_N_STRIDE
    weight_base = weight + (col_start * WEIGHT_N_STRIDE + head * WEIGHT_HEAD_STRIDE)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(latent_first, latent_base, latent_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(weight_first, weight_base, weight_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(latent_second, latent_base + BK, latent_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(2)
    a_first = gl.amd.cdna4.async_copy.load_shared_relaxed(
        latent_first.reshape([BM, BK]), gl.DotOperandLayout(0, mma_layout, 8)
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(weight_second, weight_base + BK, weight_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    exponent = gl.exp2(gate_value * -1.4426950408889634)
    gl.amd.cdna4.async_copy.wait_group(2)
    b_first = gl.amd.cdna4.async_copy.load_shared_relaxed(
        weight_first.reshape([BN, BK]).permute((1, 0)), gl.DotOperandLayout(1, mma_layout, 8)
    )
    sigmoid = 1.0 / (1.0 + exponent)
    accumulator = gl.full((BM, BN), 0, gl.float32, mma_layout)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_second = gl.amd.cdna4.async_copy.load_shared_relaxed(
        latent_second.reshape([BM, BK]), gl.DotOperandLayout(0, mma_layout, 8)
    )
    gl.amd.cdna4.async_copy.wait_group(0)
    b_second = gl.amd.cdna4.async_copy.load_shared_relaxed(
        weight_second.reshape([BN, BK]).permute((1, 0)), gl.DotOperandLayout(1, mma_layout, 8)
    )
    accumulator = gl.amd.cdna4.mfma(a_first, b_first, accumulator)
    accumulator = gl.amd.cdna4.mfma(a_second, b_second, accumulator)

    sigmoid = sigmoid.to(gl.bfloat16).to(gl.float32)
    result = _m32_rounded_gate_product(accumulator, sigmoid)
    gl.store(output + offsets, result, cache_modifier=".wt")


def mla_vc_output_gate_m32(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:

    m, h, k = latent.shape
    n = weight.shape[2]
    weight_head_stride, weight_k_stride, weight_n_stride = weight.stride()
    assert m in (32, 64) and k == 512 and n % 32 == 0
    assert weight_k_stride == 1

    assert (
        weight_head_stride % 8 == 0
        and weight_n_stride % 8 == 0
        and (
            (h - 1) * weight_head_stride + (n - 1) * weight_n_stride + k
            < 2**30
        )
        and m * h * k < 2**30
    )
    output = torch.empty((m, h * n), device=latent.device, dtype=latent.dtype)
    _m32_flat_m32_reconstruct_and_gate[(h * (m // 16) * (n // 16),)](
        latent, weight, gate, output, m, h, n,
        weight_head_stride, weight_n_stride,
        num_warps=1,
        enable_fp_fusion=False,
        waves_per_eu=0,
    )
    return output


@gluon.jit
def _m4_fold_four_gate(
    X, W, G, Y,
    H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
    BM: gl.constexpr,
):

    PM: gl.constexpr = BM * 4
    TRANSPOSED: gl.constexpr = BM == 2
    PACK: gl.constexpr = 16 if BM == 4 else 8
    if BM == 1:
        head = gl.program_id(0)
        row_start = gl.program_id(1)
    else:
        row_start = gl.program_id(0) * BM
        head = gl.program_id(1)
    col_start = gl.program_id(2) * 8
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=TRANSPOSED,
        warps_per_cta=[1, 1],
    )
    la: gl.constexpr = gl.DotOperandLayout(0, mma, PACK)
    lb: gl.constexpr = gl.DotOperandLayout(1, mma, PACK)
    vr = gl.arange(0, PM, gl.SliceLayout(1, la))
    ka = gl.arange(0, 128, gl.SliceLayout(0, la))
    kb = gl.arange(0, 128, gl.SliceLayout(1, lb))
    vc = gl.arange(0, 32, gl.SliceLayout(0, lb))
    axk = (ka[None, :] // PACK * 4 + vr[:, None] % 4) * PACK + ka[None, :] % PACK
    bwk = (kb[:, None] // PACK * 4 + vc[None, :] % 4) * PACK + kb[:, None] % PACK
    if BM == 4:
        xp = X + (row_start * H + head) * K
    else:
        xp = X + row_start * H * K + head * K
    wp = W + head * WH + col_start * WN
    xoff = (vr // 4)[:, None] * H * K + axk
    woff = bwk * WK + (vc // 4)[None, :] * WN
    if TRANSPOSED:
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [], [[0, 4], [0, 0], [1 % BM, 0], [2 % BM, 0], [0, 1], [0, 2]],
            [], [], [BM, 8],
        )
    else:
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [], [[0, 4], [0, 0], [0, 1], [0, 2], [1 % BM, 0], [2 % BM, 0]],
            [], [], [BM, 8],
        )
    rr = gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    cc = gl.arange(0, 8, gl.SliceLayout(0, out_layout))
    if BM == 4:
        base = (row_start * H + head) * N + col_start
    else:
        base = row_start * H * N + head * N + col_start
    offsets = rr[:, None] * H * N + cc[None, :]
    if BM != 4:
        gate = gl.amd.cdna4.buffer_load(G + base, offsets).to(gl.float32)
        sigmoid = _m1_rounded_sigmoid(gate)
    a = gl.load(xp + xoff)
    b = gl.load(wp + woff)
    acc = gl.amd.cdna4.mfma(a, b, gl.zeros((PM, 32), gl.float32, mma))
    if TRANSPOSED:
        even, odd = gl.split(acc.reshape((PM, 8, 2, 2)))
        a0, a2 = gl.split(even)
        a1, a3 = gl.split(odd)
        rows = gl.arange(0, PM, gl.SliceLayout(1, a0.type.layout))
        lower = gl.where(rows[:, None] % 4 < 2, a0, a2)
        upper = gl.where(rows[:, None] % 4 < 2, a1, a3)
        partial = gl.where(rows[:, None] % 2 == 0, lower, upper)
        compact: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 1, 0]],
            [[0, 0, 4], [0, 2, 0], [1 % BM, 0, 0], [2 % BM, 0, 0], [0, 0, 1], [0, 0, 2]],
            [], [], [BM, 4, 8],
        )
        folded = gl.convert_layout(partial.reshape((BM, 4, 8)), compact)
        value = gl.sum(folded, 1)
    else:
        even, odd = gl.split(acc.reshape((BM, 2, 2, 32)).permute((0, 1, 3, 2)))
        a0, a2 = gl.split(even.permute((0, 2, 1)))
        a1, a3 = gl.split(odd.permute((0, 2, 1)))
        cols = gl.arange(0, 32, gl.SliceLayout(0, a0.type.layout))
        lower = gl.where(cols[None, :] % 4 < 2, a0, a2)
        upper = gl.where(cols[None, :] % 4 < 2, a1, a3)
        partial = gl.where(cols[None, :] % 2 == 0, lower, upper)
        compact: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 0, 1]],
            [[0, 4, 0], [0, 0, 2], [0, 1, 0], [0, 2, 0], [1 % BM, 0, 0], [2 % BM, 0, 0]],
            [], [], [BM, 8, 4],
        )
        folded = gl.convert_layout(partial.reshape((BM, 8, 4)), compact)
        value = gl.sum(folded, 2)
    value = gl.convert_layout(value, out_layout).to(gl.bfloat16).to(gl.float32)
    if BM == 4:
        gate = gl.amd.cdna4.buffer_load(G + base, offsets).to(gl.float32)
        sigmoid = _m1_rounded_sigmoid(gate)
    gl.amd.cdna4.buffer_store((value * sigmoid).to(gl.bfloat16), Y + base, offsets)


def mla_vc_output_gate_m4(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:

    assert latent.ndim == weight.ndim == 3
    m, h, k = latent.shape
    assert m > 0 and h > 0 and k > 0 and weight.shape[:2] == (h, k)
    n = weight.shape[2]
    assert n > 0 and gate.shape == (m, h * n)
    assert latent.dtype is weight.dtype is gate.dtype is torch.bfloat16
    assert latent.device == weight.device == gate.device
    assert latent.is_contiguous() and gate.is_contiguous()
    assert all(stride > 0 for stride in weight.stride())
    wh, wk, wn = weight.stride()
    output = torch.empty((m, h * n), device=latent.device, dtype=latent.dtype)


    offset_bound = max(
        wh, wk, wn,
        latent.storage_offset() + max(m, 8) * h * k,
        gate.storage_offset() + max(m, 8) * h * n,
        weight.storage_offset() + (h - 1) * wh + (k - 1) * wk + (n - 1) * wn + 1,
    )
    if k == 512 and n % 8 == 0 and m in (1, 2, 4, 8, 16) and offset_bound < 2**29:
        grid = (2, h, n // 8)
        _m4_fold_four_gate[grid](
            latent, weight, gate, output, h, k, n, wh, wk, wn,
            m // 2, num_warps=1, enable_fp_fusion=False,
        )
    else:
        _m1_general_gate[(h, triton.cdiv(m, 16), triton.cdiv(n, 16))](
            latent, weight, gate, output, m, h, k, n, wh, wk, wn,
            num_warps=1, enable_fp_fusion=False,
        )
    return output


@gluon.jit
def _m64_vc_gate_async(
    X, W, G, Y, WH: gl.constexpr, WN: gl.constexpr, SMALL_M: gl.constexpr,
):

    BN: gl.constexpr = 16 if SMALL_M else 32
    NW: gl.constexpr = 1 if SMALL_M else 2
    RM: gl.constexpr = 2 if SMALL_M else 4
    CN: gl.constexpr = 128 // BN
    pid = gl.program_id(0).to(gl.uint32)
    gl.assume(pid < 192)
    col0 = (pid % CN) * BN
    row0 = ((pid // CN) % RM) * 16
    head = pid // (CN * RM)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=SMALL_M,
        warps_per_cta=[1, NW],
    )
    dot_x: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_w: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    copy_x: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, NW], [1, 0])
    copy_w: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, NW], [0, 1])

    shared_x: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16],
         [1, 0], [2, 0], [4, 0], [8, 0], [0, 32], [0, 64]],
        [], [16, 128],
    )
    if SMALL_M:
        shared_w: gl.constexpr = gl.PaddedSharedLayout(
            [[512, 32]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
             [0, 1], [0, 2], [0, 4], [0, 8], [32, 0], [64, 0]],
            [], [128, BN],
        )
    else:
        shared_w: gl.constexpr = gl.PaddedSharedLayout(
            [[512, 32]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
             [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [32, 0], [64, 0]],
            [], [128, BN],
        )

    xr = gl.arange(0, 16, layout=gl.SliceLayout(1, copy_x))
    xk = gl.arange(0, 128, layout=gl.SliceLayout(0, copy_x))
    wk = gl.arange(0, 128, layout=gl.SliceLayout(1, copy_w))
    wc = gl.arange(0, BN, layout=gl.SliceLayout(0, copy_w))
    x_base = X + (head + row0 * 12) * 512
    if WH == 128 * WN:
        w_base = W + (head * 128 + col0) * WN
    else:
        w_base = W + head * WH + col0 * WN
    x_offsets = xr[:, None] * 6144 + xk[None, :]
    w_offsets = wk[:, None] + wc[None, :] * WN
    rows = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
    cols = gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
    if not SMALL_M and WH == 128 * WN:

        out_base = (head + row0 * 12) * 128 + col0
    else:
        out_base = row0 * 1536 + head * 128 + col0
    out_offsets = rows[:, None] * 1536 + cols[None, :]
    gate = gl.load(G + out_base + out_offsets).to(gl.float32)

    a0s = gl.allocate_shared_memory(gl.bfloat16, [16, 128], shared_x)
    b0s = gl.allocate_shared_memory(gl.bfloat16, [128, BN], shared_w)
    a1s = gl.allocate_shared_memory(gl.bfloat16, [16, 128], shared_x)
    b1s = gl.allocate_shared_memory(gl.bfloat16, [128, BN], shared_w)
    a2s = gl.allocate_shared_memory(gl.bfloat16, [16, 128], shared_x)
    b2s = gl.allocate_shared_memory(gl.bfloat16, [128, BN], shared_w)
    a3s = gl.allocate_shared_memory(gl.bfloat16, [16, 128], shared_x)
    b3s = gl.allocate_shared_memory(gl.bfloat16, [128, BN], shared_w)


    gl.amd.cdna4.async_copy.buffer_load_to_shared(a0s, x_base, x_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b0s, w_base, w_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a1s, x_base + 128, x_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b1s, w_base + 128, w_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a2s, x_base + 256, x_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b2s, w_base + 256, w_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a3s, x_base + 384, x_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b3s, w_base + 384, w_offsets)
    gl.amd.cdna4.async_copy.commit_group()


    denominator = 1.0 + gl.exp2(gate * -1.4426950408889634)
    gl.amd.cdna4.async_copy.wait_group(1)
    if SMALL_M:
        a1 = gl.amd.cdna4.async_copy.load_shared_relaxed(a1s, dot_x)
        a0 = gl.amd.cdna4.async_copy.load_shared_relaxed(a0s, dot_x)
        b0 = gl.amd.cdna4.async_copy.load_shared_relaxed(b0s, dot_w)
        b1 = gl.amd.cdna4.async_copy.load_shared_relaxed(b1s, dot_w)
    else:
        a0 = gl.amd.cdna4.async_copy.load_shared_relaxed(a0s, dot_x)
        a1 = gl.amd.cdna4.async_copy.load_shared_relaxed(a1s, dot_x)
        b0 = gl.amd.cdna4.async_copy.load_shared_relaxed(b0s, dot_w)
        b1 = gl.amd.cdna4.async_copy.load_shared_relaxed(b1s, dot_w)
    sigmoid = (1.0 / denominator).to(gl.bfloat16).to(gl.float32)
    acc = gl.zeros((16, BN), gl.float32, layout=mma)
    if SMALL_M:
        acc = gl.amd.cdna4.mfma(a0, b0, acc)
        acc = gl.amd.cdna4.mfma(a1, b1, acc)
        gl.amd.cdna4.async_copy.wait_group(0)
        a2 = gl.amd.cdna4.async_copy.load_shared_relaxed(a2s, dot_x)
        a3 = gl.amd.cdna4.async_copy.load_shared_relaxed(a3s, dot_x)
        b2 = gl.amd.cdna4.async_copy.load_shared_relaxed(b2s, dot_w)
        b3 = gl.amd.cdna4.async_copy.load_shared_relaxed(b3s, dot_w)
    else:
        acc = gl.amd.cdna4.mfma(a0, b0, acc)
        gl.amd.cdna4.async_copy.wait_group(0)
        b2 = gl.amd.cdna4.async_copy.load_shared_relaxed(b2s, dot_w)
        b3 = gl.amd.cdna4.async_copy.load_shared_relaxed(b3s, dot_w)
        a2 = gl.amd.cdna4.async_copy.load_shared_relaxed(a2s, dot_x)
        a3 = gl.amd.cdna4.async_copy.load_shared_relaxed(a3s, dot_x)
        acc = gl.amd.cdna4.mfma(a1, b1, acc)

    acc = gl.amd.cdna4.mfma(a2, b2, acc)
    acc = gl.amd.cdna4.mfma(a3, b3, acc)
    result = (acc.to(gl.bfloat16).to(gl.float32) * sigmoid).to(gl.bfloat16)
    if SMALL_M:
        gl.store(Y + out_base + out_offsets, result, cache_modifier=".wt")
    elif WH == 128 * WN:

        gl.amd.cdna4.buffer_store(
            result, Y + out_base, out_offsets, cache=".wt",
        )
    else:
        gl.store(Y + out_base + out_offsets, result)


def mla_vc_output_gate_m64(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor,
) -> torch.Tensor:

    m, h, k = latent.shape
    assert m in (32, 64) and (h, k) == (12, 512)
    assert weight.shape == (12, 512, 128) and gate.shape == (m, 1536)
    assert latent.dtype is weight.dtype is gate.dtype is torch.bfloat16
    assert latent.device == weight.device == gate.device
    assert latent.is_contiguous() and gate.is_contiguous()
    wh, wk, wn = weight.stride()
    assert wk == 1 and wn > 0 and wh > 0 and wn % 8 == 0 and wh % 8 == 0
    assert (11 * wh + 511 + 127 * wn + 1) * 2 < 2**31
    assert latent.storage_offset() % 8 == weight.storage_offset() % 8 == 0
    output = torch.empty((m, 1536), dtype=latent.dtype, device=latent.device)
    _m64_vc_gate_async[(192,)](
        latent, weight, gate, output, wh, wn, m == 32,
        num_warps=2, waves_per_eu=3,
        enable_fp_fusion=False,
    )
    return output


@gluon.jit
def _m8_fold4_diagonal(acc, VCOLS: gl.constexpr):

    FOLD: gl.constexpr = 4
    layout: gl.constexpr = gl.BlockedLayout(
        [4, 1], [4, 16], [1, gl.num_warps()], [1, 0],
    )
    acc = gl.convert_layout(acc, layout)
    packed = gl.permute(gl.reshape(acc, (16 // FOLD, FOLD, VCOLS)), (0, 2, 1))
    even, odd = gl.split(gl.reshape(packed, (4, VCOLS, 2, 2)))
    p0, p2 = gl.split(even)
    p1, p3 = gl.split(odd)
    cc = gl.arange(0, VCOLS, gl.SliceLayout(0, p0.type.layout))
    low = gl.where((cc[None, :] & 1) == 0, p0, p1)
    high = gl.where((cc[None, :] & 1) == 0, p2, p3)
    partials = gl.where((cc[None, :] & 2) == 0, low, high)
    return gl.sum(gl.reshape(partials, (16 // FOLD, VCOLS // FOLD, FOLD)), 2)


@gluon.jit
def _m8_direct_fold2(acc):

    layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, 1], [1, 0])
    acc = gl.convert_layout(acc, layout)
    packed = gl.permute(gl.reshape(acc, (4, 4, 32)), (0, 2, 1))
    even, odd = gl.split(gl.reshape(packed, (4, 32, 2, 2)))
    p0, p2 = gl.split(even)
    p1, p3 = gl.split(odd)
    cc = gl.arange(0, 32, gl.SliceLayout(0, p0.type.layout))
    own = gl.where((cc[None, :] & 1) == 0, p0, p3)
    other = gl.where((cc[None, :] & 1) == 0, p2, p1)


    partner = gl.inline_asm_elementwise(
        "s_nop 1\n\tv_mov_b32_dpp $0, $2 quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
        "v_mov_b32_dpp $1, $3 quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf bound_ctrl:1\n\ts_nop 0",
        constraints="=&v,=v,v,v", args=[other], dtype=gl.float32,
        is_pure=True, pack=2,
    )
    folded = own + partner
    return gl.reshape(gl.permute(gl.reshape(folded, (4, 16, 2)), (0, 2, 1)), (8, 16))


@gluon.jit
def _m8_decode_vc_gate(
    X, W, G, Y,
    M: gl.constexpr, H: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    WH: gl.constexpr, WK: gl.constexpr, WN: gl.constexpr,
):
    FOLD: gl.constexpr = 4 if M == 8 else 2
    BK: gl.constexpr = 128
    BM: gl.constexpr = 16 // FOLD
    BN: gl.constexpr = 32 // FOLD
    pid = gl.program_id(0)
    if M == 8:
        row_base = pid % (M // BM) * BM
        head = pid // (M // BM) % H
    else:

        pid = pid.to(gl.uint32)
        head = pid % H
        row_base = pid // H % (M // BM) * BM
    col_base = pid // (H * (M // BM)) * BN
    al: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [0, 1])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 1], [1, 0])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[1, 1],
    )
    if M == 8:
        gate_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[],
            lane_bases=[[0, 4], [0, 0], [0, 1], [0, 2], [1, 0], [2, 0]],
            warp_bases=[], block_bases=[], shape=[4, 8],
        )
    else:
        gate_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 8]],
            lane_bases=[[1, 0], [0, 1], [0, 2], [0, 4], [2, 0], [4, 0]],
            warp_bases=[], block_bases=[], shape=[8, 16],
        )
    om = row_base + gl.arange(0, BM, gl.SliceLayout(1, gate_layout))
    on = gl.arange(0, BN, gl.SliceLayout(0, gate_layout))
    output_base = head * N + col_base
    offsets = om[:, None] * H * N + on[None, :]
    if M == 16:
        gate = gl.amd.cdna4.buffer_load(G + output_base, offsets).to(gl.float32)
        sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    ar = gl.arange(0, 16, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    bc = gl.arange(0, 32, gl.SliceLayout(0, bl))
    a_k = (ar[:, None] % FOLD) * 8 + (ak[None, :] // 8) * (FOLD * 8) + ak[None, :] % 8
    b_k = (bc[None, :] % FOLD) * 8 + (bk[:, None] // 8) * (FOLD * 8) + bk[:, None] % 8
    a_base = X + head * K + row_base * H * K
    b_base = W + head * WH + col_base * WN
    a_offsets = ar[:, None] // FOLD * H * K + a_k
    b_offsets = b_k * WK + bc[None, :] // FOLD * WN


    acc = gl.zeros((16, 32), gl.float32, mma)
    for ki in range(K // (FOLD * BK)):
        a = gl.load(a_base + a_offsets + ki * BK * FOLD)
        b = gl.load(b_base + b_offsets + ki * BK * FOLD * WK)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
        acc = gl.amd.cdna4.mfma(a, b, acc)
    if M == 16:
        values = _m8_direct_fold2(acc)
    else:
        values = _m8_fold4_diagonal(acc, 32)
    values = gl.convert_layout(values, gate_layout)
    if M == 8:
        gate = gl.amd.cdna4.buffer_load(G + output_base, offsets).to(gl.float32)
        sigmoid = (1.0 / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    reconstructed = values.to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        ptr=Y + output_base, offsets=offsets,
        stored_value=(reconstructed * sigmoid).to(gl.bfloat16),
    )


def mla_vc_output_gate_m8(
    latent: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor,
) -> torch.Tensor:

    m, h, k = latent.shape
    n = weight.shape[2]
    output = torch.empty((m, h * n), dtype=latent.dtype, device=latent.device)
    fold = 4
    grid = (h * (m // (16 // fold)) * (n // (32 // fold)),)
    _m8_decode_vc_gate[grid](
        latent, weight, gate, output, m, h, k, n, *weight.stride(),
        num_warps=1, enable_fp_fusion=False,
    )
    return output
