# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
# Modified by NVIDIA: add static FP8 output after activation-dtype rounding.
# Modified by NVIDIA: expose preallocated launches and publish the graph dependency
# before waiting for input readiness; the normalization arithmetic is unchanged.

"""Preallocated gated normalization with an early dependent-launch trigger."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.layernorm_gated import calc_rows_per_block


@triton.jit
def _gated_norm_early_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    Q=None,
    S=None,
    QUANT: tl.constexpr = False,
    STATIC_QUANT: tl.constexpr = False,
):
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        if Z_IS_3D:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        Z_base = Z + z_row_offsets + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # Compute mean and variance per row (reduce along axis 1)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        # Store mean for each row
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        # Broadcast mean back to 2D for subtraction
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        mean = 0.0  # Placeholder for RMS norm

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        if Z_IS_3D:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        Z_base = Z + z_row_offsets + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    if STATIC_QUANT:
        y = y.to(Y.dtype.element_ty).to(tl.float32)
        y_s = tl.load(S).to(tl.float32)
        y_q = tl.clamp(y * (1.0 / y_s), -448.0, 448.0).to(Q.dtype.element_ty)
        tl.store(Q + rows[:, None] * stride_y_row + col_offsets, y_q, mask=mask)
    elif QUANT:  # the block is one token: per-token FP8 as _per_token_group_quant_8bit
        y = tl.where(mask, y.to(Y.dtype.element_ty).to(tl.float32), 0.0)
        y_s = tl.maximum(tl.max(tl.abs(y)), 1e-10) / 448.0
        y_q = tl.clamp(y * (1.0 / y_s), -448.0, 448.0).to(Q.dtype.element_ty)
        tl.store(Q + rows[:, None] * N + col_offsets, y_q, mask=mask)
        tl.store(S + tl.program_id(0), y_s)
    else:
        # Write output
        tl.store(Y_base, y, mask=mask)


def norm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    quant_out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
) -> None:
    """Write gated RMSNorm followed by static FP8 conversion into caller buffers.

    ``x`` is BF16 with shape ``[tokens * heads, 128]``. The BF16 gate ``z``
    has shape ``[tokens, heads, 128]`` and may have strided token/head axes.
    ``weight`` is contiguous with 128 elements, ``scale`` is a device FP32
    scalar, and both destinations are contiguous. The caller must provide
    non-overlapping output storage on the same CUDA device.

    This entry point requires programmatic dependent launch support. It
    publishes its dependent at entry, then waits before reading activations.
    Normalization rounds through BF16 before static FP8 conversion, as in
    the unfused operator. No output or workspace is allocated.
    """
    assert x.ndim == 2 and x.shape[1] == 128
    assert z.ndim == 3 and z.shape[1] in (8, 16)
    heads = z.shape[1]
    assert x.shape[0] > 0 and x.shape[0] % heads == 0
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert z.shape == (x.shape[0] // heads, heads, 128)
    assert z.dtype == torch.bfloat16 and z.stride(-1) == 1
    assert weight.shape == (128,) and weight.stride(0) == 1
    assert weight.dtype in (torch.bfloat16, torch.float32)
    assert scale.numel() == 1 and scale.dtype == torch.float32
    assert quant_out.shape == x.shape and quant_out.is_contiguous()
    assert quant_out.dtype == torch.float8_e4m3fn
    assert rstd.shape == (x.shape[0],) and rstd.is_contiguous()
    assert rstd.dtype == torch.float32
    assert x.is_cuda and all(
        tensor.device == x.device for tensor in (z, weight, scale, quant_out, rstd)
    )
    rows = x.shape[0]
    rows_per_block = calc_rows_per_block(rows, x.device)
    # Y supplies the activation dtype only; STATIC_QUANT does not write it.
    _gated_norm_early_kernel[(triton.cdiv(rows, rows_per_block), 1)](
        x,
        x,
        weight,
        None,
        z,
        None,
        rstd,
        x.stride(0),
        quant_out.stride(0),
        0,
        z.stride(0),
        z.stride(1),
        rows,
        128,
        eps,
        BLOCK_N=128,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=False,
        HAS_Z=True,
        Z_IS_3D=True,
        Z_HEADS=heads,
        NORM_BEFORE_GATE=True,
        IS_RMS_NORM=True,
        ACTIVATION="swish",
        USE_GDC=True,
        Q=quant_out,
        S=scale,
        STATIC_QUANT=True,
        num_warps=1,
        launch_pdl=True,
    )
