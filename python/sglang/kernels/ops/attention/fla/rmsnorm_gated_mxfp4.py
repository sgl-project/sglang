# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Advanced Micro Devices, Inc.

"""Fused per-head gated RMSNorm and MXFP4 activation quantization."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.layernorm_gated import calc_rows_per_block
from sglang.srt.utils import cdiv, device_context

MXFP4_BLOCK_SIZE = 32
MXFP4_ROUND_UP = 1
MXFP4_ROUND_EVEN = 2
_MXFP4_SHUFFLE_GROUP_MULTIPLE = 8


def mxfp4_scale_shape(
    num_rows: int, num_groups: int, *, shuffle: bool
) -> tuple[int, int]:
    """Return the row-major or AITER ASM scale-buffer shape."""
    if not shuffle:
        return num_rows, num_groups
    return cdiv(num_rows, 256) * 256, cdiv(num_groups, 8) * 8


def _is_mxfp4_shuffled_width_supported(input_size: int) -> bool:
    """Check AITER ASM's eight-scale-group activation row alignment.

    The shuffled public producer uses its padded scale-group count as the
    packed-activation row pitch, so a contiguous output is valid only when
    K/32 needs no padding to the next eight-group tile.
    """
    return (
        input_size > 0
        and input_size % (MXFP4_BLOCK_SIZE * _MXFP4_SHUFFLE_GROUP_MULTIPLE) == 0
    )


def _is_rmsnorm_gated_mxfp4_layout_supported(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_heads: int,
    activation: str,
    shuffle_scales: bool = False,
) -> bool:
    """Check the shape, dtype, and strides required by the fused kernel."""
    if x.ndim != 2 or z.ndim not in (2, 3) or weight.ndim != 1:
        return False
    if num_heads <= 0 or x.shape[0] == 0 or x.shape[0] % num_heads != 0:
        return False

    head_dim = x.shape[1]
    if (
        head_dim == 0
        or head_dim % MXFP4_BLOCK_SIZE != 0
        or head_dim > 65536 // x.element_size()
    ):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16) or z.dtype != x.dtype:
        return False
    if weight.dtype != x.dtype or weight.shape != (head_dim,):
        return False
    if x.stride(-1) != 1 or z.stride(-1) != 1 or weight.stride(-1) != 1:
        return False

    if z.ndim == 2 and z.shape != x.shape:
        return False
    if z.ndim == 3:
        num_tokens = x.shape[0] // num_heads
        if z.shape != (num_tokens, num_heads, head_dim):
            return False
    if shuffle_scales and not _is_mxfp4_shuffled_width_supported(num_heads * head_dim):
        return False
    return activation in ("silu", "swish", "sigmoid")


def can_use_rmsnorm_gated_mxfp4(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_heads: int,
    activation: str,
    shuffle_scales: bool = False,
) -> bool:
    """Check the complete device and layout contract for the fused kernel."""
    return (
        x.is_cuda
        and x.device == z.device
        and x.device == weight.device
        and _is_rmsnorm_gated_mxfp4_layout_supported(
            x,
            z,
            weight,
            num_heads=num_heads,
            activation=activation,
            shuffle_scales=shuffle_scales,
        )
    )


@triton.jit
def _mxfp4_quantize(
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP4_BLOCK_SIZE: tl.constexpr,
    ROUND_MODE: tl.constexpr,
    ROUND_UP_MODE: tl.constexpr,
):
    """Quantize FP32 rows into packed E2M1 values plus E8M0 scales."""
    num_groups: tl.constexpr = BLOCK_N // FP4_BLOCK_SIZE
    x = tl.reshape(x, [BLOCK_M, num_groups, FP4_BLOCK_SIZE])
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)

    if ROUND_MODE == ROUND_UP_MODE:
        # AITER's default MX mode: ceil_pow2(amax / 6), with the same nonzero
        # floor used by dynamic_per_group_scaled_quant.
        amax = tl.maximum(amax, 1.0e-10)
        scale_bits = (amax * (1.0 / 6.0)).to(tl.uint32, bitcast=True)
        scale_exp = (scale_bits >> 23) & 0xFF
        has_mantissa = (scale_bits & 0x7FFFFF) != 0
        scale_exp += (has_mantissa & (scale_exp < 0xFF)).to(tl.uint32)
        scale = (scale_exp << 23).to(tl.float32, bitcast=True)
        scale_e8m0 = scale_exp.to(tl.uint8)
        quant_scale = 1.0 / scale
    else:
        # AITER dynamic_mxfp4_quant's Quark-EVEN mode.
        rounded = amax.to(tl.int32, bitcast=True)
        rounded = (rounded + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        rounded = rounded.to(tl.float32, bitcast=True)
        scale_unbiased = tl.floor(tl.log2(rounded)) - 2
        scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
        scale_e8m0 = scale_unbiased.to(tl.uint8) + 127
        quant_scale = tl.exp2(-scale_unbiased)

    qx = (x * quant_scale).to(tl.uint32, bitcast=True)

    sign = qx & 0x80000000
    qx = qx ^ sign
    qx_fp32 = qx.to(tl.float32, bitcast=True)

    saturate_mask = qx_fp32 >= 6.0
    denormal_mask = (~saturate_mask) & (qx_fp32 < 1.0)
    normal_mask = ~(saturate_mask | denormal_mask)

    denorm_exp: tl.constexpr = (127 - 1) + (23 - 1) + 1
    denorm_mask_int: tl.constexpr = denorm_exp << 23
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)
    denormal_x = (qx_fp32 + denorm_mask_float).to(tl.uint32, bitcast=True)
    denormal_x = (denormal_x - denorm_mask_int).to(tl.uint8)

    mantissa_odd = (qx >> (23 - 1)) & 1
    rounding_bias: tl.constexpr = ((1 - 127) << 23) + (1 << 21) - 1
    # qx has had its sign bit removed, so signed arithmetic can apply the
    # negative exponent-bias adjustment without an unsigned wraparound.
    normal_x = qx.to(tl.int32, bitcast=True)
    normal_x = normal_x + rounding_bias + mantissa_odd.to(tl.int32)
    normal_x = (normal_x >> (23 - 1)).to(tl.uint8)

    e2m1 = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1 = tl.where(normal_mask, normal_x, e2m1)
    e2m1 = tl.where(denormal_mask, denormal_x, e2m1)
    e2m1 = e2m1 | (sign >> 28).to(tl.uint8)

    e2m1 = tl.reshape(e2m1, [BLOCK_M, num_groups, 16, 2])
    evens, odds = tl.split(e2m1)
    packed = (evens | (odds << 4)).reshape(BLOCK_M, BLOCK_N // 2)
    return packed, scale_e8m0.reshape(BLOCK_M, num_groups)


@triton.jit
def _rmsnorm_gated_mxfp4_kernel(
    X,
    Z,
    W,
    Out,
    Scales,
    stride_x_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    num_head_rows,
    eps,
    SCALE_N_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    FP4_BLOCK_SIZE: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    ACTIVATION: tl.constexpr,
    ROUND_MODE: tl.constexpr,
    ROUND_UP_MODE: tl.constexpr,
    SHUFFLE_SCALES: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows[:, None] < num_head_rows
    col_mask = cols[None, :] < HEAD_DIM
    mask = row_mask & col_mask

    x = tl.load(
        X + rows[:, None] * stride_x_row + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / HEAD_DIM
    rstd = tl.rsqrt(variance + eps)
    weight = tl.load(W + cols, mask=cols < HEAD_DIM, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * weight[None, :]

    token_rows = rows // NUM_HEADS
    head_rows = rows % NUM_HEADS
    if Z_IS_3D:
        z_row_offsets = (
            token_rows[:, None] * stride_z_token + head_rows[:, None] * stride_z_head
        )
    else:
        z_row_offsets = rows[:, None] * stride_z_row
    z = tl.load(
        Z + z_row_offsets + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if ACTIVATION == "silu" or ACTIVATION == "swish":
        y *= z * tl.sigmoid(z)
    else:
        y *= tl.sigmoid(z)

    # The unfused chain stores gated RMSNorm in the activation dtype before
    # quantizing it. Preserve that rounding point without the HBM round-trip.
    y = y.to(X.dtype.element_ty).to(tl.float32)
    packed, scales = _mxfp4_quantize(
        y,
        BLOCK_M=ROWS_PER_BLOCK,
        BLOCK_N=BLOCK_N,
        FP4_BLOCK_SIZE=FP4_BLOCK_SIZE,
        ROUND_MODE=ROUND_MODE,
        ROUND_UP_MODE=ROUND_UP_MODE,
    )

    packed_cols = tl.arange(0, BLOCK_N // 2)
    packed_mask = (rows[:, None] < num_head_rows) & (
        packed_cols[None, :] < HEAD_DIM // 2
    )
    tl.store(
        Out + rows[:, None] * (HEAD_DIM // 2) + packed_cols[None, :],
        packed,
        mask=packed_mask,
    )

    groups_per_head: tl.constexpr = HEAD_DIM // FP4_BLOCK_SIZE
    block_groups: tl.constexpr = BLOCK_N // FP4_BLOCK_SIZE
    local_groups = tl.arange(0, block_groups)
    global_groups = head_rows[:, None] * groups_per_head + local_groups[None, :]
    scale_mask = (rows[:, None] < num_head_rows) & (
        local_groups[None, :] < groups_per_head
    )
    if SHUFFLE_SCALES:
        scale_offsets = (
            (token_rows[:, None] // 32 * SCALE_N_PAD) * 32
            + (global_groups // 8) * 256
            + (global_groups % 4) * 64
            + (token_rows[:, None] % 16) * 4
            + ((global_groups % 8) // 4) * 2
            + ((token_rows[:, None] % 32) // 16)
        )
    else:
        scale_offsets = token_rows[:, None] * SCALE_N_PAD + global_groups
    tl.store(Scales + scale_offsets, scales, mask=scale_mask)


def rmsnorm_gated_mxfp4_quant(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    num_heads: int,
    activation: str,
    round_mode: int,
    shuffle_scales: bool,
    use_native_dtypes: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the ``(packed_fp4, e8m0_scales)`` linear-input contract."""
    if not can_use_rmsnorm_gated_mxfp4(
        x,
        z,
        weight,
        num_heads=num_heads,
        activation=activation,
        shuffle_scales=shuffle_scales,
    ):
        raise ValueError("unsupported gated RMSNorm MXFP4 device, shape, or layout")
    if round_mode not in (MXFP4_ROUND_UP, MXFP4_ROUND_EVEN):
        raise ValueError(f"unsupported MXFP4 round mode: {round_mode}")

    num_head_rows, head_dim = x.shape
    num_rows = num_head_rows // num_heads
    num_groups = num_heads * head_dim // MXFP4_BLOCK_SIZE
    scale_m, scale_n = mxfp4_scale_shape(num_rows, num_groups, shuffle=shuffle_scales)
    out = torch.empty(
        (num_rows, num_heads * head_dim // 2),
        dtype=torch.uint8,
        device=x.device,
    )
    scales = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=x.device)

    block_n = triton.next_power_of_2(head_dim)
    rows_per_block = calc_rows_per_block(num_head_rows, x.device)
    num_warps = min(max(block_n // 256, 1), 8)
    z_is_3d = z.ndim == 3

    with device_context(x.device):
        _rmsnorm_gated_mxfp4_kernel[(cdiv(num_head_rows, rows_per_block),)](
            X=x,
            Z=z,
            W=weight,
            Out=out,
            Scales=scales,
            stride_x_row=x.stride(0),
            stride_z_row=z.stride(0) if not z_is_3d else 0,
            stride_z_token=z.stride(0) if z_is_3d else 0,
            stride_z_head=z.stride(1) if z_is_3d else 0,
            num_head_rows=num_head_rows,
            eps=eps,
            SCALE_N_PAD=scale_n,
            HEAD_DIM=head_dim,
            NUM_HEADS=num_heads,
            BLOCK_N=block_n,
            ROWS_PER_BLOCK=rows_per_block,
            FP4_BLOCK_SIZE=MXFP4_BLOCK_SIZE,
            Z_IS_3D=z_is_3d,
            ACTIVATION=activation,
            ROUND_MODE=round_mode,
            ROUND_UP_MODE=MXFP4_ROUND_UP,
            SHUFFLE_SCALES=shuffle_scales,
            num_warps=num_warps,
        )

    if use_native_dtypes:
        from aiter.utility import dtypes

        out = out.view(dtypes.fp4x2)
        scales = scales.view(dtypes.fp8_e8m0)
    return out, scales
