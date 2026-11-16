# SPDX-License-Identifier: Apache-2.0
"""BF16-native RMSNorm fusions shared by diffusion transformer models."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.fast_launch import replay_launches

MAX_HIDDEN_SIZE = 8192


@triton.jit
def _tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _rmsnorm_scale_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    scale_ptr,
    x_row_stride,
    scale_row_stride,
    seq_len,
    dim: tl.constexpr,
    eps: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_dim)
    mask = offsets < dim

    x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
    square = (x * x).to(tl.bfloat16)
    mean_square = (tl.sum(square, axis=0) / dim).to(tl.bfloat16)
    rstd = tl.rsqrt((mean_square + eps).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    batch = row // seq_len
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(
        scale_ptr + batch * scale_row_stride + offsets, mask=mask, other=0.0
    )
    y = (((x * rstd).to(tl.bfloat16) * weight).to(tl.bfloat16) * scale).to(tl.bfloat16)
    tl.store(y_ptr + row * dim + offsets, y, mask=mask)


@triton.jit
def _rmsnorm_tanh_residual_kernel(
    y_ptr,
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    x_row_stride,
    gate_row_stride,
    residual_row_stride,
    seq_len,
    dim: tl.constexpr,
    eps: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_dim)
    mask = offsets < dim

    x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
    square = (x * x).to(tl.bfloat16)
    mean_square = (tl.sum(square, axis=0) / dim).to(tl.bfloat16)
    rstd = tl.rsqrt((mean_square + eps).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    batch = row // seq_len
    gate = tl.load(gate_ptr + batch * gate_row_stride + offsets, mask=mask, other=0.0)
    residual = tl.load(
        residual_ptr + row * residual_row_stride + offsets, mask=mask, other=0.0
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    norm = ((x * rstd).to(tl.bfloat16) * weight).to(tl.bfloat16)
    gated = (_tanh(gate.to(tl.float32)).to(tl.bfloat16) * norm).to(tl.bfloat16)
    y = (residual + gated).to(tl.bfloat16)
    tl.store(y_ptr + row * dim + offsets, y, mask=mask)


@triton.jit
def _sum2_rmsnorm_tanh_residual_kernel(
    y_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    scale_weight_ptr,
    scale_ptr,
    gate_row_stride,
    residual_row_stride,
    scale_row_stride,
    seq_len,
    dim: tl.constexpr,
    eps: tl.constexpr,
    block_dim: tl.constexpr,
    with_scale: tl.constexpr,
):
    # _rmsnorm_tanh_residual_kernel on x = a + b, then _rmsnorm_scale_kernel on
    # its result: the same expressions, so the same bits as the three kernels
    row = tl.program_id(0)
    offsets = tl.arange(0, block_dim)
    mask = offsets < dim

    a = tl.load(a_ptr + row * dim + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + row * dim + offsets, mask=mask, other=0.0)
    x = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    square = (x * x).to(tl.bfloat16)
    mean_square = (tl.sum(square, axis=0) / dim).to(tl.bfloat16)
    rstd = tl.rsqrt((mean_square + eps).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    batch = row // seq_len
    gate = tl.load(gate_ptr + batch * gate_row_stride + offsets, mask=mask, other=0.0)
    residual = tl.load(
        residual_ptr + row * residual_row_stride + offsets, mask=mask, other=0.0
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    norm = ((x * rstd).to(tl.bfloat16) * weight).to(tl.bfloat16)
    gated = (_tanh(gate.to(tl.float32)).to(tl.bfloat16) * norm).to(tl.bfloat16)
    y = (residual + gated).to(tl.bfloat16)
    tl.store(y_ptr + row * dim + offsets, y, mask=mask)

    if with_scale:
        y = tl.where(mask, y, 0.0)
        square = (y * y).to(tl.bfloat16)
        mean_square = (tl.sum(square, axis=0) / dim).to(tl.bfloat16)
        rstd = tl.rsqrt((mean_square + eps).to(tl.bfloat16).to(tl.float32)).to(
            tl.bfloat16
        )
        scale_weight = tl.load(scale_weight_ptr + offsets, mask=mask, other=0.0)
        scale = tl.load(
            scale_ptr + batch * scale_row_stride + offsets, mask=mask, other=0.0
        )
        z = (((y * rstd).to(tl.bfloat16) * scale_weight).to(tl.bfloat16) * scale).to(
            tl.bfloat16
        )
        tl.store(z_ptr + row * dim + offsets, z, mask=mask)


def _flat_row_stride(x: torch.Tensor) -> int | None:
    if x.dim() < 2 or x.stride(-1) != 1:
        return None
    row_stride = x.stride(-2)
    expected_stride = row_stride * x.shape[-2]
    for dim in range(x.dim() - 3, -1, -1):
        if x.stride(dim) != expected_stride:
            return None
        expected_stride *= x.shape[dim]
    return row_stride


def _can_use_operand(
    x: torch.Tensor, weight: torch.Tensor, other: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and weight.is_cuda
        and other.is_cuda
        and x.device == weight.device == other.device
        and x.dtype == weight.dtype == other.dtype == torch.bfloat16
        and x.dim() >= 2
        and 0 < x.shape[-1] <= MAX_HIDDEN_SIZE
        and x.numel() > 0
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
        and other.dim() >= 2
        and other.shape[-1] == x.shape[-1]
        and other.numel() > 0
        and _flat_row_stride(x) is not None
        and _flat_row_stride(other) is not None
    )


@replay_launches
def rmsnorm_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Apply BF16-native ``RMSNorm(x) * scale`` or return ``None``."""
    if not _can_use_operand(x, weight, scale):
        return None

    dim = x.shape[-1]
    x_rows = x.numel() // dim
    scale_rows = scale.numel() // dim
    if x_rows % scale_rows != 0:
        return None

    x_row_stride = _flat_row_stride(x)
    scale_row_stride = _flat_row_stride(scale)
    if x_row_stride is None or scale_row_stride is None:
        return None

    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.get_device_module().device(x.device):
        _rmsnorm_scale_kernel[(x_rows,)](
            out.reshape(-1, dim),
            x,
            weight,
            scale,
            x_row_stride,
            scale_row_stride,
            x_rows // scale_rows,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            num_warps=8,
        )
    return out


@replay_launches
def rmsnorm_tanh_residual(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Apply BF16-native gated RMSNorm residual fusion or return ``None``."""
    if not _can_use_operand(x, weight, gate):
        return None
    if (
        residual.device != x.device
        or residual.dtype != x.dtype
        or residual.shape != x.shape
        or _flat_row_stride(residual) is None
    ):
        return None

    dim = x.shape[-1]
    x_rows = x.numel() // dim
    gate_rows = gate.numel() // dim
    if x_rows % gate_rows != 0:
        return None

    x_row_stride = _flat_row_stride(x)
    gate_row_stride = _flat_row_stride(gate)
    residual_row_stride = _flat_row_stride(residual)
    if x_row_stride is None or gate_row_stride is None or residual_row_stride is None:
        return None

    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.get_device_module().device(x.device):
        _rmsnorm_tanh_residual_kernel[(x_rows,)](
            out.reshape(-1, dim),
            x,
            gate,
            residual,
            weight,
            x_row_stride,
            gate_row_stride,
            residual_row_stride,
            x_rows // gate_rows,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            num_warps=8,
        )
    return out


@replay_launches
def rmsnorm_tanh_residual_sum2(
    a: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale_weight: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """``rmsnorm_tanh_residual(a + b, ...)`` in one kernel, or ``None``.

    ``a`` and ``b`` are a row-parallel layer's two partial sums as contiguous
    ``[rows, dim]``; either may be mapped from another device. With ``scale``,
    also returns ``rmsnorm_scale`` of the result under ``scale_weight``.
    """
    if not _can_use_operand(residual, weight, gate):
        return None
    dim = residual.shape[-1]
    rows = residual.numel() // dim
    gate_rows = gate.numel() // dim
    gate_row_stride = _flat_row_stride(gate)
    residual_row_stride = _flat_row_stride(residual)
    if (
        a.shape != (rows, dim)
        or b.shape != (rows, dim)
        or a.dtype != residual.dtype
        or b.dtype != residual.dtype
        or a.device != residual.device
        or b.device != residual.device
        or not a.is_contiguous()
        or not b.is_contiguous()
        or rows % gate_rows != 0
        or gate_row_stride is None
        or residual_row_stride is None
    ):
        return None
    with_scale = scale is not None
    scale_row_stride = 0
    if with_scale:
        if (
            scale_weight is None
            or not _can_use_operand(residual, scale_weight, scale)
            or scale.numel() // dim != gate_rows
        ):
            return None
        scale_row_stride = _flat_row_stride(scale)
        if scale_row_stride is None:
            return None

    out = torch.empty_like(residual, memory_format=torch.contiguous_format)
    scaled = (
        torch.empty_like(residual, memory_format=torch.contiguous_format)
        if with_scale
        else out
    )
    with torch.get_device_module().device(residual.device):
        _sum2_rmsnorm_tanh_residual_kernel[(rows,)](
            out.reshape(-1, dim),
            scaled.reshape(-1, dim),
            a,
            b,
            gate,
            residual,
            weight,
            scale_weight if with_scale else weight,
            scale if with_scale else gate,
            gate_row_stride,
            residual_row_stride,
            scale_row_stride,
            rows // gate_rows,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            with_scale=with_scale,
            num_warps=8,
        )
    return (out, scaled) if with_scale else out


__all__ = [
    "MAX_HIDDEN_SIZE",
    "rmsnorm_scale",
    "rmsnorm_tanh_residual",
    "rmsnorm_tanh_residual_sum2",
]
