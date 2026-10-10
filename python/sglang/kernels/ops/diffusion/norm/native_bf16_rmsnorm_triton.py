# SPDX-License-Identifier: Apache-2.0
"""BF16-native RMSNorm fusions shared by diffusion transformer models."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

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


import struct

from sglang.kernels.ops.diffusion.norm import fast_launch as _fast_launch

_RMSNORM_SCALE_LAUNCH = _fast_launch.CachedLaunch()
_RMSNORM_TANH_LAUNCH = _fast_launch.CachedLaunch()


def _f32_bits(value: float) -> int:
    """An fp32 kernel argument as the 4 bytes the driver will read."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _scale_signature(x, weight, scale, eps):
    """What a recorded rmsnorm_scale launch stays valid for.

    Strides belong in the key because the kernel indexes rows with them, and
    dtype because the compiled kernel is specialized on it. Buffer addresses do
    not: those are rebound per call, which is the whole point.
    """
    return (
        x.shape,
        x.stride(),
        x.dtype,
        weight.shape,
        scale.shape,
        scale.stride(),
        eps,
    )


def _tanh_signature(x, gate, residual, weight, eps):
    """What a recorded gated-residual launch stays valid for."""
    return (
        x.shape,
        x.stride(),
        x.dtype,
        gate.shape,
        gate.stride(),
        residual.stride(),
        weight.shape,
        eps,
    )


def rmsnorm_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Apply BF16-native ``RMSNorm(x) * scale`` or return ``None``."""
    # Operand validation, stride derivation and the power-of-two block below
    # are decided by the signature, never by the values. At this kernel's size
    # that bookkeeping costs more than the launch it guards, so a signature
    # seen before goes straight to rebinding pointers and replaying.
    if _fast_launch.available():
        cached = _RMSNORM_SCALE_LAUNCH.lookup(_scale_signature(x, weight, scale, eps))
        if cached is not None:
            out = torch.empty_like(x, memory_format=torch.contiguous_format)
            _RMSNORM_SCALE_LAUNCH.replay(
                cached,
                [
                    out.data_ptr(),
                    x.data_ptr(),
                    weight.data_ptr(),
                    scale.data_ptr(),
                ],
            )
            return out

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
    out_flat = out.reshape(-1, dim)
    rows_per_scale = x_rows // scale_rows
    with torch.get_device_module().device(x.device):
        compiled = _rmsnorm_scale_kernel[(x_rows,)](
            out_flat,
            x,
            weight,
            scale,
            x_row_stride,
            scale_row_stride,
            rows_per_scale,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            num_warps=8,
        )
    if _fast_launch.available() and compiled is not None:
        _RMSNORM_SCALE_LAUNCH.record(
            _scale_signature(x, weight, scale, eps),
            compiled,
            (x_rows, 1, 1),
            [out_flat.data_ptr(), x.data_ptr(), weight.data_ptr(), scale.data_ptr()],
            [x_row_stride, scale_row_stride, rows_per_scale, dim, _f32_bits(eps)],
        )
    return out


def rmsnorm_tanh_residual(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Apply BF16-native gated RMSNorm residual fusion or return ``None``."""
    # Same reasoning as rmsnorm_scale: the checks below are signature-decided.
    if _fast_launch.available():
        cached = _RMSNORM_TANH_LAUNCH.lookup(
            _tanh_signature(x, gate, residual, weight, eps)
        )
        if cached is not None:
            out = torch.empty_like(x, memory_format=torch.contiguous_format)
            _RMSNORM_TANH_LAUNCH.replay(
                cached,
                [
                    out.data_ptr(),
                    x.data_ptr(),
                    gate.data_ptr(),
                    residual.data_ptr(),
                    weight.data_ptr(),
                ],
            )
            return out

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
    out_flat = out.reshape(-1, dim)
    rows_per_gate = x_rows // gate_rows
    with torch.get_device_module().device(x.device):
        compiled = _rmsnorm_tanh_residual_kernel[(x_rows,)](
            out_flat,
            x,
            gate,
            residual,
            weight,
            x_row_stride,
            gate_row_stride,
            residual_row_stride,
            rows_per_gate,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            num_warps=8,
        )
    if _fast_launch.available() and compiled is not None:
        _RMSNORM_TANH_LAUNCH.record(
            _tanh_signature(x, gate, residual, weight, eps),
            compiled,
            (x_rows, 1, 1),
            [
                out_flat.data_ptr(),
                x.data_ptr(),
                gate.data_ptr(),
                residual.data_ptr(),
                weight.data_ptr(),
            ],
            [
                x_row_stride,
                gate_row_stride,
                residual_row_stride,
                rows_per_gate,
                dim,
                _f32_bits(eps),
            ],
        )
    return out


__all__ = ["MAX_HIDDEN_SIZE", "rmsnorm_scale", "rmsnorm_tanh_residual"]
