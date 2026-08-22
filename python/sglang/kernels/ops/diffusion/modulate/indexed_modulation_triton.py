# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.common.numerics import (
    cuda_rsqrtf,
    mul_rn_f32,
    round_bf16_to_fp32,
)


@triton.jit
def _indexed_scale_shift_bf16_kernel(
    output_ptr,
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
    ).to(tl.float32)

    one_plus_scale = round_bf16_to_fp32(1.0 + scale)
    scaled = round_bf16_to_fp32(x * one_plus_scale)
    tl.store(
        output_ptr + row * stride_x_row + columns,
        scaled + shift,
        mask=mask,
    )


@triton.jit
def _indexed_gate_bf16_kernel(
    output_ptr,
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_output_row,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(
        gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    other = tl.load(
        other_ptr + row * stride_other_row + columns, mask=mask, other=0.0
    ).to(tl.float32)

    gated = round_bf16_to_fp32(gate * other)
    tl.store(
        output_ptr + row * stride_output_row + columns,
        x + gated,
        mask=mask,
    )


def indexed_scale_shift_bf16_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    rows, hidden_size = x.shape
    if rows == 0:
        return x
    block_n = triton.next_power_of_2(hidden_size)
    _indexed_scale_shift_bf16_kernel[(rows,)](
        x,
        x,
        shift,
        scale,
        indices,
        hidden_size,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return x


def _indexed_gate_bf16(
    output: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    rows, hidden_size = x.shape
    if rows == 0:
        return output
    block_n = triton.next_power_of_2(hidden_size)
    _indexed_gate_bf16_kernel[(rows,)](
        output,
        x,
        gate,
        other,
        indices,
        hidden_size,
        output.stride(0),
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return output


def indexed_gate_bf16_(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return _indexed_gate_bf16(x, x, gate, other, indices)


def indexed_gate_bf16(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return _indexed_gate_bf16(torch.empty_like(x), x, gate, other, indices)


@triton.jit
def _fused_indexed_rmsnorm_scale_shift_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_out_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    eps,
    BLOCK_N: tl.constexpr,
):
    """RMSNorm(x) * (1 + scale[idx]) + shift[idx], one kernel, leave x intact."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    acc = 0.0
    offs = 0
    while offs < hidden_size:
        cols = offs + columns
        mask = cols < hidden_size
        x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        acc += tl.sum(mul_rn_f32(x, x), axis=0)
        offs += BLOCK_N
    rstd = cuda_rsqrtf(acc / hidden_size + eps)
    index = tl.load(indices_ptr + row * stride_indices)
    offs = 0
    while offs < hidden_size:
        cols = offs + columns
        mask = cols < hidden_size
        x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        shift = tl.load(
            shift_ptr + index * stride_shift_row + cols, mask=mask, other=0.0
        ).to(tl.float32)
        scale = tl.load(
            scale_ptr + index * stride_scale_row + cols, mask=mask, other=0.0
        ).to(tl.float32)
        y = round_bf16_to_fp32(x * rstd * weight)
        one_plus = round_bf16_to_fp32(1.0 + scale)
        tl.store(
            out_ptr + row * stride_out_row + cols,
            round_bf16_to_fp32(y * one_plus) + shift,
            mask=mask,
        )
        offs += BLOCK_N


def can_use_fused_indexed_rmsnorm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.shape == (x.shape[-1],)
        and shift.is_cuda
        and scale.is_cuda
        and shift.dtype == scale.dtype == torch.bfloat16
        and shift.shape == scale.shape
        and shift.shape[-1] == x.shape[-1]
        and shift.stride(-1) == 1
        and scale.stride(-1) == 1
        and indices.is_cuda
        and indices.dtype in (torch.int32, torch.int64)
        and indices.numel() == x.shape[0]
    )


def fused_indexed_rmsnorm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``norm(x) * (1 + scale[idx]) + shift[idx]`` without materializing norm(x)."""
    rows, hidden_size = x.shape
    out = torch.empty_like(x)
    if rows == 0:
        return out
    block_n = min(triton.next_power_of_2(hidden_size), 1024)
    _fused_indexed_rmsnorm_scale_shift_kernel[(rows,)](
        out,
        x,
        weight,
        shift,
        scale,
        indices,
        hidden_size,
        x.stride(0),
        out.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        float(eps),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out
