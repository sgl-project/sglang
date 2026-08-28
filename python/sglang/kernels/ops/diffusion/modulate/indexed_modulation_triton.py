# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32


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
