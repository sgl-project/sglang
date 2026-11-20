# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _awq_embedding_kernel(
    ids_ptr,
    qweight_ptr,
    qzeros_ptr,
    scales_ptr,
    output_ptr,
    num_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    group_size: tl.constexpr,
    qweight_stride: tl.constexpr,
    qzeros_stride: tl.constexpr,
    scales_stride: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row = tl.load(ids_ptr + token).to(tl.int64)
    valid_row = (row >= 0) & (row < num_rows)
    tl.device_assert(valid_row, "AWQ embedding index out of range")
    valid = (cols < hidden_size) & valid_row
    groups = cols // group_size

    packed_weight = tl.load(
        qweight_ptr + row * qweight_stride + cols // 8, mask=valid, other=0
    )
    packed_zero = tl.load(
        qzeros_ptr + row * qzeros_stride + groups // 8, mask=valid, other=0
    )
    scale = tl.load(scales_ptr + row * scales_stride + groups, mask=valid, other=0)
    weight = (packed_weight >> (4 * (cols % 8))) & 0xF
    zero = (packed_zero >> (4 * (groups % 8))) & 0xF
    result = (weight - zero).to(tl.float32) * scale.to(tl.float32)
    tl.store(
        output_ptr + token.to(tl.int64) * hidden_size + cols,
        result,
        mask=cols < hidden_size,
    )


def awq_embedding(
    input_ids: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Gather local vocabulary rows packed in consecutive INT4 nibbles."""
    assert input_ids.is_cuda and input_ids.dtype in (torch.int32, torch.int64)
    assert qweight.device == qzeros.device == scales.device == input_ids.device
    assert qweight.dtype == qzeros.dtype == torch.int32
    assert scales.dtype == torch.float16
    assert qweight.stride(1) == qzeros.stride(1) == scales.stride(1) == 1

    num_rows, packed_width = qweight.shape
    hidden_size = packed_width * 8
    num_groups = scales.shape[1]
    assert hidden_size == num_groups * group_size
    assert scales.shape[0] == num_rows
    assert qzeros.shape[0] == num_rows and qzeros.shape[1] * 8 == num_groups

    output = torch.empty(
        (*input_ids.shape, hidden_size), dtype=dtype, device=input_ids.device
    )
    num_tokens = input_ids.numel()
    if num_tokens == 0:
        return output

    block_h = min(512, triton.next_power_of_2(hidden_size))
    grid = (num_tokens, triton.cdiv(hidden_size, block_h))
    _awq_embedding_kernel[grid](
        input_ids.contiguous(),
        qweight,
        qzeros,
        scales,
        output,
        num_rows,
        hidden_size,
        group_size,
        qweight.stride(0),
        qzeros.stride(0),
        scales.stride(0),
        BLOCK_H=block_h,
        num_warps=4,
    )
    return output
