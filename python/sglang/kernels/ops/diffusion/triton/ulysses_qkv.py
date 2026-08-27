# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_qkv_destination_major_kernel(
    output_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    total_elements,
    rows,
    local_heads,
    head_size,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    dim = offsets % head_size
    head_slot = offsets // head_size
    local_head = head_slot % local_heads
    row_slot = head_slot // local_heads
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * local_heads + local_head

    q = tl.load(
        q_ptr + row * stride_q_row + global_head * stride_q_head + dim,
        mask=mask,
    )
    k = tl.load(
        k_ptr + row * stride_k_row + global_head * stride_k_head + dim,
        mask=mask,
    )
    v = tl.load(
        v_ptr + row * stride_v_row + global_head * stride_v_head + dim,
        mask=mask,
    )
    output_base = head_slot * (3 * head_size) + dim
    tl.store(output_ptr + output_base, q, mask=mask)
    tl.store(output_ptr + output_base + head_size, k, mask=mask)
    tl.store(output_ptr + output_base + 2 * head_size, v, mask=mask)


def pack_qkv_destination_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    rows, global_heads, head_size = q.shape
    local_heads = global_heads // world_size
    expected_shape = (world_size, rows, local_heads, 3 * head_size)
    if out is not None:
        assert out.shape == expected_shape and out.is_contiguous()
        assert out.dtype == q.dtype and out.device == q.device
        output = out
    else:
        output = torch.empty(
            expected_shape,
            dtype=q.dtype,
            device=q.device,
        )
    total_elements = rows * global_heads * head_size
    if total_elements == 0:
        return output

    block_size = 1024
    _pack_qkv_destination_major_kernel[(triton.cdiv(total_elements, block_size),)](
        output,
        q,
        k,
        v,
        total_elements,
        rows,
        local_heads,
        head_size,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output
