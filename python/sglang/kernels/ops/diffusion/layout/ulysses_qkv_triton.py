# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

# The extra grid dimension only pays off for production-scale payloads.
_CONTIGUOUS_FAST_PATH_MIN_ELEMENTS = 1 << 23


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


@triton.jit
def _pack_qkv_destination_major_contiguous_kernel(
    output_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    rows,
    HEAD_SIZE: tl.constexpr,
    FEATURES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Keep destination in the grid so each program only divides by the
    # compile-time local feature count, rather than recovering destination,
    # row, head, and dim from one global element index.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    destination = tl.program_id(1)
    mask = offsets < rows * FEATURES
    row = offsets // FEATURES
    feature = offsets - row * FEATURES
    local_head = feature // HEAD_SIZE
    dim = feature - local_head * HEAD_SIZE
    source_offset = (row * tl.num_programs(1) + destination) * FEATURES + feature
    output_offset = (
        (destination * rows + row) * (3 * FEATURES) + local_head * (3 * HEAD_SIZE) + dim
    )

    q = tl.load(q_ptr + source_offset, mask=mask)
    k = tl.load(k_ptr + source_offset, mask=mask)
    v = tl.load(v_ptr + source_offset, mask=mask)
    tl.store(output_ptr + output_offset, q, mask=mask)
    tl.store(output_ptr + output_offset + HEAD_SIZE, k, mask=mask)
    tl.store(output_ptr + output_offset + 2 * HEAD_SIZE, v, mask=mask)


def pack_qkv_destination_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack matching ``[rows, global_heads, head_size]`` Q/K/V tensors."""
    if q.dim() != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same 3D shape")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, and v must be CUDA tensors")
    if not (q.device == k.device == v.device and q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, and v must have the same device and dtype")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must be contiguous in head_size")
    if world_size < 1 or q.shape[1] % world_size != 0:
        raise ValueError("world_size must be positive and divide global_heads")

    rows, global_heads, head_size = q.shape
    local_heads = global_heads // world_size
    expected_shape = (world_size, rows, local_heads, 3 * head_size)
    if out is not None:
        if not (
            out.shape == expected_shape
            and out.is_contiguous()
            and out.dtype == q.dtype
            and out.device == q.device
        ):
            raise ValueError(
                "out must be a contiguous tensor with the expected shape, "
                "device, and dtype"
            )
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

    features = local_heads * head_size
    if (
        total_elements >= _CONTIGUOUS_FAST_PATH_MIN_ELEMENTS
        and torch.version.hip is None
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and world_size in (2, 4, 8)
    ):
        with torch.get_device_module().device(q.device):
            block_size = 1024
            _pack_qkv_destination_major_contiguous_kernel[
                (triton.cdiv(rows * features, block_size), world_size)
            ](
                output,
                q,
                k,
                v,
                rows,
                HEAD_SIZE=head_size,
                FEATURES=features,
                BLOCK_SIZE=block_size,
                num_warps=8,
                num_stages=1,
            )
        return output

    block_size = 1024
    with torch.get_device_module().device(q.device):
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


__all__ = ["pack_qkv_destination_major"]
