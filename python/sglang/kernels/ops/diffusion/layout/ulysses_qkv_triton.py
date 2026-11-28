# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

_MAX_INT32 = 2**31 - 1

# Devices with a Triton backend for this kernel. "cuda" also covers ROCm.
_SUPPORTED_DEVICE_TYPES = ("cuda", "xpu")


@triton.jit(do_not_specialize=["rows"])
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
    IDX64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if IDX64:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        ).to(tl.int64)
    else:
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
    """Pack matching ``[rows, global_heads, head_size]`` Q/K/V tensors."""
    if q.dim() != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same 3D shape")
    if not all(t.device.type in _SUPPORTED_DEVICE_TYPES for t in (q, k, v)):
        raise ValueError(
            f"q, k, and v must be on one of {_SUPPORTED_DEVICE_TYPES}, "
            f"got {q.device.type}"
        )
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

    # Input views may be strided (e.g. slices of a fused qkv projection), so
    # bound each tensor by one past the largest element offset it is addressed at.
    max_offset = max(
        output.numel(),
        *(
            (rows - 1) * t.stride(0) + (global_heads - 1) * t.stride(1) + head_size
            for t in (q, k, v)
        ),
    )
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
            IDX64=max_offset > _MAX_INT32,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    return output


_NORM_ROPE_HEAD_SIZE = 128
_NORM_ROPE_VECTORS_PER_PROGRAM = 4


@triton.jit(do_not_specialize=["num_vectors", "total_rows", "row_offset"])
def _pack_qkv_norm_rope_kernel(
    output_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    rope_ptr,
    num_vectors,
    global_heads,
    local_heads,
    total_rows,
    row_offset,
    stride_q_row,
    stride_k_row,
    stride_v_row,
    eps,
    HEAD_SIZE: tl.constexpr,
    BLOCK_VECTORS: tl.constexpr,
):
    vector = tl.program_id(0).to(tl.int64) * BLOCK_VECTORS + tl.arange(
        0, BLOCK_VECTORS
    ).to(tl.int64)
    cols = tl.arange(0, HEAD_SIZE)
    mask = (vector < num_vectors)[:, None]
    segment_row = vector // global_heads
    head = vector % global_heads
    row = row_offset + segment_row
    dtype = output_ptr.dtype.element_ty

    head_offset = head[:, None] * HEAD_SIZE + cols[None, :]
    q = tl.load(
        q_ptr + segment_row[:, None] * stride_q_row + head_offset, mask=mask, other=0.0
    ).to(tl.float32)
    k = tl.load(
        k_ptr + segment_row[:, None] * stride_k_row + head_offset, mask=mask, other=0.0
    ).to(tl.float32)
    v = tl.load(
        v_ptr + segment_row[:, None] * stride_v_row + head_offset, mask=mask, other=0.0
    )

    # RMSNorm, rounded after the normalization and after the weight like eager.
    q = (q / tl.sqrt(tl.sum(q * q, axis=1) / HEAD_SIZE + eps)[:, None]).to(dtype)
    k = (k / tl.sqrt(tl.sum(k * k, axis=1) / HEAD_SIZE + eps)[:, None]).to(dtype)
    q = (q.to(tl.float32) * tl.load(q_weight_ptr + cols).to(tl.float32)[None, :]).to(
        dtype
    )
    k = (k.to(tl.float32) * tl.load(k_weight_ptr + cols).to(tl.float32)[None, :]).to(
        dtype
    )

    # Interleaved (non-NeoX) rotation; the table holds (cos, sin) per pair.
    rotation = tl.load(
        rope_ptr + row[:, None] * HEAD_SIZE + cols[None, :], mask=mask, other=0.0
    )
    cos, sin = tl.split(tl.reshape(rotation, (BLOCK_VECTORS, HEAD_SIZE // 2, 2)))
    q_re, q_im = tl.split(
        tl.reshape(q.to(tl.float32), (BLOCK_VECTORS, HEAD_SIZE // 2, 2))
    )
    k_re, k_im = tl.split(
        tl.reshape(k.to(tl.float32), (BLOCK_VECTORS, HEAD_SIZE // 2, 2))
    )
    q = tl.reshape(
        tl.join(q_re * cos - q_im * sin, q_re * sin + q_im * cos),
        (BLOCK_VECTORS, HEAD_SIZE),
    )
    k = tl.reshape(
        tl.join(k_re * cos - k_im * sin, k_re * sin + k_im * cos),
        (BLOCK_VECTORS, HEAD_SIZE),
    )

    destination = head // local_heads
    local_head = head % local_heads
    base = ((destination * total_rows + row) * local_heads + local_head) * (
        3 * HEAD_SIZE
    )
    out = output_ptr + base[:, None] + cols[None, :]
    tl.store(out, q.to(dtype), mask=mask)
    tl.store(out + HEAD_SIZE, k.to(dtype), mask=mask)
    tl.store(out + 2 * HEAD_SIZE, v, mask=mask)


def _norm_rope_segment_ok(segment, global_heads, dtype, device) -> bool:
    q, k, v, q_weight, k_weight = segment
    if not all(t.dim() == 3 for t in (q, k, v)) or not (q.shape == k.shape == v.shape):
        return False
    rows, heads, head_size = q.shape
    return (
        heads == global_heads
        and head_size == _NORM_ROPE_HEAD_SIZE
        and all(
            t.dtype == dtype
            and t.device == device
            and t.stride(-1) == 1
            and t.stride(-2) == _NORM_ROPE_HEAD_SIZE
            for t in (q, k, v)
        )
        and all(
            w.shape == (_NORM_ROPE_HEAD_SIZE,)
            and w.dtype == dtype
            and w.device == device
            and w.is_contiguous()
            for w in (q_weight, k_weight)
        )
    )


def can_use_pack_qkv_norm_rope(segments, rope: torch.Tensor, world_size: int) -> bool:
    """Whether ``pack_qkv_norm_rope_destination_major`` serves these inputs."""
    if not segments:
        return False
    first = segments[0][0]
    if first.dim() != 3 or first.device.type != "cuda":
        return False
    global_heads = first.shape[1]
    total_rows = sum(segment[0].shape[0] for segment in segments)
    return (
        first.dtype in (torch.float16, torch.bfloat16)
        and world_size >= 1
        and global_heads % world_size == 0
        and total_rows > 0
        and rope.dtype == torch.float32
        and rope.device == first.device
        and rope.is_contiguous()
        and rope.shape == (total_rows, _NORM_ROPE_HEAD_SIZE // 2, 2)
        and all(
            _norm_rope_segment_ok(segment, global_heads, first.dtype, first.device)
            for segment in segments
        )
    )


def pack_qkv_norm_rope_destination_major(
    segments,
    rope: torch.Tensor,
    eps: float,
    world_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm + interleaved RoPE on Q/K, packed with V for the Ulysses exchange.

    ``segments`` lists ``(q, k, v, q_weight, k_weight)`` in token order, each
    Q/K/V a ``[rows_i, global_heads, 128]`` view with unit head-dim stride and
    heads 128 apart (e.g. slices of a fused QKV projection), so text and image
    segments with their own norm weights are joined by the pack itself. ``rope``
    is the real view ``[sum(rows_i), 64, 2]`` of the complex rotation table. The
    output layout matches ``pack_qkv_destination_major`` on the normalized and
    rotated Q/K. The rounding points follow the eager chain; the RMS reduction
    order differs, so the result matches eager to rounding error.
    """
    if not can_use_pack_qkv_norm_rope(segments, rope, world_size):
        raise ValueError("invalid input for pack_qkv_norm_rope_destination_major")
    first = segments[0][0]
    global_heads = first.shape[1]
    local_heads = global_heads // world_size
    total_rows = rope.shape[0]
    expected_shape = (world_size, total_rows, local_heads, 3 * _NORM_ROPE_HEAD_SIZE)
    if out is None:
        out = torch.empty(expected_shape, dtype=first.dtype, device=first.device)
    elif not (
        out.shape == expected_shape
        and out.is_contiguous()
        and out.dtype == first.dtype
        and out.device == first.device
    ):
        raise ValueError("out must be a contiguous tensor with the expected shape")
    if out.numel() > _MAX_INT32:
        raise ValueError("pack_qkv_norm_rope_destination_major uses 32-bit offsets")

    row_offset = 0
    with torch.get_device_module().device(first.device):
        for q, k, v, q_weight, k_weight in segments:
            num_vectors = q.shape[0] * global_heads
            if num_vectors:
                _pack_qkv_norm_rope_kernel[
                    (triton.cdiv(num_vectors, _NORM_ROPE_VECTORS_PER_PROGRAM),)
                ](
                    out,
                    q,
                    k,
                    v,
                    q_weight,
                    k_weight,
                    rope,
                    num_vectors,
                    global_heads,
                    local_heads,
                    total_rows,
                    row_offset,
                    q.stride(0),
                    k.stride(0),
                    v.stride(0),
                    eps,
                    HEAD_SIZE=_NORM_ROPE_HEAD_SIZE,
                    BLOCK_VECTORS=_NORM_ROPE_VECTORS_PER_PROGRAM,
                    num_warps=4,
                )
            row_offset += q.shape[0]
    return out


__all__ = [
    "can_use_pack_qkv_norm_rope",
    "pack_qkv_destination_major",
    "pack_qkv_norm_rope_destination_major",
]
