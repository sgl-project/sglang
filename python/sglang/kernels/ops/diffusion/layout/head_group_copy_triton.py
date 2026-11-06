# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

_MAX_INT32 = 2**31 - 1


@triton.jit(do_not_specialize=["rows"])
def _copy_head_groups_kernel(
    src_ptr,
    dst_ptr,
    rows,
    batch,
    width,
    stride_src_g,
    stride_src_b,
    stride_src_s,
    stride_dst_g,
    stride_dst_b,
    stride_dst_s,
    IDX64: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    group_batch = tl.program_id(1)
    g = group_batch // batch
    b = group_batch % batch
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_WIDTH)
    if IDX64:
        g = g.to(tl.int64)
        b = b.to(tl.int64)
        row = row.to(tl.int64)
    mask = (row[:, None] < rows) & (col[None, :] < width)
    src = (
        src_ptr
        + g * stride_src_g
        + b * stride_src_b
        + row[:, None] * stride_src_s
        + col[None, :]
    )
    dst = (
        dst_ptr
        + g * stride_dst_g
        + b * stride_dst_b
        + row[:, None] * stride_dst_s
        + col[None, :]
    )
    tl.store(dst, tl.load(src, mask=mask), mask=mask)


def copy_head_groups(src: torch.Tensor, dst: torch.Tensor) -> None:
    """``dst.copy_(src)`` for ``[group, batch, rows, width]`` views whose last dim
    is contiguous and whose other strides are arbitrary: one pass that moves the
    head groups of a sequence shard between per-group blocks and a token-major
    tensor (a same-device copy per block costs a launch and runs at a fraction
    of the bandwidth)."""
    if src.dim() != 4 or src.shape != dst.shape:
        raise ValueError(
            f"src {tuple(src.shape)} and dst {tuple(dst.shape)} must be the same 4D shape"
        )
    if src.stride(-1) != 1 or dst.stride(-1) != 1:
        raise ValueError("src and dst must be contiguous in the last dim")
    if src.dtype != dst.dtype or src.device != dst.device:
        raise ValueError("src and dst must share dtype and device")
    if src.numel() == 0:
        return
    groups, batch, rows, width = src.shape
    bound = lambda t: sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
    block_width = triton.next_power_of_2(width)
    block_rows = max(1, 8192 // block_width)
    with torch.get_device_module().device(src.device):
        _copy_head_groups_kernel[(triton.cdiv(rows, block_rows), groups * batch)](
            src,
            dst,
            rows,
            batch,
            width,
            *src.stride()[:3],
            *dst.stride()[:3],
            IDX64=max(bound(src), bound(dst)) > _MAX_INT32,
            BLOCK_ROWS=block_rows,
            BLOCK_WIDTH=block_width,
            num_warps=8,
        )


__all__ = ["copy_head_groups"]
