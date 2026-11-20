"""Offset/size-driven device memcpy kernel, migrated from
``sglang.srt.layers.dp_attention`` (RFC #29630, Phase 2.5).
"""

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    offset_src: tl.constexpr,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


@triton.jit
def memcpy_triton_with_zero_fill_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    dst_numel,
    offset_src: tl.constexpr,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    dst_idx = start_index + offs
    dst_mask = dst_idx < dst_numel

    if offset_src:
        in_copy = dst_idx < sz
        src_idx = tl.where(in_copy, offset + dst_idx, 0)
    else:
        in_copy = (dst_idx >= offset) & (dst_idx < offset + sz)
        src_idx = tl.where(in_copy, dst_idx - offset, 0)

    data = tl.load(src_ptr + src_idx, mask=in_copy & dst_mask, other=0)
    tl.store(dst_ptr + dst_idx, data, mask=dst_mask)


def prod(x):
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)


def memcpy_triton_with_zero_fill(
    dst: torch.Tensor,
    src: torch.Tensor,
    dim: int,
    offset: torch.Tensor,
    sz: torch.Tensor,
    offset_src: bool,
) -> None:
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    dst_numel: int = dst.numel()
    chunk_size: int = prod(src.shape[1:])
    BLOCK_SIZE: int = 8192
    grid = (triton.cdiv(dst_numel, BLOCK_SIZE),)

    memcpy_triton_with_zero_fill_kernel[grid](
        dst_ptr=dst,
        src_ptr=src,
        offset_ptr=offset,
        sz_ptr=sz,
        dst_numel=dst_numel,
        offset_src=offset_src,
        chunk_size=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
