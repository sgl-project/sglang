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


@triton.jit
def _copy_region_and_zero(
    dst,
    src,
    offset_ptr,
    size_ptr,
    dst_size: tl.constexpr,
    src_size: tl.constexpr,
    row_size: tl.constexpr,
    offset_src: tl.constexpr,
    copy_enabled: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if copy_enabled:
        offset = tl.load(offset_ptr).to(tl.int64) * row_size
        size = tl.load(size_ptr).to(tl.int64) * row_size
        if offset_src:
            j = i + offset
            valid = (i < size) & (j < src_size)
        else:
            j = i - offset
            valid = (j >= 0) & (j < size) & (j < src_size)
        value = tl.load(src + j, mask=(i < dst_size) & valid, other=0)
    else:
        value = tl.full((BLOCK,), 0, dst.dtype.element_ty)
    tl.store(dst + i, value, mask=i < dst_size)


def copy_region_and_zero(dst, src, offset, size, *, offset_src, copy_enabled=True):
    """Copy a row slice and zero every destination element outside that slice."""
    assert dst.is_contiguous() and src.is_contiguous()
    assert dst.shape[1:] == src.shape[1:]
    assert dst.dtype == src.dtype
    if dst.numel() == 0:
        return
    _copy_region_and_zero[(triton.cdiv(dst.numel(), 4096),)](
        dst,
        src,
        offset,
        size,
        dst.numel(),
        src.numel(),
        prod(src.shape[1:]),
        offset_src,
        copy_enabled,
        4096,
    )


@triton.jit
def _copy_bf16_canonical_zero(dst, src, n: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(src + i, mask=i < n, other=0).to(tl.uint16, bitcast=True)
    # The push allreduce uses negative zero as a transport marker. Match the
    # positive zero produced by the original reduce-scatter SUM. Test bits,
    # not floating-point equality, to preserve BF16 subnormal values.
    bits = tl.where(bits == 0x8000, 0, bits).to(tl.uint16)
    tl.store(dst + i, bits.to(tl.bfloat16, bitcast=True), mask=i < n)


def copy_bf16_canonical_zero(dst, src):
    assert dst.dtype == src.dtype == torch.bfloat16
    assert dst.is_contiguous() and src.is_contiguous() and dst.shape == src.shape
    _copy_bf16_canonical_zero[(triton.cdiv(dst.numel(), 4096),)](
        dst, src, dst.numel(), 4096
    )
