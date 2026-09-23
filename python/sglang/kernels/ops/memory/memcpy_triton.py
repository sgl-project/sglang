"""Offset/size-driven device memcpy kernel, migrated from
``sglang.srt.layers.dp_attention`` (RFC #29630, Phase 2.5).
"""

import functools

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
    src_numel,
    dst_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Clamp to both tensors, as memcpy_cpu does: a rank count larger than the
    # local rows (the EAGLE draft extend reuses the target's input-logprob
    # counts) must not read or write past either tensor.
    if offset_src:
        mask = (idx < sz) & (offset + idx < src_numel) & (idx < dst_numel)
        data = tl.load(src_ptr + offset + idx, mask=mask)
        tl.store(dst_ptr + idx, data, mask=mask)
    else:
        mask = (idx < sz) & (idx < src_numel) & (offset + idx < dst_numel)
        data = tl.load(src_ptr + idx, mask=mask)
        tl.store(dst_ptr + offset + idx, data, mask=mask)


def prod(x):
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    memcpy_triton_kernel[grid](
        dst,
        src,
        offset,
        sz,
        offset_src,
        chunk_size,
        src.numel(),
        dst.numel(),
        BLOCK_SIZE,
    )
