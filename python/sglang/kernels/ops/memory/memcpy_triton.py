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
def memcpy_scatter_zero_rest_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    chunk_size,  # multiplied for offset and sz
    n_elements,  # dst.numel()
    src_n_elements,  # src.numel()
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size
    # Clamp like ``memcpy_cpu`` does: a padded ``src`` may hold fewer rows than
    # the device-side row count asks for.
    sz = tl.minimum(sz, src_n_elements.to(tl.int64))

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    dst_mask = idx < n_elements
    # Rows inside [offset, offset + sz) come from src; every other row is zero.
    src_idx = idx - offset
    from_src = (src_idx >= 0) & (src_idx < sz)
    data = tl.load(src_ptr + src_idx, mask=dst_mask & from_src, other=0)
    tl.store(dst_ptr + idx, data, mask=dst_mask)


def memcpy_scatter_zero_rest_triton(dst, src, dim, offset, sz):
    """``dst[offset:offset+sz] = src[:sz]``, and zero every other row of ``dst``.

    The fused form of ``dst.fill_(0)`` plus :func:`memcpy_triton` with
    ``offset_src=False``. ``offset`` and ``sz`` are device scalars, so this stays
    CUDA-graph capturable; ``dst`` must not alias ``src``.
    """
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    assert src.dtype == dst.dtype, "src and dst must have the same dtype"
    chunk_size = prod(src.shape[1:])
    n_elements = dst.numel()
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    memcpy_scatter_zero_rest_kernel[grid](
        dst, src, offset, sz, chunk_size, n_elements, src.numel(), BLOCK_SIZE
    )
