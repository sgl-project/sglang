"""Masked tile addressing for the MiniMax sparse attention kernels.

Triton >= 3.9 removed ``tl.make_block_ptr``. These helpers return the
``(pointers, mask)`` pair that a block pointer with ``boundary_check`` on every
dimension described, so

    tl.load(make_block_ptr(...), boundary_check=..., padding_option="zero")
    tl.store(make_block_ptr(...), x, boundary_check=...)

become

    ptrs, mask = tile_ptrs_2d(...)
    tl.load(ptrs, mask=mask, other=0.0)
    tl.store(ptrs, x, mask=mask)

Offsets are widened to int64 like block-pointer offsets, so large strides do not
overflow. ``tl.advance`` becomes an explicit offset update at the call site; the
shape bounds stay absolute, exactly as block pointers checked them.
"""

import triton
import triton.language as tl


@triton.jit
def tile_ptrs_1d(base, n, stride, off, BLOCK: tl.constexpr):
    idx = off + tl.arange(0, BLOCK).to(tl.int64)
    return base + idx * stride, idx < n


@triton.jit
def tile_ptrs_2d(
    base,
    n_rows,
    n_cols,
    stride_r,
    stride_c,
    row_off,
    col_off,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = row_off + tl.arange(0, BLOCK_R).to(tl.int64)
    cols = col_off + tl.arange(0, BLOCK_C).to(tl.int64)
    ptrs = base + rows[:, None] * stride_r + cols[None, :] * stride_c
    mask = (
        (rows[:, None] >= 0)
        & (rows[:, None] < n_rows)
        & (cols[None, :] >= 0)
        & (cols[None, :] < n_cols)
    )
    return ptrs, mask


@triton.jit
def tile_ptrs_3d(
    base,
    n0,
    n1,
    n2,
    stride0,
    stride1,
    stride2,
    off0,
    BLOCK0: tl.constexpr,
    BLOCK1: tl.constexpr,
    BLOCK2: tl.constexpr,
):
    i0 = off0 + tl.arange(0, BLOCK0).to(tl.int64)
    i1 = tl.arange(0, BLOCK1).to(tl.int64)
    i2 = tl.arange(0, BLOCK2).to(tl.int64)
    ptrs = (
        base
        + i0[:, None, None] * stride0
        + i1[None, :, None] * stride1
        + i2[None, None, :] * stride2
    )
    mask = (
        (i0[:, None, None] < n0) & (i1[None, :, None] < n1) & (i2[None, None, :] < n2)
    )
    return ptrs, mask
