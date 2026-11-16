"""Refresh the layer-dependent tail of TRT-LLM's sparse attention tables."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_sparse_tail(
    INDICES,
    LENGTHS,
    TABLE,
    TOTAL_LENGTHS,
    INDEX_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    TOTAL_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    SWA_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_LENGTHS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(INDICES + row * INDEX_STRIDE + col, col < WIDTH, -1)
    tl.store(TABLE + row * TABLE_STRIDE + SWA_WIDTH + col, values, col < WIDTH)
    if WRITE_LENGTHS:
        length = tl.load(LENGTHS + row * LENGTH_STRIDE)
        tl.store(TOTAL_LENGTHS + row * TOTAL_STRIDE, SWA_WIDTH + length)


def pack_sparse_tail(
    indices: torch.Tensor,
    lengths: Optional[torch.Tensor],
    table: torch.Tensor,
    total_lengths: Optional[torch.Tensor],
    swa_width: int = 128,
) -> None:
    """Copy compressed indices and add the fixed SWA width in one launch.

    The caller owns the tile-padded buffers and their invariant SWA columns.
    Only active rows are modified, preserving inert graph/tile padding.

    Compressed entries start at column swa_width even for short sequences.
    total_lengths describes the table span to scan, not the valid-token count:
    adding only the actual SWA length would truncate the compressed tail.
    Unused SWA slots retain their -1 sentinel and are ignored by attention.
    With lengths and total_lengths None only the table is written.
    """
    rows, width = indices.shape
    assert table.shape == (rows, swa_width + width)
    assert indices.stride(1) == table.stride(1) == 1
    assert indices.dtype == table.dtype == torch.int32
    assert indices.device == table.device
    write_lengths = total_lengths is not None
    assert (lengths is not None) == write_lengths
    if write_lengths:
        assert lengths.shape == total_lengths.shape == (rows,)
        assert total_lengths.dtype == torch.int32
        assert lengths.dtype in (torch.int32, torch.int64)
        assert lengths.device == total_lengths.device == table.device
    if rows == 0:
        return
    _pack_sparse_tail[(rows,)](
        indices,
        lengths if write_lengths else table,
        table,
        total_lengths if write_lengths else table,
        indices.stride(0),
        table.stride(0),
        lengths.stride(0) if write_lengths else 0,
        total_lengths.stride(0) if write_lengths else 0,
        width,
        swa_width,
        triton.next_power_of_2(width),
        WRITE_LENGTHS=write_lengths,
        num_warps=4,
    )
