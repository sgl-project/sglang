"""Refresh the layer-dependent tail of TRT-LLM's sparse attention tables."""

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
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(INDICES + row * INDEX_STRIDE + col, col < WIDTH, -1)
    tl.store(TABLE + row * TABLE_STRIDE + SWA_WIDTH + col, values, col < WIDTH)
    length = tl.load(LENGTHS + row * LENGTH_STRIDE)
    tl.store(TOTAL_LENGTHS + row * TOTAL_STRIDE, SWA_WIDTH + length)


def pack_sparse_tail(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    table: torch.Tensor,
    total_lengths: torch.Tensor,
    swa_width: int = 128,
) -> None:
    """Copy compressed indices and add the fixed SWA width in one launch.

    The caller owns the tile-padded buffers and their invariant SWA columns.
    Only active rows are modified, preserving inert graph/tile padding.

    Compressed entries start at column swa_width even for short sequences.
    total_lengths describes the table span to scan, not the valid-token count:
    adding only the actual SWA length would truncate the compressed tail.
    Unused SWA slots retain their -1 sentinel and are ignored by attention.
    """
    rows, width = indices.shape
    assert table.shape == (rows, swa_width + width)
    assert lengths.shape == total_lengths.shape == (rows,)
    assert indices.stride(1) == table.stride(1) == 1
    assert indices.dtype == table.dtype == total_lengths.dtype == torch.int32
    assert lengths.dtype in (torch.int32, torch.int64)
    assert indices.device == lengths.device == table.device == total_lengths.device
    if rows == 0:
        return
    _pack_sparse_tail[(rows,)](
        indices,
        lengths,
        table,
        total_lengths,
        indices.stride(0),
        table.stride(0),
        lengths.stride(0),
        total_lengths.stride(0),
        width,
        swa_width,
        triton.next_power_of_2(width),
        num_warps=4,
    )
