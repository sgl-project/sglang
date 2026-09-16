"""Stable QSA block expansion for the NPU Torch fallback contract."""

import torch
import triton
import triton.language as tl

# Inclusive limit used when comparing int32 block ids with int64 lengths.
_INT32_MAX: tl.constexpr = 2**31 - 1


@triton.jit
def _block_prefix(
    blocks,
    lengths,
    prefix,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    BLOCKS: tl.constexpr,
    RATIO: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, TILE)
    block = tl.load(
        blocks + row * ROW_STRIDE + cols * COL_STRIDE,
        mask=cols < BLOCKS,
        other=-1,
    )
    length = tl.load(lengths + row * LENGTH_STRIDE).to(tl.int64)
    full_blocks = tl.maximum(length, 0) // RATIO
    remainder = (tl.maximum(length, 0) % RATIO).to(tl.int32)
    if block.dtype == tl.int32:
        boundary = tl.minimum(full_blocks, _INT32_MAX).to(tl.int32)
        beyond_int32 = full_blocks > _INT32_MAX
    else:
        boundary = full_blocks
        beyond_int32 = False
    count = tl.where(
        (block < boundary) | beyond_int32,
        RATIO,
        tl.where(block == boundary, remainder, 0),
    )
    count = tl.minimum(count, TOPK - cols * RATIO)
    count = tl.where((cols < BLOCKS) & (block >= 0), count, 0).to(tl.int32)
    inclusive = tl.cumsum(count, 0)
    tl.store(
        prefix + row * (BLOCKS + 1) + cols,
        inclusive - count,
        mask=cols < BLOCKS,
    )
    tl.store(prefix + row * (BLOCKS + 1) + BLOCKS, tl.sum(count, 0))


@triton.jit
def _expand_blocks(
    blocks,
    positions,
    lengths,
    prefix,
    output,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(lengths + row * LENGTH_STRIDE).to(tl.int64)
    position = tl.load(positions + row * POSITION_STRIDE).to(tl.int64)
    visible = position + 1
    # Match Torch floor division even for negative padded query positions.
    remainder = ((visible % RATIO) + RATIO) % RATIO
    tail_start = visible - remainder
    block_total = tl.load(prefix + row * (BLOCKS + 1) + BLOCKS)
    tail_extent = tl.minimum(tl.maximum(length - tail_start, 0), remainder)
    tail_extent = tail_extent.to(tl.int32)
    tail_count = tl.where(tail_start >= 0, tail_extent, 0)
    total = block_total + tail_count
    block_col = tl.minimum(cols // RATIO, BLOCKS - 1)
    before = tl.load(prefix + row * (BLOCKS + 1) + block_col)
    after = tl.load(prefix + row * (BLOCKS + 1) + block_col + 1)
    preceding = before + tl.minimum(cols % RATIO, after - before)
    preceding = tl.where(
        cols < TOPK,
        preceding,
        block_total + tl.minimum(cols - TOPK, tail_count),
    )
    block = tl.load(
        blocks + row * ROW_STRIDE + (cols // RATIO) * COL_STRIDE,
        mask=cols < TOPK,
        other=-1,
    ).to(tl.int32)
    expanded_valid = (cols < TOPK) & (cols % RATIO < after - before)
    tail_offset = cols - TOPK
    tail_valid = (cols >= TOPK) & (cols < WIDTH)
    tail_valid = tail_valid & (tail_offset < tail_extent)
    # Validity is decided before narrowing indices to the int32 output dtype.
    token = tl.where(
        expanded_valid,
        block * RATIO + cols % RATIO,
        tl.where(tail_valid, tail_start.to(tl.int32) + tail_offset, -1),
    )
    valid = expanded_valid | (tail_valid & (tail_start >= 0))
    rank = tl.where(valid, preceding, total + cols - preceding)
    tl.store(output + row * WIDTH + rank, token, mask=cols < WIDTH)


def can_run_block_expansion(blocks, positions, lengths, ratio, topk):
    integers = (torch.int32, torch.int64)
    return (
        isinstance(ratio, int)
        and isinstance(topk, int)
        and ratio > 0
        and topk > 0
        # Bound compilation and temporary storage to the tested output widths.
        and topk + ratio - 1 <= 8192
        and blocks.device.type == "npu"
        and blocks.ndim == 2
        # Larger prefill batches retain the Torch fallback until benchmarked.
        and blocks.shape[0] <= 128
        and blocks.shape[1] == triton.cdiv(topk, ratio)
        and blocks.shape[1] <= 1024
        and positions.shape == lengths.shape == (blocks.shape[0],)
        and all(t.device == blocks.device for t in (positions, lengths))
        and all(t.dtype in integers for t in (blocks, positions, lengths))
    )


def expand_blocks(blocks, positions, lengths, ratio, topk):
    """Expand, clip, append the incomplete tail, and stably compact valid tokens."""
    if not can_run_block_expansion(blocks, positions, lengths, ratio, topk):
        raise ValueError("Unsupported NPU block expansion inputs")
    rows = blocks.shape[0]
    width = topk + ratio - 1
    output = torch.empty((rows, width), device=blocks.device, dtype=torch.int32)
    if rows:
        block_topk = blocks.shape[1]
        prefix = torch.empty(
            (rows, block_topk + 1), device=blocks.device, dtype=torch.int32
        )
        _block_prefix[(rows,)](
            blocks,
            lengths,
            prefix,
            blocks.stride(0),
            blocks.stride(1),
            lengths.stride(0),
            block_topk,
            ratio,
            topk,
            triton.next_power_of_2(block_topk),
        )
        _expand_blocks[(rows, triton.cdiv(width, 256))](
            blocks,
            positions,
            lengths,
            prefix,
            output,
            blocks.stride(0),
            blocks.stride(1),
            positions.stride(0),
            lengths.stride(0),
            ratio,
            topk,
            width,
            block_topk,
            256,
        )
    return output
