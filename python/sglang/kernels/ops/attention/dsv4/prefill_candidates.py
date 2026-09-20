"""Candidate-source reductions for the dense FP4 prefill indexer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_block_max(
    LOGITS,
    LENGTHS,
    SCORES,
    ROW_STRIDE,
    COL_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    WIDTH,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
    POSITION_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    blocks = tl.program_id(1) * BLOCK_TILE + tl.arange(0, BLOCK_TILE)
    positions = tl.arange(0, POSITION_TILE)
    columns = blocks[:, None] * BLOCK_SIZE + positions[None, :]
    length = tl.load(LENGTHS + row * LENGTH_STRIDE)
    valid = (
        (blocks[:, None] < NUM_BLOCKS)
        & (positions[None, :] < BLOCK_SIZE)
        & (columns < WIDTH)
        & (columns < length)
    )
    values = tl.load(
        LOGITS + row * ROW_STRIDE + columns * COL_STRIDE,
        mask=valid,
        other=-float("inf"),
    )
    # Match torch.amax, including propagation of NaNs in visible positions.
    scores = tl.max(values, axis=1)
    scores = tl.where(
        tl.sum((values != values).to(tl.int32), axis=1) > 0, float("nan"), scores
    )
    newest = (length - 1) // BLOCK_SIZE
    scores = tl.where((length > 0) & (blocks == newest), float("inf"), scores)
    tl.store(
        SCORES + row * NUM_BLOCKS + blocks,
        scores,
        mask=blocks < NUM_BLOCKS,
    )


def causal_block_max(
    logits: torch.Tensor, compress_lens: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Pool visible positions without modifying or padding the input logits.

    compress_lens contains one nonnegative visible length per query, at most
    the input width. The block containing the newest visible position is given
    +inf, as in the reference candidate selector.
    """
    assert logits.ndim == 2 and logits.dtype == torch.float32
    assert block_size > 0
    rows, width = logits.shape
    lens = compress_lens.reshape(-1)
    assert lens.numel() == rows and lens.device == logits.device
    num_blocks = triton.cdiv(width, block_size)
    scores = torch.empty((rows, num_blocks), dtype=logits.dtype, device=logits.device)
    if rows == 0 or width == 0:
        return scores
    position_tile = triton.next_power_of_2(block_size)
    block_tile = max(1, 1024 // position_tile)
    _causal_block_max[(rows, triton.cdiv(num_blocks, block_tile))](
        logits,
        lens,
        scores,
        logits.stride(0),
        logits.stride(1),
        lens.stride(0),
        width,
        num_blocks,
        block_size,
        block_tile,
        position_tile,
        num_warps=4,
    )
    return scores


def select_prefill_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Prefill-only candidate selection; retain the reference token-mask API."""
    scores = causal_block_max(logits, compress_lens, block_size)
    top = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., : logits.shape[-1]]


def select_prefill_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Publish block IDs using the fused causal block reduction."""
    scores = causal_block_max(logits, compress_lens, block_size)
    top = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)
