"""Scheduler metadata for MiniMax sparse prefill attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _build_query_block_to_req_kernel(
    extend_seq_lens_ptr,
    query_block_to_req_ptr,
    num_query_blocks,
    NUM_REQS: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    query_blocks = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    req_for_block = tl.full((BLOCK_SIZE,), -1, tl.int32)
    block_end = 0
    for req_id in tl.static_range(NUM_REQS):
        seq_len = tl.load(extend_seq_lens_ptr + req_id)
        next_block_end = block_end + tl.cdiv(seq_len, BLOCK_SIZE_Q)
        req_for_block = tl.where(
            (query_blocks >= block_end) & (query_blocks < next_block_end),
            req_id,
            req_for_block,
        )
        block_end = next_block_end
    tl.store(
        query_block_to_req_ptr + query_blocks,
        req_for_block,
        mask=query_blocks < num_query_blocks,
    )


@torch.no_grad()
def build_query_block_to_req(
    query_block_to_req: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    block_size_q: int,
) -> None:
    """Map packed query blocks to their active request."""
    assert query_block_to_req.dtype == torch.int32 and query_block_to_req.dim() == 1
    assert extend_seq_lens.dim() == 1
    assert block_size_q > 0
    if query_block_to_req.numel() == 0:
        return

    block_size = 256
    _build_query_block_to_req_kernel[
        (triton.cdiv(query_block_to_req.numel(), block_size),)
    ](
        extend_seq_lens,
        query_block_to_req,
        query_block_to_req.numel(),
        NUM_REQS=extend_seq_lens.shape[0],
        BLOCK_SIZE_Q=block_size_q,
        BLOCK_SIZE=block_size,
    )
