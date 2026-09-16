"""Filter selected indexer scores and map logical positions to KV slots."""

import torch
import triton
import triton.language as tl


@triton.jit
def _filter_topk_pages(
    SCORES,
    INDICES,
    PAGES,
    OUT,
    RAW,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SS: tl.constexpr,
    SI: tl.constexpr,
    SP: tl.constexpr,
    SO: tl.constexpr,
    SR: tl.constexpr,
    WRITE_RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(INDICES + row * SI + col, col < TOPK, -1).to(tl.int64)
    in_bounds = (index >= 0) & (index < WIDTH) & (col < TOPK)
    score = tl.load(SCORES + row * SS + index, in_bounds, -float("inf"))
    # This comparison also rejects NaN; +inf remains a valid score.
    valid = in_bounds & (score > -float("inf"))
    page = tl.load(PAGES + row * SP + index // PAGE_SIZE, valid, 0)
    slot = (page * PAGE_SIZE).to(tl.int64) + index % PAGE_SIZE
    tl.store(OUT + row * SO + col, tl.where(valid, slot, -1), col < TOPK)
    if WRITE_RAW:
        tl.store(RAW + row * SR + col, tl.where(valid, index, -1), col < TOPK)


def filter_topk_pages(
    scores: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    raw_indices: torch.Tensor | None = None,
) -> None:
    """Preserve top-k order, write -1 for invalid scores, and map valid slots."""
    rows, topk = indices.shape
    assert scores.ndim == page_table.ndim == page_indices.ndim == 2
    assert scores.shape[0] == page_table.shape[0] == page_indices.shape[0] == rows
    assert page_indices.shape[1] == topk and scores.shape[1] > 0
    assert page_table.shape[1] * page_size >= scores.shape[1]
    assert all(t.stride(1) == 1 for t in (scores, indices, page_table, page_indices))
    if raw_indices is not None:
        assert raw_indices.shape == indices.shape and raw_indices.stride(1) == 1
    _filter_topk_pages[(rows, triton.cdiv(topk, 256))](
        scores,
        indices,
        page_table,
        page_indices,
        raw_indices,
        scores.shape[1],
        topk,
        page_size,
        scores.stride(0),
        indices.stride(0),
        page_table.stride(0),
        page_indices.stride(0),
        raw_indices.stride(0) if raw_indices is not None else 0,
        raw_indices is not None,
        256,
        num_warps=4,
    )
