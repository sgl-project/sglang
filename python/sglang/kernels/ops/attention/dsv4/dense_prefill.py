"""The DeepGEMM dense prefill scores of the ratio-1/2 index layers, tile by
tile under a memory budget, and the top-k over them: plain, or publishing /
consuming candidate block ids per request."""

from __future__ import annotations

from typing import Iterator

import torch

from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.utils.common import ceil_align

from .candidate_blocks import (
    candidate_block_mask,
    mask_topk_scores,
    select_candidate_block_ids,
)
from .topk import topk_transform_ragged_v2

# TODO: use a per-forward mqa_logits_budget_bytes() budget that also
# leaves room for candidate masks and block-selection scratch.
_SCORE_BUDGET_BYTES = 2 << 30


def dense_prefill_topk(
    *,
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    request_lengths: list[tuple[int, int]],
    topk: int,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    publish_candidates: bool,
    candidates: list[torch.Tensor] | None,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    selected = torch.full(
        (q[0].shape[0], topk), -1, dtype=torch.int32, device=weights.device
    )
    published = [] if publish_candidates else None
    request_ranges = []
    row = 0
    for query_length, context_length in request_lengths:
        request_ranges.append((row, row + query_length, context_length))
        if published is not None:
            num_blocks = (
                context_length + candidate_block_size - 1
            ) // candidate_block_size
            published.append(
                torch.empty(
                    (query_length, min(candidate_topk_blocks, num_blocks)),
                    dtype=torch.int32,
                    device=weights.device,
                )
            )
        row += query_length
    width = ceil_align(max((n for _, n in request_lengths), default=0), 4)
    if row == 0 or width == 0:
        return selected, published
    rows_per_chunk = _rows_per_chunk(row, width, heads=q[0].shape[1])
    for offset in range(0, row, rows_per_chunk):
        rows = slice(offset, min(offset + rows_per_chunk, row))
        _select_tile(
            q=(q[0][rows], q[1][rows]),
            kv=kv,
            weights=weights[rows],
            starts=starts[rows],
            lengths=lengths[rows],
            width=width,
            selected=selected[rows],
            block_size=candidate_block_size,
            row_offset=offset,
            request_ranges=request_ranges,
            publish=published,
            consume=candidates,
        )
    return selected, published


def _rows_per_chunk(rows: int, width: int, *, heads: int) -> int:
    """Query rows per logits tile so one fp32 [rows, width] tile fits the
    budget; the row count stays a multiple of the kernel's row alignment."""
    row_alignment = 128 // heads
    rows_per_chunk = mqa_logits_rows_per_chunk(
        num_rows=ceil_align(rows, row_alignment),
        row_bytes=mqa_logits_row_bytes(width),
        budget_bytes=_SCORE_BUDGET_BYTES,
    )
    if rows_per_chunk is None:
        return rows
    return max(row_alignment, rows_per_chunk // row_alignment * row_alignment)


def score_tiles(
    *,
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    context_lengths: list[int],
    width_align: int = 4,
) -> Iterator[tuple[slice, torch.Tensor]]:
    """The dense scores of the chunk row tile by row tile under the budget:
    ``(rows, logits)`` with fp32 ``logits[i, j]`` the score of query row
    ``rows.start + i`` against ``kv[starts + j]``, garbage past the row's
    ``lengths``; the width is the largest of ``context_lengths`` (compressed
    positions at each request's newest token) aligned to ``width_align``
    columns."""
    from deep_gemm import fp8_fp4_mqa_logits

    rows = q[0].shape[0]
    width = ceil_align(max(context_lengths, default=0), width_align)
    if rows == 0 or width == 0:
        return
    rows_per_chunk = _rows_per_chunk(rows, width, heads=q[0].shape[1])
    for offset in range(0, rows, rows_per_chunk):
        tile = slice(offset, min(offset + rows_per_chunk, rows))
        tile_starts = starts[tile]
        yield (
            tile,
            fp8_fp4_mqa_logits(
                (q[0][tile], q[1][tile]),
                kv,
                weights[tile],
                tile_starts,
                tile_starts + lengths[tile],
                False,
                width,
            ),
        )


def _select_tile(
    *,
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
    selected: torch.Tensor,
    block_size: int,
    row_offset: int,
    request_ranges: list[tuple[int, int, int]],
    publish: list[torch.Tensor] | None,
    consume: list[torch.Tensor] | None,
) -> None:
    from deep_gemm import fp8_fp4_mqa_logits

    logits = fp8_fp4_mqa_logits(q, kv, weights, starts, starts + lengths, False, width)
    if publish is not None or consume is not None:
        for request, (start, end, context_length) in enumerate(request_ranges):
            begin, stop = max(start, row_offset), min(end, row_offset + logits.shape[0])
            if begin >= stop or context_length == 0:
                continue
            rows = slice(begin - row_offset, stop - row_offset)
            request_rows = slice(begin - start, stop - start)
            scores = logits[rows, :context_length]
            if publish is not None:
                lens = lengths[rows, None]
                scores.masked_fill_(
                    torch.arange(context_length, device=logits.device)[None, :] >= lens,
                    -torch.inf,
                )
                blocks = publish[request][request_rows]
                blocks.copy_(
                    select_candidate_block_ids(
                        logits=scores,
                        compress_lens=lens,
                        topk_blocks=blocks.shape[1],
                        block_size=block_size,
                    )
                )
            else:
                scores.masked_fill_(
                    ~candidate_block_mask(
                        blocks=consume[request][request_rows],
                        width=context_length,
                        block_size=block_size,
                    ),
                    -torch.inf,
                )
    topk_transform_ragged_v2(logits, lengths, out_offsets=starts, out_indices=selected)
    if consume is not None:
        selected.copy_(
            mask_topk_scores(scores=logits, indices=selected, offsets=starts)
        )
