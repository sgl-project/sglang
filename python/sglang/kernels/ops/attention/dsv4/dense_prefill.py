"""The DeepGEMM dense prefill scores of the ratio-1/2 index layers, tile by
tile under a caller-given memory budget."""

from __future__ import annotations

from typing import Iterator

import torch

from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.utils.common import ceil_align


def _rows_per_chunk(rows: int, width: int, *, heads: int, budget_bytes: int) -> int:
    """Query rows per logits tile so one fp32 [rows, width] tile fits
    ``budget_bytes``; the row count stays a multiple of the kernel's row alignment."""
    row_alignment = 128 // heads
    rows_per_chunk = mqa_logits_rows_per_chunk(
        num_rows=ceil_align(rows, row_alignment),
        row_bytes=mqa_logits_row_bytes(width),
        budget_bytes=budget_bytes,
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
    budget_bytes: int,
    width_align: int = 4,
) -> Iterator[tuple[slice, torch.Tensor]]:
    """The dense scores of the chunk row tile by row tile, each fp32 tile within
    ``budget_bytes``:
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
    rows_per_chunk = _rows_per_chunk(
        rows, width, heads=q[0].shape[1], budget_bytes=budget_bytes
    )
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
