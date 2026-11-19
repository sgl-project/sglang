# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Append the trailing sliding window to DSA top-k rows (Triton).

For DSA layers that attend over the indexer top-k and the last ``window`` tokens
with one softmax. The indexer selects only positions before that window, so the
sets are disjoint; each row is ``[top-k prefix | window]`` from column 0, then
``-1`` padding to ``topk + window`` rounded up to ``alignment``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _append_window_to_topk_kernel(
    topk_ptr,
    seq_lens_ptr,
    page_table_ptr,
    page_rows_ptr,
    out_ptr,
    out_lengths_ptr,
    topk_stride,
    page_table_stride,
    out_stride,
    live_rows,
    page_table_rows,
    page_table_cols,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAS_PAGE_ROWS: tl.constexpr,
    RAGGED: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    live = row < live_rows
    seq_len = tl.maximum(tl.load(seq_lens_ptr + row, mask=live, other=0), 0)
    topk_len = tl.minimum(tl.maximum(seq_len - WINDOW, 0), TOPK)
    window_len = tl.minimum(seq_len, WINDOW)

    # Skips the window columns written below, so the two stores never overlap.
    for start in tl.static_range(0, OUT_WIDTH, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        keep = (cols < OUT_WIDTH) & ((cols < topk_len) | (cols >= topk_len + WINDOW))
        values = tl.load(
            topk_ptr + row * topk_stride + cols,
            mask=keep & (cols < topk_len),
            other=-1,
        )
        tl.store(out_ptr + row * out_stride + cols, values, mask=keep)

    offs = tl.arange(0, WINDOW_BLOCK)
    in_window = offs < WINDOW
    position = seq_len - window_len + offs
    valid = live & in_window & (offs < window_len)
    if RAGGED:
        # Ragged top-k indexes the request's flattened KV from its row offset.
        slots = tl.load(page_rows_ptr + row, mask=live, other=0) + position
    else:
        if HAS_PAGE_ROWS:
            page_row = tl.load(page_rows_ptr + row, mask=live, other=0).to(tl.int64)
        else:
            page_row = row.to(tl.int64)
        page_id = position // PAGE_SIZE
        valid = valid & (page_row < page_table_rows) & (page_id < page_table_cols)
        page = tl.load(
            page_table_ptr + page_row * page_table_stride + page_id,
            mask=valid,
            other=0,
        )
        slots = page * PAGE_SIZE + position % PAGE_SIZE
    tl.store(
        out_ptr + row * out_stride + topk_len + offs,
        tl.where(valid, slots, -1),
        mask=in_window,
    )
    tl.store(out_lengths_ptr + row, topk_len + window_len)


def append_window_to_topk(
    topk_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    window: int,
    page_size: int,
    alignment: int,
    page_rows: Optional[torch.Tensor] = None,
    ragged_offsets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(slots, lengths)`` for ``top-k + last window`` per query row.

    ``topk_slots`` is ``[num_rows, topk]`` int32 KV slot ids, ``seq_lens`` the
    per-query visible length (``seq_lens.numel()`` live rows; rows past it are
    padding and come out empty). ``page_table`` is ``[rows, pages]`` page ids
    at ``page_size``; ``page_rows`` maps a query row to its page-table row and
    is None when the table already has one row per query. With
    ``ragged_offsets`` the top-k indexes the flattened per-request KV instead,
    and the window is ``offset + position``; ``page_table`` and ``page_rows``
    are then unused.
    """
    num_rows, topk = topk_slots.shape
    live_rows = seq_lens.numel()
    if ragged_offsets is not None:
        assert ragged_offsets.numel() >= live_rows, (
            f"ragged_offsets has {ragged_offsets.numel()} rows, need {live_rows}"
        )
        page_rows = ragged_offsets.to(torch.int32)
    elif page_rows is None:
        assert page_table.shape[0] >= live_rows, (
            f"page table has {page_table.shape[0]} rows for {live_rows} queries "
            "and no row map"
        )
    else:
        assert page_rows.numel() >= live_rows, (
            f"page_rows has {page_rows.numel()} rows, need {live_rows}"
        )
    assert topk_slots.stride(1) == 1 and page_table.stride(1) == 1, (
        "topk_slots and page_table must be row-major"
    )

    out_width = triton.cdiv(topk + window, alignment) * alignment
    out = torch.empty(
        (num_rows, out_width), dtype=torch.int32, device=topk_slots.device
    )
    out_lengths = torch.empty(num_rows, dtype=torch.int32, device=topk_slots.device)
    if num_rows == 0:
        return out, out_lengths
    _append_window_to_topk_kernel[(num_rows,)](
        topk_slots,
        seq_lens,
        page_table,
        page_rows if page_rows is not None else seq_lens,
        out,
        out_lengths,
        topk_slots.stride(0),
        page_table.stride(0),
        out.stride(0),
        live_rows,
        page_table.shape[0],
        page_table.shape[1],
        TOPK=topk,
        WINDOW=window,
        PAGE_SIZE=page_size,
        HAS_PAGE_ROWS=page_rows is not None,
        RAGGED=ragged_offsets is not None,
        OUT_WIDTH=out_width,
        # 1024 over 8 warps keeps 4 elements per lane.
        BLOCK=min(1024, triton.next_power_of_2(out_width)),
        WINDOW_BLOCK=triton.next_power_of_2(window),
        num_warps=8,
    )
    return out, out_lengths
