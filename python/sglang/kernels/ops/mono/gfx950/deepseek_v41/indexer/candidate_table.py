# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 68506823: atom/model_ops/deepseek_v41/candidate_table.py
# (``_candidate_table_kernel``; outputs into caller-owned buffers, so a replayed graph reads the captured addresses)
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The REINDEX layers' key table over the candidate producer's blocks: each kept
block's virtual 8-row page (``score_fp4._keys``) and how far into its
``topk_blocks * 8`` columns a row may read."""

import triton
import triton.language as tl


@triton.jit
def _candidate_table_kernel(
    candidates,
    block_tables,
    batch_ids,
    visible,
    table,
    context,
    candidate_stride,
    page_stride,
    batch_stride,
    topk_blocks,
    UNITS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token: translate its kept blocks, then bound them.

    A slot past the row's kept count, and every slot of a padding token (batch
    id -1, its PAGE table never read), is written as block 0 rather than left
    at `-1`: `context` stops short of it so the scorer never reads it, and a
    negative id would be a real out-of-bounds read if anything ever did.
    """
    token = tl.program_id(0)
    batch = tl.load(batch_ids + token * batch_stride)
    slots = tl.arange(0, BLOCK)
    mask = slots < topk_blocks
    cand = tl.load(candidates + token * candidate_stride + slots, mask=mask, other=-1)
    live = cand >= 0

    page = tl.load(
        block_tables + batch * page_stride + cand // UNITS,
        mask=live & (batch >= 0),
        other=0,
    )
    unit = tl.where(live & (batch >= 0), page * UNITS + cand % UNITS, 0)
    tl.store(table + token * candidate_stride + slots, unit, mask=mask)

    seen = tl.load(visible + token).to(tl.int32)
    kept = tl.sum(live.to(tl.int32))
    last_selected = tl.max(tl.where(live, cand, -1))
    # A kept block before the request's newest block is full, not the entire
    # gap to `seen`. Bound its contribution by one block even if the pin was
    # lost upstream; empty rows still have no visible columns.
    last_span = tl.minimum(tl.maximum(seen - last_selected * ROWS, 0), ROWS)
    bound = (kept - 1) * ROWS + last_span
    tl.store(context + token, tl.where(kept > 0, bound, 0))


@triton.jit
def _reindex_table_kernel(
    ids,
    block_tables,
    batch_ids,
    visible,
    ids_out,
    table,
    context,
    ids_stride,
    page_stride,
    topk_blocks,
    UNITS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """SGLang's candidate blocks (no order, -1 after the kept ones) for scoring: the
    newest block (whose rows past ``visible`` are not keys yet) moved to the last
    kept slot, so ``context`` -- the full blocks, then the newest one's visible
    rows -- stops the scorer short of them; ``ids_out`` is that order (the
    compact selection's map back to positions), ``table`` its virtual pages."""
    token = tl.program_id(0)
    batch = tl.load(batch_ids + token)
    slots = tl.arange(0, BLOCK)
    mask = slots < topk_blocks
    cand = tl.load(ids + token * ids_stride + slots, mask=mask, other=-1)
    live = cand >= 0
    kept = tl.sum(live.to(tl.int32))
    seen = tl.load(visible + token).to(tl.int32)
    newest = (seen - 1) // ROWS
    at = tl.max(tl.where(live & (cand == newest), slots, -1))
    last = kept - 1
    moved = tl.sum(tl.where(slots == last, cand, 0))
    swap = at >= 0
    cand = tl.where(
        swap & (slots == at), moved, tl.where(swap & (slots == last), newest, cand)
    )
    tl.store(ids_out + token * ids_stride + slots, cand, mask=mask)
    page = tl.load(
        block_tables + batch * page_stride + cand // UNITS, mask=live, other=0
    )
    tl.store(
        table + token * ids_stride + slots,
        tl.where(live, page * UNITS + cand % UNITS, 0),
        mask=mask,
    )
    tail = tl.sum(tl.where(slots == last, cand, 0))
    last_span = tl.minimum(tl.maximum(seen - tail * ROWS, 0), ROWS)
    tl.store(context + token, tl.where(kept > 0, (kept - 1) * ROWS + last_span, 0))


def reindex_table(
    ids,
    block_tables,
    batch_ids,
    units_per_page,
    visible,
    ids_out,
    table,
    context,
    *,
    rows_per_block,
):
    """``_reindex_table_kernel`` over ``ids`` [tokens, topk_blocks] (SGLang ``CandidateBlocks.ids``)."""
    tokens, topk_blocks = ids.shape
    assert ids.stride(1) == 1 and ids_out.stride(0) == table.stride(0) == ids.stride(
        0
    ), (ids.stride(), table.stride())
    _reindex_table_kernel[(tokens,)](
        ids, block_tables, batch_ids, visible, ids_out, table, context, ids.stride(0), block_tables.stride(0),
        topk_blocks, UNITS=units_per_page, ROWS=rows_per_block, BLOCK=triton.next_power_of_2(topk_blocks),
    )  # fmt: skip


def candidate_block_table(candidates, block_tables, batch_ids, units_per_page, visible, table, context, *,
                          rows_per_block):  # fmt: skip
    """``table`` [tokens, topk_blocks] <- the kept blocks' virtual pages, ``context`` [tokens] <- the columns a row
    may read. ``block_tables`` rows are indexed by ``batch_ids`` (a row's request's first row here)."""
    tokens, topk_blocks = candidates.shape
    assert (
        table.shape[0] >= tokens
        and table.stride(0) == candidates.stride(0)
        and context.shape[0] >= tokens
    )
    _candidate_table_kernel[(tokens,)](
        candidates,
        block_tables,
        batch_ids,
        visible,
        table,
        context,
        candidates.stride(0),
        block_tables.stride(0),
        batch_ids.stride(0),
        topk_blocks,
        UNITS=units_per_page,
        ROWS=rows_per_block,
        BLOCK=triton.next_power_of_2(topk_blocks),
    )
