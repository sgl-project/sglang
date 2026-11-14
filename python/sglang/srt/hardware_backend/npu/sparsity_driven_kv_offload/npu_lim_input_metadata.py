"""Pre-LIM request metadata for decode and chain-MTP LIM + Copy-SFA."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _prepare_kernel(
    seq_ptr,
    positions_ptr,
    slot_ptr,
    last_ptr,
    src_table_ptr,
    out_table_ptr,
    safe_slot_ptr,
    length_ptr,
    prefix_ptr,
    sparse_ptr,
    valid_ptr,
    state_ptr,
    next_prefix_ptr,
    cache_size_ptr,
    ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    SRC_COLS: tl.constexpr,
    DST_COLS: tl.constexpr,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int32)
    row = tl.program_id(1).to(tl.int32)
    cols = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int32)

    # Every destination column is assigned once: preserve every source column,
    # including PAD columns, and zero only the destination's trailing columns.
    if tile * BLOCK < DST_COLS:
        value = tl.load(
            src_table_ptr + row * SRC_S0 + cols * SRC_S1,
            mask=(cols < SRC_COLS) & (cols < DST_COLS),
            other=0,
        ).to(tl.int32)
        tl.store(
            out_table_ptr + row * DST_S0 + cols * DST_S1, value, mask=cols < DST_COLS
        )

    if tile == 0:
        raw_slot = tl.load(slot_ptr + row).to(tl.int64)
        active_length = tl.load(seq_ptr + row).to(tl.int64)
        raw_length = active_length
        if HAS_POSITIONS:
            raw_length = (
                tl.load(positions_ptr + row * QUERY_WIDTH + QUERY_WIDTH - 1).to(
                    tl.int64
                )
                + 1
            )
        valid = (
            (raw_slot > 0) & (raw_slot < ROWS) & (active_length > 0) & (raw_length > 0)
        )
        safe_slot = tl.where(valid, raw_slot, ROWS).to(tl.int32)
        length = tl.where(valid, raw_length, 0).to(tl.int32)
        prefix = length
        sparse = valid & (prefix >= CAPACITY)
        previous = tl.load(last_ptr + safe_slot).to(tl.int32)
        warm = sparse & (previous >= 0) & (prefix >= previous)
        state = tl.where(sparse, tl.where(warm, -1, -2), -3).to(tl.int32)
        next_prefix = tl.where(sparse, prefix, -1).to(tl.int32)
        cache_size = tl.where(sparse, CAPACITY, 0).to(tl.int32)
        tl.store(safe_slot_ptr + row, safe_slot)
        tl.store(length_ptr + row, length)
        tl.store(prefix_ptr + row, prefix)
        tl.store(sparse_ptr + row, sparse)
        tl.store(valid_ptr + row, valid)
        tl.store(state_ptr + row, state)
        tl.store(next_prefix_ptr + row, next_prefix)
        tl.store(cache_size_ptr + row, cache_size)


def fused_prepare_lim_metadata(
    *,
    seq_lens,
    request_slots,
    last_prefix,
    block_table,
    out_block_table,
    safe_slots,
    lengths,
    prefixes,
    sparse_rows,
    valid_rows,
    request_states,
    next_prefixes,
    cache_sizes,
    capacity: int,
    request_rows: int,
    query_width: int = 1,
    positions=None,
) -> None:
    """Fill request inputs and source pages; positions supply the actual visible length."""
    batch = seq_lens.numel()
    cols = out_block_table.shape[1]
    block = 128
    _prepare_kernel[(triton.cdiv(cols, block), batch)](
        seq_lens,
        positions if positions is not None else seq_lens,
        request_slots,
        last_prefix,
        block_table,
        out_block_table,
        safe_slots,
        lengths,
        prefixes,
        sparse_rows,
        valid_rows,
        request_states,
        next_prefixes,
        cache_sizes,
        request_rows,
        capacity,
        query_width,
        positions is not None,
        block_table.shape[1],
        cols,
        block_table.stride(0),
        block_table.stride(1),
        out_block_table.stride(0),
        out_block_table.stride(1),
        block,
    )


__all__ = ["fused_prepare_lim_metadata"]
