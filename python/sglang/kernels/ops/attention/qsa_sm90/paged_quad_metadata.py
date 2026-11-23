# SPDX-License-Identifier: Apache-2.0
"""Resolve the physical KV slots and token masks for each M64 query tile."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "GROUPS",
        "PACK_CAPACITY",
        "CAPACITY",
        "TILES",
        "MAX_UNION",
        "WIDTH",
        "CACHE_ROWS",
    ]
)
def _compact_quads(
    MASK,
    TABLE,
    CU_Q,
    SEQ_LENS,
    SLOTS,
    PACKED_MASKS,
    COUNTS,
    GROUPS,
    PACK_CAPACITY,
    CAPACITY,
    TILES,
    MAX_UNION,
    WIDTH,
    CACHE_ROWS,
    HEADS_PER_KV: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    QUAD_MASK_BITS: tl.constexpr = 16
    QUAD_MASK: tl.constexpr = (1 << QUAD_MASK_BITS) - 1
    tile, batch = tl.program_id(0), tl.program_id(1)
    first_row = (tile * 64) // HEADS_PER_KV
    length = tl.load(CU_Q + batch + 1) - tl.load(CU_Q + batch)
    total = 0
    if first_row < length:
        last_row = (tile * 64 + 63) // HEADS_PER_KV
        first = first_row // 8
        last = last_row // 8
        # Use one compiled scan width for short and long requests.
        # Each outer iteration appends the next ascending range of quads.
        row_capacity = tl.minimum(CAPACITY, tl.cdiv(tl.load(SEQ_LENS + batch), 4))
        row_words = tl.cdiv(row_capacity, 2)
        for word_start in range(0, row_words, BLOCK_WORDS):
            word = word_start + tl.arange(0, BLOCK_WORDS)
            combined = tl.full((BLOCK_WORDS,), 0, tl.int32)
            quad_masks_lo = tl.full((BLOCK_WORDS,), 0, tl.uint64)
            quad_masks_hi = tl.full((BLOCK_WORDS,), 0, tl.uint64)
            # Include the partial query groups at the tile boundaries.
            rows_per_tile: tl.constexpr = tl.cdiv(64 + HEADS_PER_KV - 1, HEADS_PER_KV)
            groups_per_tile: tl.constexpr = tl.cdiv(rows_per_tile + 7, 8)
            for offset in tl.static_range(groups_per_tile):
                group = first + offset
                valid = (group < GROUPS) & (group <= last) & (word < row_words)
                data = tl.load(
                    MASK + (batch.to(tl.int64) * GROUPS + group) * PACK_CAPACITY + word,
                    valid,
                    other=0,
                )
                row_begin = tl.minimum(tl.maximum(first_row - group * 8, 0), 8)
                row_end = tl.maximum(tl.minimum(last_row + 1 - group * 8, 8), row_begin)
                row_bits = ((1 << (2 * row_end)) - (1 << (2 * row_begin))).to(tl.uint32)
                row_bits = row_bits | (row_bits << QUAD_MASK_BITS)
                data = (data & row_bits.to(tl.int32)).to(tl.uint32)
                combined = combined | data.to(tl.int32)
                shift = offset * QUAD_MASK_BITS
                quad_masks_lo = quad_masks_lo | (
                    (data & QUAD_MASK).to(tl.uint64) << shift
                )
                quad_masks_hi = quad_masks_hi | (
                    (data >> QUAD_MASK_BITS).to(tl.uint64) << shift
                )
            first_valid = (word * 2 < row_capacity) & ((combined & QUAD_MASK) != 0)
            second_valid = (word * 2 + 1 < row_capacity) & (
                (combined >> QUAD_MASK_BITS) != 0
            )
            counts = first_valid.to(tl.int32) + second_valid.to(tl.int32)
            prefix = total + tl.cumsum(counts, axis=0) - counts
            for half in tl.static_range(2):
                quad = word * 2 + half
                valid = first_valid if half == 0 else second_valid
                dst = prefix if half == 0 else prefix + first_valid.to(tl.int32)
                active = valid & (dst < MAX_UNION)
                address = tl.load(
                    TABLE + batch.to(tl.int64) * WIDTH + quad * 4,
                    active & (quad * 4 < WIDTH),
                    other=0,
                )
                offset = (batch.to(tl.int64) * TILES + tile) * MAX_UNION + dst
                bits = quad_masks_lo if half == 0 else quad_masks_hi
                # Rebase the mask to the first query in this tile.
                bits = bits >> ((first_row % 8) * 2)
                tl.store(PACKED_MASKS + offset, bits, active)
                tl.store(
                    SLOTS + offset,
                    tl.maximum(0, tl.minimum(address, CACHE_ROWS - 4)),
                    active,
                )
            total += tl.sum(counts, axis=0)
    tl.store(COUNTS + batch * TILES + tile, total)


def compact_quads(masks, token_slots, cu_q, seq_lens, max_q, heads_per_kv, cache_rows):
    width = token_slots.shape[1]
    batch, groups, packed_capacity = masks.shape
    capacity = triton.cdiv(width, 4)
    tiles = triton.cdiv(max_q * heads_per_kv, 64)
    # A tile can span at most this many query rows, each selecting 512 full
    # blocks and one causal tail. No data-dependent allocation or truncation.
    rows_per_tile = triton.cdiv(64 + heads_per_kv - 1, heads_per_kv)
    max_union = min(capacity, rows_per_tile * 513)
    slots = torch.empty(
        (batch, tiles, max_union), dtype=torch.int32, device=masks.device
    )
    mask_dtype = torch.int32 if rows_per_tile <= 16 else torch.int64
    packed_masks = torch.empty_like(slots, dtype=mask_dtype)
    counts = torch.empty((batch, tiles), dtype=torch.int32, device=masks.device)
    _compact_quads[(tiles, batch)](
        masks,
        token_slots,
        cu_q,
        seq_lens,
        slots,
        packed_masks,
        counts,
        groups,
        packed_capacity,
        capacity,
        tiles,
        max_union,
        width,
        cache_rows,
        heads_per_kv,
        1024,
        num_warps=8,
    )
    return slots, packed_masks, counts
