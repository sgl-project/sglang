"""NPU QSA expansion and physical mapping using the existing LI page view."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _vector_cores(device):
    return triton.runtime.driver.active.utils.get_device_properties(device)[
        "num_vectorcore"
    ]


@triton.jit
def _map_paged_slots(
    indices,
    sequence_ids,
    lengths,
    pages,
    positions,
    output,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    RATIO: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    SS: tl.constexpr,
    LS: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PS: tl.constexpr,
    COMPACT: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(WIDTH, TILE)
    for task in range(
        tl.program_id(0), triton.cdiv(ROWS, BLOCK_M) * TILES, tl.num_programs(0)
    ):
        row = ((task // TILES) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        col = task % TILES * TILE + tl.arange(0, TILE)
        seq = tl.load(sequence_ids + row * SS, mask=row < ROWS, other=0).to(
            tl.int64
        )
        length = tl.load(lengths + seq * LS, mask=row < ROWS, other=0).to(
            tl.int32
        )
        if COMPACT:
            group_cols = tl.arange(0, GROUPS)
            groups = tl.load(
                indices + row[:, None] * IS0 + group_cols[None, :] * IS1,
                mask=row[:, None] < ROWS,
                other=-1,
            ).to(tl.int32)
            token_count = tl.sum((groups >= 0).to(tl.int32), 1) * RATIO
            source = tl.broadcast_to(
                tl.minimum(col // RATIO, GROUPS - 1)[None, :], (BLOCK_M, TILE)
            )
            group = tl.gather(
                groups.to(tl.float32, bitcast=True), source, axis=1
            ).to(tl.int32, bitcast=True)
            visible = tl.load(
                positions + row * PS, mask=row < ROWS, other=-1
            ).to(tl.int32)
            visible += 1
            tail_start = visible // RATIO * RATIO
            tail_offset = col[None, :] - token_count[:, None]
            tail_count = tl.minimum(
                visible - tail_start, tl.maximum(length - tail_start, 0)
            )
            logical = tl.where(
                col[None, :] < token_count[:, None],
                group * RATIO + col[None, :] % RATIO,
                tl.where(
                    (tail_offset >= 0) & (tail_offset < tail_count[:, None]),
                    tail_start[:, None] + tail_offset,
                    -1,
                ),
            )
        else:
            logical = tl.load(
                indices + row[:, None] * IS0 + col[None, :] * IS1,
                mask=(row[:, None] < ROWS) & (col[None, :] < WIDTH),
                other=-1,
            )
        valid = (
            (col[None, :] < WIDTH) & (logical >= 0) & (logical < length[:, None])
        )
        safe = tl.minimum(tl.maximum(logical, 0), NUM_PAGES * PAGE_SIZE - 1).to(
            tl.int32
        )
        # One coalesced page-row load, then local gather. Random global scalar
        # loads of every selected token are slow on NPU vector cores.
        page_cols = tl.arange(0, PAGE_BLOCK)
        page_ids = tl.load(
            pages + seq[:, None] * TS0 + page_cols[None, :] * TS1,
            mask=(row[:, None] < ROWS) & (page_cols[None, :] < NUM_PAGES),
            other=0,
        ).to(tl.int32)
        physical_page = tl.gather(
            page_ids.to(tl.float32, bitcast=True), safe // PAGE_SIZE, axis=1
        ).to(tl.int32, bitcast=True)
        slot = physical_page * PAGE_SIZE + safe % PAGE_SIZE
        tl.store(
            output + row[:, None] * WIDTH + col[None, :],
            tl.where(valid, slot, -1),
            mask=(row[:, None] < ROWS) & (col[None, :] < WIDTH),
        )


def qsa_paged_indices_to_slots(
    indices,
    sequence_ids,
    lengths,
    page_table,
    query_positions=None,
    page_size=128,
    *,
    tile=None,
    block_m=None,
):
    """Map token IDs or complete groups using replay-refreshed LI metadata.

    All tensors are contiguous int32 NPU tensors.
    page_table holds full-KV page IDs (the same IDs used for ratio4 compressed
    KV pages). sequence_ids selects a valid metadata row. The caller refreshes
    this table when requests/slots change, exactly as for LightningIndexer.
    query_positions enables 512-group expansion, preserving order, duplicates,
    the incomplete causal tail and -1 padding. Lengths fit the table capacity.
    """
    rows, width = indices.shape
    compact = query_positions is not None
    tensors = [indices, sequence_ids, lengths, page_table]
    tensors += [] if not compact else [query_positions]
    if indices.device.type != "npu" or any(
        t.device != indices.device
        or t.dtype != torch.int32
        or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError(
            "QSA slot mapping requires contiguous int32 tensors on one NPU"
        )
    if sequence_ids.shape != (rows,) or lengths.ndim != 1 or page_table.ndim != 2:
        raise ValueError("QSA mapping rows/table do not match")
    if page_table.shape[0] != lengths.numel() or not page_table.shape[1]:
        raise ValueError(
            "QSA page table must have one nonempty row per sequence length"
        )
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("QSA full-KV page size must be a power of two")
    if compact and (width != 512 or query_positions.shape != (rows,)):
        raise ValueError("Compact NPU mapping requires 512 complete groups of four")
    if any(s < 0 for t in tensors for s in t.stride()):
        raise ValueError("QSA slot mapping requires nonnegative strides")
    out_width = 2051 if compact else width
    output = torch.empty((rows, out_width), dtype=torch.int32, device=indices.device)
    if not rows or not out_width:
        return output
    if tile is None or block_m is None:
        small_tile = rows < 4
        tile = 1024 if small_tile else 512
        block_m = 1 if small_tile else (8 if compact else 4)
    grid = min(
        triton.cdiv(rows, block_m) * triton.cdiv(out_width, tile),
        _vector_cores(indices.device.index),
    )
    _map_paged_slots[(grid,)](
        indices,
        sequence_ids,
        lengths,
        page_table,
        query_positions,
        output,
        rows,
        out_width,
        width if compact else 0,
        4,
        *indices.stride(),
        sequence_ids.stride(0),
        lengths.stride(0),
        *page_table.stride(),
        page_table.shape[1],
        page_size,
        query_positions.stride(0) if compact else 0,
        compact,
        triton.next_power_of_2(page_table.shape[1]),
        tile,
        block_m,
    )
    return output
