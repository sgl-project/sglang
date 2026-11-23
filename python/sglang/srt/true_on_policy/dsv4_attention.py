"""Stands in for the stock paged FP8 decode (flash_mla_with_kvcache), whose tile scheduler splits each query's KV
across SM partitions by a plan over the whole batch's lengths.

DeepSeek-V4 sparse attention as one TOP program for prefill, decode and the trainer.

Each query reads its valid compressed slots in the given order (ascending logical
blocks from the ordered indexer or the c128 schedule), then its SWA window newest
first. The selected FP8 rows are dequantized per query and read by
flash_mla_sparse_fwd with the attention sink; there is no paged split-KV decode.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged_dsv4,
)

# Query rows per launch; flash_mla_sparse_fwd treats every query row independently.
# 4096 rows bound the BF16 workspace to about 2.7 GB at 640 slots.
ROWS = 4096


class KVSelection:
    """Per-query slots into one paged FP8 cache: [rows, width] indices, -1 or past length invalid."""

    def __init__(self, *, cache, indices, lengths, page_size, layout):
        self.cache = cache
        self.indices = indices
        self.lengths = lengths
        self.page_size = page_size
        self.layout = layout

    def rows(self, start, end):
        return KVSelection(
            cache=self.cache,
            indices=self.indices[start:end],
            lengths=self.lengths[start:end],
            page_size=self.page_size,
            layout=self.layout,
        )


@triton.jit
def _pack_selection_kernel(
    idx_ptr,
    len_ptr,
    slots_ptr,
    local_ptr,
    count_ptr,
    pos_ptr,
    width,
    idx_stride,
    local_stride,
    base,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Append one selection's valid slots to each row, in column order.

    Valid means index >= 0 and column < the row's length; this is the stable
    valid-first order of the selections concatenated in call order. MODE
    _PER_CELL: every (row, column) owns a workspace row and its slot is written
    to slots_ptr. _COMPACT: the workspace row is the slot's compact position.
    _OWNERS: the workspace row is the slot's first (row, column) cell (pos_ptr
    holds it per pool slot), and only that cell's slot is written to slots_ptr.
    """
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_row = cols < width
    idx = tl.load(idx_ptr + r * idx_stride + cols, mask=in_row, other=-1)
    valid = in_row & (idx >= 0) & (cols < tl.load(len_ptr + r))
    if MODE == 1:
        compact = (
            tl.load(pos_ptr + tl.where(valid, idx, 0).to(tl.int64), mask=valid, other=1)
            - 1
        )
        local = (base + compact).to(tl.int32)
    elif MODE == 2:
        cell = r * width + cols
        owner = tl.load(
            pos_ptr + tl.where(valid, idx, 0).to(tl.int64), mask=valid, other=0
        )
        # Only the owning cell dequantizes the slot; every other cell reads its row.
        tl.store(
            slots_ptr + cell, tl.where(valid & (owner == cell), idx, -1), mask=in_row
        )
        local = (base + owner).to(tl.int32)
    else:
        # Invalid slots stay -1 so the dequant skips them; attention never reads those rows.
        tl.store(slots_ptr + r * width + cols, tl.where(valid, idx, -1), mask=in_row)
        local = (base + r * width + cols).to(tl.int32)
    start = tl.load(count_ptr + r)
    flags = valid.to(tl.int32)
    pos = start + tl.cumsum(flags, 0) - 1
    tl.store(local_ptr + r * local_stride + pos, local, mask=valid)
    tl.store(count_ptr + r, start + tl.sum(flags, 0))


@triton.jit
def _mark_slots_kernel(
    idx_ptr, len_ptr, seen_ptr, width, idx_stride, BLOCK: tl.constexpr
):
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_row = cols < width
    idx = tl.load(idx_ptr + r * idx_stride + cols, mask=in_row, other=-1)
    valid = in_row & (idx >= 0) & (cols < tl.load(len_ptr + r))
    # Racing writers all store 1.
    tl.store(seen_ptr + idx.to(tl.int64), tl.full((BLOCK,), 1, tl.int32), mask=valid)


@triton.jit
def _claim_slots_kernel(
    idx_ptr, len_ptr, owner_ptr, width, idx_stride, BLOCK: tl.constexpr
):
    """Each selected pool slot's owner: its first (row, column) cell in row-major order."""
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_row = cols < width
    idx = tl.load(idx_ptr + r * idx_stride + cols, mask=in_row, other=-1)
    valid = in_row & (idx >= 0) & (cols < tl.load(len_ptr + r))
    tl.atomic_min(
        owner_ptr + tl.where(valid, idx, 0).to(tl.int64), r * width + cols, mask=valid
    )


@triton.jit
def _compact_slots_kernel(seen_ptr, pos_ptr, uniq_ptr, num_slots, BLOCK: tl.constexpr):
    s = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = s < num_slots
    marked = tl.load(seen_ptr + s, mask=live, other=0) != 0
    compact = tl.load(pos_ptr + s, mask=live, other=1) - 1
    tl.store(uniq_ptr + compact, s.to(tl.int32), mask=live & marked)


# Selections of at least this many (row, column) cells dequantize each distinct slot once into a compact workspace;
# smaller ones keep one workspace row per cell. Every mode lays out the same bytes per query; arbitrary, not tuned.
DEDUP_MIN_CELLS = 1 << 18
_PER_CELL, _COMPACT, _OWNERS = 0, 1, 2


def _num_slots(selection):
    """Token slots addressable in the selection's paged cache: pages x page size."""
    cache = selection.cache
    return cache.numel() // cache.shape[-1] * selection.page_size


def _distinct_slots(selection, rows, width):
    """Compact position (1-based, 0 = unselected) of every pool slot the chunk selects, and the slots in order.

    Sync-free: a mark pass over the selected cells, a cumsum over the pool's slot space, a compaction pass.
    """
    device = selection.indices.device
    num_slots = _num_slots(selection)
    seen = torch.zeros(num_slots, dtype=torch.int32, device=device)
    _mark_slots_kernel[(rows,)](
        selection.indices,
        selection.lengths.to(torch.int32),
        seen,
        width,
        selection.indices.stride(0),
        BLOCK=triton.next_power_of_2(width),
    )
    pos = torch.cumsum(seen, 0, dtype=torch.int32)
    uniq = torch.full(
        (min(num_slots, rows * width),), -1, dtype=torch.int32, device=device
    )
    block = 1024
    _compact_slots_kernel[(triton.cdiv(num_slots, block),)](
        seen, pos, uniq, num_slots, BLOCK=block
    )
    return pos, uniq


def _owner_cells(selection, rows, width):
    """Per pool slot, the first (row, column) cell selecting it: one fill and one atomic-min pass over the cells."""
    owners = torch.full(
        (_num_slots(selection),),
        2**31 - 1,
        dtype=torch.int32,
        device=selection.indices.device,
    )
    _claim_slots_kernel[(rows,)](
        selection.indices,
        selection.lengths.to(torch.int32),
        owners,
        width,
        selection.indices.stride(0),
        BLOCK=triton.next_power_of_2(width),
    )
    return owners


def _gather_chunk(selections, dim, shared_rows=False):
    """Dequantize every selected slot of a row chunk; return the workspace and packed indices.

    flash_mla_sparse_fwd reads the workspace only through the packed indices, so a query sees the same
    rows in the same order whether a slot shared by many queries is dequantized once or once per query.
    ``shared_rows``: rows of one request select mostly the same slots (verify, extend), so each slot is
    dequantized once, by its first cell, in a per-cell workspace.
    """
    rows = selections[0].indices.shape[0]
    device = selections[0].indices.device
    widths = [s.indices.shape[1] for s in selections]
    modes = [
        _COMPACT
        if rows * w >= DEDUP_MIN_CELLS
        else (_OWNERS if shared_rows else _PER_CELL)
        for w in widths
    ]
    extents = [
        min(_num_slots(s), rows * w) if m == _COMPACT else rows * w
        for s, w, m in zip(selections, widths, modes, strict=True)
    ]
    kv = torch.empty((sum(extents), 1, dim), dtype=torch.bfloat16, device=device)
    # SM90's sparse kernel reads top-k in 128-slot pairs of blocks; the extra columns stay -1, past every length.
    align = 128 if torch.cuda.get_device_capability(device)[0] == 9 else 1
    packed = torch.full(
        (rows, -(-sum(widths) // align) * align), -1, dtype=torch.int32, device=device
    )
    lengths = torch.zeros(rows, dtype=torch.int32, device=device)
    base = 0
    for s, width, mode, extent in zip(selections, widths, modes, extents, strict=True):
        if mode == _COMPACT and rows:
            pos, slots = _distinct_slots(s, rows, width)
        else:
            pos = _owner_cells(s, rows, width) if mode == _OWNERS and rows else None
            slots = torch.empty(rows * width, dtype=torch.int32, device=device)
        if rows:
            _pack_selection_kernel[(rows,)](
                s.indices,
                s.lengths.to(torch.int32),
                slots,
                packed,
                lengths,
                slots if pos is None else pos,
                width,
                s.indices.stride(0),
                packed.stride(0),
                base,
                MODE=mode,
                BLOCK=triton.next_power_of_2(width),
            )
        dequantize_k_cache_paged_dsv4(
            s.cache,
            slots,
            page_size=s.page_size,
            out=kv[base : base + extent],
            layout=s.layout,
        )
        base += extent
    return kv, packed, lengths


def sparse_attention(q, selections, *, attn_sink, scale, d_v, shared_rows=False):
    """q [rows, heads, dim] BF16; ``selections`` in attention order (compressed, then SWA).

    ``shared_rows``: a hint that rows of one request share most slots (not plain decode); same bytes either way.
    """
    from sglang.kernels.ops.attention.flash_mla_interface import (
        flash_mla_sparse_fwd,
    )

    if q.dtype != torch.bfloat16 or q.ndim != 3:
        raise NotImplementedError(
            "[top] DeepSeek-V4 attention requires BF16 [rows, heads, dim]"
        )
    if any(s.indices.shape[1] % 64 for s in selections):
        raise NotImplementedError("[top] DeepSeek-V4 slot widths must be 64-aligned")
    rows = q.shape[0]
    out = q.new_empty((rows, q.shape[1], d_v))
    for start in range(0, rows, ROWS):
        end = min(start + ROWS, rows)
        kv, indices, lengths = _gather_chunk(
            [s.rows(start, end) for s in selections], q.shape[2], shared_rows
        )
        out[start:end] = flash_mla_sparse_fwd(
            q=q[start:end],
            kv=kv,
            indices=indices.unsqueeze(1),
            sm_scale=scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=lengths,
        )[0]
    return out


def pool_sparse_attention(
    q,
    *,
    token_to_kv_pool,
    layer_id,
    compress_ratio,
    swa_indices,
    swa_lengths,
    extra_indices,
    extra_lengths,
    attn_sink,
    scale,
    d_v,
    shared_rows=False,
):
    """The serving entry: selections over the DS4 pool's SWA and compressed caches."""
    selections = []
    if compress_ratio != 0:
        selections.append(
            KVSelection(
                cache=token_to_kv_pool.get_extra_key_buffer(layer_id),
                indices=extra_indices,
                lengths=extra_lengths,
                page_size=token_to_kv_pool.get_extra_key_page_size(layer_id),
                layout=token_to_kv_pool.get_extra_key_layout(layer_id),
            )
        )
    selections.append(
        KVSelection(
            cache=token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
            indices=swa_indices,
            lengths=swa_lengths,
            page_size=token_to_kv_pool.swa_kv_pool.page_size,
            layout=token_to_kv_pool.get_swa_key_layout(),
        )
    )
    return sparse_attention(
        q.reshape(q.shape[0], -1, q.shape[-1]),
        selections,
        attn_sink=attn_sink,
        scale=scale,
        d_v=d_v,
        shared_rows=shared_rows,
    )
