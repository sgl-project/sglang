"""DCP-local view builder for the DSA indexer (sharded-indexer port of
vllm-project/vllm#46076).

Ownership is assigned at PAGE granularity (``(slot // page_size) % dcp_size``),
not per-token. This is required, not cosmetic: the paged-MQA-logits kernel
addresses the (now genuinely dcp_size-smaller) index K-cache via
``block_tables`` at ``page_size``-row granularity, meaning every 64-row window
of the compacted local table must be one physically contiguous page. Per-token
ownership (``slot % dcp_size``) does NOT give you that -- it interleaves
``page_size / dcp_size`` rows from ``dcp_size`` different, generally
non-adjacent global pages into every compacted window, so the kernel would
silently read garbage from unrelated physical rows for most of the window.
Whole-page ownership sidesteps this: an owned page's ``page_size`` rows are
copied verbatim (already contiguous, since a physical page always is one),
so the compacted table trivially preserves "one window == one page".

This does not weaken the merge's exactness: the pigeonhole argument (a token
in the global top-k must also be in its owner's local top-k) only requires
that ownership partition tokens into disjoint groups, each scored in full --
it is indifferent to whether that partition is done per-token or per-page.

sglang's DCP ownership is physical-slot-valued (see
``memory_pool.py:set_kv_buffer``'s per-token analog for the main KV buffer),
not position-valued like vLLM's ``(pos // interleave) % world``. Physical slot
order is not guaranteed to track sequence position (radix-cache prefix reuse
can allocate a shared prefix's slots out of order relative to a request's own
suffix), so there is no closed-form local-length formula analogous to vLLM's
``get_dcp_local_seq_lens``. Instead this derives the local view from an
explicit, order-preserving compaction: an inclusive prefix count of the
ownership mask both gives the local causal length for a query at any position
and the destination slot for that position's compacted entry, so the
compacted local page table is written in original sequence order without any
cross-tile synchronization.
"""

from __future__ import annotations

import torch


def _owned_mask(
    page_table_1: torch.Tensor, dcp_size: int, dcp_rank: int, page_size: int
):
    valid = page_table_1 >= 0
    if dcp_size == 1:
        return valid
    page_id = torch.div(page_table_1.clamp(min=0), page_size, rounding_mode="floor")
    return valid & (page_id % dcp_size == dcp_rank)


def dcp_compact_store_loc(
    out_cache_loc: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    page_size: int,
    dummy_loc: int,
) -> torch.Tensor:
    """Compact a write address for the (now genuinely dcp_size-smaller) index
    K-cache, without any data-dependent shape (CUDA-graph-safe: decode's
    store runs inside the captured graph, and boolean-mask filtering /
    ``.all()`` host syncs are not permitted during capture).

    Must use the SAME page-granular ownership as ``dcp_localize_page_table``
    (see module docstring): every row is unconditionally mapped through
    ``_local_physical_slot`` (owned or not); rows this rank does NOT own get
    routed to ``dummy_loc`` (a scratch row reserved by the caller, e.g.
    ``pool.index_buf_size``) instead of being filtered out, so writing their
    (garbage, never-read) key/scale data there can't collide with or corrupt
    another row's rightful compacted address.
    """
    owned = _owned_mask(out_cache_loc, dcp_size, dcp_rank, page_size)
    local_loc = _local_physical_slot(out_cache_loc, dcp_size, page_size)
    return torch.where(owned, local_loc, torch.full_like(local_loc, dummy_loc))


def _local_physical_slot(page_table_1: torch.Tensor, dcp_size: int, page_size: int):
    """Global slot -> physical row in the compacted (dcp_size-smaller) index
    buffer, for an OWNED slot. Whole pages are compacted verbatim: page P's
    ``page_size`` rows land at local page ``P // dcp_size``, at the same
    within-page offset -- so 64 consecutive global rows of one owned page
    always become 64 consecutive local rows, unlike dividing the raw slot by
    dcp_size (which interleaves rows from dcp_size different global pages
    into every local window)."""
    page_id = torch.div(page_table_1.clamp(min=0), page_size, rounding_mode="floor")
    local_page_id = torch.div(page_id, dcp_size, rounding_mode="floor")
    within_page = page_table_1.clamp(min=0) % page_size
    return local_page_id * page_size + within_page


def dcp_localize_page_table(
    page_table_1: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    local_capacity: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build this DCP rank's local view of a global (page_size=1) page table.

    Args:
        page_table_1: ``[num_rows, max_len]`` global physical slot ids (one
            row per request or per CUDA-graph row); ``-1`` marks padding.
        dcp_size: DCP world size.
        dcp_rank: this rank's index in the DCP group.
        local_capacity: fixed output width -- ``dcp_local_capacity(max_len,
            dcp_size, page_size)`` is the tight (worst-case-owned-pages) bound;
            callers pass a fixed value across calls so the output shape
            doesn't depend on data, keeping this graph-capturable.
        page_size: the physical page size (e.g. 64) the paged-MQA-logits
            kernel addresses ``block_tables`` at. Ownership and the compacted
            layout are both page-aligned to this (see module docstring).

    Returns:
        local_page_table: ``[num_rows, local_capacity]``, this rank's owned
            pages compacted to the front in original position order (each
            page's ``page_size`` rows verbatim, at their new local physical
            address); ``-1`` past each row's local length.
        local_to_global: same shape/order, the original global slot id for
            each compacted entry (``-1`` past each row's local length) — the
            inverse mapping the pack step needs to recover global token ids
            for the merge, since compaction order here isn't arithmetic.
        local_causal_count: ``[num_rows, max_len]``, inclusive prefix count of
            owned entries up to and including each original position. For a
            query causally bounded at original position ``t`` (attends to
            ``[0, t]``), ``local_causal_count[row, t]`` is the correct local
            sequence length: since compaction preserves position order,
            ``local_page_table[row, :local_causal_count[row, t]]`` is exactly
            the set of local KV entries that query is allowed to see.
    """
    assert page_table_1.dim() == 2
    assert dcp_size >= 1
    assert 0 <= dcp_rank < dcp_size
    assert page_size >= 1

    if dcp_size == 1:
        local_causal_count = torch.cumsum(
            (page_table_1 >= 0).to(torch.int32), dim=1
        ).to(torch.int32)
        return page_table_1, page_table_1, local_causal_count

    owned = _owned_mask(page_table_1, dcp_size, dcp_rank, page_size)
    local_slot = _local_physical_slot(page_table_1, dcp_size, page_size)

    local_causal_count = torch.cumsum(owned.to(torch.int32), dim=1).to(torch.int32)

    # `local_causal_count - 1` is each owned entry's destination column
    # (0-indexed, strictly increasing along the row so no two entries collide
    # and no atomics are needed). Route non-owned entries to a scratch
    # overflow column instead of masking with fancy indexing, so this stays a
    # fixed-shape op (CUDA-graph-capturable regardless of how many tokens this
    # rank owns).
    overflow_col = local_capacity
    dest = torch.where(
        owned,
        (local_causal_count - 1).clamp(max=overflow_col),
        torch.full_like(local_causal_count, overflow_col),
    )

    padded_width = local_capacity + 1
    local_page_table = page_table_1.new_full((page_table_1.shape[0], padded_width), -1)
    local_to_global = page_table_1.new_full((page_table_1.shape[0], padded_width), -1)
    local_page_table.scatter_(1, dest, local_slot)
    local_to_global.scatter_(1, dest, page_table_1)

    return (
        local_page_table[:, :local_capacity],
        local_to_global[:, :local_capacity],
        local_causal_count,
    )


def dcp_local_capacity(max_len: int, dcp_size: int, page_size: int) -> int:
    """Fixed working-buffer width for the compacted local page table.

    This is NOT the persistent index K-cache pool's memory footprint (that
    stays correctly shrunk via ``index_buf_size`` regardless of this value) --
    it is a per-forward-step scratch array, and the kernel's actual compute
    cost is governed by the data-dependent local causal length
    (``seqlens_32``), not by this buffer's width. So there is no reason to
    assume ownership divides evenly across ranks by page count: with
    physically scattered (non-sequential) global pages -- the normal case
    once a request's blocks aren't all contiguous -- one rank can by chance
    own far more pages than ``num_pages / dcp_size``, up to and including
    every page. Size to that true worst case (round max_len up to a whole
    page) rather than gambling on a balanced split and silently truncating
    data when it isn't.
    """
    del dcp_size  # capacity must not depend on an assumed even split; see above
    return (max_len + page_size - 1) // page_size * page_size


def dcp_pack_local_to_global(
    page_table_1: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    row_causal_lens: torch.Tensor,
    row_offsets: torch.Tensor,
    total_size: int,
    page_size: int,
) -> torch.Tensor:
    """Flat-packed sibling of ``dcp_localize_page_table`` for the ragged/
    chunked-prefill K layout (``get_index_k_scale_buffer`` packs each
    request's local K into one flat ``[total_size]`` buffer at
    ``row_offsets[row]``, rather than a fixed-width per-row array like
    decode's page table). Produces the global-id side buffer for that same
    flat layout, so the pack step can recover global token ids from a ragged
    local-logits column position.

    Ownership must use the SAME page-granular definition as
    ``dcp_localize_page_table`` (see module docstring) -- this function's
    output is consumed alongside a ``local_block_tables`` built from that
    function's ``local_page_table``, and the two must agree on exactly which
    global tokens this rank owns. Unlike ``local_page_table``, the scattered
    value here is the raw global slot id (id lookup for the merge step, not a
    physical buffer address), so it does not need page-alignment itself.

    Args:
        page_table_1: ``[num_rows, max_len]`` global physical slot ids;
            ``-1`` marks padding.
        row_causal_lens: ``[num_rows]`` each row's own causal bound (only
            columns ``< row_causal_lens[row]`` are eligible -- packing must
            respect the row's own context length, not scan the padded width).
        row_offsets: ``[num_rows]`` each row's start offset in the flat
            output (an exclusive prefix sum of per-row local counts bounded
            by ``row_causal_lens``).
        total_size: total flat output width (sum of per-row local counts).
        page_size: must match the ``page_size`` passed to
            ``dcp_localize_page_table`` for the same forward step.

    Returns:
        packed_local_to_global: ``[total_size]``, the global slot id at each
            flat packed position; positions belonging to no row's owned set
            are unused (any row's own segment is fully covered by that row's
            owned count, by construction of ``row_offsets``).
    """
    assert page_table_1.dim() == 2
    assert dcp_size >= 1
    assert 0 <= dcp_rank < dcp_size

    col_ids = torch.arange(page_table_1.shape[1], device=page_table_1.device).unsqueeze(
        0
    )
    in_window = col_ids < row_causal_lens.unsqueeze(1)
    owned = _owned_mask(page_table_1, dcp_size, dcp_rank, page_size) & in_window
    local_causal_count = torch.cumsum(owned.to(torch.int32), dim=1).to(torch.int32)

    dest = row_offsets.unsqueeze(1) + local_causal_count - 1
    # Padded overflow slot: rows with zero owned tokens would otherwise
    # scatter at dest=-1 (row_offsets + 0 - 1); route those (and any other
    # masked-out lane, including anything outside the causal window) to a
    # discarded extra slot rather than relying on their value never being
    # read.
    dest = torch.where(owned, dest, torch.full_like(dest, total_size))

    packed = page_table_1.new_full((total_size + 1,), -1)
    packed.scatter_(0, dest.reshape(-1), page_table_1.reshape(-1))
    return packed[:total_size]
