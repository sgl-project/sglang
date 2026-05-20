"""Per-forward transient KV-cache row management for CP KV-resharding
(DESIGN_kv_reshard.md §6, dynamic variant).

Under CP-resharding each rank stores permanent rows only for the pages it
owns. Positions owned by other ranks would normally be sentinel slots in
``req_to_token``; for paged FlashAttention to read through the same page
table, those positions need *real* pool rows during the forward call. We
allocate those rows from the regular ``TokenToKVPoolAllocator`` at
``init_forward_metadata`` and free them in the post-forward epilogue, so
the pool is shared between persistent and transient KV without a
reserved scratch region.

This module contains only pure-tensor helpers; the integration points
(``init_forward_metadata``, model-runner epilogue, per-layer K/V writes)
live in Chunk B.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def compute_cp_non_owned_positions(
    cp_owner_per_pages: Sequence[torch.Tensor],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    page_size: int,
    cp_rank: int,
    req_pool_indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identify every (req_pool_idx, position) slot in the batch that is
    *not* owned by ``cp_rank`` and therefore needs a transient row.

    For each request ``s``, scans positions ``[prefix_lens[s],
    prefix_lens[s] + extend_lens[s])`` and emits the ones whose owning
    page (``cp_owner_per_pages[s][position // page_size]``) is some
    rank other than ``cp_rank``.

    Returns two parallel ``int64`` tensors of the same length:
    ``req_pool_idxs_flat`` (each entry's row in ``req_to_token``) and
    ``position_idxs`` (each entry's column). Same order across both, so
    they can be used together as a scatter target.
    """
    _validate_lengths(cp_owner_per_pages, prefix_lens, extend_lens, req_pool_indices)

    req_idx_chunks: List[torch.Tensor] = []
    pos_chunks: List[torch.Tensor] = []
    for s in range(len(prefix_lens)):
        owner = cp_owner_per_pages[s]
        prefix = int(prefix_lens[s])
        extend = int(extend_lens[s])
        if extend <= 0:
            continue

        positions = torch.arange(prefix, prefix + extend, dtype=torch.int64)
        page_ids = positions // page_size
        if page_ids.numel() > 0 and int(page_ids.max()) >= owner.numel():
            raise IndexError(
                f"request {s}: position range "
                f"[{prefix}, {prefix + extend}) needs page "
                f"{int(page_ids.max())} but cp_owner_per_page covers "
                f"only {owner.numel()} pages"
            )

        owners = owner[page_ids].to(torch.int64)
        non_owned_mask = owners != cp_rank
        if not non_owned_mask.any():
            continue

        non_owned_positions = positions[non_owned_mask]
        req_idx = int(req_pool_indices[s])
        req_idx_chunks.append(
            torch.full_like(non_owned_positions, req_idx, dtype=torch.int64)
        )
        pos_chunks.append(non_owned_positions)

    if not req_idx_chunks:
        empty = torch.empty((0,), dtype=torch.int64)
        return empty, empty.clone()

    return torch.cat(req_idx_chunks), torch.cat(pos_chunks)


def cp_alloc_transient_rows(
    req_to_token: torch.Tensor,
    allocator,
    req_pool_idxs_flat: torch.Tensor,
    position_idxs: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Allocate ``len(req_pool_idxs_flat)`` rows from ``allocator`` and
    scatter them into ``req_to_token`` at the given coordinates.

    Returns the allocated row indices (``int64`` tensor) on success, or
    ``None`` if the allocator cannot satisfy the request. On ``None``,
    ``req_to_token`` is not modified, so the caller can evict + retry
    without first undoing a partial scatter.

    When the request is empty (no non-owned positions), returns an
    empty tensor on ``req_to_token``'s device without touching the
    allocator.
    """
    _validate_pair_lengths(req_pool_idxs_flat, position_idxs)
    total = req_pool_idxs_flat.numel()
    if total == 0:
        return torch.empty((0,), dtype=torch.int64, device=req_to_token.device)

    transient_rows = allocator.alloc(total)
    if transient_rows is None:
        return None

    # Allocator and req_to_token may live on the same device; coerce types
    # without forcing a host round-trip.
    req_to_token[req_pool_idxs_flat, position_idxs] = transient_rows.to(
        dtype=req_to_token.dtype
    )
    return transient_rows


def cp_free_transient_rows(
    req_to_token: torch.Tensor,
    allocator,
    transient_rows: Optional[torch.Tensor],
    req_pool_idxs_flat: torch.Tensor,
    position_idxs: torch.Tensor,
) -> None:
    """Reverse ``cp_alloc_transient_rows``: free the rows back to the
    allocator and rewrite the corresponding ``req_to_token`` slots to
    slot-0 sentinel so subsequent ``cache_*_req`` paths skip them.

    A no-op when ``transient_rows`` is ``None`` or empty (which happens
    when the batch had no non-owned positions).
    """
    _validate_pair_lengths(req_pool_idxs_flat, position_idxs)
    if transient_rows is None or transient_rows.numel() == 0:
        return
    if transient_rows.numel() != req_pool_idxs_flat.numel():
        raise ValueError(
            f"transient_rows length {transient_rows.numel()} does not "
            f"match scatter target length {req_pool_idxs_flat.numel()}"
        )
    allocator.free(transient_rows)
    req_to_token[req_pool_idxs_flat, position_idxs] = 0


def _validate_lengths(
    cp_owner_per_pages: Sequence[torch.Tensor],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    req_pool_indices: Sequence[int],
) -> None:
    n = len(prefix_lens)
    for name, seq in (
        ("cp_owner_per_pages", cp_owner_per_pages),
        ("extend_lens", extend_lens),
        ("req_pool_indices", req_pool_indices),
    ):
        if len(seq) != n:
            raise ValueError(
                f"length mismatch: prefix_lens has {n} entries but "
                f"{name} has {len(seq)}"
            )


def _validate_pair_lengths(
    req_pool_idxs_flat: torch.Tensor, position_idxs: torch.Tensor
) -> None:
    if req_pool_idxs_flat.shape != position_idxs.shape:
        raise ValueError(
            f"req_pool_idxs_flat shape {tuple(req_pool_idxs_flat.shape)} "
            f"must match position_idxs shape {tuple(position_idxs.shape)}"
        )


# --- ForwardBatch-aware wrappers (Chunk B) -----------------------------------


def _resolve_per_request_ranges(
    forward_batch,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """Extract per-request ``(prefix_len, extend_len, req_pool_idx)`` tuples
    for the requests in ``forward_batch`` that are admitted under CP-resharding
    (i.e., have a non-``None`` ``cp_owner_per_page``).

    Returns ``None`` when no request in the batch is CP-reshard-admitted, so
    the caller can short-circuit.
    """
    owners = forward_batch.cp_owner_per_pages
    if owners is None or all(o is None for o in owners):
        return None

    # ForwardBatch carries:
    #   - extend_prefix_lens_cpu (List[int]) — prefix length per request
    #   - extend_seq_lens_cpu    (List[int]) — newly-processed token count
    #   - req_pool_indices       (torch.Tensor[bs]) — row in req_to_token
    # Decode and idle modes don't populate the extend_* fields; in that case
    # there are no new positions to write transient rows for in this forward.
    if forward_batch.extend_seq_lens_cpu is None:
        return None

    prefix_lens = list(forward_batch.extend_prefix_lens_cpu or [0] * len(owners))
    extend_lens = list(forward_batch.extend_seq_lens_cpu)
    req_pool_indices = forward_batch.req_pool_indices.tolist()
    return prefix_lens, extend_lens, req_pool_indices


def cp_alloc_req_transient(
    req_to_token: torch.Tensor,
    allocator,
    cp_owner_per_page: torch.Tensor,
    prefix_len: int,
    extend_len: int,
    req_pool_idx: int,
    cp_rank: int,
    page_size: int,
    tree_cache=None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Per-request transient allocation (Chunk B/C).

    Computes this request's non-owned positions for the forward range,
    allocates transient pool rows, scatters them into ``req_to_token``.
    On allocator OOM, evicts from ``tree_cache`` (if provided) and
    retries once.

    Returns ``(rows, req_idxs_flat, positions)``:

    - ``rows`` is the allocated row indices on the same device as
      ``req_to_token``, or ``None`` if alloc failed even after eviction
      (caller should drop this request from the batch).
    - ``req_idxs_flat`` / ``positions`` are the scatter coordinates --
      empty when the request had no non-owned positions, in which case
      ``rows`` is an empty tensor on the right device (not ``None``).

    Designed to be called in a loop over the batch's CP-admitted requests
    so a single starved request doesn't sink the whole batch.
    """
    req_idxs, positions = compute_cp_non_owned_positions(
        cp_owner_per_pages=[cp_owner_per_page],
        prefix_lens=[prefix_len],
        extend_lens=[extend_len],
        page_size=page_size,
        cp_rank=cp_rank,
        req_pool_indices=[req_pool_idx],
    )
    req_idxs = req_idxs.to(device=req_to_token.device)
    positions = positions.to(device=req_to_token.device)

    if req_idxs.numel() == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=req_to_token.device)
        return empty, req_idxs, positions

    rows = cp_alloc_transient_rows(req_to_token, allocator, req_idxs, positions)
    if rows is None and tree_cache is not None:
        # Evict just enough to cover this request's deficit and retry once.
        needed = req_idxs.numel()
        deficit = needed - allocator.available_size()
        if deficit > 0:
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            tree_cache.evict(EvictParams(num_tokens=deficit))
        rows = cp_alloc_transient_rows(req_to_token, allocator, req_idxs, positions)

    return rows, req_idxs, positions


def cp_alloc_forward_transient(
    forward_batch,
    allocator,
    cp_rank: int,
    page_size: int,
    tree_cache=None,
) -> List[int]:
    """Pre-forward transient allocation across all CP-admitted requests in
    ``forward_batch``. Calls :func:`cp_alloc_req_transient` once per
    request so a single starved request doesn't drag the whole batch.

    Successfully-allocated rows / scatter coordinates from every request
    are concatenated and stashed on ``forward_batch.cp_transient_*``.

    Returns: list of batch indices ``s`` whose allocation failed (even
    after optional eviction via ``tree_cache``). The caller drops those
    requests from the forward and re-queues them.

    No-ops:
      - No request in the batch is CP-admitted (``cp_owner_per_pages``
        is ``None`` or all-``None``) -> returns ``[]``, leaves
        ``cp_transient_*`` as ``None``.
      - Decode/idle mode (no extend range) -> returns ``[]``.
    """
    ranges = _resolve_per_request_ranges(forward_batch)
    if ranges is None:
        return []
    prefix_lens, extend_lens, req_pool_indices = ranges
    owners = forward_batch.cp_owner_per_pages

    req_to_token = forward_batch.req_to_token_pool.req_to_token

    all_rows: List[torch.Tensor] = []
    all_req_idxs: List[torch.Tensor] = []
    all_positions: List[torch.Tensor] = []
    dropped: List[int] = []

    for s, owner in enumerate(owners):
        if owner is None:
            continue

        rows, req_idxs, positions = cp_alloc_req_transient(
            req_to_token=req_to_token,
            allocator=allocator,
            cp_owner_per_page=owner,
            prefix_len=prefix_lens[s],
            extend_len=extend_lens[s],
            req_pool_idx=req_pool_indices[s],
            cp_rank=cp_rank,
            page_size=page_size,
            tree_cache=tree_cache,
        )
        if rows is None:
            dropped.append(s)
            continue
        if rows.numel() > 0:
            all_rows.append(rows)
            all_req_idxs.append(req_idxs)
            all_positions.append(positions)

    if all_rows:
        forward_batch.cp_transient_rows = torch.cat(all_rows)
        forward_batch.cp_transient_req_indices = torch.cat(all_req_idxs)
        forward_batch.cp_transient_position_indices = torch.cat(all_positions)

    return dropped


def cp_free_forward_transient(forward_batch, allocator) -> None:
    """Reverse of ``cp_alloc_forward_transient``. Safe to call when no
    transient rows were allocated (no-op)."""
    if forward_batch.cp_transient_rows is None:
        return
    cp_free_transient_rows(
        forward_batch.req_to_token_pool.req_to_token,
        allocator,
        forward_batch.cp_transient_rows,
        forward_batch.cp_transient_req_indices,
        forward_batch.cp_transient_position_indices,
    )
    forward_batch.cp_transient_rows = None
    forward_batch.cp_transient_req_indices = None
    forward_batch.cp_transient_position_indices = None
