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

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import torch


@dataclass
class CpTransientState:
    """Per-forward CP KV-resharding state attached to ``ForwardBatch``.

    Aggregates four pieces of bookkeeping needed by the CP-resharded
    forward path:

    1. ``owner_per_pages`` — per-request page-owner arrays mirrored from
       ``Req.cp_owner_per_page``. Populated by ``ForwardBatch.init_new``.
    2. Union view (``rows`` / ``req_indices`` / ``position_indices``) —
       every transient pool row allocated this forward. Consumed by the
       free policy in the post-forward epilogue.
    3. Prefix subset (``prefix_*``) — transient rows for non-owned prefix
       positions. Consumed by the per-layer prefix-fill stage.
    4. ``full_out_cache_loc`` — canonical-order union of owned-permanent
       and non-owned-transient row IDs for the current step's new tokens.
       Target of the per-layer extend-save ``set_kv_buffer``.
    """

    owner_per_pages: Optional[List[Optional[torch.Tensor]]] = None
    rows: Optional[torch.Tensor] = None
    req_indices: Optional[torch.Tensor] = None
    position_indices: Optional[torch.Tensor] = None
    prefix_rows: Optional[torch.Tensor] = None
    prefix_req_indices: Optional[torch.Tensor] = None
    prefix_position_indices: Optional[torch.Tensor] = None
    full_out_cache_loc: Optional[torch.Tensor] = None
    # Persistent (out_cache_loc) rows that were written into req_to_token at
    # non-owned EXTEND positions and then overwritten by transient rows. They
    # hold no live KV under reshard (the model writes to full_out_cache_loc =
    # owned + transient) and become orphaned; parallel to ``rows`` so the
    # post-transfer free reclaims them, leaving only this rank's local pages.
    displaced_rows: Optional[torch.Tensor] = None

    def has_transient_rows(self) -> bool:
        return self.rows is not None and self.rows.numel() > 0

    def has_prefix_subset(self) -> bool:
        return self.prefix_rows is not None and self.prefix_rows.numel() > 0

    def attach_allocation(self, allocation: "CpTransientState") -> None:
        self.rows = allocation.rows
        self.req_indices = allocation.req_indices
        self.position_indices = allocation.position_indices
        self.prefix_rows = allocation.prefix_rows
        self.prefix_req_indices = allocation.prefix_req_indices
        self.prefix_position_indices = allocation.prefix_position_indices
        self.full_out_cache_loc = allocation.full_out_cache_loc
        self.displaced_rows = allocation.displaced_rows

    def clear_transient(self) -> None:
        self.rows = None
        self.req_indices = None
        self.position_indices = None
        self.prefix_rows = None
        self.prefix_req_indices = None
        self.prefix_position_indices = None
        self.full_out_cache_loc = None
        self.displaced_rows = None


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


def cp_free_transient_for_request(
    state: Optional["CpTransientState"],
    req_pool_idx: int,
    allocator,
    req_to_token: torch.Tensor,
) -> int:
    """Free only the transient rows belonging to one request.

    Used by disagg-prefill to defer per-request freeing until that
    request's KV transfer reaches a terminal state (Success/Failed) —
    the rows must remain mapped in ``req_to_token`` while the transfer
    reads them. Returns the number of rows freed (0 if no work).

    Idempotent: rows already released are marked with ``req_indices = -1``
    and skipped on subsequent calls.
    """
    if state is None or not state.has_transient_rows():
        return 0
    mask = state.req_indices == req_pool_idx
    if not bool(mask.any()):
        return 0
    rows = state.rows[mask]
    positions = state.position_indices[mask]
    req_idxs = state.req_indices[mask]
    cp_free_transient_rows(req_to_token, allocator, rows, req_idxs, positions)
    # Reclaim the orphaned out_cache_loc rows displaced by the transient rows
    # at this request's non-owned extend positions. They hold no live KV and
    # are referenced by nothing after the transfer (the cached radix node was
    # sentinelized to slot-0 for non-owned pages at insert time), so freeing
    # them leaves only this rank's local pages resident. Slot-0 / matched-prefix
    # entries were zeroed at capture time.
    displaced = getattr(state, "displaced_rows", None)
    if displaced is not None:
        disp = displaced[mask]
        disp = disp[disp != 0]
        if disp.numel() > 0:
            allocator.free(disp)
    # Mark these slots so a second call (e.g. from leak-recovery on
    # scheduler exit) is a no-op.
    state.req_indices[mask] = -1
    return int(rows.numel())


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
    state = getattr(forward_batch, "cp_transient", None)
    if state is None or state.owner_per_pages is None:
        return None
    owners = state.owner_per_pages
    if all(o is None for o in owners):
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


def cp_alloc_extend_transient(
    *,
    req_to_token: torch.Tensor,
    allocator,
    cp_owner_per_pages: Sequence[Optional[torch.Tensor]],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    req_pool_indices: Sequence[int],
    cp_rank: int,
    page_size: int,
    tree_cache=None,
) -> Tuple[CpTransientState, List[int]]:
    """Pre-forward transient allocation for an extend batch (scheduler entry).
    For each CP-admitted request (``cp_owner_per_pages[s] is not None``),
    allocates transient pool rows for every non-owned position in
    ``[0, prefix_len + extend_len)``, then partitions the result into the
    prefix subset (positions ``< prefix_len``) and the extend subset.

    Only the prefix subset is stashed on the returned state — that
    subset is consumed by the per-layer prefix-fill stage to receive peer
    K/V via allgather. The extend subset is reachable implicitly through
    ``full_out_cache_loc``, which the extend save targets directly.

    Returns ``(CpTransientState, dropped)``. The returned state holds only
    the transient fields (``owner_per_pages`` stays ``None``); the consumer
    on ``ForwardBatch`` fills owners in separately via
    :meth:`CpTransientState.attach_allocation`. ``dropped`` lists batch
    indices whose allocation failed even after optional eviction via
    ``tree_cache``; the caller drops those requests and re-queues them.

    The function is a no-op (returns an empty state, ``[]``) when no
    request in the batch is CP-admitted or the batch has no extend tokens.
    """
    allocation = CpTransientState()
    if not cp_owner_per_pages or all(o is None for o in cp_owner_per_pages):
        return allocation, []

    device = req_to_token.device

    all_rows: List[torch.Tensor] = []
    all_req_idxs: List[torch.Tensor] = []
    all_positions: List[torch.Tensor] = []
    all_is_prefix: List[torch.Tensor] = []
    all_displaced: List[torch.Tensor] = []
    dropped: List[int] = []

    for s, owner in enumerate(cp_owner_per_pages):
        if owner is None:
            continue

        prefix_len = int(prefix_lens[s])
        extend_len = int(extend_lens[s])
        req_pool_idx = int(req_pool_indices[s])

        req_idxs, positions = compute_cp_non_owned_positions(
            cp_owner_per_pages=[owner],
            prefix_lens=[0],
            extend_lens=[prefix_len + extend_len],
            page_size=page_size,
            cp_rank=cp_rank,
            req_pool_indices=[req_pool_idx],
        )
        req_idxs = req_idxs.to(device=device)
        positions = positions.to(device=device)

        total = req_idxs.numel()
        if total == 0:
            continue

        rows = allocator.alloc(total)
        if rows is None and tree_cache is not None:
            deficit = total - allocator.available_size()
            if deficit > 0:
                from sglang.srt.mem_cache.base_prefix_cache import EvictParams

                tree_cache.evict(EvictParams(num_tokens=deficit))
            rows = allocator.alloc(total)
        if rows is None:
            dropped.append(s)
            continue

        # Capture the out_cache_loc rows currently mapped at these non-owned
        # positions BEFORE overwriting them with transient rows. Only the
        # EXTEND-range (positions >= prefix_len) rows are this request's fresh
        # out_cache_loc orphans; prefix-range positions point at shared/matched
        # tree rows or slot-0 sentinels, which must never be freed — zero those
        # so the post-transfer free skips them.
        displaced = req_to_token[req_idxs, positions].to(dtype=torch.int64).clone()
        displaced[positions < prefix_len] = 0

        req_to_token[req_idxs, positions] = rows.to(dtype=req_to_token.dtype)

        all_rows.append(rows)
        all_req_idxs.append(req_idxs)
        all_positions.append(positions)
        all_is_prefix.append(positions < prefix_len)
        all_displaced.append(displaced)

    if all_rows:
        rows_cat = torch.cat(all_rows)
        req_idxs_cat = torch.cat(all_req_idxs)
        positions_cat = torch.cat(all_positions)
        prefix_mask = torch.cat(all_is_prefix)

        allocation.rows = rows_cat
        allocation.req_indices = req_idxs_cat
        allocation.position_indices = positions_cat
        allocation.displaced_rows = torch.cat(all_displaced)

        if prefix_mask.any():
            allocation.prefix_rows = rows_cat[prefix_mask]
            allocation.prefix_req_indices = req_idxs_cat[prefix_mask]
            allocation.prefix_position_indices = positions_cat[prefix_mask]

    return allocation, dropped


def cp_build_full_out_cache_loc_for_extend(
    req_to_token: torch.Tensor,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    req_pool_indices: Sequence[int],
) -> Optional[torch.Tensor]:
    """Build the union of owned-permanent + non-owned-transient row IDs
    for the current step's new tokens, in canonical batch order.

    Must be called *after* :func:`cp_alloc_extend_transient` has scattered
    transient rows into ``req_to_token``. Reads
    ``req_to_token[req_pool_idx, prefix_len : prefix_len + extend_len]``
    for each request and concatenates.

    Returns ``None`` when the batch has no extend tokens (decode/idle).
    """
    chunks: List[torch.Tensor] = []
    for s in range(len(extend_lens)):
        extend = int(extend_lens[s])
        if extend <= 0:
            continue
        prefix = int(prefix_lens[s])
        chunks.append(req_to_token[int(req_pool_indices[s]), prefix : prefix + extend])

    if not chunks:
        return None
    return torch.cat(chunks)


def cp_alloc_forward_transient(
    forward_batch,
    allocator,
    cp_rank: int,
    page_size: int,
    tree_cache=None,
) -> List[int]:
    """ForwardBatch-facing wrapper kept for unit tests. Delegates to
    :func:`cp_alloc_extend_transient` and merges the result into the
    existing ``forward_batch.cp_transient`` state (which the caller is
    expected to have populated with ``owner_per_pages``)."""
    ranges = _resolve_per_request_ranges(forward_batch)
    if ranges is None:
        return []
    state: CpTransientState = forward_batch.cp_transient
    prefix_lens, extend_lens, req_pool_indices = ranges
    allocation, dropped = cp_alloc_extend_transient(
        req_to_token=forward_batch.req_to_token_pool.req_to_token,
        allocator=allocator,
        cp_owner_per_pages=state.owner_per_pages,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        req_pool_indices=req_pool_indices,
        cp_rank=cp_rank,
        page_size=page_size,
        tree_cache=tree_cache,
    )
    state.attach_allocation(allocation)
    return dropped


def cp_build_full_out_cache_loc(forward_batch) -> Optional[torch.Tensor]:
    """ForwardBatch-facing wrapper kept for unit tests."""
    if forward_batch.extend_seq_lens_cpu is None:
        return None
    extend_lens = list(forward_batch.extend_seq_lens_cpu)
    prefix_lens = list(forward_batch.extend_prefix_lens_cpu or [0] * len(extend_lens))
    return cp_build_full_out_cache_loc_for_extend(
        req_to_token=forward_batch.req_to_token_pool.req_to_token,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        req_pool_indices=forward_batch.req_pool_indices.tolist(),
    )


# --- Free policy + manager ---------------------------------------------------


@dataclass
class CpTransientFreeStats:
    """Counters returned by a single :meth:`CpTransientFreePolicy.free` call.

    Aggregated by :class:`CpTransientManager` for observability.
    """

    freed_rows: int = 0
    retained_rows: int = 0


class CpTransientFreePolicy(Protocol):
    """Strategy for releasing per-forward transient rows.

    Implementations decide whether to return every row to the allocator
    (v1 default) or to keep some rows around for the next forward.
    Future policies can consult the manager's hit-rate counters to make
    that decision.
    """

    def free(
        self,
        state: CpTransientState,
        allocator,
        req_to_token: torch.Tensor,
    ) -> CpTransientFreeStats: ...


class FreeAllPolicy:
    """Default v1 policy: free every transient row at the end of each
    forward. Matches DESIGN_kv_reshard.md §6's lifecycle ("transient rows
    are freed back to the allocator and req_to_token at those positions
    is rewritten to slot-0 sentinel")."""

    def free(
        self,
        state: CpTransientState,
        allocator,
        req_to_token: torch.Tensor,
    ) -> CpTransientFreeStats:
        if not state.has_transient_rows():
            state.clear_transient()
            return CpTransientFreeStats()
        freed = int(state.rows.numel())
        cp_free_transient_rows(
            req_to_token,
            allocator,
            state.rows,
            state.req_indices,
            state.position_indices,
        )
        state.clear_transient()
        return CpTransientFreeStats(freed_rows=freed)


class CpTransientManager:
    """Per-runtime owner of the CP-resharding transient free policy and
    accumulated KV-cache-hit statistics.

    Lives on ``ModelRunner`` so hit-rate counters survive across forwards.
    The free policy is pluggable: :class:`FreeAllPolicy` (default)
    reproduces v1's free-everything behavior; future policies can use the
    recorded hit rate to decide whether retaining transient rows for the
    next forward is worthwhile.
    """

    def __init__(self, policy: Optional[CpTransientFreePolicy] = None) -> None:
        self.policy: CpTransientFreePolicy = policy or FreeAllPolicy()
        self._hit_count: int = 0
        self._lookup_count: int = 0
        self._freed_rows_total: int = 0
        self._retained_rows_total: int = 0

    @property
    def hit_rate(self) -> float:
        if self._lookup_count == 0:
            return 0.0
        return self._hit_count / self._lookup_count

    def record_lookups(self, hits: int, total: int) -> None:
        self._hit_count += hits
        self._lookup_count += total

    def stats_snapshot(self) -> dict:
        return {
            "freed_rows_total": self._freed_rows_total,
            "retained_rows_total": self._retained_rows_total,
            "hit_rate": self.hit_rate,
            "lookup_count": self._lookup_count,
        }

    def free(self, forward_batch, allocator) -> None:
        state: Optional[CpTransientState] = getattr(forward_batch, "cp_transient", None)
        if state is None or not state.has_transient_rows():
            return
        stats = self.policy.free(
            state, allocator, forward_batch.req_to_token_pool.req_to_token
        )
        self._freed_rows_total += stats.freed_rows
        self._retained_rows_total += stats.retained_rows

    def free_for_request(
        self,
        state: Optional[CpTransientState],
        req_pool_idx: int,
        allocator,
        req_to_token: torch.Tensor,
    ) -> int:
        """Per-request free for the disagg-prefill deferred-release path.

        Returns the number of rows freed. Idempotent (safe to call
        multiple times for the same req_pool_idx).
        """
        n = cp_free_transient_for_request(state, req_pool_idx, allocator, req_to_token)
        self._freed_rows_total += n
        return n


def cp_free_forward_transient(forward_batch, allocator) -> None:
    """Thin convenience shim around ``FreeAllPolicy`` for unit tests.
    Production code goes through :class:`CpTransientManager` on
    ``ModelRunner``."""
    state: Optional[CpTransientState] = getattr(forward_batch, "cp_transient", None)
    if state is None or not state.has_transient_rows():
        if state is not None:
            state.clear_transient()
        return
    FreeAllPolicy().free(state, allocator, forward_batch.req_to_token_pool.req_to_token)
