"""Verifier-side ``rid -> seat`` bookkeeping for decoupled enumeration spec.

This module is the pure-host (torch-free) half of the decoupled enumeration data
plane. It answers one question for the recv daemon: *which GPU seat does an
incoming enumeration block's request currently occupy?* The GPU landing itself
lives in :mod:`decoupled_enum_buffer` (which needs torch); keeping the routing
decision here makes it unit-testable without a GPU.

Background:

* On the verifier, a request's ``req_pool_idx`` (its ``ReqToTokenPool`` seat) is
  assigned once at prefill and is stable across every decode round -- it only
  changes on retraction / re-admission (see ``ReqToTokenPool.alloc`` /
  ``.free``). So the ``rid -> seat`` binding is written rarely (per request
  lifetime), not per round.
* Seats are recycled: when a request finishes / is retracted, its seat is freed
  and later handed to a *different* request, and ``ReqToTokenPool.req_generation``
  is bumped for that seat. The captured ``generation`` therefore lets a consumer
  tell a stale binding (previous occupant) from the live one.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.speculative.decoupled_spec_io import DraftEnumerationBufferBatch


class SlotBinding(msgspec.Struct, frozen=True, gc=False):
    """A request's current seat plus the generation captured when it was bound.

    ``(pool_idx, generation)`` identifies one occupancy episode of a
    ``ReqToTokenPool`` seat: if the seat is freed and reassigned to another
    request, ``req_generation[pool_idx]`` is bumped, so this binding's
    ``generation`` no longer matches the seat's live generation and any data
    landed under it is rejected downstream.
    """

    pool_idx: int
    generation: int


class DecoupledSlotTable:
    """Thread-safe ``rid -> SlotBinding`` map shared between scheduler and daemon.

    Writer: the scheduler main loop -- :meth:`assign` at prefill / re-open,
    :meth:`remove` at close / retract. Both are per-request-lifetime events, so
    the table is *not* touched on the hot per-round schedule path.

    Reader: the decoupled recv daemon -- :meth:`lookup` once per incoming
    enumeration block, to find the seat to scatter into.

    A single lock suffices: writes are rare, and a reader holds the lock only for
    the dict access (the daemon's GPU write happens outside the lock), so the
    scheduler's overlap loop is never stalled by the daemon.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rid_to_slot: dict[str, SlotBinding] = {}

    def assign(self, rid: str, pool_idx: int, generation: int) -> None:
        """Bind ``rid`` to a seat.

        Called at the request's prefill and again at re-open after a retraction
        (with the new seat / generation). Last write wins, so a re-open cleanly
        overwrites a stale binding.
        """
        binding = SlotBinding(pool_idx=int(pool_idx), generation=int(generation))
        with self._lock:
            self._rid_to_slot[rid] = binding

    def remove(self, rid: str) -> None:
        """Drop ``rid`` at close / retract.

        A block that arrives after removal finds no binding and is dropped by
        :func:`plan_landing` -- its seat may already belong to a new request.
        Idempotent: removing an absent rid is a no-op.
        """
        with self._lock:
            self._rid_to_slot.pop(rid, None)

    def lookup(self, rid: str) -> Optional[SlotBinding]:
        """Return the live binding for ``rid``, or ``None`` if it is not bound."""
        with self._lock:
            return self._rid_to_slot.get(rid)

    def __len__(self) -> int:
        with self._lock:
            return len(self._rid_to_slot)


class PlannedWrite(msgspec.Struct, frozen=True, gc=False):
    """One enumeration-block row routed to a live seat, ready for the GPU scatter."""

    row_index: int  # index of the row within the block (0 .. batch_size-1)
    pool_idx: int  # seat to scatter the row into
    generation: int  # occupancy stamp to write beside the row
    base_committed_len: (
        int  # freshness stamp (committed length the row was drafted from)
    )


class LandingPlan(msgspec.Struct, frozen=True, gc=False):
    """Result of routing an enumeration block against the live slot table."""

    writes: list[PlannedWrite]  # rows that found a live seat
    dropped_rids: list[str]  # rows whose rid is absent (finished / retracted)


def plan_landing(
    block: DraftEnumerationBufferBatch,
    slot_table: DecoupledSlotTable,
) -> LandingPlan:
    """Route an incoming enumeration block to seats -- pure host, no GPU.

    For each row, resolve its rid to a live seat. Rows whose rid is absent (the
    request finished or was retracted before its block landed) are dropped, so
    their late data never touches a seat that may have been reused by a different
    request. The GPU scatter of the surviving rows is performed by
    :meth:`decoupled_enum_buffer.DecoupledEnumBuffer.land`.
    """
    writes: list[PlannedWrite] = []
    dropped: list[str] = []
    for i in range(block.batch_size):
        rid = block.rids[i]
        binding = slot_table.lookup(rid)
        if binding is None:
            dropped.append(rid)
            continue
        writes.append(
            PlannedWrite(
                row_index=i,
                pool_idx=binding.pool_idx,
                generation=binding.generation,
                base_committed_len=int(block.base_committed_lens[i]),
            )
        )
    return LandingPlan(writes=writes, dropped_rids=dropped)
