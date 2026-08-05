"""Verifier-side rid -> seat map for decoupled enumeration spec (torch-free).

Tells the recv daemon which GPU seat (req_pool_idx) an incoming block's request
holds; the GPU landing itself lives in decoupled_enum_buffer. A request's
req_pool_idx is assigned once at prefill and stays put until it frees / retracts,
so this map is written rarely (per request lifetime), not per round.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.speculative.decoupled_spec_io import DraftEnumerationBufferBatch


class DecoupledSlotTable:
    """Thread-safe rid -> pool_idx map. The scheduler writes (assign / remove /
    clear at prefill / finish / flush); the recv daemon reads (lookup per block).
    One lock, held only for the dict access, so the schedule loop never stalls on
    the daemon.

    remove() on every finish / retract path is the primary defense against a late
    block landing in a reused seat -- it drops the binding before the block can
    reach the seat. This includes streaming-session turn boundaries, which recycle
    a seat without an alloc/free. TODO(phase 5b): wire remove() on all such paths.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rid_to_pool_idx: dict[str, int] = {}

    def assign(self, rid: str, pool_idx: int) -> None:
        # Called at prefill and at re-open after a retraction; last write wins.
        with self._lock:
            self._rid_to_pool_idx[rid] = int(pool_idx)

    def remove(self, rid: str) -> None:
        # Called at close / retract; idempotent.
        with self._lock:
            self._rid_to_pool_idx.pop(rid, None)

    def clear(self) -> None:
        # Called wherever ReqToTokenPool.clear() runs (cache flush).
        with self._lock:
            self._rid_to_pool_idx.clear()

    def lookup(self, rid: str) -> Optional[int]:
        with self._lock:
            return self._rid_to_pool_idx.get(rid)

    def lookup_many(self, rids: list[str]) -> list[Optional[int]]:
        # One lock acquisition to route a whole block; positionally aligned with
        # rids, None where a rid is not bound.
        with self._lock:
            return [self._rid_to_pool_idx.get(rid) for rid in rids]

    def __len__(self) -> int:
        with self._lock:
            return len(self._rid_to_pool_idx)


class PlannedWrite(msgspec.Struct, frozen=True, gc=False):
    """One block row routed to a live seat, ready for the GPU scatter."""

    row_index: int  # the row's index within the block
    pool_idx: int  # seat to scatter the row into
    base_committed_len: int  # committed length the row was drafted from


class LandingPlan(msgspec.Struct, frozen=True, gc=False):
    writes: list[PlannedWrite]  # rows with a live seat
    dropped_rids: list[str]  # rows whose request has already left


def plan_landing(
    block: DraftEnumerationBufferBatch,
    slot_table: DecoupledSlotTable,
    *,
    verifier_rank: int,
) -> LandingPlan:
    """Route a block to seats (host-only, no GPU).

    Rows whose rid is no longer bound (the request finished / retracted before its
    block landed) are dropped so their late data never reaches a reused seat;
    survivors are scattered by DecoupledEnumBuffer.land. Rejects a block routed to
    another verifier: a request_id is only unique within its owning verifier (see
    DraftReqKey), so a misrouted / M:N block whose rid aliases a live local rid
    would otherwise land in the local seat.

    NOTE: the raises run on the recv daemon thread, whose loop dies on an uncaught
    exception (same as VerifierCommitSegment.append_message). TODO(phase 5c):
    quarantine the offending request instead of crashing the thread.
    """
    if int(block.dst_verifier_rank) != int(verifier_rank):
        raise RuntimeError(
            "enumeration block routed to the wrong verifier: "
            f"dst_verifier_rank={block.dst_verifier_rank} "
            f"verifier_rank={verifier_rank} "
            f"src_drafter_rank={block.src_drafter_rank} "
            f"batch_size={block.batch_size}"
        )
    block.validate()  # also rejects duplicate rids
    # One lock acquisition for the whole block. This rid -> seat step is CPU-side
    # (rid is a string; pool_idx is scheduler-owned) and can be removed later by
    # having the drafter echo the pool_idx it was told at DraftSync, at the cost
    # of a retraction re-sync + the base_committed_len staleness guard.
    pool_indices = slot_table.lookup_many(block.rids)
    writes: list[PlannedWrite] = []
    dropped: list[str] = []
    for i, (rid, pool_idx) in enumerate(zip(block.rids, pool_indices)):
        if pool_idx is None:
            dropped.append(rid)
            continue
        writes.append(
            PlannedWrite(
                row_index=i,
                pool_idx=pool_idx,
                base_committed_len=int(block.base_committed_lens[i]),
            )
        )
    return LandingPlan(writes=writes, dropped_rids=dropped)
