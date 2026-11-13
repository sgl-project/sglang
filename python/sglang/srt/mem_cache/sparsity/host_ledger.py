"""One owner of a request's HiSparse host slots.

A host slot passes through three owners: the request that allocated it, the L2
radix tree once an insert claims it, and -- for a matched prefix -- the request
that borrowed it. Keeping all three in one object keyed by ``req_pool_idx`` is
what makes release computable in a single place, and what makes a leak or a
double free visible (``verify``) instead of silent.

The pool still owns the page arithmetic (``alloc_paged_token_slots``); the
ledger owns the accounting, and is the only writer of the allocated length.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class HostSlotRecord:
    """Where one request's host slots are.

    borrowed_len    [0, borrowed_len) came from a matched radix prefix and is
                    owned by that node's refcount, not by this request.
    owned_len       highest end handed to the tree by an insert.
    discarded       insert holes inside [borrowed_len, owned_len): allocated by
                    this request, skipped by an insert, so the tree never
                    reclaims them and release must.
    nodes           host locks this request holds (the borrow and every publish).
    """

    borrowed_len: int = 0
    owned_len: int = 0
    discarded: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    nodes: list[tuple[Any, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class HostReleasePlan:
    """What releasing one request still owes: ranges to free and locks to drop."""

    free: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    nodes: list[tuple[Any, Any]] = dataclasses.field(default_factory=list)


class HiSparseHostLedger:
    """Every host slot and host lock a request holds, keyed by req_pool_idx.

    Keying by slot (not by rid) is deliberate: a retracted request that is
    re-bootstrapped keeps its rid, and must not inherit the previous
    incarnation's slots or tree locks.
    """

    def __init__(
        self,
        mem_pool_host: HiSparseDSATokenToKVPool,
        req_to_host_pool: torch.Tensor,
        allocated_len: torch.Tensor,
    ):
        self._pool = mem_pool_host
        self._table = req_to_host_pool
        self._allocated = allocated_len
        self._records: dict[int, HostSlotRecord] = {}

    # ---------------------------------------------------------------- queries
    def _record(self, req_pool_idx: int) -> HostSlotRecord:
        rec = self._records.get(req_pool_idx)
        if rec is None:
            rec = HostSlotRecord()
            self._records[req_pool_idx] = rec
        return rec

    def allocated(self, req_pool_idx: int) -> int:
        return int(self._allocated[req_pool_idx])

    def borrowed_len(self, req_pool_idx: int) -> int:
        rec = self._records.get(req_pool_idx)
        return 0 if rec is None else rec.borrowed_len

    def borrowed_node(self, req_pool_idx: int) -> Optional[Any]:
        """The node this request borrowed a prefix from, if any."""
        rec = self._records.get(req_pool_idx)
        if rec is None or not rec.nodes:
            return None
        return rec.nodes[0][0]

    def tracks(self, req_pool_idx: int) -> bool:
        return req_pool_idx in self._records

    # ------------------------------------------------------------------ moves
    def alloc(self, req_pool_idx: int, start: int, num_tokens: int) -> torch.Tensor:
        """Allocate host slots for [start, start + num_tokens) and record them."""
        locs = self._pool.alloc_paged_token_slots(
            self._table, self._allocated, req_pool_idx, start, num_tokens
        )
        self._record(req_pool_idx)
        return locs

    def adopt(self, req_pool_idx: int, prefix_len: int, indices: torch.Tensor) -> None:
        """Point this request's host table at a borrowed prefix's slots."""
        self._table[req_pool_idx, :prefix_len] = indices[:prefix_len].to(
            self._table.device, non_blocking=True
        )
        # Never shrink what the request already holds: adopt runs before its own
        # allocation in production, but a caller may adopt on a slot it owns.
        self._allocated[req_pool_idx] = max(
            self.allocated(req_pool_idx), int(prefix_len)
        )
        self._record(req_pool_idx).borrowed_len = prefix_len

    def hand_to_tree(self, req_pool_idx: int, matched_len: int, key_len: int) -> None:
        """Record that an insert claimed slots for the tree.

        Slots between what the request already handed over (or borrowed) and the
        insert's match point were skipped: the tree will not reclaim them, so
        they become holes release must free.
        """
        rec = self._record(req_pool_idx)
        start = max(rec.owned_len, rec.borrowed_len)
        end = max(start, int(matched_len))
        if end > start:
            rec.discarded.append((start, end))
        rec.owned_len = max(rec.owned_len, int(key_len))

    def borrow_node(self, req_pool_idx: int, node_id: Any, params: Any) -> None:
        self._record(req_pool_idx).nodes.append((node_id, params))

    def pop_nodes(self, req_pool_idx: int) -> list[tuple[Any, Any]]:
        """Take this request's host locks without touching its slots."""
        rec = self._records.get(req_pool_idx)
        if rec is None or not rec.nodes:
            return []
        nodes, rec.nodes = rec.nodes, []
        return nodes

    # ---------------------------------------------------------------- release
    def release(self, req_pool_idx: int) -> HostReleasePlan:
        """Close out this request: the ranges to free and the locks to drop.

        Frees the insert holes and the private tail; the borrowed prefix and the
        tree-claimed band stay with their owners (the node's refcount).
        """
        rec = self._records.pop(req_pool_idx, None) or HostSlotRecord()
        allocated = self.allocated(req_pool_idx)
        free: list[tuple[int, int]] = []
        for start, end in rec.discarded:
            end = min(end, allocated)
            if end > start:
                free.append((start, end))
        tail = max(rec.borrowed_len, rec.owned_len)
        if allocated > tail:
            free.append((tail, allocated))
        self._allocated[req_pool_idx] = 0
        return HostReleasePlan(free=free, nodes=rec.nodes)

    # ------------------------------------------------------------ diagnostics
    def verify(self) -> list[str]:
        """Invariant check: violations as strings, empty when healthy.

        ``allocated`` must cover every band a record names, holes must sit
        inside the allocated range, and no record may outlive its slots.
        """
        bad: list[str] = []
        for req_pool_idx, rec in self._records.items():
            allocated = self.allocated(req_pool_idx)
            if rec.borrowed_len > allocated:
                bad.append(
                    f"idx={req_pool_idx}: borrowed {rec.borrowed_len} > allocated {allocated}"
                )
            if rec.owned_len > allocated:
                bad.append(
                    f"idx={req_pool_idx}: owned {rec.owned_len} > allocated {allocated}"
                )
            for start, end in rec.discarded:
                if not (0 <= start < end <= allocated):
                    bad.append(
                        f"idx={req_pool_idx}: hole [{start},{end}) outside [0,{allocated})"
                    )
        return bad

    def snapshot(self) -> dict[int, HostSlotRecord]:
        """Copy of the records, for logging a rank's host account."""
        return {idx: dataclasses.replace(rec) for idx, rec in self._records.items()}
