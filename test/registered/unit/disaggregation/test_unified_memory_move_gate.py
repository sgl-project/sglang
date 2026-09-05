"""Regression tests for the unified-memory PD compaction move gate.

The gate decides when lazy compaction may relocate physical pages. A page is
exposed to the peer from the moment its address is published until the transfer
concludes, and for part of that lifetime the request sits in NEITHER end's
queue. Both cases below are exactly those windows: an earlier version of the
predicates looked only at `disagg_prefill_inflight_queue` /
`disagg_decode_transfer_queue` (plus `scheduler.chunked_req`) and returned True
here, letting compaction move pages under in-flight RDMA -- silent KV
corruption with no crash.
"""

import unittest
from typing import List, Optional, Set

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    unified_memory_disagg_move_gate,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _FakeTransferQueue:
    def __init__(self):
        self.queue: List[object] = []


class _FakePreallocQueue:
    """Mirrors the real queue's published-destination bookkeeping."""

    def __init__(self):
        self._num_published_destinations = 0

    @property
    def has_published_destinations(self) -> bool:
        return self._num_published_destinations > 0

    def note_destinations_published(self) -> None:
        self._num_published_destinations += 1

    def note_destinations_queued(self, count: int) -> None:
        self._num_published_destinations = max(
            0, self._num_published_destinations - count
        )


class _FakeScheduler:
    def __init__(self, mode: DisaggregationMode):
        self.disaggregation_mode = mode
        self.chunked_req: Optional[object] = None
        self.disagg_prefill_inflight_queue: List[object] = []
        self.disagg_prefill_pending_chunk_rids: Set[str] = set()
        self.disagg_decode_transfer_queue = _FakeTransferQueue()
        self.disagg_decode_prealloc_queue = _FakePreallocQueue()


class TestDecodeMoveGate(CustomTestCase):
    def test_closed_while_destination_published_but_not_queued(self):
        """`pop_preallocated` publishes request A's destination addresses via
        `send_metadata`, then keeps allocating for request B in the same loop;
        the batch only reaches the transfer queue after the loop returns. B's
        allocation can urgently flush the peer sub-allocator, so the gate must
        stay closed across that window even though the transfer queue is empty.
        """
        scheduler = _FakeScheduler(DisaggregationMode.DECODE)
        gate = unified_memory_disagg_move_gate(scheduler)
        self.assertTrue(gate(), "idle decode node should allow compaction")

        # A's destination is now visible to prefill; transfer queue still empty.
        scheduler.disagg_decode_prealloc_queue.note_destinations_published()
        self.assertFalse(scheduler.disagg_decode_transfer_queue.queue)
        self.assertFalse(gate())

        # Handing the batch to the transfer queue transfers responsibility.
        scheduler.disagg_decode_transfer_queue.queue.append(object())
        scheduler.disagg_decode_prealloc_queue.note_destinations_queued(1)
        self.assertFalse(gate(), "transfer queue still holds it")

        scheduler.disagg_decode_transfer_queue.queue.clear()
        self.assertTrue(gate())


class TestPrefillMoveGate(CustomTestCase):
    def test_closed_after_final_chunk_clears_chunked_req(self):
        """Scheduling the final chunk clears `scheduler.chunked_req`, but the
        request only reaches `disagg_prefill_inflight_queue` later in the result
        path. Earlier middle chunks may still be draining in that window, so the
        gate must not key off `chunked_req` alone.
        """
        scheduler = _FakeScheduler(DisaggregationMode.PREFILL)
        gate = unified_memory_disagg_move_gate(scheduler)
        self.assertTrue(gate(), "idle prefill node should allow compaction")

        # A middle chunk went out for rid "r0".
        scheduler.chunked_req = object()
        scheduler.disagg_prefill_pending_chunk_rids.add("r0")
        self.assertFalse(gate())

        # Final chunk scheduled: chunked_req cleared, not yet inflight-queued.
        scheduler.chunked_req = None
        self.assertFalse(scheduler.disagg_prefill_inflight_queue)
        self.assertFalse(gate())

        # Last chunk sent: the request is on the inflight queue, which covers it.
        scheduler.disagg_prefill_inflight_queue.append(object())
        scheduler.disagg_prefill_pending_chunk_rids.discard("r0")
        self.assertFalse(gate())

        scheduler.disagg_prefill_inflight_queue.clear()
        self.assertTrue(gate())

    def test_reopens_when_middle_sent_request_is_retired_without_final_chunk(self):
        """A request aborted after a middle chunk never reaches a `last_chunk`
        send, so its rid is only dropped by the abort/release cleanup. Without
        that discard the gate stays closed for the process lifetime and lazy
        compaction never packs the free list again -- a liveness leak that ends
        in allocation failure despite reclaimable space.
        """
        scheduler = _FakeScheduler(DisaggregationMode.PREFILL)
        gate = unified_memory_disagg_move_gate(scheduler)

        scheduler.chunked_req = object()
        scheduler.disagg_prefill_pending_chunk_rids.add("r0")
        self.assertFalse(gate())

        # Aborted mid-chunking: chunked_req dropped, no final send, never queued.
        scheduler.chunked_req = None
        scheduler.disagg_prefill_pending_chunk_rids.discard("r0")
        self.assertTrue(gate(), "abort cleanup must let compaction resume")


class TestGatedPeerHolesAreNotSchedulable(CustomTestCase):
    """`schedulable_available_size` credits holes a peer urgent-flush would
    release. While the move gate is closed that flush relocates nothing, so
    crediting them lets the scheduler admit work `_flush_peer_for_alloc` cannot
    satisfy; the alloc then returns None and the decode prealloc path treats
    that as a memory-estimation bug and aborts the scheduler.
    """

    class _Peer:
        def __init__(self, gate):
            self.lazy_compaction = True
            self._free_phys_pages = [0, 1, 2, 3]  # only len() is read
            self.entry_bytes_per_page = 512
            self.disagg_move_gate = gate

        def _is_frontier_transparent(self):
            return False

    class _Owner:
        """Stands in for a grow-up END pool: the credit walks the chain from
        `_growth_side_neighbor()`, so the stub must expose what that walk reads,
        not the pre-chain `_peer` slot it used to."""

        def __init__(self, peer):
            self.grow_direction = "up"
            self.high_peer = peer
            self.low_peer = None

        _growth_side_neighbor = MultiEndedAllocator._growth_side_neighbor

    def _credit(self, gate):
        peer = self._Peer(gate)
        owner = self._Owner(peer)
        return MultiEndedAllocator._peer_drainable_hole_bytes(owner)

    def test_credit_follows_the_gate(self):
        # No PD gate installed (non-disagg): holes are realizable as before.
        self.assertEqual(self._credit(gate=None), 4 * 512)
        # Gate open: peer can compact, so the credit stands.
        self.assertEqual(self._credit(gate=lambda: True), 4 * 512)
        # Gate closed: an urgent flush would move nothing, so credit nothing.
        self.assertEqual(self._credit(gate=lambda: False), 0)


class TestMoveGateRejectsNonPdNode(CustomTestCase):
    def test_null_mode_is_rejected(self):
        """The gate is only meaningful on a PD node; a NULL-mode scheduler is a
        wiring bug and must not silently produce an always-open predicate."""
        scheduler = _FakeScheduler(DisaggregationMode.NULL)
        with self.assertRaises(ValueError):
            unified_memory_disagg_move_gate(scheduler)


if __name__ == "__main__":
    unittest.main()
