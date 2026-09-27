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
from types import SimpleNamespace
from typing import List, Optional, Set

import torch

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    unified_memory_disagg_move_gate,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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
        self.decode_offload_manager = None


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
    crediting them lets the scheduler admit work `_relieve_for_alloc` cannot
    satisfy; the alloc then returns None and the decode prealloc path treats
    that as a memory-estimation bug and aborts the scheduler.
    """

    class _Peer:
        def __init__(self, gate, host_gate=None):
            self.lazy_compaction = True
            self._free_phys_pages = [0, 1, 2, 3]  # only len() is read
            self.entry_bytes_per_page = 512
            self.disagg_move_gate = gate
            self.host_transfer_move_gate = host_gate

        def _is_frontier_transparent(self):
            return False

        # Exercise the production predicate when checking each gate.
        moves_blocked = MultiEndedAllocator.moves_blocked

    class _Owner:
        """Stands in for a grow-up END pool: the credit walks the chain from
        `_growth_side_neighbor()`, so the stub must expose what that walk reads,
        not the pre-chain `_peer` slot it used to."""

        def __init__(self, peer):
            self.grow_direction = "up"
            self.high_peer = peer
            self.low_peer = None

        _growth_side_neighbor = MultiEndedAllocator._growth_side_neighbor

    def _credit(self, gate, host_gate=None):
        peer = self._Peer(gate, host_gate)
        owner = self._Owner(peer)
        return MultiEndedAllocator._peer_drainable_hole_bytes(owner)

    def test_credit_follows_the_gate(self):
        # No PD gate installed (non-disagg): holes are realizable as before.
        self.assertEqual(self._credit(gate=None), 4 * 512)
        # Gate open: peer can compact, so the credit stands.
        self.assertEqual(self._credit(gate=lambda: True), 4 * 512)
        # Gate closed: an urgent flush would move nothing, so credit nothing.
        self.assertEqual(self._credit(gate=lambda: False), 0)
        # Either the RDMA gate or the HiCache gate can block compaction.
        self.assertEqual(self._credit(gate=None, host_gate=lambda: True), 4 * 512)
        self.assertEqual(self._credit(gate=None, host_gate=lambda: False), 0)
        self.assertEqual(self._credit(gate=lambda: True, host_gate=lambda: False), 0)


class TestMoveGateRejectsNonPdNode(CustomTestCase):
    def test_null_mode_is_rejected(self):
        """The gate is only meaningful on a PD node; a NULL-mode scheduler is a
        wiring bug and must not silently produce an always-open predicate."""
        scheduler = _FakeScheduler(DisaggregationMode.NULL)
        with self.assertRaises(ValueError):
            unified_memory_disagg_move_gate(scheduler)


class TestUnifiedAllocatorsPublishTheTransferContract(CustomTestCase):
    """Unified composites must translate virtual IDs before PD transfer.

    The implementation may be inherited from a shared unified allocator base,
    but inheriting the static allocator's identity would put virtual IDs on the
    wire and silently corrupt KV. Gate installation must reach every member.
    """

    @staticmethod
    def _allocator_class(name):
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedMambaSWATokenToKVPoolAllocator,
            UnifiedSWATokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.allocator.unified_mamba import (
            UnifiedMambaTokenToKVPoolAllocator,
        )

        classes = (
            UnifiedMambaTokenToKVPoolAllocator,
            UnifiedSWATokenToKVPoolAllocator,
            UnifiedMambaSWATokenToKVPoolAllocator,
        )
        return {cls.__name__: cls for cls in classes}[name]

    def test_transfer_translate_is_not_inherited_identity(self):
        virtual = torch.tensor([1, 3], dtype=torch.int32)
        for name in self._EXPECTED_COVERAGE:
            with self.subTest(composite=name), get_parallel().override(attn_dcp_size=1):
                alloc = object.__new__(self._allocator_class(name))
                alloc.full_attn_allocator = SimpleNamespace(
                    translate_kv_loc=lambda ids: ids + 16
                )
                physical = alloc.translate_kv_indices_for_transfer(virtual)
                self.assertEqual(physical.dtype, torch.int64)
                self.assertEqual(physical.tolist(), [17, 19])

    # Every sub-allocator attribute a composite can hold. The stub carries all
    # of them regardless of composite, so the assertion is on what installation
    # REACHES rather than on what the stub was given.
    _MEMBER_ATTRS = ("full_attn_allocator", "swa_attn_allocator", "mamba_allocator")

    # Inherited gate setters must cover every member, including tri-pool Mamba.
    _EXPECTED_COVERAGE = {
        "UnifiedMambaTokenToKVPoolAllocator": {
            "full_attn_allocator",
            "mamba_allocator",
        },
        "UnifiedSWATokenToKVPoolAllocator": {
            "full_attn_allocator",
            "swa_attn_allocator",
        },
        "UnifiedMambaSWATokenToKVPoolAllocator": {
            "full_attn_allocator",
            "swa_attn_allocator",
            "mamba_allocator",
        },
    }

    def _members_reached(self, cls_name: str, slot: str) -> Set[str]:
        """Install one gate on a stub composite and report which members got it.

        `object.__new__` skips `__init__` (which needs a GPU); the setter reads
        only `lazy_compaction` and the member attributes.
        """
        cls = self._allocator_class(cls_name)
        alloc = object.__new__(cls)
        alloc.lazy_compaction = True
        for attr in self._MEMBER_ATTRS:
            member = type("_Member", (), {})()
            member.disagg_move_gate = None
            member.host_transfer_move_gate = None
            setattr(alloc, attr, member)

        def gate() -> bool:
            return True

        if slot == "disagg_move_gate":
            alloc.set_disagg_move_gate(gate)
        else:
            alloc.set_host_transfer_move_gate(gate)
        return {
            attr
            for attr in self._MEMBER_ATTRS
            if getattr(getattr(alloc, attr), slot) is gate
        }

    def test_every_gate_reaches_every_member(self):
        """Both transfer gates must protect every sub-pool from relocation."""
        for name, expected in self._EXPECTED_COVERAGE.items():
            for slot in ("disagg_move_gate", "host_transfer_move_gate"):
                with self.subTest(composite=name, slot=slot):
                    self.assertEqual(
                        self._members_reached(name, slot),
                        expected,
                        f"{name}.{slot} does not cover every member",
                    )

    def test_gate_setters_do_not_enumerate_members_themselves(self):
        """The structural half of the rule above: a setter that names its
        members is one a new member silently escapes. Installation must go
        through the shared helper, which drives off `_move_gate_targets`.
        """
        import inspect

        for name in self._EXPECTED_COVERAGE:
            cls = self._allocator_class(name)
            for setter in ("set_disagg_move_gate", "set_host_transfer_move_gate"):
                if setter not in vars(cls):
                    continue  # inherited, and the inherited one is checked above
                with self.subTest(composite=name, setter=setter):
                    body = inspect.getsource(getattr(cls, setter))
                    self.assertIn("install_move_gate", body)
                    self.assertNotIn("_move_gate = ", body)

    def test_swa_composite_translates_the_swa_side_separately(self):
        """The SWA sub-pool runs its OWN compaction, so a full-side physical id
        does not name the SWA page holding the same virtual token. The read-path
        `translate_loc_from_full_to_swa` cannot stand in either: it returns
        kernel-facing ids, and the transfer addresses raw page envelopes."""
        virtual = torch.tensor([1, 3], dtype=torch.int32)
        for name in (
            "UnifiedSWATokenToKVPoolAllocator",
            "UnifiedMambaSWATokenToKVPoolAllocator",
        ):
            with self.subTest(composite=name), get_parallel().override(attn_dcp_size=1):
                alloc = object.__new__(self._allocator_class(name))
                alloc.full_attn_allocator = SimpleNamespace(
                    translate_kv_loc=lambda ids: ids + 16
                )
                alloc.swa_attn_allocator = SimpleNamespace(
                    translate_kv_loc=lambda ids: ids + 32,
                    translate_kv_loc_for_kernel=lambda ids: ids + 64,
                )
                physical = alloc.translate_swa_indices_for_transfer(virtual)
                self.assertEqual(physical.dtype, torch.int64)
                self.assertEqual(physical.tolist(), [33, 35])


class TestEverySwaAllocatorAnswersTheTransferTranslate(CustomTestCase):
    """Any allocator with a full->SWA read translate needs the transfer sibling.

    `_swa_payload` on both PD sides calls
    `translate_swa_indices_for_transfer` on whatever allocator the scheduler
    holds. Most get it by inheriting `SWATokenToKVPoolAllocator`, but a
    composite that merely DELEGATES the read translate (the DSV4 HiSparse
    allocator derives from `BaseTokenToKVPoolAllocator`) inherits neither the
    default nor an override, and PD aborts with an AttributeError the moment a
    sliding-window payload is built.

    Derived from the live class tree rather than a hand-kept list: a list would
    pass forever the day someone adds the next delegating composite.
    """

    @staticmethod
    def _allocator_classes():
        import importlib
        import inspect
        import pkgutil

        import sglang.srt.mem_cache.allocator as pkg
        from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

        found = {}
        for mod_info in pkgutil.iter_modules(pkg.__path__):
            try:
                mod = importlib.import_module(
                    f"sglang.srt.mem_cache.allocator.{mod_info.name}"
                )
            except Exception:
                continue  # optional backends need hardware this runner may lack
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if issubclass(cls, BaseTokenToKVPoolAllocator):
                    found[cls.__name__] = cls
        return found

    def test_read_translate_implies_transfer_translate(self):
        classes = self._allocator_classes()
        # Guard the guard: an import failure that empties this set would make
        # the assertion below vacuous.
        self.assertIn("SWATokenToKVPoolAllocator", classes)
        for name, cls in sorted(classes.items()):
            if not hasattr(cls, "translate_loc_from_full_to_swa"):
                continue
            with self.subTest(allocator=name):
                self.assertTrue(
                    hasattr(cls, "translate_swa_indices_for_transfer"),
                    f"{name} translates full->SWA for reads but cannot answer "
                    "translate_swa_indices_for_transfer; PD's _swa_payload "
                    "calls it on whatever allocator the scheduler holds",
                )


if __name__ == "__main__":
    unittest.main()
