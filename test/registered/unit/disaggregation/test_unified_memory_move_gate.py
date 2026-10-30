"""Regression tests for the unified-memory PD pin source.

The source decides when lazy compaction may relocate physical pages. A page is
exposed to the peer from the moment its address is published until the transfer
concludes, and for part of that lifetime the request sits in NEITHER end's
queue. Both cases below are exactly those windows: an earlier version of the
predicates looked only at `disagg_prefill_inflight_queue` /
`disagg_decode_transfer_queue` (plus `scheduler.chunked_req`) and reported idle
here, letting compaction move pages under in-flight RDMA -- silent KV
corruption with no crash.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from typing import List, Optional, Set
from unittest import mock

import torch

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    disagg_transfers_in_flight,
)
from sglang.srt.mem_cache.kv.alloc.band import Band
from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        allocation can urgently flush the peer sub-allocator, so the source must
        stay busy across that window even though the transfer queue is empty.
        """
        scheduler = _FakeScheduler(DisaggregationMode.DECODE)
        busy = disagg_transfers_in_flight(scheduler)
        self.assertFalse(busy(), "idle decode node should allow compaction")

        # A's destination is now visible to prefill; transfer queue still empty.
        scheduler.disagg_decode_prealloc_queue.note_destinations_published()
        self.assertFalse(scheduler.disagg_decode_transfer_queue.queue)
        self.assertTrue(busy())

        # Handing the batch to the transfer queue transfers responsibility.
        scheduler.disagg_decode_transfer_queue.queue.append(
            SimpleNamespace(host_staged=False)
        )
        scheduler.disagg_decode_prealloc_queue.note_destinations_queued(1)
        self.assertTrue(busy(), "transfer queue still holds it")

        scheduler.disagg_decode_transfer_queue.queue.clear()
        self.assertFalse(busy())


class TestPrefillMoveGate(CustomTestCase):
    def test_closed_after_final_chunk_clears_chunked_req(self):
        """Scheduling the final chunk clears `scheduler.chunked_req`, but the
        request only reaches `disagg_prefill_inflight_queue` later in the result
        path. Earlier middle chunks may still be draining in that window, so the
        source must not key off `chunked_req` alone.
        """
        scheduler = _FakeScheduler(DisaggregationMode.PREFILL)
        busy = disagg_transfers_in_flight(scheduler)
        self.assertFalse(busy(), "idle prefill node should allow compaction")

        # A middle chunk went out for rid "r0".
        scheduler.chunked_req = object()
        scheduler.disagg_prefill_pending_chunk_rids.add("r0")
        self.assertTrue(busy())

        # Final chunk scheduled: chunked_req cleared, not yet inflight-queued.
        scheduler.chunked_req = None
        self.assertFalse(scheduler.disagg_prefill_inflight_queue)
        self.assertTrue(busy())

        # Last chunk sent: the request is on the inflight queue, which covers it.
        scheduler.disagg_prefill_inflight_queue.append(object())
        scheduler.disagg_prefill_pending_chunk_rids.discard("r0")
        self.assertTrue(busy())

        scheduler.disagg_prefill_inflight_queue.clear()
        self.assertFalse(busy())

    def test_reopens_when_middle_sent_request_is_retired_without_final_chunk(self):
        """A request aborted after a middle chunk never reaches a `last_chunk`
        send, so its rid is only dropped by the abort/release cleanup. Without
        that discard the source stays busy for the process lifetime and lazy
        compaction never packs the free list again -- a liveness leak that ends
        in allocation failure despite reclaimable space.
        """
        scheduler = _FakeScheduler(DisaggregationMode.PREFILL)
        busy = disagg_transfers_in_flight(scheduler)

        scheduler.chunked_req = object()
        scheduler.disagg_prefill_pending_chunk_rids.add("r0")
        self.assertTrue(busy())

        # Aborted mid-chunking: chunked_req dropped, no final send, never queued.
        scheduler.chunked_req = None
        scheduler.disagg_prefill_pending_chunk_rids.discard("r0")
        self.assertFalse(busy(), "abort cleanup must let compaction resume")


class TestPinnedPeerHolesAreNotSchedulable(CustomTestCase):
    """`schedulable_available_size` credits holes a peer urgent-flush would
    release. While a pin source is busy that flush relocates nothing, so
    crediting them lets the scheduler admit work `relieve_for_alloc` cannot
    satisfy; the alloc then returns None and the decode prealloc path treats
    that as a memory-estimation bug and aborts the scheduler.
    """

    class _Peer:
        def __init__(self, sources):
            self.lazy_compaction = True
            self._free_phys_pages = [0, 1, 2, 3]  # only len() is read
            self._load_pinned_pages = 0
            self.entry_bytes_per_page = 512
            self.pin_sources = sources

        def _is_frontier_transparent(self):
            return False

        # Exercise the production predicates when checking each source.
        moves_blocked = Band.moves_blocked
        pinned_by_source = Band.pinned_by_source

    class _Owner:
        """Stands in for a grow-up END pool: the credit walks the chain from
        `_growth_side_neighbor()`, so the stub must expose what that walk reads,
        not the pre-chain `_peer` slot it used to."""

        def __init__(self, peer):
            self.grow_direction = "up"
            self.high_peer = peer
            self.low_peer = None

        _growth_side_neighbor = Band._growth_side_neighbor

    def _credit(self, *sources):
        peer = self._Peer(sources)
        owner = self._Owner(peer)
        return Band._peer_drainable_hole_bytes(owner)

    def test_credit_follows_the_pins(self):
        # No pin source (non-disagg): holes are realizable as before.
        self.assertEqual(self._credit(), 4 * 512)
        # Source idle: peer can compact, so the credit stands.
        self.assertEqual(self._credit(lambda: False), 4 * 512)
        # Source busy: an urgent flush would move nothing, so credit nothing.
        self.assertEqual(self._credit(lambda: True), 0)
        # Any one busy source (RDMA or HiCache) blocks compaction.
        self.assertEqual(self._credit(lambda: False, lambda: False), 4 * 512)
        self.assertEqual(self._credit(lambda: False, lambda: True), 0)
        self.assertEqual(self._credit(lambda: True, lambda: False), 0)


class TestUnifiedAllocatorsPublishTheTransferContract(CustomTestCase):
    """The unified allocator must translate virtual IDs before PD transfer.

    Inheriting the static allocator's identity would put virtual IDs on the
    wire and silently corrupt KV. A pin source over every sub-pool must reach
    every band of every chain, and so must every event that orders a
    sub-pool's page moves.
    """

    # The layers of each chain shape: full + window, full + state, all three.
    _CHAINS = {
        "swa": dict(swa_layer_ids=(1,)),
        "mamba": dict(state_layer_ids=(1,)),
        "tri": dict(swa_layer_ids=(1,), state_layer_ids=(2,)),
    }

    @classmethod
    def _chain(cls, chain: str) -> ChainAllocator:
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain=chain,
                full_layer_ids=(0,),
                lazy=True,
                device="cpu",
                **cls._CHAINS[chain],
            )
        )
        return bundle.token_to_kv_pool_allocator

    @staticmethod
    def _sub_pools(alloc: ChainAllocator) -> Set[str]:
        return {band.name for band in alloc.arena.bands}

    def test_transfer_translate_is_not_inherited_identity(self):
        virtual = torch.tensor([1, 3], dtype=torch.int32)
        with get_parallel().override(attn_dcp_size=1):
            alloc = object.__new__(ChainAllocator)
            alloc.full_attn_allocator = SimpleNamespace(
                ids=SimpleNamespace(translate=lambda ids: ids + 16),
                shards_under_dcp=True,
            )
            physical = alloc.translate_kv_indices_for_transfer(virtual)
        self.assertEqual(physical.dtype, torch.int64)
        self.assertEqual(physical.tolist(), [17, 19])

    def test_every_pin_source_reaches_every_member(self):
        """A transfer's pin source, registered over every sub-pool the storage
        lists, must protect every band from relocation."""

        def busy() -> bool:
            return True

        for chain in self._CHAINS:
            with self.subTest(chain=chain):
                alloc = self._chain(chain)
                storage = alloc.storage
                storage.add_pin_source(storage.sub_pool_names(), busy)
                self.assertEqual(
                    {m.sub_pool_name for m in alloc.members if busy in m.pin_sources},
                    self._sub_pools(alloc),
                    f"{chain}'s pin source does not cover every sub-pool",
                )
                self.assertTrue(all(m.moves_blocked() for m in alloc.members))

    def test_role_switch_keeps_current_pd_pins_and_independent_hicache_pins(self):
        """Role changes must release old queues without losing protection for
        new PD transfers or an independent HiCache transfer."""
        for chain in self._CHAINS:
            with self.subTest(chain=chain):
                alloc = self._chain(chain)
                scheduler = _FakeScheduler(DisaggregationMode.DECODE)
                storage = alloc.storage
                storage.add_pin_source(
                    storage.sub_pool_names(), disagg_transfers_in_flight(scheduler)
                )
                hicache = SimpleNamespace(busy=False)
                storage.add_pin_source(storage.sub_pool_names(), lambda: hicache.busy)

                def capacity_keys():
                    return tuple(m._schedulable_capacity_key() for m in alloc.members)

                idle_keys = capacity_keys()

                def check_pins(blocked):
                    self.assertEqual(
                        [m.moves_blocked() for m in alloc.members],
                        [blocked] * len(alloc.members),
                    )
                    if blocked:
                        self.assertNotEqual(capacity_keys(), idle_keys)
                    else:
                        self.assertEqual(capacity_keys(), idle_keys)

                scheduler.disagg_decode_prealloc_queue.note_destinations_published()
                check_pins(True)
                scheduler.disagg_decode_prealloc_queue.note_destinations_queued(1)
                check_pins(False)

                # Role teardown removes the decode queues; the pool survives.
                scheduler.disaggregation_mode = DisaggregationMode.PREFILL
                scheduler.disagg_decode_transfer_queue = None
                scheduler.disagg_decode_prealloc_queue = None
                check_pins(False)
                scheduler.disagg_prefill_pending_chunk_rids.add("prefill")
                check_pins(True)
                scheduler.disagg_prefill_pending_chunk_rids.clear()
                hicache.busy = True
                check_pins(True)
                hicache.busy = False
                check_pins(False)

                scheduler.disaggregation_mode = DisaggregationMode.DECODE
                scheduler.disagg_prefill_inflight_queue = None
                scheduler.disagg_decode_transfer_queue = _FakeTransferQueue()
                scheduler.disagg_decode_prealloc_queue = _FakePreallocQueue()
                check_pins(False)
                scheduler.disagg_decode_transfer_queue.queue.append(
                    SimpleNamespace(host_staged=False)
                )
                check_pins(True)
                scheduler.disagg_decode_transfer_queue.queue.clear()
                offload = SimpleNamespace(busy=True)
                scheduler.decode_offload_manager = SimpleNamespace(
                    has_inflight_device_transfer=lambda: offload.busy
                )
                check_pins(True)
                offload.busy = False
                check_pins(False)

    def test_every_ordering_event_reaches_every_member(self):
        """Every member that can relocate pages must hear each event its moves
        wait on: a member that misses a HiCache copy's event may move a page the
        copy still reads or writes. Recurrent state is written by its own
        kernels, so its member gets no in-flight token ids."""
        for chain in self._CHAINS:
            alloc = self._chain(chain)
            received = {m.sub_pool_name: {} for m in alloc.members}
            with ExitStack() as stack:
                for member in alloc.members:
                    got = received[member.sub_pool_name]
                    for name, hook in (
                        ("set_hicache_transfer_done_event", "hicache"),
                        ("set_latest_forward_done_event", "forward_done"),
                        ("set_inflight_forward", "inflight"),
                    ):
                        stack.enter_context(
                            mock.patch.object(
                                member,
                                name,
                                side_effect=lambda *args, got=got, hook=hook: (
                                    got.__setitem__(hook, args[-1])
                                ),
                            )
                        )
                event, ids = object(), torch.tensor([5, 6])
                alloc.set_hicache_transfer_done_event("load", event)
                alloc.set_latest_forward_done_event(event)
                alloc.set_inflight_forward(event, ids)
            for hook in ("hicache", "forward_done", "inflight"):
                with self.subTest(chain=chain, hook=hook):
                    self.assertEqual(
                        {name for name, got in received.items() if hook in got},
                        self._sub_pools(alloc),
                        f"{chain} does not send {hook} to every member",
                    )
            with self.subTest(chain=chain, hook="inflight ids"):
                for name, got in received.items():
                    if name == "mamba":
                        self.assertIsNone(got["inflight"])
                    else:
                        self.assertEqual(got["inflight"].tolist(), [5, 6])

    def test_swa_chain_translates_the_swa_side_separately(self):
        """The SWA sub-pool runs its OWN compaction, so a full-side physical id
        does not name the SWA page holding the same virtual token: the transfer
        translate must go through the SWA sub-pool's own table."""
        virtual = torch.tensor([1, 3], dtype=torch.int32)
        with get_parallel().override(attn_dcp_size=1):
            alloc = object.__new__(ChainAllocator)
            alloc.full_attn_allocator = SimpleNamespace(
                ids=SimpleNamespace(translate=lambda ids: ids + 16)
            )
            alloc.swa_attn_allocator = SimpleNamespace(
                ids=SimpleNamespace(translate=lambda ids: ids + 32),
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

        import sglang.srt.mem_cache.allocator as allocator_pkg
        import sglang.srt.mem_cache.kv.alloc as chain_pkg
        from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

        found = {}
        for pkg in (allocator_pkg, chain_pkg):
            for mod_info in pkgutil.iter_modules(pkg.__path__):
                try:
                    mod = importlib.import_module(f"{pkg.__name__}.{mod_info.name}")
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
        self.assertIn("ChainAllocator", classes)
        for name, cls in sorted(classes.items()):
            # A chain translates window reads through its id spaces.
            reads_window = (
                hasattr(cls, "translate_loc_from_full_to_swa")
                or name == "ChainAllocator"
            )
            if not reads_window:
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
