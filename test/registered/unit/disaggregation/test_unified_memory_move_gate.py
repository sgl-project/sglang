"""Regression tests for unified-memory PD transfer ordering.

A PD transfer addresses device pages by physical id from the moment the ids are
published (decode) or translated for a send (prefill) until its engine is done,
and for part of that lifetime the request sits in NEITHER end's queue. An
earlier pin keyed off the queues missed those windows and let compaction move
pages under in-flight RDMA; a later fix pinned every band while any transfer was
in flight, and another kept each addressed page in place. Pages a transfer
reads or writes now move like any page unfinished work touches: a copy of a
page it writes waits its completion flag, a page it reads that a move vacates is
reused only after it, and nothing the transfer does changes pricing.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Set
from unittest import mock

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import (
    end_network_transfer,
    record_network_transfer,
)
from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    chain_snapshot,
    hold_every_band,
    release_every_band,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _tri_pool():
    setup_allocator_context()
    bundle = build_unified_bundle(
        UnifiedGeometry(
            chain="tri",
            full_layer_ids=(0,),
            swa_layer_ids=(1,),
            state_layer_ids=(2,),
            lazy=True,
            device="cpu",
        )
    )
    allocator = bundle.token_to_kv_pool_allocator
    return allocator, allocator.storage


def _physical_pages(band, ids) -> list:
    pages = band.ids.virtual_to_physical[ids // band.page_size] * band.page_size
    return (pages // band.pool_page_size).tolist()


class _Manager(CommonKVManager):
    """The status path of a connection, without its sockets."""

    def __init__(self):
        self.request_status = {}
        self.deferred_bootstrap = None
        self._deferred_ack_targets = {}
        self._deferred_ack_poisoned_rooms = set()
        self._staging_outstanding = {}


def _record(allocator, req, pages, *, writes, manager=None, failed_delay=None):
    record_network_transfer(
        allocator,
        req,
        kv_manager=_Manager() if manager is None else manager,
        writes=writes,
        page_indices=np.array(pages, dtype=np.int32),
        page_size=allocator.page_size,
        failed_delay=failed_delay,
    )


class TestNetworkTransfers(CustomTestCase):
    def test_transfers_never_change_pricing_or_block_a_band(self):
        """A rank prices its pool from counts and geometry only, whatever its
        transfers are doing: rank 0 prices another stage from the same."""
        allocator, storage = _tri_pool()
        owner, state = allocator.full_attn_allocator, allocator.mamba_allocator
        ids = owner.alloc(8 * allocator.page_size)
        owner.free(ids[: 2 * allocator.page_size])
        slots = state.alloc(2)
        before = chain_snapshot(allocator)
        req = SimpleNamespace(rid="r0", bootstrap_room=1)
        _record(allocator, req, _physical_pages(owner, ids[-1:]), writes=True)
        record_network_transfer(
            allocator,
            req,
            kv_manager=_Manager(),
            writes=True,
            page_indices=np.array([], dtype=np.int32),
            page_size=allocator.page_size,
            state_types=[StateType.MAMBA],
            state_indices=[[np.array(_physical_pages(state, slots[:1]))]],
        )
        self.assertEqual(chain_snapshot(allocator), before)
        self.assertFalse(any(m.moves_blocked() for m in allocator.members))
        self.assertFalse(any(m.fixed_pages() for m in allocator.members))
        end_network_transfer(allocator, req)
        self.assertEqual(chain_snapshot(allocator), before)

    def test_a_read_page_moves_and_its_old_row_waits_for_the_transfer(self):
        """A send's row moves like a page a forward reads; the row it vacates
        is not handed out until the send is released."""
        allocator, storage = _tri_pool()
        owner = allocator.full_attn_allocator
        page = allocator.page_size
        ids = owner.alloc(8 * page)
        # The page nearest the frontier is the first a compaction moves.
        top = _physical_pages(owner, ids[:1])
        req = SimpleNamespace(rid="r0", bootstrap_room=1)
        _record(allocator, req, top, writes=False)
        owner.free(ids[2 * page : 6 * page])  # interior holes
        self.assertGreater(owner.flush_opportunistic(), 0)
        self.assertNotEqual(_physical_pages(owner, ids[:1]), top, "it moved")
        self.assertIn(top[0], owner.usage.pending_pages)
        self.assertNotIn(top[0], owner._free_pages_snapshot())
        end_network_transfer(allocator, req)
        owner.flush_opportunistic()
        self.assertNotIn(top[0], owner.usage.pending_pages)

    def test_only_a_needed_move_takes_a_written_page(self):
        """An opportunistic flush leaves a page a receive still writes; a
        flush an allocation needs moves it, ordered after the receive."""
        allocator, storage = _tri_pool()
        owner = allocator.full_attn_allocator
        page = allocator.page_size
        ids = owner.alloc(8 * page)
        # The page nearest the frontier is the first a compaction moves.
        top = _physical_pages(owner, ids[:1])
        req = SimpleNamespace(rid="r0", bootstrap_room=1)
        _record(allocator, req, top, writes=True)
        owner.free(ids[2 * page : 6 * page])  # interior holes
        self.assertEqual(owner.flush_opportunistic(), 0)
        self.assertEqual(_physical_pages(owner, ids[:1]), top)
        self.assertGreater(owner.flush_for_allocation(), 0)
        self.assertNotEqual(_physical_pages(owner, ids[:1]), top)

    def test_releasing_the_row_ends_the_transfer(self):
        """A request aborted mid-transfer never commits; its row release is
        main's release point, where the transfer ends and its flag publishes."""
        allocator, storage = _tri_pool()
        owner = allocator.full_attn_allocator
        ids = owner.alloc(4)
        req = SimpleNamespace(rid="r0", bootstrap_room=1)
        manager = _Manager()
        _record(
            allocator,
            req,
            _physical_pages(owner, ids[:1]),
            writes=True,
            manager=manager,
        )
        flag, release = storage._network_transfers[req.rid]
        self.assertFalse(flag.query())
        with mock.patch.object(storage.requests, "free"):
            storage.release_row(req)
        self.assertTrue(flag.query())
        self.assertTrue(release.released)
        self.assertFalse(storage._network_transfers)

    def test_the_status_path_publishes_the_flag(self):
        """The flag is published by the thread applying the transfer's status,
        never only by the scheduler loop, which may itself wait on it."""
        allocator, storage = _tri_pool()
        owner = allocator.full_attn_allocator
        ids = owner.alloc(4)
        for status, delay in ((KVPoll.Success, 30.0), (KVPoll.Failed, None)):
            with self.subTest(status=status):
                manager = _Manager()
                req = SimpleNamespace(rid=f"r{status}", bootstrap_room=status)
                manager.update_status(req.bootstrap_room, KVPoll.Bootstrapping)
                _record(
                    allocator,
                    req,
                    _physical_pages(owner, ids[:1]),
                    writes=True,
                    manager=manager,
                    failed_delay=delay,
                )
                flag, _ = storage._network_transfers[req.rid]
                self.assertFalse(flag.query())
                manager.update_status(req.bootstrap_room, status)
                self.assertTrue(flag.query())
        # A failed receive publishes only after the engine's deadline.
        manager = _Manager()
        req = SimpleNamespace(rid="late", bootstrap_room=7)
        manager.update_status(7, KVPoll.Bootstrapping)
        with mock.patch("threading.Timer") as timer:
            _record(
                allocator,
                req,
                _physical_pages(owner, ids[:1]),
                writes=True,
                manager=manager,
                failed_delay=30.0,
            )
            flag, _ = storage._network_transfers[req.rid]
            manager.update_status(7, KVPoll.Failed)
            self.assertFalse(flag.query())
            delay, publish = timer.call_args.args
            self.assertEqual(delay, 30.0)
            publish()
        self.assertTrue(flag.query())


class TestUnifiedAllocatorsPublishTheTransferContract(CustomTestCase):
    """The unified allocator must translate virtual IDs before PD transfer.

    Inheriting the static allocator's identity would put virtual IDs on the
    wire and silently corrupt KV. Every event that orders a sub-pool's page
    moves must reach every band of every chain.
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

    def test_transfers_block_nothing_and_holding_still_does(self):
        """Network transfers order moves without blocking a band, so a role
        switch has nothing to drop or leak; holding every band still keeps each
        in place on its own."""
        for chain in self._CHAINS:
            with self.subTest(chain=chain):
                alloc = self._chain(chain)
                storage = alloc.storage
                owner = alloc.full_attn_allocator

                def capacity_keys():
                    return alloc.pricing_snapshot()

                ids = owner.alloc(alloc.page_size)
                keys = capacity_keys()
                pages = owner.ids.virtual_to_physical[ids[:1] // owner.page_size]
                req = SimpleNamespace(rid="r0", bootstrap_room=1)
                _record(alloc, req, pages.tolist(), writes=True)
                self.assertFalse(any(m.moves_blocked() for m in alloc.members))
                self.assertEqual(capacity_keys(), keys)
                end_network_transfer(alloc, req)
                self.assertEqual(capacity_keys(), keys)
                hold_every_band(alloc)
                # A hold fixes live pages: a band with none has nothing to keep.
                self.assertTrue(
                    all(m.moves_blocked() for m in alloc.members if m.live_page_count)
                )
                self.assertNotEqual(capacity_keys(), keys)
                release_every_band(alloc)
                self.assertFalse(any(m.moves_blocked() for m in alloc.members))

    def test_every_ordering_event_reaches_every_member(self):
        """Every member that can relocate pages must hear each event its moves
        wait on: a member that misses a HiCache copy's event may move a page the
        copy still reads or writes. Recurrent state is written by its own
        kernels, so its member gets the state slots, not the token ids."""
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
                                side_effect=lambda *args, got=got, hook=hook, **_: (
                                    got.__setitem__(hook, args[-1])
                                ),
                            )
                        )
                event, ids = object(), torch.tensor([5, 6])
                alloc.set_hicache_transfer_done_event("load", event)
                alloc.set_latest_forward_done_event(event)
                alloc.set_inflight_forward(event, ids, torch.tensor([3]))
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
                        self.assertEqual(got["inflight"].tolist(), [3])
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
