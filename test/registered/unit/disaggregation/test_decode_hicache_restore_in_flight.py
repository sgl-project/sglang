"""Decode HiCache restores must not turn READY on pages a load-back is still writing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFIX_LEN = 8
PENDING = HiCacheRestoreResult.PENDING
READY = HiCacheRestoreResult.READY


class _FakeTreeCache:
    """UnifiedRadixCache's load-back bookkeeping for one host-resident prefix.

    As in ``load_back``, a restore attaches its device pages to the tree when the
    load is queued; its H2D copy finishes only when the test says so.
    """

    def __init__(self, on_device=False, state_only_load=False, free_slot=True):
        self.device_pages = torch.arange(100, 100 + PREFIX_LEN)
        self.on_device = on_device
        self.state_only_load = state_only_load
        self.free_slot = free_slot
        self.ongoing_load_back = {}
        self.cache_controller = SimpleNamespace(
            load_queue=[], layer_done_counter=SimpleNamespace(producer_index=-1)
        )
        self.done = set()

    def has_free_load_back_slot(self):
        return self.free_slot

    def init_load_back(self, params):
        if self.state_only_load:
            # KV is on device; only this request's Mamba state comes from host.
            self.state_only_load = False
            self.add_load()
            return 0, "node"
        if self.on_device:
            return 0, "node"
        self.on_device = True
        self.add_load()
        return PREFIX_LEN, "node"

    def add_load(self):
        self.ongoing_load_back[len(self.ongoing_load_back)] = "node"
        self.cache_controller.load_queue.append("op")

    def ready_to_load_host_cache(self):
        if not self.cache_controller.load_queue:
            return -1
        self.cache_controller.load_queue.clear()
        counter = self.cache_controller.layer_done_counter
        counter.producer_index = (counter.producer_index + 1) % 3
        return counter.producer_index

    def latest_load_back_index(self):
        return UnifiedRadixCache.latest_load_back_index(self)

    def is_load_back_event_done(self, index):
        return index < 0 or index in self.done

    def finish_loads(self):
        self.done.add(self.cache_controller.layer_done_counter.producer_index)
        self.ongoing_load_back.clear()

    def path_device_indices(self, node):
        return self.device_pages

    def lock(self, node):
        return f"lock-{node}"


def _host_hit(rid):
    """A request whose prefix is host-only at prealloc."""
    no_device_prefix = torch.tensor([], dtype=torch.int64)
    return _decode_req(rid, DecodePrefixMatch(no_device_prefix, PREFIX_LEN, 0, "root"))


def _device_hit(rid, tree):
    """A request whose prealloc match already found the prefix on device."""
    return _decode_req(rid, DecodePrefixMatch(tree.device_pages, 0, 0, "node"))


def _decode_req(rid, prefix_match):
    req = SimpleNamespace(
        rid=rid,
        cache_request_handle=CacheRequestHandle(rid, 0),
        origin_input_ids=list(range(PREFIX_LEN + 4)),
        last_node="root",
    )
    return SimpleNamespace(
        req=req,
        prefix_match=prefix_match,
        hicache_restore_status=PENDING,
        hicache_restore_lock=None,
        hicache_restored_kv_indices=None,
        hicache_load_consumer_index=-1,
    )


class TestDecodeHiCacheRestoreInFlight(CustomTestCase):
    def _tick(self, tree, decode_reqs, kv_on_device=None):
        harness = SimpleNamespace(tree_cache=tree)
        for name in (
            "_try_hicache_queue_load_back",
            "_advance_hicache_local_restores",
            "_gate_reused_device_pages",
        ):
            method = getattr(DecodeHiCacheTransferMixin, name, None)
            if method is not None:
                setattr(harness, name, method.__get__(harness))

        def rematch(req, tree_cache, *args, **kwargs):
            on_device = tree_cache.on_device if kv_on_device is None else kv_on_device
            return SimpleNamespace(
                device_prefix_len=PREFIX_LEN if on_device else 0,
                best_match_node="node",
                host_hit_length=0 if on_device else PREFIX_LEN,
            )

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_kv_cache",
            side_effect=rematch,
        ):
            DecodeHiCacheTransferMixin._process_hicache_local_restores(
                harness, decode_reqs
            )

    def test_rematch_onto_pages_of_an_in_flight_load_waits_for_it(self):
        tree = _FakeTreeCache()
        loader, follower = _host_hit("loader"), _host_hit("follower")

        self._tick(tree, [loader, follower])
        # The loader queued the DMA; the follower rematched onto its pages.
        self.assertEqual(
            follower.hicache_restored_kv_indices.tolist(), tree.device_pages.tolist()
        )
        self.assertEqual(loader.hicache_restore_status, PENDING)
        self.assertEqual(follower.hicache_restore_status, PENDING)

        self._tick(tree, [loader, follower])
        self.assertEqual(follower.hicache_restore_status, PENDING)

        tree.finish_loads()
        self._tick(tree, [loader, follower])
        self.assertEqual(loader.hicache_restore_status, READY)
        self.assertEqual(follower.hicache_restore_status, READY)

    def test_prealloc_device_hit_on_an_in_flight_load_waits_for_it(self):
        tree = _FakeTreeCache()
        loader = _host_hit("loader")
        self._tick(tree, [loader])

        # Prealloc matched the loader's pages as L1 while their H2D is pending.
        late = _device_hit("late", tree)
        self._tick(tree, [loader, late])
        self.assertEqual(late.hicache_restore_status, PENDING)

        tree.finish_loads()
        self._tick(tree, [loader, late])
        self.assertEqual(late.hicache_restore_status, READY)

    def test_state_only_load_back_holds_loader_and_reuser(self):
        # KV is on device; only the loader's Mamba state is restored from host.
        tree = _FakeTreeCache(on_device=True, state_only_load=True)
        loader, reuser = _host_hit("loader"), _device_hit("reuser", tree)

        self._tick(tree, [loader, reuser], kv_on_device=True)
        # The state DMA was submitted, not left in the queue.
        self.assertEqual(tree.cache_controller.load_queue, [])
        self.assertEqual(loader.hicache_restore_status, PENDING)
        self.assertEqual(reuser.hicache_restore_status, PENDING)

        tree.finish_loads()
        self._tick(tree, [loader, reuser], kv_on_device=True)
        self.assertEqual(loader.hicache_restore_status, READY)
        self.assertEqual(reuser.hicache_restore_status, READY)

    def test_device_hit_waits_even_when_no_load_slot_is_free(self):
        tree = _FakeTreeCache()
        loader = _host_hit("loader")
        self._tick(tree, [loader])

        tree.free_slot = False
        late = _device_hit("late", tree)
        self._tick(tree, [loader, late])
        self.assertEqual(late.hicache_restore_status, PENDING)

    def test_device_hits_without_in_flight_load_are_ready_at_once(self):
        tree = _FakeTreeCache(on_device=True)
        rematched, prealloc_hit = _host_hit("rematched"), _device_hit("hit", tree)

        self._tick(tree, [rematched, prealloc_hit])

        self.assertEqual(rematched.hicache_restore_status, READY)
        self.assertEqual(prealloc_hit.hicache_restore_status, READY)
        self.assertEqual(rematched.hicache_restore_lock, "lock-node")


if __name__ == "__main__":
    unittest.main()
