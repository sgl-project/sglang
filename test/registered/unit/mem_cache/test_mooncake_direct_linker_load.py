"""Unit tests for the Mooncake direct linker's grouped, concurrent layer loads."""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    PrefetchStatus,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SLICE = 10


class FakePool:
    def __init__(self, skip_layers=()):
        self.skip_layers = set(skip_layers)

    def prepare_locations(self, host_indices):
        return list(host_indices)

    def get_prepared_layer_range_meta(self, locations, layer):
        if layer in self.skip_layers:
            return None
        ptrs = [[(row, layer)] for row in locations]
        # DevicePoolEntry shares these read-only rows across locations.
        sizes = [[SLICE]] * len(locations)
        offsets = [[layer * SLICE]] * len(locations)
        return ptrs, sizes, offsets


class FakeStore:
    def __init__(self, delay=None, fail_key=None):
        self.delay = delay or (lambda keys: 0)
        self.fail_key = fail_key
        self.lock = threading.Lock()
        self.ranges = []
        self.calls = []
        self.events = []

    def batch_get_session_start(self, keys):
        return [0] * len(keys)

    def batch_get_session_end(self, keys):
        with self.lock:
            self.events.append(("end", time.monotonic()))

    def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
        time.sleep(self.delay(keys))
        with self.lock:
            self.calls.append(list(keys))
            for key, key_ptrs, key_sizes, key_offsets in zip(
                keys, ptrs, sizes, offsets
            ):
                assert len(key_ptrs) == len(key_sizes) == len(key_offsets)
                assert all(size == SLICE for size in key_sizes)
                for ptr, offset in zip(key_ptrs, key_offsets):
                    self.ranges.append((key, ptr, offset))
            self.events.append(("get", time.monotonic()))
        if self.fail_key in keys:
            return [0] * len(keys)
        return [sum(item) for item in sizes]


class FakeStorage:
    def __init__(self, store):
        self.store = store

    def _get_hybrid_page_component_keys(self, keys, transfer):
        return keys, None

    def _tag_keys(self, keys):
        return keys


class RecordingCounter(LayerWiseLoadCounter):
    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.order = []

    def complete(self, index, layer):
        self.order.append(layer)
        super().complete(index, layer)


def make_linker(store, *, num_layers, layer_group, threads, pools):
    linker = object.__new__(MooncakeDirectLinker)
    linker.num_layers = num_layers
    linker.load_layer_group = layer_group
    linker.load_threads = threads
    linker.load_io_executor = ThreadPoolExecutor(threads)
    linker.pools = pools
    linker.storage = FakeStorage(store)
    linker.layer_done_counter = RecordingCounter(num_layers)
    return linker


def transfer(name, keys):
    return PoolTransfer(name=name, keys=keys, host_indices=list(range(len(keys))))


def load(linker, request_transfers):
    index = linker.layer_done_counter.update_producer()
    linker.load_layer_wise(index, request_transfers)
    return index


class TestGroupedLoad(CustomTestCase):
    def test_every_layer_range_is_read_once_per_key(self):
        store = FakeStore()
        linker = make_linker(
            store,
            num_layers=5,
            layer_group=2,
            threads=2,
            pools={PoolName.KV: FakePool(), PoolName.INDEXER: FakePool({1, 2})},
        )
        self.addCleanup(linker.load_io_executor.shutdown)
        keys = [f"k{i}" for i in range(3)]
        index = load(
            linker,
            [[transfer(PoolName.KV, keys), transfer(PoolName.INDEXER, keys[:2])]],
        )
        self.assertTrue(linker.layer_done_counter.finish(index))
        expected = [
            (key, (row, layer), layer * SLICE)
            for row, key in enumerate(keys)
            for layer in range(5)
        ] + [
            (key, (row, layer), layer * SLICE)
            for row, key in enumerate(keys[:2])
            for layer in (0, 3, 4)
        ]
        self.assertCountEqual(store.ranges, expected)
        # Groups of two layers carry two ranges per key; keys split over two threads.
        self.assertTrue(all(len(call) <= 2 for call in store.calls))
        self.assertEqual(linker.layer_done_counter.order, [0, 1, 2, 3, 4])

    def test_layers_complete_in_order_when_later_groups_finish_first(self):
        store = FakeStore(delay=lambda keys: 0.05 if keys == ["k0"] else 0)
        linker = make_linker(
            store,
            num_layers=4,
            layer_group=1,
            threads=4,
            pools={PoolName.KV: FakePool()},
        )
        self.addCleanup(linker.load_io_executor.shutdown)
        index = load(linker, [[transfer(PoolName.KV, ["k0"])]])
        self.assertTrue(linker.layer_done_counter.finish(index))
        self.assertEqual(linker.layer_done_counter.order, [0, 1, 2, 3])

    def test_failure_waits_for_inflight_reads_before_releasing_consumers(self):
        """Failed loads must stop writing before consumers can recycle their rows."""
        for failure in ("read", "metadata", "submit"):
            with self.subTest(failure=failure):
                self._check_failure_drains_reads(failure)

    def _check_failure_drains_reads(self, failure):
        slow_started = threading.Event()
        release_slow = threading.Event()
        failure_triggered = threading.Event()

        def fail():
            if not slow_started.wait(5):
                raise TimeoutError("Slow read did not start")
            failure_triggered.set()
            raise RuntimeError("Injected load failure")

        class BlockingStore(FakeStore):
            def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
                if keys == ["slow"]:
                    slow_started.set()
                    if not release_slow.wait(5):
                        raise TimeoutError("Slow read was not released")
                else:
                    fail()
                return super().batch_get_into_multi_buffer_ranges(
                    keys, ptrs, sizes, offsets
                )

        class FailingPool(FakePool):
            def get_prepared_layer_range_meta(self, locations, layer):
                fail()

        store = BlockingStore()
        pools = {PoolName.KV: FakePool()}
        keys = ["bad", "slow"] if failure == "read" else ["slow", "bad"]
        transfers = [transfer(PoolName.KV, keys)]
        if failure == "metadata":
            pools[PoolName.INDEXER] = FailingPool()
            transfers = [
                transfer(PoolName.KV, ["slow"]),
                transfer(PoolName.INDEXER, ["bad"]),
            ]
        linker = make_linker(store, num_layers=1, layer_group=1, threads=2, pools=pools)
        self.addCleanup(linker.load_io_executor.shutdown)
        submit = linker.load_io_executor.submit

        def submit_or_fail(fn, name, keys, *args):
            if failure == "submit" and keys == ["bad"]:
                fail()
            return submit(fn, name, keys, *args)

        index = linker.layer_done_counter.update_producer()
        with patch.object(linker.load_io_executor, "submit", submit_or_fail):
            with ThreadPoolExecutor(2) as consumers:
                loading = consumers.submit(linker.load_layer_wise, index, [transfers])
                finishing = consumers.submit(linker.finish_layer_wise_loading, index)
                try:
                    self.assertTrue(failure_triggered.wait(5))
                    # The failed request still owns rows that the slow read writes.
                    with self.assertRaises(TimeoutError):
                        finishing.result(timeout=0.1)
                    self.assertFalse(any(kind == "end" for kind, _ in store.events))
                finally:
                    release_slow.set()
                loading.result(timeout=5)
                self.assertFalse(finishing.result(timeout=5))

        self.assertEqual(linker.layer_done_counter.order, [])
        last_get = max(t for kind, t in store.events if kind == "get")
        first_end = min(t for kind, t in store.events if kind == "end")
        self.assertGreater(first_end, last_get)


class _FakeReplicaDesc:
    def __init__(self, *, memory: bool, status: int = 3):
        self._memory = memory
        self.status = status

    def is_memory_replica(self) -> bool:
        return self._memory


class TestMooncakeStoreMemoryPrefetch(CustomTestCase):
    COMPLETE = 3
    PROCESSING = 2
    FAILED = 5

    def _make_linker_with_storage(self, replicas_by_key, *, exist_options_cls=None):
        import torch

        calls = {"exist": [], "replicas": []}

        class _StoreClient:
            def batch_is_exist(self, keys, *args):
                calls["exist"].append((list(keys), args))
                return [1] * len(keys)

            def batch_get_replica_desc(self, keys):
                calls["replicas"].append(list(keys))
                return {k: replicas_by_key[k] for k in keys if k in replicas_by_key}

        storage = MooncakeStore.__new__(MooncakeStore)
        storage.store = _StoreClient()
        storage.config_prefix = "model"
        storage.mla_suffix = "tp0_cp0_pp0"
        storage.mha_suffix = "tp0_cp0_pp0"
        storage._replica_complete_status = self.COMPLETE
        storage._replica_terminal_statuses = (self.FAILED,)
        if exist_options_cls is not None:
            opts = exist_options_cls()
            opts.prefetch_to_memory = True
            storage._prefetch_exist_options = opts

        kv_pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[torch.zeros((8, 4), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=1,
            rows_are_pages=False,
        )
        swa_pool = DevicePoolEntry(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            device_pool=None,
            components=[[torch.zeros((8, 4), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=1,
            rows_are_pages=False,
        )
        pool_group = DevicePoolGroup([kv_pool, swa_pool], num_layers=1, page_size=1)
        storage.mem_pool_host = pool_group
        storage.registered_pools = pool_group.entry_map

        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.load_mode = "layer_wise"
        linker.pool_group = pool_group
        linker.pools = pool_group.entry_map
        linker.storage = storage
        linker.pending_loads = {}
        linker.pending_prefetches = {}
        linker.broadcast_loads = {}
        linker.stats = {"lookup": 0, "load": 0, "offload": 0}
        return linker, calls

    def test_lookup_prefetch_passes_exist_options_and_prunes_promoted_keys(self):
        """Subsequent readiness polls must query only keys not yet in COMPLETE MEMORY."""

        class _ExistOptions:
            def __init__(self):
                self.prefetch_to_memory = False

        k0 = "model_p0_tp0_cp0_pp0_k"
        k1 = "model_p1_tp0_cp0_pp0_k"
        k2 = "model_p2_tp0_cp0_pp0_k"
        s1 = "model_p1_tp0_cp0_pp0_swa"
        s2 = "model_p2_tp0_cp0_pp0_swa"
        replicas = {
            k0: [_FakeReplicaDesc(memory=True, status=self.COMPLETE)],
            k1: [
                _FakeReplicaDesc(memory=False, status=self.COMPLETE),
                _FakeReplicaDesc(memory=True, status=self.PROCESSING),
            ],
            k2: [_FakeReplicaDesc(memory=False, status=self.COMPLETE)],
            s1: [_FakeReplicaDesc(memory=True, status=self.COMPLETE)],
            s2: [_FakeReplicaDesc(memory=False, status=self.COMPLETE)],
        }
        linker, calls = self._make_linker_with_storage(
            replicas, exist_options_cls=_ExistOptions
        )
        transfers = [
            PoolTransfer(name=PoolName.KV, keys=["p0", "p1", "p2"]),
            PoolTransfer(
                name=PoolName.SWA,
                keys=["p2"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        # Restrict restorable boundary to 2 pages so p2 keys are not tracked,
        # and trailing SWA window of 1 page selects only p1 (s1), not p0.
        with patch.object(
            linker.storage,
            "batch_exists_v2",
            wraps=linker.storage.batch_exists_v2,
        ):
            orig_exist = linker.storage._batch_exist
            linker.storage._batch_exist = lambda keys, prefetch_to_memory=False: [
                0 if key.startswith("model_p2_") else r
                for key, r in zip(
                    keys, orig_exist(keys, prefetch_to_memory=prefetch_to_memory)
                )
            ]
            restorable = linker.lookup("r1", transfers, prefetch=True)

        self.assertEqual(restorable, [1, 2])
        self.assertEqual(len(calls["exist"]), 1)
        _, exist_args = calls["exist"][0]
        self.assertEqual(len(exist_args), 1)
        self.assertTrue(exist_args[0].prefetch_to_memory)
        # Tracked keys at 2-page boundary: KV [k0, k1] + trailing SWA [s1].
        self.assertEqual(linker.pending_prefetches["r1"], [k0, k1, s1])

        # Poll 1: k0 and s1 are COMPLETE MEMORY -> pruned; k1 is PROCESSING -> kept.
        self.assertEqual(linker.prefetch_status("r1"), PrefetchStatus.PENDING)
        self.assertEqual(calls["replicas"][-1], [k0, k1, s1])
        self.assertEqual(linker.pending_prefetches["r1"], [k1])

        # Poll 2: queries ONLY k1; once k1 becomes COMPLETE MEMORY, rid retires.
        replicas[k1] = [_FakeReplicaDesc(memory=True, status=self.COMPLETE)]
        self.assertEqual(linker.prefetch_status("r1"), PrefetchStatus.READY)
        self.assertEqual(calls["replicas"][-1], [k1])
        self.assertNotIn("r1", linker.pending_prefetches)

        # Non-prefetch lookup does not pass ExistOptions or record pending_prefetches.
        linker.lookup("r2", transfers, prefetch=False)
        self.assertEqual(calls["exist"][-1][1], ())
        self.assertNotIn("r2", linker.pending_prefetches)

    def test_evicted_or_failed_replicas_do_not_block_and_lifecycle_cleans_up(self):
        """Evicted or terminal-failed keys must report READY instead of stalling admission."""
        k0 = "model_p0_tp0_cp0_pp0_k"
        k1 = "model_p1_tp0_cp0_pp0_k"
        replicas = {
            k0: [],
            k1: [_FakeReplicaDesc(memory=False, status=self.FAILED)],
        }
        linker, _ = self._make_linker_with_storage(replicas)
        linker.storage.batch_exists_v2 = (
            lambda keys, expanded, prefetch_to_memory=False: PoolTransferResult(
                2, {PoolName.KV: 2}, [1, 2]
            )
        )
        transfers = [
            PoolTransfer(name=PoolName.KV, keys=["p0", "p1"]),
            PoolTransfer(
                name=PoolName.SWA, keys=["p1"], hit_policy=PoolHitPolicy.TRAILING_PAGES
            ),
        ]
        linker.lookup("missing", transfers, prefetch=True)
        self.assertEqual(linker.prefetch_status("missing"), PrefetchStatus.READY)
        self.assertEqual(linker.pending_prefetches, {})

        # Active pending key is cleared by load(), cancel_prefetch(), and reset().
        replicas[k0] = [_FakeReplicaDesc(memory=False, status=self.COMPLETE)]
        linker.lookup("load-rid", transfers, prefetch=True)
        self.assertEqual(linker.prefetch_status("load-rid"), PrefetchStatus.PENDING)
        self.assertTrue(
            linker.load(
                "load-rid",
                [PoolTransfer(name=PoolName.KV, keys=["p0"], device_indices=None)],
            )
        )
        self.assertNotIn("load-rid", linker.pending_prefetches)

        linker.lookup("cancel-rid", transfers, prefetch=True)
        linker.cancel_prefetch("cancel-rid")
        self.assertEqual(linker.prefetch_status("cancel-rid"), PrefetchStatus.READY)


if __name__ == "__main__":
    unittest.main()
