"""Unit tests for the Mooncake direct linker's grouped, concurrent layer loads."""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
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
        sizes = [[SLICE] for _ in locations]
        offsets = [[layer * SLICE] for _ in locations]
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
            for key, key_ptrs, key_offsets in zip(keys, ptrs, offsets):
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


if __name__ == "__main__":
    unittest.main()
