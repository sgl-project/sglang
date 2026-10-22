"""Unit tests for the Mooncake direct linker's layer-wise load sessions."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

POOLS = ("kv", "index")


class _Store:
    def __init__(self, start):
        self.start = start
        self.ended = []

    def batch_get_session_start(self, keys):
        return self.start(keys)

    def batch_get_session_end(self, keys):
        self.ended.append(list(keys))


class _Counter:
    def __init__(self):
        self.completed = []
        self.failures = []

    def complete(self, index, layer):
        self.completed.append(layer)

    def fail(self, index, error):
        self.failures.append(error)


def _linker(start):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.storage = SimpleNamespace(
        store=_Store(start),
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, None),
        _tag_keys=lambda keys: keys,
    )
    pool = SimpleNamespace(
        prepare_locations=lambda host_indices: list(host_indices),
        get_prepared_layer_range_meta=lambda locations, layer: None,
    )
    linker.pools = {name: pool for name in POOLS}
    linker.num_layers = 2
    linker.layer_done_counter = _Counter()
    linker.session_executor = ThreadPoolExecutor(max_workers=len(POOLS))
    return linker


def _transfers():
    return [
        [
            SimpleNamespace(name=name, keys=[f"{name}-page"], host_indices=[0])
            for name in POOLS
        ]
    ]


class TestLayerWiseLoadSessions(CustomTestCase):
    def test_pool_session_starts_run_concurrently(self):
        # Each start waits for the other pool's start: a serial issue times out.
        both_started = threading.Barrier(len(POOLS), timeout=5)

        def start(keys):
            both_started.wait()
            return [0] * len(keys)

        linker = _linker(start)
        self.addCleanup(linker.session_executor.shutdown)
        linker.load_layer_wise(0, _transfers())

        self.assertEqual(linker.layer_done_counter.failures, [])
        self.assertEqual(linker.layer_done_counter.completed, [0, 1])
        self.assertCountEqual(
            linker.storage.store.ended, [[f"{name}-page"] for name in POOLS]
        )

    def test_failed_start_still_ends_the_sessions_that_started(self):
        def start(keys):
            return [1] if keys == ["index-page"] else [0]

        linker = _linker(start)
        self.addCleanup(linker.session_executor.shutdown)
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
            level="ERROR",
        ) as logs:
            linker.load_layer_wise(0, _transfers())

        self.assertIn("Mooncake layer-wise load batch failed", logs.output[0])
        self.assertEqual(linker.layer_done_counter.completed, [])
        self.assertEqual(len(linker.layer_done_counter.failures), 1)
        self.assertIn(
            "get session start failed", str(linker.layer_done_counter.failures[0])
        )
        self.assertEqual(linker.storage.store.ended, [["kv-page"]])


if __name__ == "__main__":
    unittest.main()
