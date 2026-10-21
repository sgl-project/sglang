"""Unit tests for the Mooncake direct linker's prefetched existence answers."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from unittest.mock import patch

from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    EXIST_CACHE_TTL_S,
    MooncakeDirectLinker,
    _ExistenceCache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Fetcher:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, keys):
        self.calls.append(list(keys))
        return [self.answers[key] for key in keys]


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TestExistenceCache(CustomTestCase):
    def setUp(self):
        self.clock = _Clock()
        patcher = patch.object(mooncake_direct_linker.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fetches_only_keys_without_a_fresh_answer(self):
        cache = _ExistenceCache(ttl_s=5.0)
        fetch = _Fetcher({"a": 1, "b": 0, "c": 1})
        cache.fetch_and_store(["a", "b"], fetch)

        self.clock.now += 4.9
        self.assertEqual(cache.query(["a", "b", "c"], fetch), [1, 0, 1])
        self.assertEqual(fetch.calls[-1], ["c"])

        # "a" and "b" were stamped before "c" and expire first.
        self.clock.now += 0.2
        self.assertEqual(cache.query(["a", "b", "c"], fetch), [1, 0, 1])
        self.assertEqual(fetch.calls[-1], ["a", "b"])

    def test_rpc_errors_are_never_answered_from_the_cache(self):
        cache = _ExistenceCache(ttl_s=5.0)
        fetch = _Fetcher({"a": -1, "b": 1})
        self.assertEqual(cache.query(["a", "b"], fetch), [-1, 1])

        fetch.answers["a"] = 1
        self.assertEqual(cache.query(["a", "b"], fetch), [1, 1])
        self.assertEqual(fetch.calls[-1], ["a"])

    def test_answer_outlives_neither_the_mooncake_lease_nor_its_refresh(self):
        """A cached hit stands in for the lease the RPC took: Mooncake keeps a key
        for 10 s (master default_kv_lease_ttl) after batch_is_exist reports it."""
        self.assertLess(EXIST_CACHE_TTL_S, 10.0)
        self.assertLess(
            MooncakeDirectLinker.lookup_prefetch_refresh_s, EXIST_CACHE_TTL_S
        )


class TestResetDrainsPrefetch(CustomTestCase):
    def test_prefetch_in_flight_at_reset_cannot_refill_the_cache(self):
        """A flush clears the store; an answer fetched before it must not survive."""
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pending_loads = {}
        linker.exist_cache = _ExistenceCache(EXIST_CACHE_TTL_S)
        linker.lookup_prefetcher = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(linker.lookup_prefetcher.shutdown)
        linker.load_queue = Queue()
        linker.offload_queue = Queue()
        linker.offload_results = Queue()
        linker.broadcast_loads = {}
        linker.layer_done_counter = type("Counter", (), {"reset": lambda self: None})()

        fetch_started, release_fetch = threading.Event(), threading.Event()

        def blocked_fetch(keys):
            fetch_started.set()
            release_fetch.wait()
            return [1] * len(keys)

        linker.lookup_prefetcher.submit(
            linker.exist_cache.fetch_and_store, ["page"], blocked_fetch
        )
        self.assertTrue(fetch_started.wait(timeout=5))
        reset_done = threading.Event()
        resetter = threading.Thread(target=lambda: (linker.reset(), reset_done.set()))
        resetter.start()
        self.assertFalse(reset_done.wait(timeout=0.2))

        release_fetch.set()
        resetter.join(timeout=5)
        self.assertTrue(reset_done.is_set())
        self.assertEqual(linker.exist_cache.entries, {})


if __name__ == "__main__":
    unittest.main()
