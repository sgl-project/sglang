"""
Unit tests for the deferred unlock of LMCache MP-mode async stores.

In MP mode ``cache_finished_req`` submits the store to the LMCache daemon via
``store_kv_async`` and returns without waiting, leaving the radix node pinned.
``evict()`` then settles those futures and unpins the nodes. These tests cover
``LMCRadixCache._reconcile_mp_stores``, which owns that unpin:

1. Successful store    -> node unpinned, end_session fired, list drained.
2. Failed store (False) -> node still unpinned (logged, not raised).
3. Future raises        -> node still unpinned (logged, not raised).
4. Reconcile with nothing in flight is a no-op.

Cases 2 and 3 are the point of the file: a store failure that skipped
``dec_lock_ref`` would pin those KV slots for the lifetime of the process,
exactly when memory is already scarce.

Usage:
    python -m pytest test/registered/unit/mem_cache/test_lmc_mp_async_store.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

# LMCache is an optional dependency and the module raises at import time when
# it is absent; skip the whole file rather than fail collection.
try:
    from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import LMCRadixCache

    LMCACHE_AVAILABLE = True
except Exception:
    LMCACHE_AVAILABLE = False


class _FakeNode:
    """Minimal stand-in for TreeNode: only lock_ref is exercised here."""

    def __init__(self):
        self.lock_ref = 0


class _FakeFuture:
    """Stand-in for MessagingFuture with a scripted result()."""

    def __init__(self, result=True, exc=None):
        self._result = result
        self._exc = exc
        self.result_calls = 0

    def result(self, timeout=None):
        self.result_calls += 1
        if self._exc is not None:
            raise self._exc
        return self._result


def _make_cache():
    """Build an LMCRadixCache without running __init__.

    __init__ opens a real LMCache connector (MP needs a live daemon), so set
    only the attributes _reconcile_mp_stores touches.
    """
    import threading

    cache = object.__new__(LMCRadixCache)
    cache._node_lock = threading.Lock()
    cache._in_flight_mp_stores = []
    cache._mp_load_back_markers = {}
    cache.lmcache_connector = MagicMock()
    cache.dec_lock_ref = lambda node: setattr(node, "lock_ref", node.lock_ref - 1)
    return cache


@unittest.skipUnless(LMCACHE_AVAILABLE, "LMCache is not installed")
class TestReconcileMPStores(unittest.TestCase):
    def test_successful_store_unpins_and_ends_session(self):
        cache = _make_cache()
        node = _FakeNode()
        node.lock_ref = 1
        future = _FakeFuture(result=True)
        cache._mp_load_back_markers["req-0"] = object()
        cache._in_flight_mp_stores.append((node, future, "req-0"))

        cache._reconcile_mp_stores()

        self.assertEqual(future.result_calls, 1)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])
        self.assertNotIn("req-0", cache._mp_load_back_markers)
        cache.lmcache_connector.end_session.assert_called_once_with("req-0")

    def test_failed_store_still_unpins(self):
        cache = _make_cache()
        node = _FakeNode()
        node.lock_ref = 1
        cache._in_flight_mp_stores.append((node, _FakeFuture(result=False), "req-1"))

        cache._reconcile_mp_stores()

        # A failed store must not leak the lock, and must not raise on the
        # eviction path.
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])
        cache.lmcache_connector.end_session.assert_called_once_with("req-1")

    def test_raising_future_still_unpins(self):
        cache = _make_cache()
        node = _FakeNode()
        node.lock_ref = 1
        future = _FakeFuture(exc=RuntimeError("daemon died"))
        cache._in_flight_mp_stores.append((node, future, "req-2"))

        cache._reconcile_mp_stores()

        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])

    def test_multiple_stores_all_settled(self):
        cache = _make_cache()
        nodes = []
        for i in range(3):
            node = _FakeNode()
            node.lock_ref = 1
            nodes.append(node)
            # Middle store fails; the others must be unaffected.
            result = i != 1
            cache._in_flight_mp_stores.append(
                (node, _FakeFuture(result=result), f"req-{i}")
            )

        cache._reconcile_mp_stores()

        for node in nodes:
            self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])
        self.assertEqual(cache.lmcache_connector.end_session.call_count, 3)

    def test_empty_reconcile_is_noop(self):
        cache = _make_cache()

        cache._reconcile_mp_stores()

        self.assertEqual(cache._in_flight_mp_stores, [])
        cache.lmcache_connector.end_session.assert_not_called()

    def test_shared_node_unpinned_once_per_store(self):
        # Concurrent requests can share a prefix node, so lock_ref is a count:
        # two stores against one node must produce two dec_lock_ref calls.
        cache = _make_cache()
        node = _FakeNode()
        node.lock_ref = 2
        cache._in_flight_mp_stores.append((node, _FakeFuture(), "req-a"))
        cache._in_flight_mp_stores.append((node, _FakeFuture(), "req-b"))

        cache._reconcile_mp_stores()

        self.assertEqual(node.lock_ref, 0)


if __name__ == "__main__":
    unittest.main()
