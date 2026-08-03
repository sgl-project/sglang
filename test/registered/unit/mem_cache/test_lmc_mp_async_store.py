"""
Unit tests for the deferred unlock of LMCache MP-mode async stores.

In MP mode ``cache_finished_req`` submits the store to the LMCache daemon via
``store_kv_async`` and returns without waiting, leaving the radix node pinned.
The pin is released in two places:

* ``_reap_completed_mp_stores`` -- non-blocking, called from
  ``evictable_size()`` so pins are released *before* the scheduler decides a
  batch does not fit.
* ``_reconcile_mp_stores`` -- blocking, called from ``evict()`` as the backstop
  when the caller is about to choose eviction victims.

``test_evictable_size_releases_finished_stores`` is the regression test for a
livelock observed under sustained load: pinning a node moves its bytes out of
the evictable total, the scheduler derives its token budget from that total,
and eviction only runs when an allocation falls short -- so releasing pins only
from eviction is circular. The server stayed up, answered health checks, and
scheduled nothing.

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
    """Minimal stand-in for TreeNode: only lock_ref and byte size matter."""

    def __init__(self, size=1):
        self.lock_ref = 0
        self.size = size


class _FakeFuture:
    """Stand-in for MessagingFuture with a scripted query() and result()."""

    def __init__(self, result=True, exc=None, done=True):
        self._result = result
        self._exc = exc
        self._done = done
        self.result_calls = 0
        self.query_calls = 0

    def query(self):
        self.query_calls += 1
        return self._done

    def result(self, timeout=None):
        self.result_calls += 1
        if self._exc is not None:
            raise self._exc
        return self._result


def _make_cache():
    """Build an LMCRadixCache without running __init__.

    __init__ opens a real LMCache connector (MP needs a live daemon), so set
    only the attributes the reap/reconcile paths touch. dec_lock_ref mirrors
    RadixCache's bookkeeping: unpinning returns the node's bytes to
    ``evictable_size_``, which is what the scheduler budget reads.
    """
    import threading

    cache = object.__new__(LMCRadixCache)
    cache._node_lock = threading.Lock()
    cache._in_flight_mp_stores = []
    cache._mp_load_back_markers = {}
    cache.evictable_size_ = 0
    cache.lmcache_connector = MagicMock()

    def _dec_lock_ref(node):
        node.lock_ref -= 1
        if node.lock_ref == 0:
            cache.evictable_size_ += node.size

    cache.dec_lock_ref = _dec_lock_ref
    return cache


def _pin(cache, node, future):
    """Simulate what cache_finished_req does to a node in MP mode."""
    node.lock_ref += 1
    cache._in_flight_mp_stores.append((node, future))


@unittest.skipUnless(LMCACHE_AVAILABLE, "LMCache is not installed")
class TestReapCompletedMPStores(unittest.TestCase):
    """The non-blocking reap called from evictable_size()."""

    def test_evictable_size_releases_finished_stores(self):
        """A finished store must reach the scheduler budget without eviction.

        Pre-fix, the unpin ran only from ``evict()``, which the scheduler
        reaches only after an allocation falls short -- so a finished store
        stayed invisible to the budget that decides whether to allocate at all.
        """
        cache = _make_cache()
        node = _FakeNode(size=64)
        _pin(cache, node, _FakeFuture(done=True))

        self.assertEqual(cache.evictable_size(), 64)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])

    def test_unfinished_store_stays_pinned_without_blocking(self):
        """Still-copying stores must not be waited on from the budget path."""
        cache = _make_cache()
        node = _FakeNode(size=64)
        future = _FakeFuture(done=False)
        _pin(cache, node, future)

        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(node.lock_ref, 1)
        self.assertEqual(cache._in_flight_mp_stores, [(node, future)])
        # result() blocks; reaching it from evictable_size() would stall the
        # scheduler on another process every step.
        self.assertEqual(future.result_calls, 0)

    def test_reap_queries_each_future_once(self):
        """Two query() calls per entry race the daemon and can drop the entry.

        Partitioning with two comprehensions lets a future that completes
        between them fall into neither list, leaking the pin permanently.
        """
        cache = _make_cache()
        future = _FakeFuture(done=False)
        _pin(cache, _FakeNode(), future)

        cache._reap_completed_mp_stores()

        self.assertEqual(future.query_calls, 1)

    def test_reap_settles_only_the_finished_ones(self):
        cache = _make_cache()
        done_node, pending_node = _FakeNode(size=8), _FakeNode(size=8)
        pending_future = _FakeFuture(done=False)
        _pin(cache, done_node, _FakeFuture(done=True))
        _pin(cache, pending_node, pending_future)

        cache._reap_completed_mp_stores()

        self.assertEqual(done_node.lock_ref, 0)
        self.assertEqual(pending_node.lock_ref, 1)
        self.assertEqual(cache._in_flight_mp_stores, [(pending_node, pending_future)])

    def test_failed_store_still_unpins(self):
        """A store the daemon reports as failed must not leak its pin."""
        cache = _make_cache()
        node = _FakeNode()
        _pin(cache, node, _FakeFuture(result=False, done=True))

        cache._reap_completed_mp_stores()

        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])

    def test_raising_future_still_unpins(self):
        """A dead daemon must not raise on a path the scheduler runs each step."""
        cache = _make_cache()
        node = _FakeNode()
        _pin(cache, node, _FakeFuture(exc=RuntimeError("daemon died"), done=True))

        cache._reap_completed_mp_stores()

        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])


@unittest.skipUnless(LMCACHE_AVAILABLE, "LMCache is not installed")
class TestReconcileMPStores(unittest.TestCase):
    """The blocking backstop called from evict()."""

    def test_reconcile_settles_even_unfinished_stores(self):
        """evict() is about to pick victims, so it waits rather than deferring."""
        cache = _make_cache()
        node = _FakeNode()
        future = _FakeFuture(done=False)
        _pin(cache, node, future)

        cache._reconcile_mp_stores()

        self.assertEqual(future.result_calls, 1)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])

    def test_one_failed_store_does_not_strand_the_others(self):
        cache = _make_cache()
        nodes = []
        for i in range(3):
            node = _FakeNode()
            nodes.append(node)
            # Middle store fails; the others must still settle.
            _pin(cache, node, _FakeFuture(result=i != 1))

        cache._reconcile_mp_stores()

        for node in nodes:
            self.assertEqual(node.lock_ref, 0)
        self.assertEqual(cache._in_flight_mp_stores, [])

    def test_shared_node_unpinned_once_per_store(self):
        """lock_ref is a count, not a flag: concurrent requests share prefixes.

        Unpinning a shared node once per *node* rather than once per *store*
        would free slots another in-flight store is still copying from.
        """
        cache = _make_cache()
        node = _FakeNode()
        _pin(cache, node, _FakeFuture())
        _pin(cache, node, _FakeFuture())
        self.assertEqual(node.lock_ref, 2)

        cache._reconcile_mp_stores()

        self.assertEqual(node.lock_ref, 0)


if __name__ == "__main__":
    unittest.main()
