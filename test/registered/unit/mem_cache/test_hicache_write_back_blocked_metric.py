"""sglang:hicache_write_back_backup_blocked_tokens_total counts device KV a
write-back eviction could neither back up (the host tier is full and its only
host-only node is pinned) nor drop (that pinned node sits in the victim's
subtree), on both tree cores."""

import unittest
from collections import Counter

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    PAGE,
    insert_seq,
    match_seq,
    write_back_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CacheMetrics:
    """The cache counters behind the blocked and dropped token metrics."""

    def __init__(self):
        self.totals = Counter()

    def increment_write_back_backup_blocked_tokens(self, num_tokens):
        self.totals["blocked"] += num_tokens

    def increment_dropped_tokens(self, num_tokens, reason, pool):
        self.totals["dropped", reason] += num_tokens

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _WriteBackBackupBlockedMetric:
    tree_core_backend: str

    def test_pinned_subtree_under_full_host_counts_each_blocked_attempt(self):
        cache, allocator, host_pool = write_back_cache(
            self.tree_core_backend, device_tokens=16 * PAGE, host_tokens=4 * PAGE
        )
        metrics = cache.metrics_collector = _CacheMetrics()
        parent_seq = [1000 + t for t in range(2 * PAGE)]
        child_seq = parent_seq + [2000 + t for t in range(PAGE)]
        insert_seq(cache, allocator, parent_seq)
        insert_seq(cache, allocator, child_seq)
        parent = match_seq(cache, parent_seq).last_device_node
        child = match_seq(cache, child_seq).last_device_node

        # Evicting the only leaf backs the child up and leaves it host-only
        # under the still-unbacked device parent.
        cache.evict(EvictParams(num_tokens=PAGE))
        self.assertTrue(cache.tree_core.is_full_device_evicted(child))
        self.assertFalse(cache.tree_core.is_backuped(parent))

        # Pin the child and fill the host tier: the parent's backup finds no
        # host room and its drop fallback is declined by the pinned child.
        child_lock = cache.inc_host_lock_ref(child).to_dec_params()
        self.assertIsNotNone(host_pool.alloc(host_pool.available_size()))
        for attempt in (1, 2):
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
            self.assertEqual(result.num_tokens_evicted, 0)
            self.assertEqual(metrics.totals["blocked"], attempt * len(parent_seq))
        self.assertEqual(metrics.totals["dropped", "host_pressure"], 0)

        # Unpinned, the drop fallback reclaims the parent and nothing more is
        # counted as blocked.
        cache.dec_host_lock_ref(child, child_lock)
        result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(parent_seq))
        self.assertEqual(metrics.totals["blocked"], 2 * len(parent_seq))
        self.assertEqual(metrics.totals["dropped", "host_pressure"], len(parent_seq))
        cache.sanity_check()


class TestWriteBackBackupBlockedMetricRust(
    _WriteBackBackupBlockedMetric, CustomTestCase
):
    tree_core_backend = "rust"


class TestWriteBackBackupBlockedMetricPython(
    _WriteBackBackupBlockedMetric, CustomTestCase
):
    tree_core_backend = "python"


if __name__ == "__main__":
    unittest.main()
