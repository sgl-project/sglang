"""Unit tests for RadixCache.evict_dead_namespaces (weight-version KV
namespace sweep): dead extra_key namespaces are evicted wholesale, live and
unversioned namespaces survive, locked subtrees are skipped and reclaimed by a
later sweep, and size accounting stays consistent. Pure CPU via
RadixCache.create_simulated."""

import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_cache() -> RadixCache:
    return RadixCache.create_simulated(mock_allocator=MagicMock())


def _insert(cache: RadixCache, token_ids, extra_key=None) -> None:
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids), extra_key),
            value=torch.tensor(token_ids, dtype=torch.int64),
        )
    )


def _matched_len(cache: RadixCache, token_ids, extra_key=None) -> int:
    result = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids), extra_key))
    )
    return len(result.device_indices)


class TestWeightVersionNamespaceSweep(CustomTestCase):
    def test_dead_namespace_is_evicted_live_ones_survive(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3], "wv1;")
        _insert(cache, [1, 2, 3], "wv2;")
        _insert(cache, [4, 5, 6])  # unversioned

        num_evicted = cache.evict_dead_namespaces(lambda extra_key: extra_key != "wv1;")

        self.assertEqual(num_evicted, 3)
        self.assertEqual(_matched_len(cache, [1, 2, 3], "wv1;"), 0)
        self.assertEqual(_matched_len(cache, [1, 2, 3], "wv2;"), 3)
        self.assertEqual(_matched_len(cache, [4, 5, 6]), 3)
        self.assertEqual(cache.evictable_size(), 6)
        self.assertEqual(cache.total_size(), 6)

    def test_branched_namespace_subtree_is_fully_removed(self):
        cache = _make_cache()
        # Shared prefix [1, 2] splits into two branches within the namespace.
        _insert(cache, [1, 2, 3, 4], "wv1;")
        _insert(cache, [1, 2, 7, 8], "wv1;")
        self.assertEqual(cache.total_size(), 6)

        num_evicted = cache.evict_dead_namespaces(lambda extra_key: False)

        self.assertEqual(num_evicted, 6)
        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(len(cache.root_node.children), 0)

    def test_locked_subtree_survives_and_sweeps_after_unlock(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3], "wv1;")
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "wv1;"))
        )
        cache.inc_lock_ref(result.last_device_node)

        num_evicted = cache.evict_dead_namespaces(lambda extra_key: False)
        self.assertEqual(num_evicted, 0)
        self.assertEqual(_matched_len(cache, [1, 2, 3], "wv1;"), 3)

        cache.dec_lock_ref(result.last_device_node)
        num_evicted = cache.evict_dead_namespaces(lambda extra_key: False)
        self.assertEqual(num_evicted, 3)
        self.assertEqual(_matched_len(cache, [1, 2, 3], "wv1;"), 0)

    def test_locked_branch_does_not_block_unlocked_sibling(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3, 4], "wv1;")
        _insert(cache, [1, 2, 7, 8], "wv1;")
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4]), "wv1;"))
        )
        cache.inc_lock_ref(result.last_device_node)

        num_evicted = cache.evict_dead_namespaces(lambda extra_key: False)

        # The unlocked sibling branch [7, 8] is evicted; the locked chain
        # [1, 2] -> [3, 4] survives.
        self.assertEqual(num_evicted, 2)
        self.assertEqual(_matched_len(cache, [1, 2, 3, 4], "wv1;"), 4)
        self.assertEqual(_matched_len(cache, [1, 2, 7, 8], "wv1;"), 2)

    def test_disabled_cache_is_noop(self):
        cache = RadixCache.create_simulated(disable=True, mock_allocator=MagicMock())
        self.assertEqual(cache.evict_dead_namespaces(lambda extra_key: False), 0)


if __name__ == "__main__":
    unittest.main()
