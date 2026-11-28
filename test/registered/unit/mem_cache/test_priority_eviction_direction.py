"""
Unit tests for the priority direction of radix-cache eviction.

A request's priority enters the tree as an eviction sort key (larger =
evicted later). With --schedule-low-priority-values-first lower integer
values mean higher priority, so the key is order-reversed (~p = -p - 1,
a bijection on the signed int64 range) on the way in; the lowest-priority
prefix is then evicted first under both directions.

Usage:
    python test_priority_eviction_direction.py
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import unittest
import unittest.mock
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.test_utils import CustomTestCase


class TestPriorityEvictionDirection(CustomTestCase):
    """PriorityStrategy must evict the lowest-priority prefix first under
    both --schedule-low-priority-values_first directions."""

    HIGH_TOKENS = [1, 2, 3, 4]
    LOW_TOKENS = [5, 6, 7, 8]

    @staticmethod
    def _make_cache(low_first: bool) -> RadixCache:
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")
        mock_allocator.page_size = 1
        req_pool = SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int64).unsqueeze(0)
        )
        return RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=mock_allocator,
                page_size=1,
                eviction_policy="priority",
                schedule_low_priority_values_first=low_first,
            )
        )

    def _insert_req(self, cache: RadixCache, tokens, priority) -> None:
        """Insert through _insert_cache so the business priority passes the
        eviction-sort-key conversion (the code path a real request takes)."""
        req = SimpleNamespace(
            priority=priority,
            extra_key=None,
            cache_salt=None,
            kv=SimpleNamespace(
                req_pool_idx=0,
                cache_inserted_len=0,
                cache_protected_len=0,
                record_insert=lambda n: None,
            ),
        )
        cache._insert_cache(req, array("q", tokens))

    @staticmethod
    def _present(cache: RadixCache, tokens) -> bool:
        return cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).device_prefix_len == len(tokens)

    def _assert_low_evicted_high_kept(
        self, low_first: bool, high_prio: int, low_prio: int
    ) -> None:
        cache = self._make_cache(low_first)
        self._insert_req(cache, self.HIGH_TOKENS, priority=high_prio)
        self._insert_req(cache, self.LOW_TOKENS, priority=low_prio)

        cache.evict(EvictParams(num_tokens=len(self.LOW_TOKENS)))

        self.assertFalse(
            self._present(cache, self.LOW_TOKENS),
            f"low-priority prefix should be evicted first (low_first={low_first})",
        )
        self.assertTrue(
            self._present(cache, self.HIGH_TOKENS),
            f"high-priority prefix should survive (low_first={low_first})",
        )

    def test_default_direction_low_priority_evicted_first(self):
        """By default higher values are higher priority: 0 is evicted before 100."""
        self._assert_low_evicted_high_kept(low_first=False, high_prio=100, low_prio=0)

    def test_low_first_direction_low_priority_evicted_first(self):
        """With --schedule-low-priority-values-first, 100 is the low priority
        and must be evicted before 0 despite being the larger value."""
        self._assert_low_evicted_high_kept(low_first=True, high_prio=0, low_prio=100)

    def test_sentinel_priority_evicted_first_in_both_directions(self):
        """A request without an explicit priority is sentineled by the
        scheduler: -sys.maxsize-1 by default, +sys.maxsize under
        --schedule-low-priority-values-first. Both must sort as the lowest
        priority in the eviction order."""
        sentinel_cases = [
            (False, -sys.maxsize - 1, 50),
            (True, sys.maxsize, 50),
        ]
        for low_first, sentinel, other in sentinel_cases:
            with self.subTest(low_first=low_first):
                self._assert_low_evicted_high_kept(
                    low_first=low_first, high_prio=other, low_prio=sentinel
                )

    def test_int64_extremes_stay_in_sort_key_range(self):
        """The order reversal must stay inside the signed int64 range.
        ~p (= -p - 1) is a bijection, so INT64_MIN maps to INT64_MAX with
        no clamping and no value leaves the range."""
        int64_min, int64_max = -(1 << 63), (1 << 63) - 1

        # low_first: INT64_MIN is the highest priority; ~INT64_MIN = INT64_MAX,
        # the largest in-range sort key, without any clamping.
        cache = self._make_cache(low_first=True)
        self._insert_req(cache, self.HIGH_TOKENS, priority=int64_min)
        node = cache.root_node.children[1]
        self.assertEqual(node.priority, int64_max)

        # Default direction: INT64_MIN stays the smallest in-range sort key
        # and must be evicted before priority 0.
        cache = self._make_cache(low_first=False)
        self._insert_req(cache, self.HIGH_TOKENS, priority=int64_min)
        self._insert_req(cache, self.LOW_TOKENS, priority=0)
        cache.evict(EvictParams(num_tokens=len(self.HIGH_TOKENS)))
        self.assertFalse(self._present(cache, self.HIGH_TOKENS))
        self.assertTrue(self._present(cache, self.LOW_TOKENS))

    def test_int64_extremes_stay_distinct_under_reversal(self):
        """INT64_MIN and INT64_MIN + 1 are the two most urgent priorities
        under --schedule-low-priority-values-first and must not collapse
        onto the same sort key: a plain negation clamps both to INT64_MAX,
        and the LRU tiebreak then evicts whichever leaf is older — the more
        urgent one when it was inserted first."""
        int64_min = -(1 << 63)
        cache = self._make_cache(low_first=True)
        # The more urgent leaf is inserted first, so a key collision would
        # make the LRU tiebreak evict it first.
        self._insert_req(cache, self.HIGH_TOKENS, priority=int64_min)
        self._insert_req(cache, self.LOW_TOKENS, priority=int64_min + 1)

        high_node = cache.root_node.children[self.HIGH_TOKENS[0]]
        low_node = cache.root_node.children[self.LOW_TOKENS[0]]
        self.assertNotEqual(high_node.priority, low_node.priority)

        cache.evict(EvictParams(num_tokens=len(self.LOW_TOKENS)))
        self.assertFalse(
            self._present(cache, self.LOW_TOKENS),
            "INT64_MIN + 1 must be evicted before INT64_MIN under low_first",
        )
        self.assertTrue(
            self._present(cache, self.HIGH_TOKENS),
            "the most urgent priority (INT64_MIN) must survive the eviction",
        )

    def test_shared_prefix_protects_most_urgent_request_in_both_directions(self):
        """A shared prefix node aggregates the max sort key of its branches,
        so it is protected at the level of the most urgent request using it,
        and evicting the low-priority branch only removes its unique tokens."""
        # (low_first, high_prio, low_prio): under low_first 0 is the high priority.
        cases = [(False, 100, 0), (True, 0, 100)]
        for low_first, high_prio, low_prio in cases:
            with self.subTest(low_first=low_first):
                cache = self._make_cache(low_first=low_first)
                self._insert_req(cache, [1, 2, 3, 4], priority=high_prio)
                self._insert_req(cache, [1, 2, 3, 5], priority=low_prio)

                # The shared [1,2,3] node carries the max sort key, i.e. the
                # high-priority request's key — the sort-key space keeps the
                # max() propagation meaning "protect the highest priority".
                # ~x = -x - 1 reverses the order bijectively, so the max of
                # the reversed keys is the reversal of the min.
                shared = cache.root_node.children[1]
                if cache.priority_sign > 0:
                    expected = max(high_prio, low_prio)
                else:
                    expected = ~min(high_prio, low_prio)
                self.assertEqual(shared.priority, expected)

                # Evicting one token removes only the low-priority branch's
                # unique token; the shared prefix and the high-priority
                # branch survive.
                cache.evict(EvictParams(num_tokens=1))
                self.assertFalse(self._present(cache, [1, 2, 3, 5]))
                self.assertTrue(self._present(cache, [1, 2, 3, 4]))
                self.assertTrue(self._present(cache, [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
