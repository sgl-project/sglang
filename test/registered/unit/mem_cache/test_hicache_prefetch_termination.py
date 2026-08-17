"""Unit tests for HiCache prefetch termination policies."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiCachePrefetchTermination(unittest.TestCase):
    CACHE_TYPES = (HiRadixCache, UnifiedRadixCache)
    PAGE_SIZE = 16

    def _build_cache(self, cache_type, policy, timed_out):
        cache = object.__new__(cache_type)
        cache.page_size = self.PAGE_SIZE
        cache.prefetch_stop_policy = policy
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache.is_prefetch_timeout = lambda operation: timed_out
        cache._prefetch_timeout_check_linear_func = lambda operation: timed_out
        return cache

    def _build_operation(
        self,
        *,
        kv_completed,
        has_pool_transfers,
        pool_transfers_done,
        terminated=False,
    ):
        return SimpleNamespace(
            hash_value=["page"],
            completed_tokens=self.PAGE_SIZE if kv_completed else 0,
            pool_transfers=[object()] if has_pool_transfers else None,
            pool_transfers_done=pool_transfers_done,
            is_terminated=lambda: terminated,
        )

    def test_prefetch_termination_policy_matrix(self):
        cases = (
            ("best_effort", False, True, False, False, True),
            ("wait_complete", True, False, False, False, True),
            ("wait_complete", True, True, False, False, False),
            ("wait_complete", True, True, True, False, True),
            ("timeout", True, True, False, False, False),
            ("timeout", True, True, False, True, True),
            ("timeout", False, True, False, True, True),
            ("timeout", True, True, True, False, True),
            ("wait_complete", False, True, False, False, False),
        )

        for cache_type in self.CACHE_TYPES:
            for (
                policy,
                kv_completed,
                has_pool_transfers,
                pool_transfers_done,
                timed_out,
                expected,
            ) in cases:
                with self.subTest(
                    cache_type=cache_type.__name__,
                    policy=policy,
                    kv_completed=kv_completed,
                    has_pool_transfers=has_pool_transfers,
                    pool_transfers_done=pool_transfers_done,
                    timed_out=timed_out,
                ):
                    cache = self._build_cache(cache_type, policy, timed_out)
                    operation = self._build_operation(
                        kv_completed=kv_completed,
                        has_pool_transfers=has_pool_transfers,
                        pool_transfers_done=pool_transfers_done,
                    )
                    self.assertEqual(
                        cache.can_terminate_prefetch(operation), expected
                    )

    def test_terminated_operation_can_terminate_on_all_cache_types(self):
        for cache_type in self.CACHE_TYPES:
            with self.subTest(cache_type=cache_type.__name__):
                cache = self._build_cache(cache_type, "wait_complete", False)
                operation = self._build_operation(
                    kv_completed=False,
                    has_pool_transfers=True,
                    pool_transfers_done=False,
                    terminated=True,
                )
                self.assertTrue(cache.can_terminate_prefetch(operation))


if __name__ == "__main__":
    unittest.main()
