"""Tests that update_extra_pool_hit_pages records both the total hit count and
the contiguous hit prefix, so an ALL_PAGES sidecar (DSA indexer) clamps to the
prefix (not sum) and never commits past a mid-run miss like [T, F, T]."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hicache_storage import PoolTransferResult  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestExtraPoolHitPrefix(unittest.TestCase):
    """update_extra_pool_hit_pages records sum AND contiguous prefix separately."""

    def test_full_hit_prefix_equals_count(self):
        r = PoolTransferResult.empty()
        r.update_extra_pool_hit_pages({"indexer": [True, True, True, True]})
        self.assertEqual(r.extra_pool_hit_pages["indexer"], 4)
        self.assertEqual(r.extra_pool_hit_prefix["indexer"], 4)

    def test_trailing_miss_prefix_stops_before_miss(self):
        r = PoolTransferResult.empty()
        r.update_extra_pool_hit_pages({"indexer": [True, True, False, False]})
        self.assertEqual(r.extra_pool_hit_pages["indexer"], 2)
        self.assertEqual(r.extra_pool_hit_prefix["indexer"], 2)

    def test_mid_hole_prefix_diverges_from_sum(self):
        """The whole point: a non-prefix-fail hole [T,F,T,T] sums to 3 but the
        usable contiguous prefix is 1."""
        r = PoolTransferResult.empty()
        r.update_extra_pool_hit_pages({"indexer": [True, False, True, True]})
        self.assertEqual(r.extra_pool_hit_pages["indexer"], 3)  # sum (SWA/mamba view)
        self.assertEqual(r.extra_pool_hit_prefix["indexer"], 1)  # prefix (DSA view)

    def test_leading_miss_prefix_zero(self):
        r = PoolTransferResult.empty()
        r.update_extra_pool_hit_pages({"indexer": [False, True, True, True]})
        self.assertEqual(r.extra_pool_hit_prefix["indexer"], 0)

    def test_empty_result_list_prefix_zero(self):
        r = PoolTransferResult.empty()
        r.update_extra_pool_hit_pages({"indexer": []})
        self.assertEqual(r.extra_pool_hit_prefix["indexer"], 0)

    def test_multiple_pools_tracked_independently(self):
        r = PoolTransferResult.empty()
        r.update_extra_pool_hit_pages(
            {"indexer": [True, False, True], "swa": [True, True, True]}
        )
        self.assertEqual(r.extra_pool_hit_prefix["indexer"], 1)
        self.assertEqual(r.extra_pool_hit_prefix["swa"], 3)
        # sum view preserved for both
        self.assertEqual(r.extra_pool_hit_pages["indexer"], 2)
        self.assertEqual(r.extra_pool_hit_pages["swa"], 3)


if __name__ == "__main__":
    unittest.main()
