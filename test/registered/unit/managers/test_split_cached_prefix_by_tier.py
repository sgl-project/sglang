"""Unit tests for split_cached_prefix_by_tier — #20865."""

import unittest

from sglang.srt.managers.schedule_batch import split_cached_prefix_by_tier
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSplitCachedPrefixByTier(unittest.TestCase):
    def test_device_host_storage_partition(self):
        self.assertEqual(split_cached_prefix_by_tier(100, 60, 40), (40, 20, 40))

    def test_storage_clamped_to_host_hit(self):
        self.assertEqual(split_cached_prefix_by_tier(50, 30, 100), (20, 0, 30))

    def test_no_host_or_storage(self):
        self.assertEqual(split_cached_prefix_by_tier(25, 0, 0), (25, 0, 0))

    def test_prefix_fully_on_host_tiers(self):
        self.assertEqual(split_cached_prefix_by_tier(40, 40, 15), (0, 25, 15))

    def test_host_hit_exceeds_prefix(self):
        self.assertEqual(split_cached_prefix_by_tier(10, 25, 5), (0, 20, 5))

    def test_zero_prefix(self):
        self.assertEqual(split_cached_prefix_by_tier(0, 0, 0), (0, 0, 0))

    def test_storage_covers_all_host_hit(self):
        self.assertEqual(split_cached_prefix_by_tier(80, 50, 50), (30, 0, 50))


if __name__ == "__main__":
    unittest.main()
