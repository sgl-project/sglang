"""CPU unit tests for the MiniMax-M3 decode IndexCache cadence policy.

These cover the pure decision logic that drives which sparse layers recompute
the lightning indexer versus reuse the cached block selection. The helpers are
dependency-free (no torch / GPU), so this runs on the CPU CI runner.
"""

import unittest

from sglang.srt.layers.attention.minimax_sparse_ops.indexcache import (
    indexcache_enabled,
    indexcache_layer_positions,
    indexcache_should_reuse,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIndexCacheEnabled(unittest.TestCase):
    def test_off_for_zero_and_one(self):
        # 0/1 = stock behavior (indexer every layer).
        self.assertFalse(indexcache_enabled(0))
        self.assertFalse(indexcache_enabled(1))

    def test_on_for_stride_gt_one(self):
        self.assertTrue(indexcache_enabled(2))
        self.assertTrue(indexcache_enabled(4))


class TestIndexCacheLayerPositions(unittest.TestCase):
    def test_positions_are_sorted_and_zero_based(self):
        # Unsorted ids must map to 0-based execution-order positions.
        self.assertEqual(
            indexcache_layer_positions([7, 3, 5, 1]),
            {1: 0, 3: 1, 5: 2, 7: 3},
        )

    def test_empty(self):
        self.assertEqual(indexcache_layer_positions([]), {})


class TestIndexCacheShouldReuse(unittest.TestCase):
    def test_disabled_never_reuses(self):
        for stride in (0, 1):
            for pos in range(4):
                self.assertFalse(
                    indexcache_should_reuse(pos, stride, has_cached_state=True)
                )

    def test_first_layer_without_state_recomputes(self):
        # Position 0 is a cadence layer and has no cached state yet -> recompute.
        self.assertFalse(indexcache_should_reuse(0, 4, has_cached_state=False))

    def test_cadence_pattern_stride_4(self):
        # With a warm cache: recompute at 0/4/8 (pos % stride == 0), reuse elsewhere.
        expected_reuse = {
            0: False,
            1: True,
            2: True,
            3: True,
            4: False,
            5: True,
            8: False,
        }
        for pos, want in expected_reuse.items():
            self.assertEqual(
                indexcache_should_reuse(pos, 4, has_cached_state=True),
                want,
                msg=f"pos={pos}",
            )

    def test_stride_2_alternates(self):
        self.assertFalse(indexcache_should_reuse(0, 2, has_cached_state=True))
        self.assertTrue(indexcache_should_reuse(1, 2, has_cached_state=True))
        self.assertFalse(indexcache_should_reuse(2, 2, has_cached_state=True))

    def test_reuse_requires_cached_state(self):
        # A non-cadence layer still recomputes if nothing has been cached.
        self.assertFalse(indexcache_should_reuse(1, 4, has_cached_state=False))


if __name__ == "__main__":
    unittest.main()
