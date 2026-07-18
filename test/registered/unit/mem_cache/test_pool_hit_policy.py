import unittest

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    find_prefix_hit_boundary,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPoolHitPolicy(unittest.TestCase):
    def test_trailing_window_rejects_short_prefix(self):
        self.assertEqual(
            find_prefix_hit_boundary(
                [True, True], PoolHitPolicy.TRAILING_PAGES, trailing_pages=3
            ),
            0,
        )

    def test_trailing_window_accepts_complete_window(self):
        self.assertEqual(
            find_prefix_hit_boundary(
                [True, True, True], PoolHitPolicy.TRAILING_PAGES, trailing_pages=3
            ),
            3,
        )

    def test_trailing_window_finds_longest_complete_suffix(self):
        self.assertEqual(
            find_prefix_hit_boundary(
                [False, True, True, True],
                PoolHitPolicy.TRAILING_PAGES,
                trailing_pages=3,
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
