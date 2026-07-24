import unittest

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    find_prefix_hit_boundary,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPoolHitPolicy(unittest.TestCase):
    def test_all_pages_returns_leading_contiguous_prefix(self):
        self.assertEqual(
            find_prefix_hit_boundary(
                [True, True, False, True], PoolHitPolicy.ALL_PAGES
            ),
            2,
        )

    def test_single_trailing_page_accepts_mamba_hit(self):
        self.assertEqual(
            find_prefix_hit_boundary(
                [False, False, True], PoolHitPolicy.TRAILING_PAGES, trailing_pages=1
            ),
            3,
        )

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

    def test_trailing_window_falls_back_to_earlier_complete_window(self):
        self.assertEqual(
            find_prefix_hit_boundary(
                [True, True, True, False],
                PoolHitPolicy.TRAILING_PAGES,
                trailing_pages=3,
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
