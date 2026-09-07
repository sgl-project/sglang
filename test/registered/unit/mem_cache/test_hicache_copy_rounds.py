"""The copy-round screen in kvcache/hicache.py must agree with pick_group_bytes()
in kvcacheio/hicache.cuh: a size the screen admits has to compile, and a size the
kernel cannot tile has to be turned away before the JIT ever sees it."""

import unittest
from unittest import mock

from sglang.kernels.ops.kvcache import hicache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _screen(element_size: int, unroll: int, *, is_hip: bool) -> bool:
    """Run _tiles_across_lanes() as the given platform would see it."""
    group_bytes = (128, 64, 32, 16) if is_hip else (128,)
    with mock.patch.object(hicache, "GROUP_BYTES", group_bytes):
        return hicache._tiles_across_lanes(element_size, unroll)


class TestHiCacheCopyRounds(unittest.TestCase):
    def test_mla_fp8_row_is_admitted_only_on_rocm(self):
        # 576 B is MLA's fp8 row and the reason the narrow rounds exist: 128 does
        # not divide it, so the 128-only CUDA screen has to keep turning it away.
        self.assertTrue(_screen(576, unroll=4, is_hip=True))
        self.assertFalse(_screen(576, unroll=4, is_hip=False))

    def test_a_size_128_divides_is_admitted_everywhere(self):
        self.assertTrue(_screen(512, unroll=4, is_hip=True))
        self.assertTrue(_screen(512, unroll=4, is_hip=False))

    def test_rocm_admits_no_more_than_the_kernel_can_tile(self):
        # The screen mirrors group_fits(): the round must divide the element and
        # split across lanes into a package the hardware has (4, 8 or 16 B).
        for element_size in range(16, 1300, 4):
            for unroll in (1, 2, 4):
                num_threads = hicache.WARP_THREADS // unroll
                expected = any(
                    group % num_threads == 0
                    and element_size % group == 0
                    and group // num_threads in (4, 8, 16)
                    for group in (128, 64, 32, 16)
                )
                with self.subTest(element_size=element_size, unroll=unroll):
                    self.assertEqual(
                        _screen(element_size, unroll, is_hip=True), expected
                    )

    def test_an_odd_size_is_turned_away_on_both(self):
        # 100 B is divisible by no round, so no lane split can cover it.
        self.assertFalse(_screen(100, unroll=4, is_hip=True))
        self.assertFalse(_screen(100, unroll=4, is_hip=False))

    def test_the_python_lane_count_matches_the_pinned_constant(self):
        # kvcacheio/hicache.cuh static_asserts device::kWarpThreads == 32 so this
        # mirror cannot drift; assert the mirror itself here.
        self.assertEqual(hicache.WARP_THREADS, 32)


if __name__ == "__main__":
    unittest.main()
