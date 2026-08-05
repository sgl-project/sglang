"""Unit tests for minimax_h3_time_shift_sigmas schedule construction.

The MiniMax H3 Euler denoise loop performs ``len(sigmas) - 1`` updates, so a
requested ``num_inference_steps`` must map to ``num_steps + 1`` sigma points
(including the terminal 0.0) to honor the requested step count. This pins the
schedule shape, monotonicity, endpoints, and input validation.
"""

import unittest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxH3TimeShiftSigmas(CustomTestCase):
    def test_schedule_has_num_steps_plus_one_points(self):
        """The Euler loop runs len(sigmas) - 1 updates, so a request for N
        steps must produce N + 1 sigma points."""
        for num_steps in (1, 2, 5, 50):
            sigmas = minimax_h3_time_shift_sigmas(num_steps=num_steps)
            self.assertEqual(
                len(sigmas),
                num_steps + 1,
                f"num_steps={num_steps} should yield {num_steps + 1} points",
            )

    def test_exact_n_plus_one_cardinality_across_shift_scales(self):
        """The N+1 cardinality contract must hold for every shift_scale:
        the schedule is strictly monotonic, so deduplication can never be
        needed and must never shrink the returned list."""
        for shift_scale in (1e-6, 0.01, 1.0, 6.0, 12.0, 1e4, 1e6, 1e9):
            for num_steps in (1, 8, 50, 1000):
                sigmas = minimax_h3_time_shift_sigmas(
                    num_steps=num_steps, shift_scale=shift_scale
                )
                self.assertEqual(
                    len(sigmas),
                    num_steps + 1,
                    f"shift_scale={shift_scale:g}, num_steps={num_steps} "
                    f"must yield exactly {num_steps + 1} points, got {len(sigmas)}",
                )

    def test_schedule_span_and_endpoints(self):
        sigmas = minimax_h3_time_shift_sigmas(num_steps=10, shift_scale=6.0)
        self.assertAlmostEqual(sigmas[0], 1.0, places=6)
        self.assertEqual(sigmas[-1], 0.0)
        self.assertLessEqual(max(sigmas), 1.0)
        self.assertGreaterEqual(min(sigmas), 0.0)

    def test_schedule_strictly_decreasing(self):
        for shift_scale in (0.5, 1.0, 6.0, 12.0):
            sigmas = minimax_h3_time_shift_sigmas(num_steps=20, shift_scale=shift_scale)
            self.assertTrue(
                all(a > b for a, b in zip(sigmas, sigmas[1:])),
                f"shift_scale={shift_scale} schedule must be strictly decreasing",
            )

    def test_shift_scale_one_is_linear_schedule(self):
        sigmas = minimax_h3_time_shift_sigmas(num_steps=4, shift_scale=1.0)
        self.assertEqual(len(sigmas), 5)
        expected = [1.0, 0.75, 0.5, 0.25, 0.0]
        self.assertEqual(len(sigmas), len(expected))
        for actual, want in zip(sigmas, expected):
            self.assertAlmostEqual(actual, want, places=5)

    def test_larger_shift_scale_concentrates_low_sigma_steps(self):
        """A higher shift_scale pushes more sigma points toward the low-noise
        end, the standard rectified-flow time-shift trade-off."""
        sigmas_6 = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=6.0)
        sigmas_12 = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=12.0)
        mid_6 = sigmas_6[len(sigmas_6) // 2]
        mid_12 = sigmas_12[len(sigmas_12) // 2]
        self.assertGreater(mid_12, mid_6)

    def test_invalid_shift_scale_rejected(self):
        with self.assertRaises(ValueError):
            minimax_h3_time_shift_sigmas(shift_scale=0.0)
        with self.assertRaises(ValueError):
            minimax_h3_time_shift_sigmas(shift_scale=-1.0)

    def test_invalid_num_steps_rejected(self):
        with self.assertRaises(ValueError):
            minimax_h3_time_shift_sigmas(num_steps=0)
        with self.assertRaises(ValueError):
            minimax_h3_time_shift_sigmas(num_steps=-5)

    def test_single_step_is_two_points(self):
        """A one-step request still needs the terminal 0.0 so the Euler loop
        runs exactly one update."""
        sigmas = minimax_h3_time_shift_sigmas(num_steps=1)
        self.assertEqual(len(sigmas), 2)
        self.assertEqual(sigmas[0], 1.0)
        self.assertEqual(sigmas[-1], 0.0)

    def test_denoise_loop_equivalence(self):
        """The number of Euler updates implied by the schedule equals the
        requested num_inference_steps (denoise_loop uses len(sigmas) - 1)."""
        for num_steps in (1, 5, 30, 50):
            sigmas = minimax_h3_time_shift_sigmas(num_steps=num_steps)
            self.assertEqual(len(sigmas) - 1, num_steps)
            self.assertEqual(len(sigmas), num_steps + 1)


if __name__ == "__main__":
    unittest.main()
