# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TeaCache coefficient calibration math."""

import unittest

import numpy as np
import torch

from sglang.multimodal_gen.runtime.cache.teacache_calibrate import (
    SINGLE_EXPERT,
    TeaCacheCalibrator,
    _relative_diff,
    calculate_threshold,
)


class TestRelativeDiff(unittest.TestCase):
    def test_mean_abs_over_mean_abs(self) -> None:
        # mean(|3-2|) / mean(|2|) = 1 / 2 = 0.5
        t1 = torch.full((8,), 2.0)
        t2 = torch.full((8,), 3.0)
        self.assertAlmostEqual(_relative_diff(t2, t1), 0.5, places=5)

    def test_zero_change(self) -> None:
        t = torch.randn(16)
        self.assertAlmostEqual(_relative_diff(t, t), 0.0, places=6)


class TestCalculateThreshold(unittest.TestCase):
    def test_flat_run_mean_times_two(self) -> None:
        # Longest flat run is [0.1, 0.1, 0.1]; mean 0.1 * 2 = 0.2.
        y = np.array([0.1, 0.1, 0.1, 0.5])
        self.assertAlmostEqual(calculate_threshold(y), 0.2, places=6)

    def test_single_sample_is_zero(self) -> None:
        self.assertEqual(calculate_threshold(np.array([0.3])), 0.0)

    def test_no_flat_run_falls_back(self) -> None:
        # Every successive slope exceeds the tolerance -> fallback 0.2.
        y = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(calculate_threshold(y, slope_threshold=0.01), 0.2)

    def test_picks_longest_flat_run(self) -> None:
        # Two flat regions; the longer one (0.2 x4) must win: 0.2 * 2 = 0.4.
        y = np.array([0.9, 0.9, 0.2, 0.2, 0.2, 0.2])
        self.assertAlmostEqual(calculate_threshold(y), 0.4, places=6)


class TestCalibratorFit(unittest.TestCase):
    def _feed_linear_branch(self, calib: TeaCacheCalibrator, expert: str) -> None:
        # Construct steps whose modulated-input and output magnitudes grow so
        # each step yields a well-defined, nonzero relative diff.
        for step in range(6):
            e = torch.full((4,), 1.0 + 0.5 * step)
            x = torch.full((4,), 2.0 + step)
            calib.record(e, x, step_index=step, is_cfg_negative=False, expert=expert)

    def test_single_expert_shape(self) -> None:
        calib = TeaCacheCalibrator(degree=4)
        self._feed_linear_branch(calib, SINGLE_EXPERT)
        result = calib.fit()
        # Single expert collapses to a flat dict.
        self.assertIn("coefficients", result)
        self.assertIn("teacache_thresh", result)
        self.assertEqual(len(result["coefficients"]), 5)  # degree 4 -> 5 coeffs
        self.assertIsInstance(result["teacache_thresh"], float)

    def test_moe_split_returns_per_expert(self) -> None:
        calib = TeaCacheCalibrator(degree=4)
        self._feed_linear_branch(calib, "high")
        self._feed_linear_branch(calib, "low")
        result = calib.fit()
        self.assertEqual(set(result), {"high", "low"})
        self.assertEqual(len(result["high"]["coefficients"]), 5)

    def test_step_running_average(self) -> None:
        # Two branches hitting the same step index average into one row.
        calib = TeaCacheCalibrator(degree=4)
        # pos branch: prev then current
        calib.record(
            torch.full((4,), 1.0),
            torch.full((4,), 1.0),
            step_index=0,
            is_cfg_negative=False,
        )
        calib.record(
            torch.full((4,), 2.0),
            torch.full((4,), 2.0),
            step_index=1,
            is_cfg_negative=False,
        )  # diff_tm=1.0
        # neg branch: prev then current, same step 1 -> averaged
        calib.record(
            torch.full((4,), 1.0),
            torch.full((4,), 1.0),
            step_index=0,
            is_cfg_negative=True,
        )
        calib.record(
            torch.full((4,), 3.0),
            torch.full((4,), 3.0),
            step_index=1,
            is_cfg_negative=True,
        )  # diff_tm=2.0
        row = calib._experts[SINGLE_EXPERT]["rows"][1]
        # count=2, mean of diff_tm (1.0, 2.0) = 1.5
        self.assertEqual(row[0], 2)
        self.assertAlmostEqual(row[1], 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
