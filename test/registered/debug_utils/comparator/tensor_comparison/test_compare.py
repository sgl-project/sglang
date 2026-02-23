import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="default", nightly=True)


class TestCompareTensors(CustomTestCase):
    def test_compute_tensor_stats(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.compare import (
            _compute_tensor_stats,
        )

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = _compute_tensor_stats(x)

        self.assertAlmostEqual(stats.mean, 3.0, places=4)
        self.assertAlmostEqual(stats.min, 1.0, places=4)
        self.assertAlmostEqual(stats.max, 5.0, places=4)
        self.assertIsNotNone(stats.p1)
        self.assertIsNotNone(stats.p99)

    def test_compute_tensor_stats_large_tensor_skips_quantiles(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.compare import (
            QUANTILE_NUMEL_THRESHOLD,
            _compute_tensor_stats,
        )

        x = torch.randn(QUANTILE_NUMEL_THRESHOLD + 1)
        stats = _compute_tensor_stats(x)

        self.assertIsNotNone(stats.mean)
        self.assertIsNone(stats.p1)
        self.assertIsNone(stats.p5)
        self.assertIsNone(stats.p95)
        self.assertIsNone(stats.p99)

    def test_compute_diff_identical(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.compare import (
            _compute_diff,
        )

        x = torch.ones(10, 10)
        diff = _compute_diff(x_baseline=x, x_target=x)

        self.assertAlmostEqual(diff.rel_diff, 0.0, places=5)
        self.assertAlmostEqual(diff.max_abs_diff, 0.0, places=5)
        self.assertAlmostEqual(diff.mean_abs_diff, 0.0, places=5)

    def test_compute_diff_known_offset(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.compare import (
            _compute_diff,
        )

        x = torch.ones(10, 10)
        y = x.clone()
        y[3, 7] = 1.5

        diff = _compute_diff(x_baseline=x, x_target=y)

        self.assertAlmostEqual(diff.max_abs_diff, 0.5, places=4)
        self.assertEqual(diff.max_diff_coord, (3, 7))
        self.assertAlmostEqual(diff.baseline_at_max, 1.0, places=4)
        self.assertAlmostEqual(diff.target_at_max, 1.5, places=4)
        self.assertGreater(diff.mean_abs_diff, 0.0)

    def test_compare_tensors_normal(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.compare import (
            compare_tensors,
        )

        x = torch.randn(5, 5)
        y = x + torch.randn(5, 5) * 0.001

        info = compare_tensors(x_baseline=x, x_target=y, name="test")

        self.assertEqual(info.name, "test")
        self.assertEqual(info.baseline.shape, torch.Size([5, 5]))
        self.assertEqual(info.target.shape, torch.Size([5, 5]))
        self.assertFalse(info.shape_mismatch)
        self.assertIsNotNone(info.diff)
        self.assertIsNone(info.diff_downcast)

    def test_compare_tensors_shape_mismatch(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.compare import (
            compare_tensors,
        )

        x = torch.randn(3, 4)
        y = torch.randn(5, 6)

        info = compare_tensors(x_baseline=x, x_target=y, name="mismatch")

        self.assertTrue(info.shape_mismatch)
        self.assertIsNone(info.diff)

    def test_compare_tensors_dtype_mismatch(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.compare import (
            compare_tensors,
        )

        x = torch.randn(5, 5, dtype=torch.float32)
        y = torch.randn(5, 5, dtype=torch.bfloat16)

        info = compare_tensors(x_baseline=x, x_target=y, name="dtype_test")

        self.assertFalse(info.shape_mismatch)
        self.assertIsNotNone(info.diff)
        self.assertIsNotNone(info.diff_downcast)
        self.assertEqual(info.downcast_dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
