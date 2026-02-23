import tempfile
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="default", nightly=True)


# ===== utils.py tests (moved from test_dump_comparator.py, updated imports) =====


class TestUtils(CustomTestCase):
    def test_calc_rel_diff(self):
        from sglang.srt.debug_utils.comparator.utils import calc_rel_diff

        x = torch.randn(10, 10)
        self.assertAlmostEqual(calc_rel_diff(x, x).item(), 0.0, places=5)
        self.assertAlmostEqual(
            calc_rel_diff(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])).item(),
            1.0,
            places=5,
        )

    def test_argmax_coord(self):
        from sglang.srt.debug_utils.comparator.utils import argmax_coord

        x = torch.zeros(2, 3, 4)
        x[1, 2, 3] = 10.0
        self.assertEqual(argmax_coord(x), (1, 2, 3))

    def test_try_unify_shape(self):
        from sglang.srt.debug_utils.comparator.utils import try_unify_shape

        target = torch.Size([3, 4])
        self.assertEqual(
            try_unify_shape(torch.randn(1, 1, 3, 4), target).shape, target
        )
        self.assertEqual(
            try_unify_shape(torch.randn(2, 3, 4), target).shape, (2, 3, 4)
        )

    def test_compute_smaller_dtype(self):
        from sglang.srt.debug_utils.comparator.utils import compute_smaller_dtype

        self.assertEqual(
            compute_smaller_dtype(torch.float32, torch.bfloat16), torch.bfloat16
        )
        self.assertEqual(
            compute_smaller_dtype(torch.bfloat16, torch.float32), torch.bfloat16
        )
        self.assertIsNone(compute_smaller_dtype(torch.float32, torch.float32))

    def test_load_object(self):
        from sglang.srt.debug_utils.comparator.utils import load_object

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tensor.pt"
            torch.save(torch.randn(5, 5), path)
            self.assertEqual(load_object(path).shape, (5, 5))

            torch.save({"dict": 1}, path)
            self.assertIsNone(load_object(path))

        self.assertIsNone(load_object(Path("/nonexistent.pt")))


# ===== tensor_comparison/core.py tests (new) =====


class TestTensorComparisonCore(CustomTestCase):
    def test_compute_tensor_stats(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
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
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
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
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
            _compute_diff,
        )

        x = torch.ones(10, 10)
        diff = _compute_diff(x_baseline=x, x_target=x)

        self.assertAlmostEqual(diff.rel_diff, 0.0, places=5)
        self.assertAlmostEqual(diff.max_abs_diff, 0.0, places=5)
        self.assertAlmostEqual(diff.mean_abs_diff, 0.0, places=5)

    def test_compute_diff_known_offset(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
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
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
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
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
            compare_tensors,
        )

        x = torch.randn(3, 4)
        y = torch.randn(5, 6)

        info = compare_tensors(x_baseline=x, x_target=y, name="mismatch")

        self.assertTrue(info.shape_mismatch)
        self.assertIsNone(info.diff)

    def test_compare_tensors_dtype_mismatch(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
            compare_tensors,
        )

        x = torch.randn(5, 5, dtype=torch.float32)
        y = torch.randn(5, 5, dtype=torch.bfloat16)

        info = compare_tensors(x_baseline=x, x_target=y, name="dtype_test")

        self.assertFalse(info.shape_mismatch)
        self.assertIsNotNone(info.diff)
        self.assertIsNotNone(info.diff_downcast)
        self.assertEqual(info.downcast_dtype, torch.bfloat16)


# ===== tensor_comparison/printer.py tests (new) =====


class TestTensorComparisonPrinter(CustomTestCase):
    def test_print_comparison_no_error(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
            compare_tensors,
        )
        from sglang.srt.debug_utils.comparator.tensor_comparison.printer import (
            print_comparison,
        )

        x = torch.randn(5, 5)
        y = x + torch.randn(5, 5) * 0.01
        info = compare_tensors(x_baseline=x, x_target=y, name="printer_test")

        print_comparison(info=info, diff_threshold=1e-3)

    def test_print_comparison_shape_mismatch(self):
        from sglang.srt.debug_utils.comparator.tensor_comparison.core import (
            compare_tensors,
        )
        from sglang.srt.debug_utils.comparator.tensor_comparison.printer import (
            print_comparison,
        )

        x = torch.randn(3, 4)
        y = torch.randn(5, 6)
        info = compare_tensors(x_baseline=x, x_target=y, name="mismatch_print")

        print_comparison(info=info, diff_threshold=1e-3)


# ===== E2E test (moved from test_dump_comparator.py) =====


class TestEndToEnd(CustomTestCase):
    def test_main(self):
        from argparse import Namespace

        from sglang.srt.debug_utils.comparator.entrypoint import main
        from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            baseline_tensor = torch.randn(10, 10)
            target_tensor = baseline_tensor + torch.randn(10, 10) * 0.01

            dump_dirs = []
            for d, tensor in [(d1, baseline_tensor), (d2, target_tensor)]:
                dumper = _Dumper(
                    config=DumperConfig(
                        enable=True,
                        dir=d,
                        enable_http_server=False,
                    )
                )
                dumper.dump("tensor_a", tensor)
                dumper.step()
                dumper.dump("tensor_b", tensor * 2)
                dumper.step()
                dump_dirs.append(Path(d) / dumper._config.exp_name)

            args = Namespace(
                baseline_path=str(dump_dirs[0]),
                target_path=str(dump_dirs[1]),
                start_id=0,
                end_id=1,
                baseline_start_id=0,
                diff_threshold=1e-3,
                filter=None,
            )
            main(args)


if __name__ == "__main__":
    unittest.main()
