import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.tensor_comparison.compare import (
    QUANTILE_NUMEL_THRESHOLD,
    _compute_diff,
    _compute_tensor_stats,
    compare_tensors,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="default", nightly=True)


class TestComputeTensorStats:
    def test_basic_stats(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = _compute_tensor_stats(x)

        assert stats.mean == pytest.approx(3.0, abs=1e-4)
        assert stats.min == pytest.approx(1.0, abs=1e-4)
        assert stats.max == pytest.approx(5.0, abs=1e-4)
        assert stats.p1 is not None
        assert stats.p99 is not None

    def test_large_tensor_skips_quantiles(self):
        x = torch.randn(QUANTILE_NUMEL_THRESHOLD + 1)
        stats = _compute_tensor_stats(x)

        assert stats.mean is not None
        assert stats.p1 is None
        assert stats.p5 is None
        assert stats.p95 is None
        assert stats.p99 is None


class TestComputeDiff:
    def test_identical_tensors(self):
        x = torch.ones(10, 10)
        diff = _compute_diff(x_baseline=x, x_target=x)

        assert diff.rel_diff == pytest.approx(0.0, abs=1e-5)
        assert diff.max_abs_diff == pytest.approx(0.0, abs=1e-5)
        assert diff.mean_abs_diff == pytest.approx(0.0, abs=1e-5)

    def test_known_offset(self):
        x = torch.ones(10, 10)
        y = x.clone()
        y[3, 7] = 1.5

        diff = _compute_diff(x_baseline=x, x_target=y)

        assert diff.max_abs_diff == pytest.approx(0.5, abs=1e-4)
        assert diff.max_diff_coord == (3, 7)
        assert diff.baseline_at_max == pytest.approx(1.0, abs=1e-4)
        assert diff.target_at_max == pytest.approx(1.5, abs=1e-4)
        assert diff.mean_abs_diff > 0.0


class TestCompareTensors:
    def test_normal(self):
        x = torch.randn(5, 5)
        y = x + torch.randn(5, 5) * 0.001

        info = compare_tensors(x_baseline=x, x_target=y, name="test")

        assert info.name == "test"
        assert info.baseline.shape == torch.Size([5, 5])
        assert info.target.shape == torch.Size([5, 5])
        assert info.shape_mismatch is False
        assert info.diff is not None
        assert info.diff_downcast is None

    def test_shape_mismatch(self):
        x = torch.randn(3, 4)
        y = torch.randn(5, 6)

        info = compare_tensors(x_baseline=x, x_target=y, name="mismatch")

        assert info.shape_mismatch is True
        assert info.diff is None

    def test_dtype_mismatch(self):
        x = torch.randn(5, 5, dtype=torch.float32)
        y = torch.randn(5, 5, dtype=torch.bfloat16)

        info = compare_tensors(x_baseline=x, x_target=y, name="dtype_test")

        assert info.shape_mismatch is False
        assert info.diff is not None
        assert info.diff_downcast is not None
        assert info.downcast_dtype == torch.bfloat16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
