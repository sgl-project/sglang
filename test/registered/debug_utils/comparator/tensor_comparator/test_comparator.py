import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    QUANTILE_NUMEL_THRESHOLD,
    SAMPLE_DIFF_THRESHOLD,
    _compute_tensor_stats,
    compare_tensor_pair,
    compute_diff,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import DiffInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="default", nightly=True)


class TestComputeTensorStats:
    def test_basic_stats(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = _compute_tensor_stats(x)

        assert stats.mean == pytest.approx(3.0, abs=1e-4)
        assert stats.abs_mean == pytest.approx(3.0, abs=1e-4)
        assert stats.std == pytest.approx(1.5811, abs=1e-3)
        assert stats.min == pytest.approx(1.0, abs=1e-4)
        assert stats.max == pytest.approx(5.0, abs=1e-4)

    def test_abs_mean_with_negative_values(self):
        x = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        stats = _compute_tensor_stats(x)

        assert stats.mean == pytest.approx(0.0, abs=1e-4)
        assert stats.abs_mean == pytest.approx(2.0, abs=1e-4)

    def test_quantile_values(self):
        x = torch.linspace(0.0, 100.0, steps=1000)
        stats = _compute_tensor_stats(x)

        assert stats.percentiles[1] == pytest.approx(1.0, abs=0.5)
        assert stats.percentiles[5] == pytest.approx(5.0, abs=0.5)
        assert stats.percentiles[50] == pytest.approx(50.0, abs=0.5)
        assert stats.percentiles[95] == pytest.approx(95.0, abs=0.5)
        assert stats.percentiles[99] == pytest.approx(99.0, abs=0.5)

    def test_large_tensor_skips_quantiles(self):
        x = torch.randn(QUANTILE_NUMEL_THRESHOLD + 1)
        stats = _compute_tensor_stats(x)

        assert stats.mean is not None
        assert stats.percentiles == {}


class TestComputeDiff:
    def test_identical_tensors(self):
        x = torch.ones(10, 10)
        diff = compute_diff(x_baseline=x, x_target=x)

        assert diff.rel_diff == pytest.approx(0.0, abs=1e-5)
        assert diff.max_abs_diff == pytest.approx(0.0, abs=1e-5)
        assert diff.mean_abs_diff == pytest.approx(0.0, abs=1e-5)
        assert diff.abs_diff_percentiles[50] == pytest.approx(0.0, abs=1e-5)
        assert diff.abs_diff_percentiles[95] == pytest.approx(0.0, abs=1e-5)
        assert diff.abs_diff_percentiles[99] == pytest.approx(0.0, abs=1e-5)
        assert diff.passed is True

    def test_known_offset(self):
        x = torch.ones(10, 10)
        y = x.clone()
        y[3, 7] = 1.5

        diff = compute_diff(x_baseline=x, x_target=y)

        assert diff.max_abs_diff == pytest.approx(0.5, abs=1e-4)
        assert diff.max_diff_coord == [3, 7]
        assert diff.baseline_at_max == pytest.approx(1.0, abs=1e-4)
        assert diff.target_at_max == pytest.approx(1.5, abs=1e-4)
        assert diff.mean_abs_diff == pytest.approx(0.5 / 100, abs=1e-4)
        assert diff.abs_diff_percentiles[1] == pytest.approx(0.0, abs=1e-4)
        assert diff.abs_diff_percentiles[50] == pytest.approx(0.0, abs=1e-4)
        assert diff.abs_diff_percentiles[99] > 0
        assert diff.passed is False

    def test_large_tensor_skips_diff_quantiles(self):
        x = torch.randn(QUANTILE_NUMEL_THRESHOLD + 1)
        y = x + 0.001
        diff = compute_diff(x_baseline=x, x_target=y)

        assert diff.abs_diff_percentiles == {}

    def test_rel_diff_value(self):
        x = torch.tensor([1.0, 0.0])
        y = torch.tensor([0.0, 1.0])
        diff = compute_diff(x_baseline=x, x_target=y)

        assert diff.rel_diff == pytest.approx(1.0, abs=1e-5)
        assert diff.passed is False

    def test_per_token_with_seq_dim(self) -> None:
        """seq_dim provided → per_token_rel_diff is list[float]."""
        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(8, 16)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = compute_diff(
            x_baseline=x, x_target=y, diff_threshold=1e-3, seq_dim=0
        )

        assert diff.per_token_rel_diff is not None
        assert isinstance(diff.per_token_rel_diff, list)
        assert len(diff.per_token_rel_diff) == 8
        assert all(isinstance(v, float) for v in diff.per_token_rel_diff)

    def test_per_token_without_seq_dim(self) -> None:
        """No seq_dim → per_token_rel_diff is None."""
        x: torch.Tensor = torch.randn(8, 16)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = compute_diff(x_baseline=x, x_target=y, diff_threshold=1e-3)

        assert diff.per_token_rel_diff is None

    def test_per_token_json_roundtrip(self) -> None:
        """DiffInfo with per_token_rel_diff survives JSON serialization."""
        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(4, 8)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = compute_diff(
            x_baseline=x, x_target=y, diff_threshold=1e-3, seq_dim=0
        )

        json_str: str = diff.model_dump_json()
        assert "per_token_rel_diff" in json_str

        roundtripped: DiffInfo = DiffInfo.model_validate_json(json_str)
        assert roundtripped.per_token_rel_diff is not None
        assert len(roundtripped.per_token_rel_diff) == 4


class TestCompareTensors:
    def test_normal(self):
        x = torch.randn(5, 5)
        y = x + torch.randn(5, 5) * 0.001

        info = compare_tensor_pair(x_baseline=x, x_target=y, name="test")

        assert info.name == "test"
        assert info.baseline.shape == [5, 5]
        assert info.target.shape == [5, 5]
        assert info.shape_mismatch is False
        assert info.diff is not None
        assert info.diff_downcast is None

    def test_shape_mismatch(self):
        x = torch.randn(3, 4)
        y = torch.randn(5, 6)

        info = compare_tensor_pair(x_baseline=x, x_target=y, name="mismatch")

        assert info.shape_mismatch is True
        assert info.diff is None

    def test_dtype_mismatch(self):
        x = torch.randn(5, 5, dtype=torch.float32)
        y = torch.randn(5, 5, dtype=torch.bfloat16)

        info = compare_tensor_pair(x_baseline=x, x_target=y, name="dtype_test")

        assert info.shape_mismatch is False
        assert info.diff is not None
        assert info.diff_downcast is not None
        assert info.downcast_dtype == "torch.bfloat16"

    def test_shape_unification(self):
        torch.manual_seed(0)
        core = torch.randn(4, 8)
        x = core.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 8]
        y = core.clone()  # [4, 8]

        info = compare_tensor_pair(x_baseline=x, x_target=y, name="unify")

        assert info.baseline.shape == [1, 1, 4, 8]
        assert info.unified_shape == [4, 8]
        assert info.shape_mismatch is False
        assert info.diff is not None
        assert info.diff.max_abs_diff == pytest.approx(0.0, abs=1e-5)

    def test_sample_generated_when_large_diff(self):
        x = torch.zeros(5, 5)
        y = torch.ones(5, 5)

        info = compare_tensor_pair(x_baseline=x, x_target=y, name="big_diff")

        assert info.diff is not None
        assert info.diff.max_abs_diff > SAMPLE_DIFF_THRESHOLD
        assert info.baseline.sample is not None
        assert info.target.sample is not None

    def test_no_sample_when_small_diff(self):
        x = torch.ones(5, 5)
        y = x + 1e-5

        info = compare_tensor_pair(x_baseline=x, x_target=y, name="tiny_diff")

        assert info.diff is not None
        assert info.diff.max_abs_diff < SAMPLE_DIFF_THRESHOLD
        assert info.baseline.sample is None
        assert info.target.sample is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
