import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    QUANTILE_NUMEL_THRESHOLD,
    SAMPLE_DIFF_THRESHOLD,
    _compute_tensor_stats,
    compare_tensor_pair,
    compute_diff,
    compute_tensor_info,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import DiffInfo
from sglang.srt.debug_utils.comparator.threshold_dsl import DiffThresholdRule
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=1, suite="base-c-test-cpu")


class TestComputeTensorInfo:
    def test_basic_tensor_returns_correct_shape_and_dtype(self) -> None:
        tensor = torch.randn(2, 3)
        info = compute_tensor_info(tensor)
        assert info.shape == [2, 3]
        assert info.dtype == "torch.float32"
        assert info.stats.mean == pytest.approx(tensor.float().mean().item(), abs=1e-4)

    def test_include_sample_false_returns_none_sample(self) -> None:
        tensor = torch.randn(2, 3)
        info = compute_tensor_info(tensor, include_sample=False)
        assert info.sample is None

    def test_include_sample_true_returns_string_sample(self) -> None:
        tensor = torch.randn(2, 3)
        info = compute_tensor_info(tensor, include_sample=True)
        assert info.sample is not None
        assert isinstance(info.sample, str)

    def test_empty_tensor_stats_are_zero(self) -> None:
        tensor = torch.tensor([])
        info = compute_tensor_info(tensor)
        assert info.stats.mean == 0.0
        assert info.stats.std == 0.0
        assert info.shape == [0]

    def test_integer_tensor_converted_to_float_for_stats(self) -> None:
        """Integer tensors should be cast to float internally for stats computation."""
        tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        info = compute_tensor_info(tensor)
        assert info.dtype == "torch.int32"
        assert info.stats.mean == pytest.approx(2.5, abs=1e-4)
        assert info.stats.min == pytest.approx(1.0, abs=1e-4)
        assert info.stats.max == pytest.approx(4.0, abs=1e-4)

    def test_bfloat16_tensor_shape_and_stats(self) -> None:
        """bfloat16 tensors produce correct shape and dtype string."""
        tensor = torch.ones(3, 4, dtype=torch.bfloat16)
        info = compute_tensor_info(tensor)
        assert info.shape == [3, 4]
        assert info.dtype == "torch.bfloat16"
        assert info.stats.mean == pytest.approx(1.0, abs=1e-2)

    def test_multidimensional_shape(self) -> None:
        """Shape is preserved for high-rank tensors."""
        tensor = torch.randn(2, 3, 4, 5)
        info = compute_tensor_info(tensor)
        assert info.shape == [2, 3, 4, 5]

    def test_scalar_tensor(self) -> None:
        """Scalar (0-dim) tensor produces empty shape list."""
        tensor = torch.tensor(3.14)
        info = compute_tensor_info(tensor)
        assert info.shape == []
        assert info.stats.mean == pytest.approx(3.14, abs=1e-4)
        assert info.stats.min == pytest.approx(3.14, abs=1e-4)
        assert info.stats.max == pytest.approx(3.14, abs=1e-4)

    def test_include_sample_true_contains_tensor_representation(self) -> None:
        """Sample string should contain some recognizable tensor content."""
        tensor = torch.tensor([1.0, 2.0])
        info = compute_tensor_info(tensor, include_sample=True)
        assert info.sample is not None
        assert "1." in info.sample or "2." in info.sample

    def test_percentiles_present_for_small_tensor(self) -> None:
        """Small tensors (< threshold) should have percentile data."""
        tensor = torch.randn(100)
        info = compute_tensor_info(tensor)
        assert len(info.stats.percentiles) > 0
        assert 50 in info.stats.percentiles


class TestComputeTensorInfo:
    def test_basic_tensor_returns_correct_shape_and_dtype(self) -> None:
        tensor = torch.randn(2, 3)
        info = compute_tensor_info(tensor)
        assert info.shape == [2, 3]
        assert info.dtype == "torch.float32"
        assert info.stats.mean == pytest.approx(tensor.float().mean().item(), abs=1e-4)

    def test_include_sample_false_returns_none_sample(self) -> None:
        tensor = torch.randn(2, 3)
        info = compute_tensor_info(tensor, include_sample=False)
        assert info.sample is None

    def test_include_sample_true_returns_string_sample(self) -> None:
        tensor = torch.randn(2, 3)
        info = compute_tensor_info(tensor, include_sample=True)
        assert info.sample is not None
        assert isinstance(info.sample, str)

    def test_empty_tensor_stats_are_zero(self) -> None:
        tensor = torch.tensor([])
        info = compute_tensor_info(tensor)
        assert info.stats.mean == 0.0
        assert info.stats.std == 0.0
        assert info.shape == [0]

    def test_integer_tensor_converted_to_float_for_stats(self) -> None:
        """Integer tensors should be cast to float internally for stats computation."""
        tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        info = compute_tensor_info(tensor)
        assert info.dtype == "torch.int32"
        assert info.stats.mean == pytest.approx(2.5, abs=1e-4)
        assert info.stats.min == pytest.approx(1.0, abs=1e-4)
        assert info.stats.max == pytest.approx(4.0, abs=1e-4)

    def test_bfloat16_tensor_shape_and_stats(self) -> None:
        """bfloat16 tensors produce correct shape and dtype string."""
        tensor = torch.ones(3, 4, dtype=torch.bfloat16)
        info = compute_tensor_info(tensor)
        assert info.shape == [3, 4]
        assert info.dtype == "torch.bfloat16"
        assert info.stats.mean == pytest.approx(1.0, abs=1e-2)

    def test_multidimensional_shape(self) -> None:
        """Shape is preserved for high-rank tensors."""
        tensor = torch.randn(2, 3, 4, 5)
        info = compute_tensor_info(tensor)
        assert info.shape == [2, 3, 4, 5]

    def test_scalar_tensor(self) -> None:
        """Scalar (0-dim) tensor produces empty shape list."""
        tensor = torch.tensor(3.14)
        info = compute_tensor_info(tensor)
        assert info.shape == []
        assert info.stats.mean == pytest.approx(3.14, abs=1e-4)
        assert info.stats.min == pytest.approx(3.14, abs=1e-4)
        assert info.stats.max == pytest.approx(3.14, abs=1e-4)

    def test_include_sample_true_contains_tensor_representation(self) -> None:
        """Sample string should contain some recognizable tensor content."""
        tensor = torch.tensor([1.0, 2.0])
        info = compute_tensor_info(tensor, include_sample=True)
        assert info.sample is not None
        assert "1." in info.sample or "2." in info.sample

    def test_percentiles_present_for_small_tensor(self) -> None:
        """Small tensors (< threshold) should have percentile data."""
        tensor = torch.randn(100)
        info = compute_tensor_info(tensor)
        assert len(info.stats.percentiles) > 0
        assert 50 in info.stats.percentiles


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

    def test_percentiles_exact_for_uniform_range(self):
        """Percentiles of arange(0, 101) equal the percentile index exactly (linear interp)."""
        x = torch.arange(0, 101, dtype=torch.float32)
        stats = _compute_tensor_stats(x)

        for p in (1, 5, 50, 95, 99):
            assert stats.percentiles[p] == pytest.approx(float(p), abs=1e-4)

    def test_percentiles_match_torch_quantile_reference(self):
        """numpy-based percentiles must match torch.quantile on the same data within tight tolerance."""
        torch.manual_seed(0)
        x = torch.randn(5000)
        stats = _compute_tensor_stats(x)

        for p in (1, 5, 50, 95, 99):
            expected = torch.quantile(x.float(), p / 100.0).item()
            assert stats.percentiles[p] == pytest.approx(expected, abs=1e-4)

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

        diff: DiffInfo = compute_diff(x_baseline=x, x_target=y, seq_dim=0)

        assert diff.per_token_rel_diff is not None
        assert isinstance(diff.per_token_rel_diff, list)
        assert len(diff.per_token_rel_diff) == 8
        assert all(isinstance(v, float) for v in diff.per_token_rel_diff)

    def test_per_token_without_seq_dim(self) -> None:
        """No seq_dim → per_token_rel_diff is None."""
        x: torch.Tensor = torch.randn(8, 16)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = compute_diff(x_baseline=x, x_target=y)

        assert diff.per_token_rel_diff is None

    def test_per_token_json_roundtrip(self) -> None:
        """DiffInfo with per_token_rel_diff survives JSON serialization."""
        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(4, 8)
        y: torch.Tensor = x + torch.randn_like(x) * 0.01

        diff: DiffInfo = compute_diff(x_baseline=x, x_target=y, seq_dim=0)

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


class TestComputeDiffPredicate:
    @staticmethod
    def _near_zero_pair() -> tuple[torch.Tensor, torch.Tensor]:
        """Sign-flipped near-zero pair: rel_diff == 2.0 but max_abs/mean_abs == 2e-5."""
        x = torch.tensor([1e-5, -1e-5, 1e-5, -1e-5])
        return x, -x

    def test_default_predicate(self) -> None:
        """No predicate → the default 'rel <= 0.001'; near-zero pair fails and the string is recorded."""
        x, y = self._near_zero_pair()
        diff = compute_diff(x_baseline=x, x_target=y)

        assert diff.rel_diff == pytest.approx(2.0, abs=1e-4)
        assert diff.max_abs_diff == pytest.approx(2e-5, abs=1e-7)
        assert diff.predicate == "rel <= 0.001"
        assert diff.passed is False

    def test_predicate_rescues_near_zero_via_max_abs(self) -> None:
        """A 'rel or max_abs' predicate passes the near-zero pair despite a failing rel."""
        x, y = self._near_zero_pair()
        diff = compute_diff(
            x_baseline=x, x_target=y, predicate="rel <= 0.0085 or max_abs <= 1e-4"
        )

        assert diff.rel_diff > 1.0  # relative term still fails
        assert diff.passed is True
        assert diff.predicate == "rel <= 0.0085 or max_abs <= 1e-4"

    def test_predicate_does_not_rescue_real_magnitude_diff(self) -> None:
        """A real-magnitude diff fails both terms of a 'rel or max_abs' predicate."""
        x = torch.ones(10)
        y = x.clone()
        y[0] = 2.0  # max_abs_diff == 1.0, rel_diff ~0.043
        diff = compute_diff(
            x_baseline=x, x_target=y, predicate="rel <= 0.0085 or max_abs <= 1e-3"
        )

        assert diff.max_abs_diff == pytest.approx(1.0, abs=1e-4)
        assert diff.passed is False

    def test_and_predicate_requires_both(self) -> None:
        """'rel and max_abs' fails the near-zero pair (rel huge) but passes a small-both diff."""
        x, y = self._near_zero_pair()
        assert (
            compute_diff(
                x_baseline=x, x_target=y, predicate="rel <= 0.0085 and max_abs <= 1e-4"
            ).passed
            is False
        )
        small = torch.ones(10)
        small_y = small + 1e-4  # rel ~5e-9, max_abs 1e-4
        assert (
            compute_diff(
                x_baseline=small,
                x_target=small_y,
                predicate="rel <= 0.0085 and max_abs <= 1e-3",
            ).passed
            is True
        )

    def test_mean_abs_variable(self) -> None:
        """A mean_abs predicate uses the mean absolute diff (2e-5 for the near-zero pair)."""
        x, y = self._near_zero_pair()
        assert (
            compute_diff(x_baseline=x, x_target=y, predicate="mean_abs <= 1e-4").passed
            is True
        )
        assert (
            compute_diff(x_baseline=x, x_target=y, predicate="mean_abs <= 1e-6").passed
            is False
        )

    def test_boundary_le_inclusive(self) -> None:
        """<= includes the boundary, < excludes it (max_abs_diff == 0.5 exactly)."""
        x = torch.tensor([1.0, 1.0])
        y = torch.tensor([1.5, 1.5])
        assert (
            compute_diff(x_baseline=x, x_target=y, predicate="max_abs <= 0.5").passed
            is True
        )
        assert (
            compute_diff(x_baseline=x, x_target=y, predicate="max_abs < 0.5").passed
            is False
        )

    def test_predicate_recorded_for_empty_tensor(self) -> None:
        """Empty tensors short-circuit to passed=True and still record the predicate."""
        empty = torch.empty(0)
        diff = compute_diff(x_baseline=empty, x_target=empty, predicate="rel <= 0")

        assert diff.passed is True
        assert diff.predicate == "rel <= 0"

    def test_predicate_json_roundtrip(self) -> None:
        """DiffInfo.predicate survives JSON serialization."""
        x, y = self._near_zero_pair()
        diff = compute_diff(
            x_baseline=x, x_target=y, predicate="rel <= 0.0085 or max_abs <= 1e-4"
        )

        roundtripped = DiffInfo.model_validate_json(diff.model_dump_json())
        assert roundtripped.predicate == "rel <= 0.0085 or max_abs <= 1e-4"
        assert roundtripped.passed is True


class TestCompareTensorPairPredicate:
    def test_predicate_resolved_per_name(self) -> None:
        """compare_tensor_pair resolves the per-regex predicate by tensor name into the verdict."""
        x = torch.tensor([1e-5, -1e-5, 1e-5, -1e-5])
        y = -x

        without = compare_tensor_pair(x_baseline=x, x_target=y, name="g.expert.0")
        assert without.diff is not None
        assert without.diff.passed is False

        with_pred = compare_tensor_pair(
            x_baseline=x,
            x_target=y,
            name="g.expert.0",
            diff_threshold_rules=[
                DiffThresholdRule(".*expert.*", "rel <= 0.0085 or max_abs <= 1e-4")
            ],
        )
        assert with_pred.diff is not None
        assert with_pred.diff.passed is True
        assert with_pred.diff.predicate == "rel <= 0.0085 or max_abs <= 1e-4"

    def test_unmatched_name_raises(self) -> None:
        """A tensor matching no pattern raises (fail-closed)."""
        x = torch.tensor([1e-5, -1e-5, 1e-5, -1e-5])
        with pytest.raises(ValueError, match="matched no --diff-threshold pattern"):
            compare_tensor_pair(
                x_baseline=x,
                x_target=-x,
                name="g.attn.qkv",
                diff_threshold_rules=[
                    DiffThresholdRule(".*expert.*", "rel <= 0.0085 or max_abs <= 1e-4")
                ],
            )

    def test_predicate_propagates_to_downcast_diff(self) -> None:
        """When baseline/target dtypes differ, the resolved predicate also drives the downcast diff."""
        x = torch.tensor([1e-5, -1e-5, 1e-5, -1e-5])
        info = compare_tensor_pair(
            x_baseline=x,
            x_target=(-x).to(torch.bfloat16),
            name="g.expert.0",
            diff_threshold_rules=[
                DiffThresholdRule(".*expert.*", "rel <= 0.0085 or max_abs <= 1e-4")
            ],
        )
        assert info.diff_downcast is not None
        assert info.diff_downcast.predicate == "rel <= 0.0085 or max_abs <= 1e-4"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
