import json
import sys

import pytest

from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.types import (
    DiffInfo,
    TensorComparisonInfo,
    TensorInfo,
    TensorStats,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _make_stats(**overrides: float) -> TensorStats:
    defaults = dict(
        mean=0.5,
        std=1.0,
        min=-2.0,
        max=3.0,
        p1=-1.8,
        p5=-1.5,
        p95=2.5,
        p99=2.8,
    )
    defaults.update(overrides)
    return TensorStats(**defaults)


def _make_diff(**overrides) -> DiffInfo:
    defaults = dict(
        rel_diff=1e-4,
        max_abs_diff=5e-4,
        mean_abs_diff=2e-4,
        max_diff_coord=[2, 3],
        baseline_at_max=1.0,
        target_at_max=1.0005,
        passed=True,
    )
    defaults.update(overrides)
    return DiffInfo(**defaults)


def _make_tensor_info(**overrides) -> TensorInfo:
    defaults = dict(
        shape=[4, 8],
        dtype="torch.float32",
        stats=_make_stats(),
    )
    defaults.update(overrides)
    return TensorInfo(**defaults)


def _make_comparison_info(**overrides) -> TensorComparisonInfo:
    defaults = dict(
        name="test_tensor",
        baseline=_make_tensor_info(),
        target=_make_tensor_info(dtype="torch.bfloat16"),
        unified_shape=[4, 8],
        shape_mismatch=False,
        diff=_make_diff(),
    )
    defaults.update(overrides)
    return TensorComparisonInfo(**defaults)


class TestTensorStatsRoundTrip:
    def test_round_trip(self):
        original = _make_stats()
        json_str = original.model_dump_json()
        restored = TensorStats.model_validate_json(json_str)
        assert restored == original

    def test_none_quantiles(self):
        original = _make_stats(p1=None, p5=None, p95=None, p99=None)
        json_str = original.model_dump_json()
        restored = TensorStats.model_validate_json(json_str)
        assert restored.p1 is None
        assert restored.p99 is None

    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            TensorStats(mean=0.0, std=1.0, min=-1.0, max=1.0, bogus=42)


class TestDiffInfoRoundTrip:
    def test_round_trip(self):
        original = _make_diff()
        json_str = original.model_dump_json()
        restored = DiffInfo.model_validate_json(json_str)
        assert restored == original

    def test_coord_is_list(self):
        diff = _make_diff(max_diff_coord=[0, 5, 10])
        data = diff.model_dump()
        assert data["max_diff_coord"] == [0, 5, 10]

    def test_passed_field(self):
        diff_pass = _make_diff(passed=True)
        diff_fail = _make_diff(passed=False)
        assert diff_pass.passed is True
        assert diff_fail.passed is False


class TestTensorInfoRoundTrip:
    def test_round_trip(self):
        original = _make_tensor_info()
        json_str = original.model_dump_json()
        restored = TensorInfo.model_validate_json(json_str)
        assert restored == original

    def test_shape_is_list(self):
        info = _make_tensor_info(shape=[1, 2, 3, 4])
        data = info.model_dump()
        assert data["shape"] == [1, 2, 3, 4]

    def test_dtype_is_string(self):
        info = _make_tensor_info(dtype="torch.bfloat16")
        data = info.model_dump()
        assert data["dtype"] == "torch.bfloat16"


class TestTensorComparisonInfoRoundTrip:
    def test_round_trip(self):
        original = _make_comparison_info()
        json_str = original.model_dump_json()
        restored = TensorComparisonInfo.model_validate_json(json_str)
        assert restored == original

    def test_json_is_valid(self):
        info = _make_comparison_info()
        json_str = info.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["name"] == "test_tensor"
        assert parsed["baseline"]["shape"] == [4, 8]
        assert parsed["baseline"]["dtype"] == "torch.float32"
        assert parsed["diff"]["max_diff_coord"] == [2, 3]
        assert parsed["shape_mismatch"] is False

    def test_with_none_diff(self):
        info = _make_comparison_info(diff=None, shape_mismatch=True)
        json_str = info.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["diff"] is None
        assert parsed["shape_mismatch"] is True

    def test_with_downcast(self):
        info = _make_comparison_info(
            diff_downcast=_make_diff(rel_diff=1e-5),
            downcast_dtype="torch.bfloat16",
        )
        json_str = info.model_dump_json()
        restored = TensorComparisonInfo.model_validate_json(json_str)
        assert restored.downcast_dtype == "torch.bfloat16"
        assert restored.diff_downcast is not None


class TestRecordTypes:
    def test_config_record(self):
        record = ConfigRecord(
            baseline_path="/tmp/baseline",
            target_path="/tmp/target",
            diff_threshold=1e-3,
            start_step=0,
            end_step=100,
        )
        parsed = json.loads(record.model_dump_json())
        assert parsed["type"] == "config"
        assert parsed["baseline_path"] == "/tmp/baseline"

    def test_config_record_round_trip(self):
        original = ConfigRecord(
            baseline_path="/a",
            target_path="/b",
            diff_threshold=0.01,
            start_step=5,
            end_step=50,
        )
        restored = ConfigRecord.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_skip_record(self):
        record = SkipRecord(name="attention", reason="no_baseline")
        parsed = json.loads(record.model_dump_json())
        assert parsed["type"] == "skip"
        assert parsed["name"] == "attention"
        assert parsed["reason"] == "no_baseline"

    def test_comparison_record_inherits_fields(self):
        record = ComparisonRecord(
            name="hidden_states",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )
        parsed = json.loads(record.model_dump_json())
        assert parsed["type"] == "comparison"
        assert parsed["name"] == "hidden_states"
        assert "baseline" in parsed
        assert "diff" in parsed

    def test_comparison_record_round_trip(self):
        original = ComparisonRecord(
            name="mlp",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
        )
        restored = ComparisonRecord.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_summary_record(self):
        record = SummaryRecord(total=100, passed=95, failed=3, skipped=2)
        parsed = json.loads(record.model_dump_json())
        assert parsed["type"] == "summary"
        assert parsed["total"] == 100
        assert parsed["passed"] == 95

    def test_summary_record_round_trip(self):
        original = SummaryRecord(total=10, passed=8, failed=1, skipped=1)
        restored = SummaryRecord.model_validate_json(original.model_dump_json())
        assert restored == original


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
