import sys

import pytest
from pydantic import ValidationError

from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    GeneralWarning,
    SkipRecord,
    SummaryRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.types import (
    DiffInfo,
    TensorInfo,
    TensorStats,
)
from sglang.srt.debug_utils.comparator.utils import _check_equal_lengths
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestCheckEqualLengths:
    def test_all_equal(self):
        _check_equal_lengths(a=[1, 2], b=[3, 4])

    def test_empty_lists(self):
        _check_equal_lengths(a=[], b=[])

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            _check_equal_lengths(a=[1, 2], b=[3])


class TestSummaryRecord:
    def test_valid(self):
        record = SummaryRecord(total=10, passed=7, failed=2, skipped=1)
        assert record.total == 10

    def test_total_mismatch(self):
        with pytest.raises(ValidationError, match="total=10"):
            SummaryRecord(total=10, passed=5, failed=2, skipped=1)


def _make_tensor_info() -> TensorInfo:
    return TensorInfo(
        shape=[4, 4],
        dtype="float32",
        stats=TensorStats(mean=0.0, std=1.0, min=-2.0, max=2.0),
    )


def _make_diff_info(*, passed: bool) -> DiffInfo:
    return DiffInfo(
        rel_diff=0.001,
        max_abs_diff=0.01,
        mean_abs_diff=0.005,
        max_diff_coord=[0, 0],
        baseline_at_max=1.0,
        target_at_max=1.01,
        passed=passed,
    )


def _make_comparison_record(
    *,
    diff: DiffInfo | None,
    warnings: list | None = None,
) -> ComparisonRecord:
    ti: TensorInfo = _make_tensor_info()
    return ComparisonRecord(
        name="t",
        baseline=ti,
        target=ti,
        unified_shape=[4, 4],
        shape_mismatch=False,
        diff=diff,
        warnings=warnings or [],
    )


class TestOutputRecordCategories:
    def test_skip_record_with_warnings_is_failed(self) -> None:
        record = SkipRecord(
            name="t",
            reason="test",
            warnings=[GeneralWarning(category="c", message="m")],
        )
        assert record.category == "failed"

    def test_skip_record_no_warnings_is_skipped(self) -> None:
        record = SkipRecord(name="t", reason="test")
        assert record.category == "skipped"

    def test_comparison_record_diff_none_is_failed(self) -> None:
        record: ComparisonRecord = _make_comparison_record(diff=None)
        assert record.category == "failed"

    def test_comparison_record_passed_with_warnings_is_failed(self) -> None:
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
            warnings=[GeneralWarning(category="c", message="m")],
        )
        assert record.category == "failed"

    def test_comparison_record_passed_no_warnings_is_passed(self) -> None:
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        assert record.category == "passed"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
