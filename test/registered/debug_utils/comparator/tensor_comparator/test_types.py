import json
import sys

import pytest

from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonSkipRecord,
    ComparisonTensorRecord,
    ConfigRecord,
    ErrorLog,
    InfoLog,
    LogRecord,
    ReplicatedCheckResult,
    SummaryRecord,
    parse_record_json,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorInfo,
    TensorStats,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _make_stats(**overrides) -> TensorStats:
    defaults: dict = dict(
        mean=0.5,
        abs_mean=1.2,
        std=1.0,
        min=-2.0,
        max=3.0,
        percentiles={1: -1.8, 5: -1.5, 50: 0.0, 95: 2.5, 99: 2.8},
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
        diff_threshold=1e-3,
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


class TestStrictBase:
    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            TensorStats(mean=0.0, abs_mean=0.5, std=1.0, min=-1.0, max=1.0, bogus=42)

    def test_rejects_extra_fields_on_diff(self):
        with pytest.raises(Exception):
            DiffInfo(
                rel_diff=0.0,
                max_abs_diff=0.0,
                mean_abs_diff=0.0,
                max_diff_coord=[0],
                baseline_at_max=0.0,
                target_at_max=0.0,
                diff_threshold=1e-3,
                passed=True,
                extra_field=123,
            )


class TestRecordTypes:
    def test_comparison_record_inherits_tensor_fields(self):
        record = ComparisonTensorRecord(
            name="hidden_states",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )
        parsed = json.loads(record.model_dump_json())
        assert parsed["type"] == "comparison_tensor"
        assert parsed["name"] == "hidden_states"
        assert "baseline" in parsed
        assert "diff" in parsed

    def test_discriminated_union_parsing(self):
        for record in [
            ConfigRecord(
                config={
                    "baseline_path": "/a",
                    "target_path": "/b",
                    "diff_threshold": 1e-3,
                    "start_step": 0,
                    "end_step": 100,
                }
            ),
            ComparisonSkipRecord(name="attn", reason="no_baseline"),
            ComparisonTensorRecord(
                name="mlp",
                baseline=_make_tensor_info(),
                target=_make_tensor_info(),
                unified_shape=[4, 8],
                shape_mismatch=False,
            ),
            SummaryRecord(total=10, passed=8, failed=1, skipped=1),
            LogRecord(
                errors=[ErrorLog(category="test", message="test warning")],
            ),
        ]:
            restored = parse_record_json(record.model_dump_json())
            assert type(restored) is type(record)
            assert restored == record


def _make_replicated_check(**overrides) -> ReplicatedCheckResult:
    defaults: dict = dict(
        axis="tp",
        group_index=0,
        compared_index=1,
        baseline_index=0,
        passed=False,
        atol=1e-6,
        diff=_make_diff(
            rel_diff=0.1,
            max_abs_diff=0.1,
            mean_abs_diff=0.05,
            diff_threshold=1e-6,
            passed=False,
        ),
    )
    defaults.update(overrides)
    return ReplicatedCheckResult(**defaults)


class TestWarnings:
    def test_comparison_record_failed_when_diff_passed_but_errors(self):
        """ComparisonTensorRecord with diff.passed=True but errors → category=='failed'."""
        record = ComparisonTensorRecord(
            name="hidden",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(passed=True),
            errors=[ErrorLog(category="test", message="some warning")],
        )
        assert record.category == "failed"

    def test_skip_record_failed_when_errors(self):
        """ComparisonSkipRecord with errors → category=='failed' instead of 'skipped'."""
        record = ComparisonSkipRecord(
            name="x",
            reason="no_baseline",
            errors=[ErrorLog(category="test", message="some warning")],
        )
        assert record.category == "failed"

    def test_replicated_checks_all_passed(self):
        """ComparisonTensorRecord with all replicated_checks passed → category=='passed'."""
        record = ComparisonTensorRecord(
            name="hidden",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(passed=True),
            replicated_checks=[_make_replicated_check(passed=True)],
        )
        assert record.category == "passed"

    def test_replicated_checks_failed_means_record_failed(self):
        """ComparisonTensorRecord with any replicated_check.passed=False → category=='failed'."""
        record = ComparisonTensorRecord(
            name="hidden",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(passed=True),
            replicated_checks=[_make_replicated_check(passed=False)],
        )
        assert record.category == "failed"

    def test_replicated_check_json_round_trip(self):
        """ReplicatedCheckResult survives JSON round-trip via ComparisonTensorRecord."""
        check = _make_replicated_check(
            axis="cp",
            group_index=2,
            compared_index=3,
            baseline_index=0,
            passed=False,
        )
        record = ComparisonTensorRecord(
            name="mlp",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
            replicated_checks=[check],
        )

        restored = parse_record_json(record.model_dump_json())
        assert isinstance(restored, ComparisonTensorRecord)
        assert len(restored.replicated_checks) == 1

        restored_check: ReplicatedCheckResult = restored.replicated_checks[0]
        assert restored_check.axis == "cp"
        assert restored_check.group_index == 2
        assert restored_check.compared_index == 3
        assert restored_check.baseline_index == 0
        assert not restored_check.passed

    def test_any_log_discriminated_union_round_trip(self):
        """ErrorLog and InfoLog survive JSON round-trip via a LogRecord."""
        all_errors = [
            ErrorLog(
                category="rids_mismatch",
                message="rids mismatch across ranks: rank 0 has [1,2,3], "
                "rank 1 has [4,5,6]",
            ),
        ]
        all_infos = [
            InfoLog(
                category="aux_tensors_missing",
                message="Aux tensors missing, skipping token alignment",
            ),
        ]

        record = LogRecord(errors=all_errors, infos=all_infos)
        restored = parse_record_json(record.model_dump_json())
        assert isinstance(restored, LogRecord)
        assert len(restored.errors) == len(all_errors)
        assert len(restored.infos) == len(all_infos)

        for original, parsed in zip(all_errors, restored.errors):
            assert type(parsed) is type(original)
            assert parsed == original

        for original, parsed in zip(all_infos, restored.infos):
            assert type(parsed) is type(original)
            assert parsed == original


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
