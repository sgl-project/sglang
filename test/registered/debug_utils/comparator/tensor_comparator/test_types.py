import json
import sys

import pytest

from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ConfigRecord,
    GeneralWarning,
    ReplicatedMismatchWarning,
    SkipRecord,
    SummaryRecord,
    WarningRecord,
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
            SkipRecord(name="attn", reason="no_baseline"),
            ComparisonRecord(
                name="mlp",
                baseline=_make_tensor_info(),
                target=_make_tensor_info(),
                unified_shape=[4, 8],
                shape_mismatch=False,
            ),
            SummaryRecord(total=10, passed=8, failed=1, skipped=1),
            WarningRecord(
                warnings=[GeneralWarning(category="test", message="test warning")],
            ),
        ]:
            restored = parse_record_json(record.model_dump_json())
            assert type(restored) is type(record)
            assert restored == record


def _make_warning(**overrides) -> ReplicatedMismatchWarning:
    defaults: dict = dict(
        axis="tp",
        group_index=0,
        differing_index=1,
        baseline_index=0,
        max_abs_diff=0.1,
    )
    defaults.update(overrides)
    return ReplicatedMismatchWarning(**defaults)


class TestWarnings:
    def test_comparison_record_failed_when_diff_passed_but_warnings(self):
        """ComparisonRecord with diff.passed=True but warnings → category=='failed'."""
        record = ComparisonRecord(
            name="hidden",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(passed=True),
            warnings=[_make_warning()],
        )
        assert record.category == "failed"

    def test_skip_record_failed_when_warnings(self):
        """SkipRecord with warnings → category=='failed' instead of 'skipped'."""
        record = SkipRecord(
            name="x",
            reason="no_baseline",
            warnings=[_make_warning()],
        )
        assert record.category == "failed"

    def test_warnings_json_round_trip(self):
        """warnings survive model_dump_json → parse_record_json round-trip."""
        warning = _make_warning(
            axis="cp",
            group_index=2,
            differing_index=3,
            baseline_index=0,
            max_abs_diff=0.42,
        )
        record = ComparisonRecord(
            name="mlp",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
            warnings=[warning],
        )

        restored = parse_record_json(record.model_dump_json())
        assert isinstance(restored, ComparisonRecord)
        assert len(restored.warnings) == 1

        restored_warning = restored.warnings[0]
        assert restored_warning.axis == "cp"
        assert restored_warning.group_index == 2
        assert restored_warning.differing_index == 3
        assert restored_warning.baseline_index == 0
        assert restored_warning.max_abs_diff == pytest.approx(0.42)

    def test_any_warning_discriminated_union_round_trip(self):
        """All AnyWarning variants survive JSON round-trip via a WarningRecord."""
        all_warnings = [
            ReplicatedMismatchWarning(
                axis="tp",
                group_index=0,
                differing_index=1,
                baseline_index=0,
                max_abs_diff=0.1,
            ),
            GeneralWarning(
                category="aux_tensors_missing",
                message="Aux tensors missing, skipping token alignment",
            ),
            GeneralWarning(
                category="rids_mismatch",
                message="rids mismatch across ranks: rank 0 has [1,2,3], "
                "rank 1 has [4,5,6]",
            ),
        ]

        record = WarningRecord(warnings=all_warnings)
        restored = parse_record_json(record.model_dump_json())
        assert isinstance(restored, WarningRecord)
        assert len(restored.warnings) == len(all_warnings)

        for original, parsed in zip(all_warnings, restored.warnings):
            assert type(parsed) is type(original)
            assert parsed == original


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
