import json
import sys

import pytest

from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    parse_record_json,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.types import (
    DiffInfo,
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


class TestStrictBase:
    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            TensorStats(mean=0.0, std=1.0, min=-1.0, max=1.0, bogus=42)

    def test_rejects_extra_fields_on_diff(self):
        with pytest.raises(Exception):
            DiffInfo(
                rel_diff=0.0,
                max_abs_diff=0.0,
                mean_abs_diff=0.0,
                max_diff_coord=[0],
                baseline_at_max=0.0,
                target_at_max=0.0,
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
                baseline_path="/a",
                target_path="/b",
                diff_threshold=1e-3,
                start_step=0,
                end_step=100,
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
        ]:
            restored = parse_record_json(record.model_dump_json())
            assert type(restored) is type(record)
            assert restored == record


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
