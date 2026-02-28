import json
import sys

import pytest
from pydantic import ValidationError

from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    PositionalSeqId,
    TokenAlignerPlan,
    TokenAlignerSeqInfo,
    TokenAlignerStepAux,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    AxisInfo,
    ConcatParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis, TokenLayout
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    GeneralWarning,
    NonTensorRecord,
    SkipRecord,
    SummaryRecord,
    parse_record_json,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorInfo,
    TensorStats,
)
from sglang.srt.debug_utils.comparator.utils import Pair, _check_equal_lengths
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


class TestTokenAlignerStepAux:
    def test_valid(self):
        aux = TokenAlignerStepAux(
            input_ids=[10, 20, 30],
            positions=[0, 1, 2],
            seq_lens=[2, 1],
            seq_ids=[
                PositionalSeqId(step=0, seq_index=0),
                PositionalSeqId(step=0, seq_index=1),
            ],
        )
        assert len(aux.input_ids) == 3

    def test_token_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            TokenAlignerStepAux(
                input_ids=[10, 20, 30],
                positions=[0, 1],
                seq_lens=[2, 1],
                seq_ids=[
                    PositionalSeqId(step=0, seq_index=0),
                    PositionalSeqId(step=0, seq_index=1),
                ],
            )

    def test_seq_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            TokenAlignerStepAux(
                input_ids=[10, 20, 30],
                positions=[0, 1, 2],
                seq_lens=[2, 1],
                seq_ids=[PositionalSeqId(step=0, seq_index=0)],
            )

    def test_sum_seq_lens_mismatch(self):
        with pytest.raises(ValueError, match="sum\\(seq_lens\\)"):
            TokenAlignerStepAux(
                input_ids=[10, 20, 30],
                positions=[0, 1, 2],
                seq_lens=[1, 1],
                seq_ids=[
                    PositionalSeqId(step=0, seq_index=0),
                    PositionalSeqId(step=0, seq_index=1),
                ],
            )


class TestTokenAlignerSeqInfo:
    def test_valid(self):
        info = TokenAlignerSeqInfo(
            input_ids=[10, 20, 30],
            positions=[0, 1, 2],
            locator=TokenLocator(steps=[0, 0, 1], token_index_in_step=[0, 1, 0]),
        )
        assert len(info.input_ids) == 3

    def test_length_mismatch(self):
        with pytest.raises(ValidationError):
            TokenAlignerSeqInfo(
                input_ids=[10, 20, 30],
                positions=[0, 1, 2],
                locator=TokenLocator(steps=[0, 0], token_index_in_step=[0, 1, 0]),
            )

    def test_positions_not_sequential(self):
        with pytest.raises(ValidationError, match="positions must be"):
            TokenAlignerSeqInfo(
                input_ids=[10, 20, 30],
                positions=[0, 2, 1],
                locator=TokenLocator(steps=[0, 0, 1], token_index_in_step=[0, 1, 0]),
            )


class TestTokenAlignerPlan:
    def test_valid(self):
        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[0, 0, 1], token_index_in_step=[0, 1, 0]),
                y=TokenLocator(steps=[0, 1, 1], token_index_in_step=[0, 0, 1]),
            ),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )
        assert len(plan.locators.x.steps) == 3

    def test_length_mismatch(self):
        with pytest.raises(ValidationError, match="Length mismatch"):
            TokenAlignerPlan(
                locators=Pair(
                    x=TokenLocator(steps=[0, 0], token_index_in_step=[0, 1]),
                    y=TokenLocator(steps=[0, 1, 1], token_index_in_step=[0, 0, 1]),
                ),
                layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
            )


class TestSummaryRecord:
    def test_valid(self):
        record = SummaryRecord(total=10, passed=7, failed=2, skipped=1)
        assert record.total == 10

    def test_total_mismatch(self):
        with pytest.raises(ValidationError, match="total=10"):
            SummaryRecord(total=10, passed=5, failed=2, skipped=1)


class TestAxisInfo:
    def test_valid(self):
        info = AxisInfo(axis_rank=0, axis_size=4)
        assert info.axis_rank == 0

    def test_axis_size_zero(self):
        with pytest.raises(ValidationError, match="axis_size must be > 0"):
            AxisInfo(axis_rank=0, axis_size=0)

    def test_axis_size_negative(self):
        with pytest.raises(ValidationError, match="axis_size must be > 0"):
            AxisInfo(axis_rank=0, axis_size=-1)

    def test_axis_rank_negative(self):
        with pytest.raises(ValidationError, match="axis_rank must be in"):
            AxisInfo(axis_rank=-1, axis_size=4)

    def test_axis_rank_too_large(self):
        with pytest.raises(ValidationError, match="axis_rank must be in"):
            AxisInfo(axis_rank=4, axis_size=4)

    def test_axis_rank_equals_size_minus_one(self):
        info = AxisInfo(axis_rank=3, axis_size=4)
        assert info.axis_rank == 3


def _make_tensor_info() -> TensorInfo:
    return TensorInfo(
        shape=[4, 4],
        dtype="float32",
        stats=TensorStats(mean=0.0, abs_mean=0.8, std=1.0, min=-2.0, max=2.0),
    )


def _make_diff_info(*, passed: bool) -> DiffInfo:
    return DiffInfo(
        rel_diff=0.001,
        max_abs_diff=0.01,
        mean_abs_diff=0.005,
        max_diff_coord=[0, 0],
        baseline_at_max=1.0,
        target_at_max=1.01,
        diff_threshold=1e-3,
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

    def test_non_tensor_record_equal_is_passed(self) -> None:
        record = NonTensorRecord(
            name="sm_scale",
            baseline_value="0.125",
            target_value="0.125",
            baseline_type="float",
            target_type="float",
            values_equal=True,
        )
        assert record.category == "passed"

    def test_non_tensor_record_different_is_failed(self) -> None:
        record = NonTensorRecord(
            name="sm_scale",
            baseline_value="0.125",
            target_value="0.25",
            baseline_type="float",
            target_type="float",
            values_equal=False,
        )
        assert record.category == "failed"

    def test_non_tensor_record_with_warnings_is_failed(self) -> None:
        record = NonTensorRecord(
            name="sm_scale",
            baseline_value="0.125",
            target_value="0.125",
            baseline_type="float",
            target_type="float",
            values_equal=True,
            warnings=[GeneralWarning(category="c", message="m")],
        )
        assert record.category == "failed"

    def test_non_tensor_record_json_roundtrip(self) -> None:
        record = NonTensorRecord(
            name="sm_scale",
            baseline_value="0.125",
            target_value="0.25",
            baseline_type="float",
            target_type="float",
            values_equal=False,
        )
        json_str: str = record.model_dump_json()
        roundtripped = parse_record_json(json_str)
        assert isinstance(roundtripped, NonTensorRecord)
        assert roundtripped.name == "sm_scale"
        assert roundtripped.values_equal is False
        assert roundtripped.baseline_value == "0.125"
        assert roundtripped.target_value == "0.25"

    def test_non_tensor_record_text_format_equal(self) -> None:
        record = NonTensorRecord(
            name="sm_scale",
            baseline_value="0.125",
            target_value="0.125",
            baseline_type="float",
            target_type="float",
            values_equal=True,
        )
        text: str = record.to_text()
        assert "sm_scale" in text
        assert "[equal]" in text

    def test_non_tensor_record_text_format_different(self) -> None:
        record = NonTensorRecord(
            name="sm_scale",
            baseline_value="0.125",
            target_value="0.25",
            baseline_type="float",
            target_type="float",
            values_equal=False,
        )
        text: str = record.to_text()
        assert "baseline" in text
        assert "target" in text


def _make_aligner_plan() -> AlignerPlan:
    unsharder = UnsharderPlan(
        axis=ParallelAxis.TP,
        params=ConcatParams(dim_name="h"),
        groups=[[0, 1]],
    )
    return AlignerPlan(
        per_step_plans=Pair(
            x=[
                AlignerPerStepPlan(
                    step=0, input_object_indices=[0, 1], sub_plans=[unsharder]
                )
            ],
            y=[
                AlignerPerStepPlan(
                    step=0, input_object_indices=[0, 1], sub_plans=[unsharder]
                )
            ],
        ),
    )


class TestAlignerPlanInComparisonRecord:
    def test_comparison_record_with_aligner_plan(self) -> None:
        plan: AlignerPlan = _make_aligner_plan()
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        record_with_plan = record.model_copy(update={"aligner_plan": plan})
        assert record_with_plan.aligner_plan is not None
        assert record_with_plan.aligner_plan.per_step_plans.x[0].step == 0

    def test_aligner_plan_json_roundtrip(self) -> None:
        plan: AlignerPlan = _make_aligner_plan()
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        record_with_plan = record.model_copy(update={"aligner_plan": plan})

        json_str: str = record_with_plan.model_dump_json()
        parsed = json.loads(json_str)
        assert "aligner_plan" in parsed
        assert (
            parsed["aligner_plan"]["per_step_plans"]["x"][0]["sub_plans"][0]["type"]
            == "unsharder"
        )

        roundtripped: ComparisonRecord = parse_record_json(json_str)
        assert roundtripped.aligner_plan is not None
        assert (
            roundtripped.aligner_plan.per_step_plans.x[0].sub_plans[0].type
            == "unsharder"
        )

    def test_comparison_record_without_aligner_plan(self) -> None:
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        json_str: str = record.model_dump_json()
        roundtripped: ComparisonRecord = parse_record_json(json_str)
        assert roundtripped.aligner_plan is None

    def test_aligner_plan_text_format(self) -> None:
        plan: AlignerPlan = _make_aligner_plan()
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        record_with_plan = record.model_copy(update={"aligner_plan": plan})

        text: str = record_with_plan.to_text()
        assert "Aligner Plan:" in text
        assert "unsharder" in text


def _make_aligner_plan() -> AlignerPlan:
    unsharder = UnsharderPlan(
        axis=ParallelAxis.TP,
        params=ConcatParams(dim_name="h"),
        groups=[[0, 1]],
    )
    return AlignerPlan(
        per_step_plans=Pair(
            x=[
                AlignerPerStepPlan(
                    step=0, input_object_indices=[0, 1], sub_plans=[unsharder]
                )
            ],
            y=[
                AlignerPerStepPlan(
                    step=0, input_object_indices=[0, 1], sub_plans=[unsharder]
                )
            ],
        ),
    )


class TestAlignerPlanInComparisonRecord:
    def test_comparison_record_with_aligner_plan(self) -> None:
        plan: AlignerPlan = _make_aligner_plan()
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        record_with_plan = record.model_copy(update={"aligner_plan": plan})
        assert record_with_plan.aligner_plan is not None
        assert record_with_plan.aligner_plan.per_step_plans.x[0].step == 0

    def test_aligner_plan_json_roundtrip(self) -> None:
        plan: AlignerPlan = _make_aligner_plan()
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        record_with_plan = record.model_copy(update={"aligner_plan": plan})

        json_str: str = record_with_plan.model_dump_json()
        parsed = json.loads(json_str)
        assert "aligner_plan" in parsed
        assert (
            parsed["aligner_plan"]["per_step_plans"]["x"][0]["sub_plans"][0]["type"]
            == "unsharder"
        )

        roundtripped: ComparisonRecord = parse_record_json(json_str)
        assert roundtripped.aligner_plan is not None
        assert (
            roundtripped.aligner_plan.per_step_plans.x[0].sub_plans[0].type
            == "unsharder"
        )

    def test_comparison_record_without_aligner_plan(self) -> None:
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        json_str: str = record.model_dump_json()
        roundtripped: ComparisonRecord = parse_record_json(json_str)
        assert roundtripped.aligner_plan is None

    def test_aligner_plan_text_format(self) -> None:
        plan: AlignerPlan = _make_aligner_plan()
        record: ComparisonRecord = _make_comparison_record(
            diff=_make_diff_info(passed=True),
        )
        record_with_plan = record.model_copy(update={"aligner_plan": plan})

        text: str = record_with_plan.to_text()
        assert "Aligner Plan:" in text
        assert "unsharder" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
