import sys
from io import StringIO

import pytest
from registered.debug_utils.comparator.testing_helpers import make_diff as _make_diff
from registered.debug_utils.comparator.testing_helpers import (
    make_tensor_info as _make_tensor_info,
)
from rich.console import Console, Group
from rich.panel import Panel

from sglang.srt.debug_utils.comparator.aligner.axis_aligner import AxisAlignerPlan
from sglang.srt.debug_utils.comparator.aligner.entrypoint.traced_types import (
    TracedAlignerPlan,
    TracedSidePlan,
    TracedStepPlan,
    TracedSubPlan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import (
    ReordererPlan,
    ZigzagToNaturalParams,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    TokenAlignerPlan,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    ConcatParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims_spec import ParallelAxis, TokenLayout
from sglang.srt.debug_utils.comparator.output_types import (
    ConfigRecord,
    ErrorLog,
    InfoLog,
    LogRecord,
    NonTensorComparisonRecord,
    RecordLocation,
    SkipComparisonRecord,
    SummaryRecord,
    TensorComparisonRecord,
    _format_aligner_plan,
    _split_logs,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _render_rich(renderable: object) -> str:
    buf: StringIO = StringIO()
    Console(file=buf, force_terminal=False, width=120).print(renderable)
    return buf.getvalue().rstrip("\n")


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_split_logs_mixed_list() -> None:
    """_split_logs correctly partitions a mixed list of ErrorLog and InfoLog."""
    errors, infos = _split_logs(
        [
            ErrorLog(category="a", message="err"),
            InfoLog(category="b", message="info"),
            ErrorLog(category="c", message="err2"),
        ]
    )
    assert len(errors) == 2
    assert len(infos) == 1
    assert errors[0].message == "err"
    assert errors[1].message == "err2"
    assert infos[0].message == "info"


def test_log_record_to_text_format() -> None:
    """LogRecord.to_text() renders errors with ✗ and infos with ℹ markers."""
    record = LogRecord(
        errors=[ErrorLog(category="a", message="bad thing")],
        infos=[InfoLog(category="b", message="fyi")],
    )
    text: str = record.to_text()
    assert "✗ bad thing" in text
    assert "ℹ fyi" in text


class TestLogRecord:
    def test_format_body_returns_empty(self) -> None:
        record: LogRecord = LogRecord()
        assert record._format_body() == ""

    def test_format_rich_body_returns_empty(self) -> None:
        record: LogRecord = LogRecord()
        assert record._format_rich_body() == ""

    def test_to_text_empty_no_logs(self) -> None:
        record: LogRecord = LogRecord()
        assert record.to_text() == ""

    def test_to_text_with_errors_and_infos(self) -> None:
        record: LogRecord = LogRecord(
            errors=[ErrorLog(category="a", message="bad thing")],
            infos=[InfoLog(category="b", message="fyi")],
        )
        text: str = record.to_text()
        assert text == "\n  ✗ bad thing\n  ℹ fyi"


# ---------------------------------------------------------------------------
# ConfigRecord
# ---------------------------------------------------------------------------


class TestConfigRecord:
    def test_format_body(self) -> None:
        record: ConfigRecord = ConfigRecord(config={"a": 1, "b": "two"})
        assert record._format_body() == "Config: {'a': 1, 'b': 'two'}"

    def test_format_rich_body(self) -> None:
        record: ConfigRecord = ConfigRecord(config={"threshold": 0.001, "mode": "fast"})
        body = record._format_rich_body()

        assert isinstance(body, Panel)
        rendered: str = _render_rich(body)
        assert rendered == (
            "╭───────────────────────────────────────────────── Comparator Config "
            "──────────────────────────────────────────────────╮\n"
            "│   threshold : 0.001"
            "                                                                                                  │\n"
            "│   mode : fast"
            "                                                                                                        │\n"
            "╰──────────────────────────────────────────────────────────────────────"
            "────────────────────────────────────────────────╯"
        )

    def test_to_text_with_errors(self) -> None:
        record: ConfigRecord = ConfigRecord(
            config={"x": 1},
            errors=[ErrorLog(category="cfg", message="bad config")],
        )
        text: str = record.to_text()
        assert text.startswith("Config: {'x': 1}")
        assert "✗ bad config" in text


# ---------------------------------------------------------------------------
# SkipComparisonRecord
# ---------------------------------------------------------------------------


class TestSkipComparisonRecord:
    def test_format_body_no_step(self) -> None:
        record: SkipComparisonRecord = SkipComparisonRecord(
            name="layer.weight",
            reason="zero-dim tensor",
        )
        assert record._format_body() == "Skip: layer.weight (zero-dim tensor)"

    def test_format_body_with_step(self) -> None:
        record: SkipComparisonRecord = SkipComparisonRecord(
            name="layer.weight",
            reason="scalar",
            location=RecordLocation(step=3),
        )
        assert record._format_body() == "Skip: layer.weight (step=3) (scalar)"

    def test_format_rich_body(self) -> None:
        record: SkipComparisonRecord = SkipComparisonRecord(
            name="attn.qkv",
            reason="no baseline",
        )
        body: str = record._format_rich_body()
        assert body == "[dim]⊘ attn.qkv ── skipped (no baseline)[/]"

    def test_category_skipped(self) -> None:
        record: SkipComparisonRecord = SkipComparisonRecord(
            name="x",
            reason="r",
        )
        assert record.category == "skipped"

    def test_category_failed(self) -> None:
        record: SkipComparisonRecord = SkipComparisonRecord(
            name="x",
            reason="r",
            errors=[ErrorLog(category="e", message="boom")],
        )
        assert record.category == "failed"


# ---------------------------------------------------------------------------
# NonTensorComparisonRecord
# ---------------------------------------------------------------------------


class TestNonTensorComparisonRecord:
    def test_format_body_equal(self) -> None:
        record: NonTensorComparisonRecord = NonTensorComparisonRecord(
            name="config.lr",
            baseline_value="0.001",
            target_value="0.001",
            baseline_type="float",
            target_type="float",
            values_equal=True,
        )
        assert record._format_body() == "NonTensor: config.lr = 0.001 (float) [equal]"

    def test_format_body_not_equal(self) -> None:
        record: NonTensorComparisonRecord = NonTensorComparisonRecord(
            name="config.lr",
            baseline_value="0.001",
            target_value="0.01",
            baseline_type="float",
            target_type="float",
            values_equal=False,
        )
        assert record._format_body() == (
            "NonTensor: config.lr\n"
            "  baseline = 0.001 (float)\n"
            "  target   = 0.01 (float)"
        )

    def test_format_rich_body_equal(self) -> None:
        record: NonTensorComparisonRecord = NonTensorComparisonRecord(
            name="config.lr",
            baseline_value="0.001",
            target_value="0.001",
            baseline_type="float",
            target_type="float",
            values_equal=True,
        )
        assert record._format_rich_body() == ("═ config.lr = 0.001 (float) [green]✓[/]")

    def test_format_rich_body_not_equal(self) -> None:
        record: NonTensorComparisonRecord = NonTensorComparisonRecord(
            name="config.lr",
            baseline_value="0.001",
            target_value="0.01",
            baseline_type="float",
            target_type="float",
            values_equal=False,
        )
        assert record._format_rich_body() == (
            "═ [bold red]config.lr[/]\n"
            "  baseline = 0.001 (float)\n"
            "  target   = 0.01 (float)"
        )

    def test_with_step(self) -> None:
        record: NonTensorComparisonRecord = NonTensorComparisonRecord(
            name="bias",
            baseline_value="True",
            target_value="True",
            baseline_type="bool",
            target_type="bool",
            values_equal=True,
            location=RecordLocation(step=5),
        )
        assert "(step=5)" in record._format_body()

    def test_category(self) -> None:
        passed: NonTensorComparisonRecord = NonTensorComparisonRecord(
            name="x",
            baseline_value="1",
            target_value="1",
            baseline_type="int",
            target_type="int",
            values_equal=True,
        )
        failed: NonTensorComparisonRecord = NonTensorComparisonRecord(
            name="x",
            baseline_value="1",
            target_value="2",
            baseline_type="int",
            target_type="int",
            values_equal=False,
        )
        assert passed.category == "passed"
        assert failed.category == "failed"


# ---------------------------------------------------------------------------
# SummaryRecord
# ---------------------------------------------------------------------------


class TestSummaryRecord:
    def test_format_body(self) -> None:
        record: SummaryRecord = SummaryRecord(
            total=10,
            passed=7,
            failed=2,
            skipped=1,
        )
        assert record._format_body() == (
            "Summary: 7 passed, 2 failed, 1 skipped (total 10)"
        )

    def test_format_rich_body(self) -> None:
        record: SummaryRecord = SummaryRecord(
            total=10,
            passed=7,
            failed=2,
            skipped=1,
        )
        body = record._format_rich_body()
        assert isinstance(body, Panel)

        rendered: str = _render_rich(body)
        assert rendered == (
            "╭────────────────────────────────────────────────────── SUMMARY "
            "───────────────────────────────────────────────────────╮\n"
            "│ 7 passed │ 2 failed │ 1 skipped │ 10 total"
            "                                                                           │\n"
            "╰──────────────────────────────────────────────────────────────────────"
            "────────────────────────────────────────────────╯"
        )

    def test_validation_error(self) -> None:
        with pytest.raises(ValueError, match="total=5 !="):
            SummaryRecord(total=5, passed=1, failed=1, skipped=1)


# ---------------------------------------------------------------------------
# TensorComparisonRecord._format_body
# ---------------------------------------------------------------------------


class TestTensorComparisonRecordFormatBody:
    def test_basic(self) -> None:
        record: TensorComparisonRecord = TensorComparisonRecord(
            name="hidden",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )
        body: str = record._format_body()

        assert body == (
            "Raw [shape] [4, 8] vs [4, 8]\t[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\tmax_abs_diff=0.0005\tmean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005"
        )

    def test_with_replicated_checks(self) -> None:
        from sglang.srt.debug_utils.comparator.output_types import ReplicatedCheckResult

        record: TensorComparisonRecord = TensorComparisonRecord(
            name="hidden",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
            replicated_checks=[
                ReplicatedCheckResult(
                    axis="tp",
                    group_index=0,
                    compared_index=1,
                    baseline_index=0,
                    passed=True,
                    atol=1e-3,
                    diff=_make_diff(
                        rel_diff=1e-6, max_abs_diff=1e-5, mean_abs_diff=1e-6
                    ),
                ),
            ],
        )
        body: str = record._format_body()

        assert body == (
            "Raw [shape] [4, 8] vs [4, 8]\t[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\tmax_abs_diff=0.0005\tmean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005\n"
            "Replicated checks:\n"
            "  ✅ axis=tp group=0 idx=1 vs 0: "
            "rel_diff=1.000000e-06 max_abs_diff=1.000000e-05 mean_abs_diff=1.000000e-06"
        )

    def test_with_aligner_plan(self) -> None:
        plan: AlignerPlan = AlignerPlan(
            per_step_plans=Pair(x=[], y=[]),
        )
        traced: TracedAlignerPlan = TracedAlignerPlan(
            plan=plan,
            per_side=Pair(
                x=TracedSidePlan(step_plans=[]),
                y=TracedSidePlan(step_plans=[]),
            ),
        )
        record: TensorComparisonRecord = TensorComparisonRecord(
            name="hidden",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
            traced_plan=traced,
        )
        body: str = record._format_body()

        assert body == (
            "Raw [shape] [4, 8] vs [4, 8]\t[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\tmax_abs_diff=0.0005\tmean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005\n"
            "Aligner Plan:\n"
            "  baseline: (no steps)\n"
            "  target: (no steps)"
        )

    def test_with_step(self) -> None:
        record: TensorComparisonRecord = TensorComparisonRecord(
            name="hidden",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
            location=RecordLocation(step=2),
        )
        body: str = record._format_body()

        assert body == (
            "[step=2] "
            "Raw [shape] [4, 8] vs [4, 8]\t[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\tmax_abs_diff=0.0005\tmean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005"
        )


# ---------------------------------------------------------------------------
# _format_aligner_plan
# ---------------------------------------------------------------------------


def _wrap_plan(plan: AlignerPlan) -> TracedAlignerPlan:
    """Wrap an AlignerPlan into a TracedAlignerPlan with no snapshots."""
    baseline_traced_steps: list[TracedStepPlan] = [
        TracedStepPlan(
            step=sp.step,
            input_object_indices=sp.input_object_indices,
            sub_plans=[TracedSubPlan(plan=sub) for sub in sp.sub_plans],
        )
        for sp in plan.per_step_plans.x
    ]
    target_traced_steps: list[TracedStepPlan] = [
        TracedStepPlan(
            step=sp.step,
            input_object_indices=sp.input_object_indices,
            sub_plans=[TracedSubPlan(plan=sub) for sub in sp.sub_plans],
        )
        for sp in plan.per_step_plans.y
    ]
    return TracedAlignerPlan(
        plan=plan,
        per_side=Pair(
            x=TracedSidePlan(step_plans=baseline_traced_steps),
            y=TracedSidePlan(step_plans=target_traced_steps),
        ),
    )


class TestFormatAlignerPlan:
    def test_passthrough(self) -> None:
        plan: AlignerPlan = AlignerPlan(
            per_step_plans=Pair(x=[], y=[]),
        )
        result: str = _format_aligner_plan(_wrap_plan(plan))

        assert result == (
            "Aligner Plan:\n" "  baseline: (no steps)\n" "  target: (no steps)"
        )

    def test_unsharder(self) -> None:
        unsharder: UnsharderPlan = UnsharderPlan(
            axis=ParallelAxis.TP,
            params=ConcatParams(dim_name="h"),
            groups=[[0, 1]],
        )
        plan: AlignerPlan = AlignerPlan(
            per_step_plans=Pair(
                x=[],
                y=[
                    AlignerPerStepPlan(
                        step=0, input_object_indices=[0, 1], sub_plans=[unsharder]
                    )
                ],
            ),
        )
        result: str = _format_aligner_plan(_wrap_plan(plan))

        assert result == (
            "Aligner Plan:\n" "  baseline: (no steps)\n" "  target: [step=0: unsharder]"
        )

    def test_reorderer(self) -> None:
        reorderer: ReordererPlan = ReordererPlan(
            params=ZigzagToNaturalParams(dim_name="s", cp_size=2),
        )
        plan: AlignerPlan = AlignerPlan(
            per_step_plans=Pair(
                x=[],
                y=[
                    AlignerPerStepPlan(
                        step=0, input_object_indices=[0], sub_plans=[reorderer]
                    )
                ],
            ),
        )
        result: str = _format_aligner_plan(_wrap_plan(plan))

        assert result == (
            "Aligner Plan:\n" "  baseline: (no steps)\n" "  target: [step=0: reorderer]"
        )

    def test_multi_step(self) -> None:
        unsharder: UnsharderPlan = UnsharderPlan(
            axis=ParallelAxis.TP,
            params=ConcatParams(dim_name="h"),
            groups=[[0, 1]],
        )
        reorderer: ReordererPlan = ReordererPlan(
            params=ZigzagToNaturalParams(dim_name="s", cp_size=2),
        )
        plan: AlignerPlan = AlignerPlan(
            per_step_plans=Pair(
                x=[],
                y=[
                    AlignerPerStepPlan(
                        step=0, input_object_indices=[0, 1], sub_plans=[unsharder]
                    ),
                    AlignerPerStepPlan(
                        step=1, input_object_indices=[0], sub_plans=[reorderer]
                    ),
                ],
            ),
        )
        result: str = _format_aligner_plan(_wrap_plan(plan))

        assert result == (
            "Aligner Plan:\n"
            "  baseline: (no steps)\n"
            "  target: [step=0: unsharder; step=1: reorderer]"
        )

    def test_with_token_aligner(self) -> None:
        ta_plan: TokenAlignerPlan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[0, 0, 0], token_index_in_step=[0, 1, 2]),
                y=TokenLocator(steps=[0, 0, 0], token_index_in_step=[0, 1, 2]),
            ),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )
        plan: AlignerPlan = AlignerPlan(
            per_step_plans=Pair(x=[], y=[]),
            token_aligner_plan=ta_plan,
        )
        result: str = _format_aligner_plan(_wrap_plan(plan))

        assert result == (
            "Aligner Plan:\n"
            "  baseline: (no steps)\n"
            "  target: (no steps)\n"
            "  token_aligner: 3 tokens aligned"
        )

    def test_with_axis_aligner(self) -> None:
        aa_plan: AxisAlignerPlan = AxisAlignerPlan(
            pattern=Pair(x="b s d -> s b d", y=None),
        )
        plan: AlignerPlan = AlignerPlan(
            per_step_plans=Pair(x=[], y=[]),
            axis_aligner_plan=aa_plan,
        )
        result: str = _format_aligner_plan(_wrap_plan(plan))

        assert result == (
            "Aligner Plan:\n"
            "  baseline: (no steps)\n"
            "  target: (no steps)\n"
            "  axis_aligner: x: b s d -> s b d"
        )


# ---------------------------------------------------------------------------
# _OutputRecord log attachment (to_text / to_rich)
# ---------------------------------------------------------------------------


class TestOutputRecordLogAttachment:
    def test_to_text_no_logs(self) -> None:
        record: ConfigRecord = ConfigRecord(config={"a": 1})
        text: str = record.to_text()

        assert text == "Config: {'a': 1}"

    def test_to_text_errors_only(self) -> None:
        record: ConfigRecord = ConfigRecord(
            config={"a": 1},
            errors=[ErrorLog(category="x", message="err1")],
        )
        text: str = record.to_text()

        assert text == "Config: {'a': 1}\n  ✗ err1"

    def test_to_text_infos_only(self) -> None:
        record: ConfigRecord = ConfigRecord(
            config={"a": 1},
            infos=[InfoLog(category="x", message="note1")],
        )
        text: str = record.to_text()

        assert text == "Config: {'a': 1}\n  ℹ note1"

    def test_to_text_mixed(self) -> None:
        record: ConfigRecord = ConfigRecord(
            config={"a": 1},
            errors=[ErrorLog(category="x", message="err1")],
            infos=[InfoLog(category="y", message="note1")],
        )
        text: str = record.to_text()

        assert text == "Config: {'a': 1}\n  ✗ err1\n  ℹ note1"

    def test_to_rich_string_body(self) -> None:
        record: SkipComparisonRecord = SkipComparisonRecord(
            name="x",
            reason="r",
            errors=[ErrorLog(category="e", message="oops")],
        )
        body = record.to_rich()

        assert isinstance(body, str)
        assert body == "[dim]⊘ x ── skipped (r)[/]\n  [red]✗ oops[/]"

    def test_to_rich_group_body(self) -> None:
        record: ConfigRecord = ConfigRecord(
            config={"a": 1},
            errors=[ErrorLog(category="e", message="oops")],
        )
        body = record.to_rich()

        # Panel body + log block → Group
        assert isinstance(body, Group)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
