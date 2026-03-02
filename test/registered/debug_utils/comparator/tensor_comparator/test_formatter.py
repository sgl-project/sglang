import sys

import pytest
from registered.debug_utils.comparator.testing_helpers import make_diff as _make_diff
from registered.debug_utils.comparator.testing_helpers import make_stats as _make_stats
from registered.debug_utils.comparator.testing_helpers import (
    make_tensor_info as _make_tensor_info,
)

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
    BundleFileInfo,
    BundleSideInfo,
    ReplicatedCheckResult,
    ShapeSnapshot,
    TensorComparisonRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
    _format_abs_diff_percentiles_rich,
    _format_bundle_section,
    _format_plan_section_rich,
    _format_stats_rich,
    format_comparison,
    format_comparison_rich,
    format_replicated_checks,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
    TensorStats,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


# Snapshot strings below are intentionally spelled out in full per test.
# The shared skeleton (stats block, diff block) looks duplicated, but keeping
# each test self-contained makes failures immediately readable without chasing
# helper functions.  Do not extract common fragments.
class TestFormatComparison:
    def test_normal(self):
        info = TensorComparisonInfo(
            name="test",
            baseline=_make_tensor_info(
                stats=_make_stats(mean=0.1, std=1.0, min=-2.0, max=2.0),
            ),
            target=_make_tensor_info(
                stats=_make_stats(mean=0.1001, std=1.0001, min=-2.0001, max=2.0001),
            ),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )

        assert format_comparison(info) == (
            "Raw [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.1000 vs 0.1001 (diff: 0.0001)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0001 (diff: 0.0001)\n"
            "[min] -2.0000 vs -2.0001 (diff: -0.0001)\n"
            "[max] 2.0000 vs 2.0001 (diff: 0.0001)\n"
            "[p1] -1.8000 vs -1.8000 (diff: 0.0000)\n"
            "[p5] -1.5000 vs -1.5000 (diff: 0.0000)\n"
            "[p50] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[p95] 1.5000 vs 1.5000 (diff: 0.0000)\n"
            "[p99] 1.8000 vs 1.8000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\tmax_abs_diff=0.0005\tmean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005"
        )

    def test_shape_mismatch(self):
        info = TensorComparisonInfo(
            name="mismatch",
            baseline=_make_tensor_info(shape=[3, 4]),
            target=_make_tensor_info(shape=[5, 6]),
            unified_shape=[3, 4],
            shape_mismatch=True,
        )

        assert format_comparison(info) == (
            "Raw [shape] [3, 4] vs [5, 6]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [3, 4] vs [5, 6]\t"
            "[dtype] torch.float32 vs torch.float32\n"
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
            "⚠️ Shape mismatch"
        )

    def test_with_downcast(self):
        info = TensorComparisonInfo(
            name="downcast",
            baseline=_make_tensor_info(),
            target=_make_tensor_info(dtype="torch.bfloat16"),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(
                rel_diff=0.002, max_abs_diff=0.005, mean_abs_diff=0.001, passed=False
            ),
            diff_downcast=_make_diff(
                rel_diff=0.0001, max_abs_diff=0.0005, mean_abs_diff=0.0002, passed=True
            ),
            downcast_dtype="torch.bfloat16",
        )

        assert format_comparison(info) == (
            "Raw [shape] [4, 8] vs [4, 8]\t"
            "[🟠dtype] torch.float32 vs torch.bfloat16\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.bfloat16\n"
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
            "❌ rel_diff=0.002\tmax_abs_diff=0.005\tmean_abs_diff=0.001\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005\n"
            "When downcast to torch.bfloat16: "
            "✅ rel_diff=0.0001\tmax_abs_diff=0.0005\tmean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005"
        )

    def test_with_shape_unification(self):
        info = TensorComparisonInfo(
            name="unify",
            baseline=_make_tensor_info(shape=[1, 1, 4, 8]),
            target=_make_tensor_info(),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )

        assert format_comparison(info) == (
            "Raw [shape] [1, 1, 4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "Unify shape: [1, 1, 4, 8] -> [4, 8] "
            "(to match [4, 8])\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
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
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005"
        )

    def test_with_samples(self):
        info = TensorComparisonInfo(
            name="samples",
            baseline=_make_tensor_info(sample="tensor([0.1, 0.2, ...])"),
            target=_make_tensor_info(sample="tensor([0.1, 0.3, ...])"),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(),
        )

        assert format_comparison(info) == (
            "Raw [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
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
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005\n"
            "[abs_diff] p1=0.0001 p5=0.0001 p50=0.0002 p95=0.0004 p99=0.0005\n"
            "x_baseline(sample)=tensor([0.1, 0.2, ...])\n"
            "x_target(sample)=tensor([0.1, 0.3, ...])"
        )

    def test_empty_percentiles(self):
        stats_no_quantiles = _make_stats(percentiles={})

        info = TensorComparisonInfo(
            name="no_quantiles",
            baseline=_make_tensor_info(stats=stats_no_quantiles),
            target=_make_tensor_info(stats=stats_no_quantiles),
            unified_shape=[4, 8],
            shape_mismatch=False,
            diff=_make_diff(abs_diff_percentiles={}),
        )

        assert format_comparison(info) == (
            "Raw [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "After unify [shape] [4, 8] vs [4, 8]\t"
            "[dtype] torch.float32 vs torch.float32\n"
            "[mean] 0.0000 vs 0.0000 (diff: 0.0000)\n"
            "[abs_mean] 0.8000 vs 0.8000 (diff: 0.0000)\n"
            "[std] 1.0000 vs 1.0000 (diff: 0.0000)\n"
            "[min] -2.0000 vs -2.0000 (diff: 0.0000)\n"
            "[max] 2.0000 vs 2.0000 (diff: 0.0000)\n"
            "✅ rel_diff=0.0001\tmax_abs_diff=0.0005\tmean_abs_diff=0.0002\n"
            "max_abs_diff happens at coord=[2, 3] with "
            "baseline=1.0 target=1.0005"
        )


def _make_comparison_record(
    name: str = "hidden_states",
    shape: list[int] | None = None,
    dtype: str = "torch.float32",
    diff: DiffInfo | None = None,
    shape_mismatch: bool = False,
    sample: str | None = None,
    diff_downcast: DiffInfo | None = None,
    downcast_dtype: str | None = None,
    replicated_checks: list[ReplicatedCheckResult] | None = None,
    raw_bundle_info: Pair[BundleSideInfo] | None = None,
    traced_plan: TracedAlignerPlan | None = None,
) -> TensorComparisonRecord:
    s: list[int] = shape if shape is not None else [4, 8]
    return TensorComparisonRecord(
        name=name,
        baseline=_make_tensor_info(shape=s, dtype=dtype, sample=sample),
        target=_make_tensor_info(shape=s, dtype=dtype, sample=sample),
        unified_shape=s,
        shape_mismatch=shape_mismatch,
        diff=diff,
        diff_downcast=diff_downcast,
        downcast_dtype=downcast_dtype,
        replicated_checks=replicated_checks or [],
        raw_bundle_info=raw_bundle_info,
        traced_plan=traced_plan,
    )


def _make_bundle_side_info(
    num_files: int = 2,
    shape: list[int] | None = None,
    dtype: str = "torch.float32",
    dims: str | None = None,
    with_parallel_info: bool = False,
) -> BundleSideInfo:
    s: list[int] = shape if shape is not None else [2, 4096]
    files: list[BundleFileInfo] = []
    for i in range(num_files):
        par: dict[str, str] | None = (
            {"tp": f"{i}/{num_files}"} if with_parallel_info else None
        )
        files.append(BundleFileInfo(shape=s, dtype=dtype, rank=i, parallel_info=par))
    return BundleSideInfo(num_files=num_files, files=files, dims=dims)


def _make_simple_aligner_plan(
    *,
    with_unsharder: bool = False,
    with_reorderer: bool = False,
    with_token_aligner: bool = False,
    with_axis_aligner: bool = False,
    axis_aligner_noop: bool = False,
) -> AlignerPlan:
    baseline_plans: list[AlignerPerStepPlan] = []
    target_plans: list[AlignerPerStepPlan] = []

    if with_unsharder:
        unsharder: UnsharderPlan = UnsharderPlan(
            axis=ParallelAxis.TP,
            params=ConcatParams(dim_name="h"),
            groups=[[0, 1]],
        )
        target_plans.append(
            AlignerPerStepPlan(
                step=0, input_object_indices=[0, 1], sub_plans=[unsharder]
            )
        )

    if with_reorderer:
        reorderer: ReordererPlan = ReordererPlan(
            params=ZigzagToNaturalParams(dim_name="s", cp_size=2),
        )
        target_plans.append(
            AlignerPerStepPlan(step=0, input_object_indices=[0], sub_plans=[reorderer])
        )

    token_aligner_plan: TokenAlignerPlan | None = None
    if with_token_aligner:
        token_aligner_plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[0, 0, 0], token_index_in_step=[0, 1, 2]),
                y=TokenLocator(steps=[0, 0, 0], token_index_in_step=[0, 1, 2]),
            ),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

    axis_aligner_plan: AxisAlignerPlan | None = None
    if with_axis_aligner:
        if axis_aligner_noop:
            axis_aligner_plan = AxisAlignerPlan(pattern=Pair(x=None, y=None))
        else:
            axis_aligner_plan = AxisAlignerPlan(
                pattern=Pair(x="b s d -> s b d", y=None)
            )

    return AlignerPlan(
        per_step_plans=Pair(x=baseline_plans, y=target_plans),
        token_aligner_plan=token_aligner_plan,
        axis_aligner_plan=axis_aligner_plan,
    )


def _make_traced_plan(
    plan: AlignerPlan,
    *,
    target_input_shapes: list[list[int]] | None = None,
    target_output_shapes: list[list[int]] | None = None,
) -> TracedAlignerPlan:
    """Build a TracedAlignerPlan by attaching snapshots to target sub_plans."""
    baseline_traced_steps: list[TracedStepPlan] = [
        TracedStepPlan(
            step=sp.step,
            input_object_indices=sp.input_object_indices,
            sub_plans=[TracedSubPlan(plan=sub) for sub in sp.sub_plans],
        )
        for sp in plan.per_step_plans.x
    ]

    target_traced_steps: list[TracedStepPlan] = []
    for sp in plan.per_step_plans.y:
        traced_subs: list[TracedSubPlan] = []
        for sub in sp.sub_plans:
            snapshot: ShapeSnapshot | None = None
            if target_input_shapes is not None or target_output_shapes is not None:
                snapshot = ShapeSnapshot(
                    input_shapes=target_input_shapes or [[2, 4096], [2, 4096]],
                    output_shapes=target_output_shapes or [[4, 4096]],
                )
            traced_subs.append(TracedSubPlan(plan=sub, snapshot=snapshot))
        target_traced_steps.append(
            TracedStepPlan(
                step=sp.step,
                input_object_indices=sp.input_object_indices,
                sub_plans=traced_subs,
            )
        )

    return TracedAlignerPlan(
        plan=plan,
        per_side=Pair(
            x=TracedSidePlan(step_plans=baseline_traced_steps),
            y=TracedSidePlan(step_plans=target_traced_steps),
        ),
    )


# ---------------------------------------------------------------------------
# Rich format snapshot tests (normal mode only)
# ---------------------------------------------------------------------------


class TestFormatComparisonRichNormal:
    """format_comparison_rich() snapshot tests."""

    def test_passed(self) -> None:
        record: TensorComparisonRecord = _make_comparison_record(
            diff=_make_diff(rel_diff=1e-4, passed=True),
        )
        result: str = format_comparison_rich(record)

        assert result == (
            "[green]✅[/] [bold green]hidden_states[/] [dim cyan]── float32  [4, 8][/]\n"
            "   [green]rel_diff=1.00e-04[/]  max_abs=5.00e-04  mean_abs=2.00e-04\n"
            "   [dim]Aligned[/]\n"
            "      [4, 8] vs [4, 8]   torch.float32 vs torch.float32\n"
            "   [dim]Stats[/]\n"
            "      [blue]mean      [/]     0.0000 vs     0.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]std       [/]     1.0000 vs     1.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0000, 2.0000]"
        )

    def test_failed(self) -> None:
        record: TensorComparisonRecord = _make_comparison_record(
            diff=_make_diff(
                rel_diff=0.5, max_abs_diff=1.0, mean_abs_diff=0.3, passed=False
            ),
        )
        result: str = format_comparison_rich(record)

        assert result == (
            "[red]❌[/] [bold red]hidden_states[/] [dim cyan]── float32  [4, 8][/]\n"
            "   [bold red]rel_diff=5.00e-01[/]  max_abs=1.00e+00  mean_abs=3.00e-01\n"
            "   max_abs @ [2, 3]: baseline=1.0  target=1.0005\n"
            "   [dim]Aligned[/]\n"
            "      [4, 8] vs [4, 8]   torch.float32 vs torch.float32\n"
            "   [dim]Stats[/]\n"
            "      [blue]mean      [/]     0.0000 vs     0.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]std       [/]     1.0000 vs     1.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0000, 2.0000]\n"
            "   [dim]Abs Diff Percentiles[/]\n"
            "      p1=1.00e-04  p5=1.00e-04  p50=2.00e-04  p95=4.00e-04  p99=5.00e-04"
        )

    def test_shape_mismatch(self) -> None:
        record: TensorComparisonRecord = _make_comparison_record(
            shape_mismatch=True,
        )
        result: str = format_comparison_rich(record)

        assert result == (
            "[red]❌[/] [bold red]hidden_states[/] [dim cyan]── float32  [4, 8][/]\n"
            "   [yellow]⚠ Shape mismatch[/]\n"
            "   [dim]Aligned[/]\n"
            "      [4, 8] vs [4, 8]   torch.float32 vs torch.float32\n"
            "   [dim]Stats[/]\n"
            "      [blue]mean      [/]     0.0000 vs     0.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]std       [/]     1.0000 vs     1.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0000, 2.0000]"
        )

    def test_with_downcast(self) -> None:
        record: TensorComparisonRecord = _make_comparison_record(
            diff=_make_diff(rel_diff=0.01, passed=False),
            diff_downcast=_make_diff(rel_diff=1e-5, passed=True),
            downcast_dtype="torch.bfloat16",
        )
        result: str = format_comparison_rich(record)

        assert result == (
            "[red]❌[/] [bold red]hidden_states[/] [dim cyan]── float32  [4, 8][/]\n"
            "   [bold red]rel_diff=1.00e-02[/]  max_abs=5.00e-04  mean_abs=2.00e-04\n"
            "   max_abs @ [2, 3]: baseline=1.0  target=1.0005\n"
            "   [green]✅[/] downcast to torch.bfloat16: rel_diff=1.00e-05\n"
            "   [dim]Aligned[/]\n"
            "      [4, 8] vs [4, 8]   torch.float32 vs torch.float32\n"
            "   [dim]Stats[/]\n"
            "      [blue]mean      [/]     0.0000 vs     0.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]std       [/]     1.0000 vs     1.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0000, 2.0000]\n"
            "   [dim]Abs Diff Percentiles[/]\n"
            "      p1=1.00e-04  p5=1.00e-04  p50=2.00e-04  p95=4.00e-04  p99=5.00e-04"
        )

    def test_with_bundle_info(self) -> None:
        bundle_info: Pair[BundleSideInfo] = Pair(
            x=_make_bundle_side_info(num_files=2, dims="b s h(tp) d"),
            y=_make_bundle_side_info(num_files=2, dims="b s h(tp) d"),
        )
        record: TensorComparisonRecord = _make_comparison_record(
            diff=_make_diff(passed=True),
            raw_bundle_info=bundle_info,
        )
        result: str = format_comparison_rich(record)

        assert result == (
            "[green]✅[/] [bold green]hidden_states[/] [dim cyan]── float32  [4, 8][/]\n"
            "   [green]rel_diff=1.00e-04[/]  max_abs=5.00e-04  mean_abs=2.00e-04\n"
            "   [dim]Bundle[/]\n"
            "      baseline  [cyan]2 files[/] × [2, 4096] float32  [dim]dims: b s h(tp) d[/]\n"
            "      target  [cyan]2 files[/] × [2, 4096] float32  [dim]dims: b s h(tp) d[/]\n"
            "   [dim]Aligned[/]\n"
            "      [4, 8] vs [4, 8]   torch.float32 vs torch.float32\n"
            "   [dim]Stats[/]\n"
            "      [blue]mean      [/]     0.0000 vs     0.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]std       [/]     1.0000 vs     1.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0000, 2.0000]"
        )

    def test_with_plan(self) -> None:
        plan: AlignerPlan = _make_simple_aligner_plan(with_unsharder=True)
        record: TensorComparisonRecord = _make_comparison_record(
            diff=_make_diff(passed=True),
            traced_plan=_make_traced_plan(plan),
        )
        result: str = format_comparison_rich(record)

        assert result == (
            "[green]✅[/] [bold green]hidden_states[/] [dim cyan]── float32  [4, 8][/]\n"
            "   [green]rel_diff=1.00e-04[/]  max_abs=5.00e-04  mean_abs=2.00e-04\n"
            "   [dim]Plan[/]\n"
            "      baseline  [dim](passthrough)[/]\n"
            "      target  [magenta]unsharder(ParallelAxis.TP)[/]\n"
            "   [dim]Aligned[/]\n"
            "      [4, 8] vs [4, 8]   torch.float32 vs torch.float32\n"
            "   [dim]Stats[/]\n"
            "      [blue]mean      [/]     0.0000 vs     0.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]std       [/]     1.0000 vs     1.0000  Δ [dim]+0.00e+00[/]\n"
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0000, 2.0000]"
        )


class TestFormatBundleSection:
    """_format_bundle_section() snapshot tests."""

    def test_single_shape(self) -> None:
        bundle: Pair[BundleSideInfo] = Pair(
            x=_make_bundle_side_info(num_files=2, shape=[2, 4096]),
            y=_make_bundle_side_info(num_files=2, shape=[2, 4096]),
        )
        lines: list[str] = _format_bundle_section(bundle)

        assert lines == [
            "      baseline  [cyan]2 files[/] × [2, 4096] float32",
            "      target  [cyan]2 files[/] × [2, 4096] float32",
        ]

    def test_mixed_shapes(self) -> None:
        side: BundleSideInfo = BundleSideInfo(
            num_files=2,
            files=[
                BundleFileInfo(shape=[2, 4096], dtype="torch.float32", rank=0),
                BundleFileInfo(shape=[3, 4096], dtype="torch.float32", rank=1),
            ],
        )
        bundle: Pair[BundleSideInfo] = Pair(x=side, y=side)
        lines: list[str] = _format_bundle_section(bundle)

        assert lines == [
            "      baseline  [cyan]2 files[/] × mixed shapes float32",
            "      target  [cyan]2 files[/] × mixed shapes float32",
        ]

    def test_no_files(self) -> None:
        empty: BundleSideInfo = BundleSideInfo(num_files=0, files=[])
        bundle: Pair[BundleSideInfo] = Pair(x=empty, y=empty)
        lines: list[str] = _format_bundle_section(bundle)

        assert lines == [
            "      baseline  [dim](no files)[/]",
            "      target  [dim](no files)[/]",
        ]

    def test_with_dims(self) -> None:
        bundle: Pair[BundleSideInfo] = Pair(
            x=_make_bundle_side_info(num_files=1, dims="b s h(tp) d"),
            y=_make_bundle_side_info(num_files=1, dims="b s h(tp) d"),
        )
        lines: list[str] = _format_bundle_section(bundle)

        assert lines == [
            "      baseline  [cyan]1 files[/] × [2, 4096] float32  [dim]dims: b s h(tp) d[/]",
            "      target  [cyan]1 files[/] × [2, 4096] float32  [dim]dims: b s h(tp) d[/]",
        ]


class TestFormatPlanSectionRich:
    """_format_plan_section_rich() snapshot tests."""

    def test_passthrough(self) -> None:
        plan: AlignerPlan = _make_simple_aligner_plan()
        traced: TracedAlignerPlan = _make_traced_plan(plan)
        lines: list[str] = _format_plan_section_rich(traced_plan=traced)

        assert lines == [
            "      baseline  [dim](passthrough)[/]",
            "      target  [dim](passthrough)[/]",
        ]

    def test_unsharder_op(self) -> None:
        plan: AlignerPlan = _make_simple_aligner_plan(with_unsharder=True)
        traced: TracedAlignerPlan = _make_traced_plan(plan)
        lines: list[str] = _format_plan_section_rich(traced_plan=traced)

        assert lines == [
            "      baseline  [dim](passthrough)[/]",
            "      target  [magenta]unsharder(ParallelAxis.TP)[/]",
        ]

    def test_reorderer_op(self) -> None:
        plan: AlignerPlan = _make_simple_aligner_plan(with_reorderer=True)
        traced: TracedAlignerPlan = _make_traced_plan(plan)
        lines: list[str] = _format_plan_section_rich(traced_plan=traced)

        assert lines == [
            "      baseline  [dim](passthrough)[/]",
            "      target  [magenta]reorderer[/]",
        ]

    def test_with_shape_traces(self) -> None:
        plan: AlignerPlan = _make_simple_aligner_plan(with_unsharder=True)
        traced: TracedAlignerPlan = _make_traced_plan(
            plan,
            target_input_shapes=[[2, 4096], [2, 4096]],
            target_output_shapes=[[4, 4096]],
        )
        lines: list[str] = _format_plan_section_rich(traced_plan=traced)

        assert lines == [
            "      baseline  [dim](passthrough)[/]",
            "      target  [magenta]unsharder(ParallelAxis.TP)[/] 2×[2, 4096] → 1×[4, 4096]",
        ]

    def test_with_token_aligner(self) -> None:
        plan: AlignerPlan = _make_simple_aligner_plan(with_token_aligner=True)
        traced: TracedAlignerPlan = _make_traced_plan(plan)
        lines: list[str] = _format_plan_section_rich(traced_plan=traced)

        assert lines == [
            "      baseline  [dim](passthrough)[/]",
            "      target  [dim](passthrough)[/]",
            "      token_aligner  [dim]3 tokens[/]",
        ]

    def test_with_axis_aligner(self) -> None:
        plan: AlignerPlan = _make_simple_aligner_plan(with_axis_aligner=True)
        traced: TracedAlignerPlan = _make_traced_plan(plan)
        lines: list[str] = _format_plan_section_rich(traced_plan=traced)

        assert lines == [
            "      baseline  [dim](passthrough)[/]",
            "      target  [dim](passthrough)[/]",
            "      axis_aligner  [dim]x=b s d -> s b d[/]",
        ]

    def test_axis_aligner_noop(self) -> None:
        plan: AlignerPlan = _make_simple_aligner_plan(
            with_axis_aligner=True, axis_aligner_noop=True
        )
        traced: TracedAlignerPlan = _make_traced_plan(plan)
        lines: list[str] = _format_plan_section_rich(traced_plan=traced)

        assert lines == [
            "      baseline  [dim](passthrough)[/]",
            "      target  [dim](passthrough)[/]",
            "      axis_aligner  [dim](no-op)[/]",
        ]


class TestFormatStatsRich:
    """_format_stats_rich() snapshot tests."""

    def test_basic(self) -> None:
        baseline: TensorStats = _make_stats(mean=0.0, std=1.0, min=-2.0, max=2.0)
        target: TensorStats = _make_stats(
            mean=0.0001, std=1.0001, min=-2.0001, max=2.0001
        )
        lines: list[str] = _format_stats_rich(baseline=baseline, target=target)

        assert lines == [
            "      [blue]mean      [/]     0.0000 vs     0.0001  Δ [dim]+1.00e-04[/]",
            "      [blue]std       [/]     1.0000 vs     1.0001  Δ [dim]+1.00e-04[/]",
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0001, 2.0001]",
        ]

    def test_large_delta(self) -> None:
        baseline: TensorStats = _make_stats(mean=0.0)
        target: TensorStats = _make_stats(mean=1.0)
        lines: list[str] = _format_stats_rich(baseline=baseline, target=target)

        assert lines == [
            "      [blue]mean      [/]     0.0000 vs     1.0000  Δ [yellow]+1.00e+00[/]",
            "      [blue]std       [/]     1.0000 vs     1.0000  Δ [dim]+0.00e+00[/]",
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0000, 2.0000]",
        ]

    def test_small_delta(self) -> None:
        baseline: TensorStats = _make_stats(mean=0.0)
        target: TensorStats = _make_stats(mean=0.001)
        lines: list[str] = _format_stats_rich(baseline=baseline, target=target)

        assert lines == [
            "      [blue]mean      [/]     0.0000 vs     0.0010  Δ [dim]+1.00e-03[/]",
            "      [blue]std       [/]     1.0000 vs     1.0000  Δ [dim]+0.00e+00[/]",
            "      [blue]range     [/] [-2.0000, 2.0000] vs [-2.0000, 2.0000]",
        ]


class TestFormatAbsDiffPercentilesRich:
    """_format_abs_diff_percentiles_rich() snapshot tests."""

    def test_normal_values(self) -> None:
        diff: DiffInfo = _make_diff()
        result: str = _format_abs_diff_percentiles_rich(diff)

        assert result == (
            "p1=1.00e-04  p5=1.00e-04  p50=2.00e-04  " "p95=4.00e-04  p99=5.00e-04"
        )

    def test_high_p99_coloring(self) -> None:
        diff: DiffInfo = _make_diff(
            abs_diff_percentiles={99: 0.5},
        )
        result: str = _format_abs_diff_percentiles_rich(diff)

        assert result == "[yellow]p99=5.00e-01[/]"

    def test_low_p99_no_coloring(self) -> None:
        diff: DiffInfo = _make_diff(
            abs_diff_percentiles={99: 0.01},
        )
        result: str = _format_abs_diff_percentiles_rich(diff)

        assert result == "p99=1.00e-02"


class TestFormatReplicatedChecks:
    """format_replicated_checks() snapshot tests."""

    def test_all_passed(self) -> None:
        checks: list[ReplicatedCheckResult] = [
            ReplicatedCheckResult(
                axis="tp",
                group_index=0,
                compared_index=1,
                baseline_index=0,
                passed=True,
                atol=1e-3,
                diff=_make_diff(rel_diff=1e-6, max_abs_diff=1e-5, mean_abs_diff=1e-6),
            ),
        ]
        result: str = format_replicated_checks(checks)

        assert result == (
            "Replicated checks:\n"
            "  ✅ axis=tp group=0 idx=1 vs 0: "
            "rel_diff=1.000000e-06 max_abs_diff=1.000000e-05 mean_abs_diff=1.000000e-06"
        )

    def test_one_failed(self) -> None:
        checks: list[ReplicatedCheckResult] = [
            ReplicatedCheckResult(
                axis="tp",
                group_index=0,
                compared_index=1,
                baseline_index=0,
                passed=False,
                atol=1e-3,
                diff=_make_diff(rel_diff=0.5, max_abs_diff=1.0, mean_abs_diff=0.3),
            ),
        ]
        result: str = format_replicated_checks(checks)

        assert result == (
            "Replicated checks:\n"
            "  ❌ axis=tp group=0 idx=1 vs 0: "
            "rel_diff=5.000000e-01 max_abs_diff=1.000000e+00 mean_abs_diff=3.000000e-01"
        )

    def test_no_diff(self) -> None:
        checks: list[ReplicatedCheckResult] = [
            ReplicatedCheckResult(
                axis="tp",
                group_index=0,
                compared_index=1,
                baseline_index=0,
                passed=True,
                atol=1e-3,
            ),
        ]
        result: str = format_replicated_checks(checks)

        assert result == (
            "Replicated checks:\n" "  ✅ axis=tp group=0 idx=1 vs 0: n/a diff"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
