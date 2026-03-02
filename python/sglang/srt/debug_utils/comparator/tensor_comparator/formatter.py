from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from rich.markup import escape

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
    TensorInfo,
    TensorStats,
)

if TYPE_CHECKING:
    from sglang.srt.debug_utils.comparator.aligner.entrypoint.traced_types import (
        TracedAlignerPlan,
        TracedSubPlan,
    )
    from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import AlignerPlan
    from sglang.srt.debug_utils.comparator.output_types import (
        BundleSideInfo,
        ReplicatedCheckResult,
        ShapeSnapshot,
        TensorComparisonRecord,
    )
    from sglang.srt.debug_utils.comparator.utils import Pair

Verbosity = Literal["minimal", "normal", "verbose"]


def _esc_shape(shape: Optional[list[int]]) -> str:
    return escape(str(shape))


def _strip_torch_prefix(dtype: str) -> str:
    return dtype.replace("torch.", "")


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def _fmt_val(value: float) -> str:
    return f"{value:.2e}"


def _fmt_diff_colored(diff: float, *, threshold: float = 1e-2) -> str:
    formatted: str = f"{diff:+.2e}"
    if abs(diff) >= threshold:
        return f"[yellow]{formatted}[/]"
    return f"[dim]{formatted}[/]"


# ---------------------------------------------------------------------------
# Passed / color / marker helper
# ---------------------------------------------------------------------------


def _category_marker(category: str) -> tuple[bool, str, str]:
    passed: bool = category == "passed"
    color: str = "green" if passed else "red"
    marker: str = f"[{color}]✅[/]" if passed else f"[{color}]❌[/]"
    return passed, color, marker


# ---------------------------------------------------------------------------
# Stats formatting helpers (shared between compact / verbose)
# ---------------------------------------------------------------------------


def _format_stat_line(stat_name: str, val_b: float, val_t: float, diff: float) -> str:
    return (
        f"      [blue]{stat_name:10s}[/] {val_b:>10.4f} vs {val_t:>10.4f}"
        f"  Δ {_fmt_diff_colored(diff)}"
    )


# ---------------------------------------------------------------------------
# Old text-only formatters (kept for to_text() backward compatibility)
# ---------------------------------------------------------------------------


def format_comparison(info: TensorComparisonInfo) -> str:
    lines: list[str] = []
    baseline = info.baseline
    target = info.target

    dtype_marker = "" if baseline.dtype == target.dtype else "🟠"
    lines.append(
        f"Raw "
        f"[shape] {baseline.shape} vs {target.shape}\t"
        f"[{dtype_marker}dtype] {baseline.dtype} vs {target.dtype}"
    )

    if info.unified_shape != baseline.shape:
        lines.append(
            f"Unify shape: {baseline.shape} -> {info.unified_shape} "
            f"(to match {target.shape})"
        )

    lines.append(
        f"After unify "
        f"[shape] {info.unified_shape} vs {target.shape}\t"
        f"[dtype] {baseline.dtype} vs {target.dtype}"
    )

    lines.extend(_format_stats_comparison(baseline=baseline.stats, target=target.stats))

    if info.shape_mismatch:
        lines.append("⚠️ Shape mismatch")
        return "\n".join(lines)

    if info.diff is not None:
        lines.extend(_format_diff(diff=info.diff))

    if info.diff_downcast is not None and info.downcast_dtype is not None:
        lines.extend(
            _format_diff(
                diff=info.diff_downcast,
                prefix_text=f"When downcast to {info.downcast_dtype}: ",
            )
        )

    if baseline.sample is not None:
        lines.append(f"x_baseline(sample)={baseline.sample}")
    if target.sample is not None:
        lines.append(f"x_target(sample)={target.sample}")

    return "\n".join(lines)


def format_replicated_checks(checks: list[ReplicatedCheckResult]) -> str:
    lines: list[str] = ["Replicated checks:"]

    for check in checks:
        marker: str = "✅" if check.passed else "❌"

        if check.diff is not None:
            detail: str = (
                f"rel_diff={check.diff.rel_diff:.6e} "
                f"max_abs_diff={check.diff.max_abs_diff:.6e} "
                f"mean_abs_diff={check.diff.mean_abs_diff:.6e}"
            )
        else:
            detail = "n/a diff"

        lines.append(
            f"  {marker} axis={check.axis} group={check.group_index} "
            f"idx={check.compared_index} vs {check.baseline_index}: "
            f"{detail}"
        )

    return "\n".join(lines)


def _format_stats_comparison(baseline: TensorStats, target: TensorStats) -> list[str]:
    lines: list[str] = []

    for stat_name in TensorStats.model_fields:
        if stat_name == "percentiles":
            continue
        value_baseline: float = getattr(baseline, stat_name)
        value_target: float = getattr(target, stat_name)
        lines.append(
            f"[{stat_name}] {value_baseline:.4f} vs {value_target:.4f} "
            f"(diff: {value_target - value_baseline:.4f})"
        )

    for p in sorted(set(baseline.percentiles) & set(target.percentiles)):
        value_baseline = baseline.percentiles[p]
        value_target = target.percentiles[p]
        lines.append(
            f"[p{p}] {value_baseline:.4f} vs {value_target:.4f} "
            f"(diff: {value_target - value_baseline:.4f})"
        )

    return lines


def _format_diff(diff: DiffInfo, prefix_text: str = "") -> list[str]:
    rel_diff_marker: str = "❌" if diff.rel_diff > diff.diff_threshold else "✅"
    lines: list[str] = [
        prefix_text
        + f"{rel_diff_marker} rel_diff={diff.rel_diff}\t"
        + f"max_abs_diff={diff.max_abs_diff}\t"
        + f"mean_abs_diff={diff.mean_abs_diff}",
        f"max_abs_diff happens at coord={diff.max_diff_coord} with "
        f"baseline={diff.baseline_at_max} "
        f"target={diff.target_at_max}",
    ]

    if diff.abs_diff_percentiles:
        quantile_parts: list[str] = [
            f"p{p}={value:.4f}"
            for p, value in sorted(diff.abs_diff_percentiles.items())
        ]
        lines.append("[abs_diff] " + " ".join(quantile_parts))

    return lines


# ---------------------------------------------------------------------------
# New Rich markup formatters
# ---------------------------------------------------------------------------


def format_comparison_rich(
    record: TensorComparisonRecord,
    verbosity: Verbosity = "normal",
) -> str:
    if verbosity == "minimal":
        return _format_comparison_minimal(record)

    return _format_comparison_normal_or_verbose(
        record=record,
        verbose=(verbosity == "verbose"),
    )


def _format_comparison_minimal(record: TensorComparisonRecord) -> str:
    passed, color, marker = _category_marker(record.category)

    name_part: str = f"[bold {color}]{escape(record.name):30s}[/]"
    if record.diff is not None:
        return f"{marker} {name_part} rel_diff={_fmt_val(record.diff.rel_diff)}"
    elif record.shape_mismatch:
        return f"{marker} {name_part} [yellow]shape mismatch[/]"
    else:
        return f"{marker} {name_part}"


def _format_comparison_normal_or_verbose(
    *,
    record: TensorComparisonRecord,
    verbose: bool,
) -> str:
    passed, color, marker = _category_marker(record.category)

    baseline: TensorInfo = record.baseline
    target: TensorInfo = record.target
    aligned_shape: str = _esc_shape(record.unified_shape)
    dtype_str: str = _strip_torch_prefix(baseline.dtype)

    lines: list[str] = []

    # L0: Header
    lines.append(
        f"{marker} [bold {color}]{escape(record.name)}[/] "
        f"[dim cyan]── {dtype_str}  {aligned_shape}[/]"
    )

    # L1: Key metrics
    if record.diff is not None:
        diff: DiffInfo = record.diff
        rel_style: str = f"bold {color}" if not passed else color
        lines.append(
            f"   [{rel_style}]rel_diff={_fmt_val(diff.rel_diff)}[/]"
            f"  max_abs={_fmt_val(diff.max_abs_diff)}"
            f"  mean_abs={_fmt_val(diff.mean_abs_diff)}"
        )

        if not passed:
            lines.append(
                f"   max_abs @ {_esc_shape(diff.max_diff_coord)}: "
                f"baseline={diff.baseline_at_max}  target={diff.target_at_max}"
            )
    elif record.shape_mismatch:
        lines.append("   [yellow]⚠ Shape mismatch[/]")

    # Downcast info
    if record.diff_downcast is not None and record.downcast_dtype is not None:
        dc: DiffInfo = record.diff_downcast
        dc_marker: str = "[green]✅[/]" if dc.passed else "[red]❌[/]"
        lines.append(
            f"   {dc_marker} downcast to {record.downcast_dtype}: "
            f"rel_diff={_fmt_val(dc.rel_diff)}"
        )

    # Bundle section
    if record.raw_bundle_info is not None:
        lines.append("   [dim]Bundle[/]")
        lines.extend(
            _format_bundle_section(bundle_info=record.raw_bundle_info, verbose=verbose)
        )

    # Plan section
    if record.traced_plan is not None:
        lines.append("   [dim]Plan[/]")
        lines.extend(
            _format_plan_section_rich(
                traced_plan=record.traced_plan,
                verbose=verbose,
            )
        )

    # Aligned section
    lines.append("   [dim]Aligned[/]")
    lines.append(
        f"      {_esc_shape(record.unified_shape)} vs {_esc_shape(target.shape)}"
        f"   {baseline.dtype} vs {target.dtype}"
    )

    # Stats section
    lines.append("   [dim]Stats[/]")
    lines.extend(
        _format_stats_rich(
            baseline=baseline.stats, target=target.stats, verbose=verbose
        )
    )

    show_detail: bool = verbose or not passed

    # Abs diff percentiles
    if show_detail and record.diff is not None and record.diff.abs_diff_percentiles:
        lines.append("   [dim]Abs Diff Percentiles[/]")
        lines.append("      " + _format_abs_diff_percentiles_rich(record.diff))

    # Samples
    if show_detail and baseline.sample is not None:
        lines.append("   [dim]Samples[/]")
        lines.append(f"      baseline  {escape(baseline.sample)}")
        if target.sample is not None:
            lines.append(f"      target    {escape(target.sample)}")

    # Replicated checks
    if show_detail and record.replicated_checks:
        lines.append("   [dim]Replicated Checks[/]")
        for check in record.replicated_checks:
            chk_marker: str = "[green]✅[/]" if check.passed else "[red]❌[/]"
            if check.diff is not None:
                lines.append(
                    f"      {chk_marker} axis={check.axis}  group={check.group_index}"
                    f"  idx={check.compared_index} vs {check.baseline_index}"
                    f"  rel_diff={_fmt_val(check.diff.rel_diff)}"
                    f"  max_abs={_fmt_val(check.diff.max_abs_diff)}"
                )
            else:
                lines.append(
                    f"      {chk_marker} axis={check.axis}  group={check.group_index}"
                    f"  idx={check.compared_index} vs {check.baseline_index}: n/a"
                )

    return "\n".join(lines)


def _format_bundle_section(
    bundle_info: Pair[BundleSideInfo], *, verbose: bool = False
) -> list[str]:
    lines: list[str] = []

    for label, side in [("baseline", bundle_info.x), ("target", bundle_info.y)]:
        if not side.files:
            lines.append(f"      {label}  [dim](no files)[/]")
            continue

        dtype_desc: str = _strip_torch_prefix(side.files[0].dtype)

        if verbose:
            dims_part: str = f"  dims: {side.dims}" if side.dims else ""
            lines.append(
                f"      {label}  [cyan]{side.num_files} files[/]"
                f" {dtype_desc}{dims_part}"
            )

            for idx, f in enumerate(side.files):
                rank_part: str = f"rank={f.rank}" if f.rank is not None else ""
                par_part: str = ""
                if f.parallel_info:
                    par_part = " " + " ".join(
                        f"{k}={v}" for k, v in f.parallel_info.items()
                    )
                lines.append(
                    f"         [{idx}] {_esc_shape(f.shape)}  {rank_part}{par_part}"
                )
        else:
            shapes: list[list[int]] = [f.shape for f in side.files]
            unique_shapes: set[str] = {str(s) for s in shapes}
            shape_desc: str
            if len(unique_shapes) == 1:
                shape_desc = _esc_shape(shapes[0])
            else:
                shape_desc = "mixed shapes"

            dims_part = f"  [dim]dims: {side.dims}[/]" if side.dims else ""
            lines.append(
                f"      {label}  [cyan]{side.num_files} files[/]"
                f" × {shape_desc} {dtype_desc}{dims_part}"
            )

    return lines


def _format_plan_section_rich(
    *,
    traced_plan: TracedAlignerPlan,
    verbose: bool = False,
) -> list[str]:
    lines: list[str] = []

    for side_label, traced_side in [
        ("baseline", traced_plan.per_side.x),
        ("target", traced_plan.per_side.y),
    ]:
        if not traced_side.step_plans:
            lines.append(f"      {side_label}  [dim](passthrough)[/]")
            continue

        parts: list[str] = [
            _format_sub_plan_rich(traced_sub)
            for traced_step in traced_side.step_plans
            for traced_sub in traced_step.sub_plans
        ]
        lines.append(f"      {side_label}  " + " → ".join(parts))

    lines.extend(_format_cross_side_plan_rich(traced_plan.plan))
    return lines


def _format_sub_plan_rich(traced_sub: TracedSubPlan) -> str:
    sub = traced_sub.plan
    snapshot: Optional[ShapeSnapshot] = traced_sub.snapshot

    op_name: str = sub.type
    axis_str: str = ""
    if isinstance(sub, UnsharderPlan):
        axis_str = f"({sub.axis})"

    shape_change: str = ""
    if snapshot:
        in_count: int = len(snapshot.input_shapes)
        out_count: int = len(snapshot.output_shapes)
        in_shape: str = (
            _esc_shape(snapshot.input_shapes[0]) if snapshot.input_shapes else "?"
        )
        out_shape: str = (
            _esc_shape(snapshot.output_shapes[0]) if snapshot.output_shapes else "?"
        )
        shape_change = f" {in_count}×{in_shape} → {out_count}×{out_shape}"

    return f"[magenta]{op_name}{axis_str}[/]{shape_change}"


def _format_cross_side_plan_rich(plan: AlignerPlan) -> list[str]:
    lines: list[str] = []

    if plan.token_aligner_plan is not None:
        num_tokens: int = len(plan.token_aligner_plan.locators.x.steps)
        lines.append(f"      token_aligner  [dim]{num_tokens} tokens[/]")

    if plan.axis_aligner_plan is not None:
        parts: list[str] = []
        if plan.axis_aligner_plan.pattern.x:
            parts.append(f"x={plan.axis_aligner_plan.pattern.x}")
        if plan.axis_aligner_plan.pattern.y:
            parts.append(f"y={plan.axis_aligner_plan.pattern.y}")
        if parts:
            lines.append(f"      axis_aligner  [dim]{', '.join(parts)}[/]")
        else:
            lines.append("      axis_aligner  [dim](no-op)[/]")

    return lines


def _format_stats_rich(
    *,
    baseline: TensorStats,
    target: TensorStats,
    verbose: bool = False,
) -> list[str]:
    lines: list[str] = []

    if verbose:
        # All stat fields
        for stat_name in TensorStats.model_fields:
            if stat_name == "percentiles":
                continue
            val_b: float = getattr(baseline, stat_name)
            val_t: float = getattr(target, stat_name)
            lines.append(_format_stat_line(stat_name, val_b, val_t, val_t - val_b))

        # Percentiles
        for p in sorted(set(baseline.percentiles) & set(target.percentiles)):
            val_b = baseline.percentiles[p]
            val_t = target.percentiles[p]
            lines.append(_format_stat_line(f"p{p}", val_b, val_t, val_t - val_b))
    else:
        # Compact: mean, std, range (min/max combined)
        for stat_name in ("mean", "std"):
            val_b = getattr(baseline, stat_name)
            val_t = getattr(target, stat_name)
            lines.append(_format_stat_line(stat_name, val_b, val_t, val_t - val_b))

        # Range line: combine min/max (escape brackets to avoid Rich markup)
        range_baseline: str = escape(f"[{baseline.min:.4f}, {baseline.max:.4f}]")
        range_target: str = escape(f"[{target.min:.4f}, {target.max:.4f}]")
        lines.append(f"      [blue]{'range':10s}[/] {range_baseline} vs {range_target}")

    return lines


def _format_abs_diff_percentiles_rich(diff: DiffInfo) -> str:
    parts: list[str] = []
    for p, value in sorted(diff.abs_diff_percentiles.items()):
        formatted: str = f"p{p}={_fmt_val(value)}"
        if p >= 99 and value > 0.1:
            formatted = f"[yellow]{formatted}[/]"
        parts.append(formatted)
    return "  ".join(parts)
