"""Formatting functions for comparator output records.

Extracted from output_types.py to separate data-structure definitions
from rendering / formatting logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from rich.console import Group
from rich.markup import escape
from rich.panel import Panel

from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
    format_comparison,
    format_replicated_checks,
)

if TYPE_CHECKING:
    from rich.console import RenderableType

    from sglang.srt.debug_utils.comparator.aligner.entrypoint.traced_types import (
        TracedAlignerPlan,
        TracedSubPlan,
    )
    from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import AlignerPlan
    from sglang.srt.debug_utils.comparator.output_types import (
        ComparisonErrorRecord,
        ComparisonNonTensorRecord,
        ComparisonSkipRecord,
        ComparisonTensorRecord,
        ConfigRecord,
        ErrorLog,
        InfoLog,
        LogRecord,
        SummaryRecord,
        _OutputRecord,
        _TableRecord,
    )

Verbosity = Literal["minimal", "normal", "verbose"]


# ── Record-level rendering (body + logs) ─────────────────────────────


def _render_record_rich(
    record: _OutputRecord, *, verbosity: Verbosity = "normal"
) -> RenderableType:
    body: RenderableType = record._format_rich_body(verbosity=verbosity)

    log_lines: list[str] = _format_log_lines_rich(
        errors=record.errors, infos=record.infos
    )

    if not log_lines:
        return body

    log_block: str = "\n".join(log_lines)
    if isinstance(body, str):
        return body + "\n" + log_block
    return Group(body, log_block)


def _render_record_text(record: _OutputRecord) -> str:
    body: str = record._format_body()

    log_suffix: str = _format_log_lines_text(errors=record.errors, infos=record.infos)

    if log_suffix:
        body += "\n" + log_suffix

    return body


def _format_log_lines_rich(
    *, errors: list[ErrorLog], infos: list[InfoLog]
) -> list[str]:
    lines: list[str] = []

    if errors:
        lines.extend(f"  [red]✗ {e.to_text()}[/]" for e in errors)
    if infos:
        lines.extend(f"  [dim]ℹ {i.to_text()}[/]" for i in infos)

    return lines


def _format_log_lines_text(*, errors: list[ErrorLog], infos: list[InfoLog]) -> str:
    lines: list[str] = []

    if errors:
        lines.extend(f"  ✗ {e.to_text()}" for e in errors)
    if infos:
        lines.extend(f"  ℹ {i.to_text()}" for i in infos)

    return "\n".join(lines)


# ── ConfigRecord ──────────────────────────────────────────────────────


def _format_config_body(record: ConfigRecord) -> str:
    return f"Config: {record.config}"


def _format_config_rich_body(
    record: ConfigRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    lines: list[str] = [f"  [bold]{k}[/] : {v}" for k, v in record.config.items()]
    return Panel("\n".join(lines), title="Comparator Config", border_style="cyan")


# ── ComparisonSkipRecord ─────────────────────────────────────────────


def _format_skip_body(record: ComparisonSkipRecord) -> str:
    return f"Skip: {record.name}{record._format_location_suffix()} ({record.reason})"


def _format_skip_rich_body(
    record: ComparisonSkipRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    suffix: str = record._format_location_suffix()
    return (
        f"[dim]⊘ {escape(record.name)}{suffix} ── skipped ({escape(record.reason)})[/]"
    )


# ── ComparisonErrorRecord ────────────────────────────────────────────


def _format_error_body(record: ComparisonErrorRecord) -> str:
    prefix: str = record._format_location_prefix()
    return (
        f"{prefix}Error: {record.name} ({record.exception_type})\n"
        f"{record.traceback_str}"
    )


def _format_error_rich_body(
    record: ComparisonErrorRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    prefix: str = record._format_location_prefix_rich()
    name: str = escape(record.name)
    header: str = (
        f"{prefix}[bold red]{name} ── errored ({escape(record.exception_type)})[/]"
    )
    if verbosity == "minimal":
        return header
    return header + f"\n[dim]{escape(record.traceback_str)}[/]"


# ── _TableRecord ─────────────────────────────────────────────────────


def _format_table_body(record: _TableRecord) -> str:
    import polars as pl

    from sglang.srt.debug_utils.comparator.display import _render_polars_as_text

    return _render_polars_as_text(
        pl.DataFrame(record.rows), title=record._table_title()
    )


def _format_table_rich_body(
    record: _TableRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    import polars as pl

    from sglang.srt.debug_utils.comparator.display import (
        _render_polars_as_rich_table,
    )

    return _render_polars_as_rich_table(
        pl.DataFrame(record.rows), title=record._table_title()
    )


# ── ComparisonTensorRecord ───────────────────────────────────────────


def _format_tensor_comparison_body(record: ComparisonTensorRecord) -> str:
    body: str = record._format_location_prefix() + format_comparison(record)
    if record.replicated_checks:
        body += "\n" + format_replicated_checks(record.replicated_checks)
    if record.traced_plan is not None:
        body += "\n" + _format_aligner_plan(record.traced_plan)
    return body


def _format_tensor_comparison_rich_body(
    record: ComparisonTensorRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
        format_comparison_rich,
    )

    return record._format_location_prefix_rich() + format_comparison_rich(
        record=record, verbosity=verbosity
    )


# ── ComparisonNonTensorRecord ────────────────────────────────────────


def _format_non_tensor_body(record: ComparisonNonTensorRecord) -> str:
    suffix: str = record._format_location_suffix()
    if record.values_equal:
        return f"NonTensor: {record.name}{suffix} = {record.baseline_value} ({record.baseline_type}) [equal]"
    return (
        f"NonTensor: {record.name}{suffix}\n"
        f"  baseline = {record.baseline_value} ({record.baseline_type})\n"
        f"  target   = {record.target_value} ({record.target_type})"
    )


def _format_non_tensor_rich_body(
    record: ComparisonNonTensorRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    suffix: str = record._format_location_suffix()
    name: str = escape(record.name)
    baseline_val: str = escape(record.baseline_value)
    target_val: str = escape(record.target_value)

    if record.values_equal:
        return (
            f"═ {name}{suffix} = {baseline_val} "
            f"({record.baseline_type}) [green]✓[/]"
        )
    return (
        f"═ [bold red]{name}{suffix}[/]\n"
        f"  baseline = {baseline_val} ({record.baseline_type})\n"
        f"  target   = {target_val} ({record.target_type})"
    )


# ── SummaryRecord ────────────────────────────────────────────────────


def _format_summary_body(record: SummaryRecord) -> str:
    text: str = (
        f"Summary: {record.passed} passed, {record.failed} failed, "
        f"{record.skipped} skipped (total {record.total})"
    )
    if record.errored > 0:
        text += f", {record.errored} errored"
    return text


def _format_summary_rich_body(
    record: SummaryRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    text: str = (
        f"[bold green]{record.passed} passed[/] │ "
        f"[bold red]{record.failed} failed[/] │ "
        f"[yellow]{record.skipped} skipped[/] │ "
        f"{record.total} total"
    )
    if record.errored > 0:
        text += f" │ [bold red]{record.errored} errored[/]"
    return Panel(text, title="SUMMARY", border_style="bold")


# ── LogRecord ────────────────────────────────────────────────────────


def _format_log_body(record: LogRecord) -> str:
    return ""


# ── Standalone helpers ───────────────────────────────────────────────


def _format_aligner_plan(traced_plan: TracedAlignerPlan) -> str:
    lines: list[str] = ["Aligner Plan:"]

    for side_label, traced_side in [
        ("baseline", traced_plan.per_side.x),
        ("target", traced_plan.per_side.y),
    ]:
        if not traced_side.step_plans:
            lines.append(f"  {side_label}: (no steps)")
            continue

        step_summaries: list[str] = []
        for traced_step in traced_side.step_plans:
            sub_strs: list[str] = [
                _format_sub_plan_text(traced_sub)
                for traced_sub in traced_step.sub_plans
            ]
            summary: str = ", ".join(sub_strs) if sub_strs else "passthrough"
            step_summaries.append(f"step={traced_step.step}: {summary}")
        lines.append(f"  {side_label}: [{'; '.join(step_summaries)}]")

    lines.extend(_format_cross_side_plan_text(traced_plan.plan))
    return "\n".join(lines)


def _format_sub_plan_text(traced_sub: TracedSubPlan) -> str:
    sub_desc: str = f"{traced_sub.plan.type}"

    if traced_sub.snapshot is not None:
        snap = traced_sub.snapshot
        in_count: int = len(snap.input_shapes)
        out_count: int = len(snap.output_shapes)
        in_shape: str = str(snap.input_shapes[0]) if snap.input_shapes else "?"
        out_shape: str = str(snap.output_shapes[0]) if snap.output_shapes else "?"
        sub_desc += f" {in_count}x{in_shape} -> {out_count}x{out_shape}"

    return sub_desc


def _format_cross_side_plan_text(plan: AlignerPlan) -> list[str]:
    lines: list[str] = []

    if plan.token_aligner_plan is not None:
        num_tokens: int = len(plan.token_aligner_plan.locators.x.steps)
        lines.append(f"  token_aligner: {num_tokens} tokens aligned")

    if plan.axis_aligner_plan is not None:
        parts: list[str] = []
        if plan.axis_aligner_plan.pattern.x:
            parts.append(f"x: {plan.axis_aligner_plan.pattern.x}")
        if plan.axis_aligner_plan.pattern.y:
            parts.append(f"y: {plan.axis_aligner_plan.pattern.y}")
        lines.append(f"  axis_aligner: {', '.join(parts)}")

    return lines
