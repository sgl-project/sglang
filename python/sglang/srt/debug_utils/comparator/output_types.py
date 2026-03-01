from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

import polars as pl
from pydantic import ConfigDict, Discriminator, Field, TypeAdapter, model_validator
from rich.console import RenderableType
from rich.markup import escape

from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
    format_comparison,
    format_replicated_checks,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
)
from sglang.srt.debug_utils.comparator.utils import Pair, _StrictBase

if TYPE_CHECKING:
    from sglang.srt.debug_utils.comparator.aligner.entrypoint.traced_types import (
        TracedAlignerPlan,
        TracedSubPlan,
    )
    from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import AlignerPlan
    from sglang.srt.debug_utils.comparator.report_sink import Verbosity


class BaseLog(_StrictBase):
    category: str
    message: str

    def to_text(self) -> str:
        return self.message


class ErrorLog(BaseLog):
    kind: Literal["error"] = "error"


class InfoLog(BaseLog):
    kind: Literal["info"] = "info"


AnyLog = Annotated[Union[ErrorLog, InfoLog], Discriminator("kind")]


def _split_logs(logs: list[BaseLog]) -> tuple[list[ErrorLog], list[InfoLog]]:
    errors: list[ErrorLog] = [log for log in logs if isinstance(log, ErrorLog)]
    infos: list[InfoLog] = [log for log in logs if isinstance(log, InfoLog)]
    return errors, infos


class ReplicatedCheckResult(_StrictBase):
    axis: str
    group_index: int
    compared_index: int
    baseline_index: int
    passed: bool
    atol: float
    diff: Optional[DiffInfo] = None


class BundleFileInfo(_StrictBase):
    """Per-file info within a bundle (one rank's raw tensor)."""

    shape: list[int]
    dtype: str
    rank: Optional[int] = None
    parallel_info: Optional[dict[str, str]] = None  # e.g. {"tp": "0/4", "ep": "1/2"}


class BundleSideInfo(_StrictBase):
    num_files: int
    files: list[BundleFileInfo]
    dims: Optional[str] = None  # e.g. "b s h(tp) d"


class ShapeSnapshot(_StrictBase):
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]


class _OutputRecord(_StrictBase):
    errors: list[ErrorLog] = Field(default_factory=list)
    infos: list[InfoLog] = Field(default_factory=list)

    @abstractmethod
    def _format_body(self) -> str: ...

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return self._format_body()

    def to_rich(self, verbosity: Verbosity = "normal") -> RenderableType:
        return self._format_body()

    def to_text(self) -> str:
        body = self._format_body()
        if self.errors:
            body += "\n" + "\n".join(f"  ✗ {e.to_text()}" for e in self.errors)
        if self.infos:
            body += "\n" + "\n".join(f"  ℹ {i.to_text()}" for i in self.infos)
        return body


class RecordLocation(_StrictBase):
    step: Optional[int] = None


class _BaseComparisonRecord(_OutputRecord):
    location: RecordLocation = Field(default_factory=RecordLocation)

    def _format_location_prefix(self) -> str:
        if self.location.step is not None:
            return f"[step={self.location.step}] "
        return ""

    def _format_location_prefix_rich(self) -> str:
        if self.location.step is not None:
            return escape(f"[step={self.location.step}]") + " "
        return ""

    def _format_location_suffix(self) -> str:
        if self.location.step is not None:
            return f" (step={self.location.step})"
        return ""


class ConfigRecord(_OutputRecord):
    type: Literal["config"] = "config"
    config: dict[str, Any]

    def _format_body(self) -> str:
        return f"Config: {self.config}"


class SkipComparisonRecord(_BaseComparisonRecord):
    type: Literal["skip"] = "skip"
    name: str
    reason: str

    @property
    def category(self) -> str:
        if self.errors:
            return "failed"
        return "skipped"

    def _format_body(self) -> str:
        return f"Skip: {self.name}{self._format_location_suffix()} ({self.reason})"


class _TableRecord(_OutputRecord):
    label: str
    rows: list[dict[str, Any]]

    @abstractmethod
    def _table_title(self) -> str: ...

    def _format_body(self) -> str:
        from sglang.srt.debug_utils.comparator.display import _render_polars_as_text

        return _render_polars_as_text(
            pl.DataFrame(self.rows), title=self._table_title()
        )


class RankInfoRecord(_TableRecord):
    type: Literal["rank_info"] = "rank_info"

    def _table_title(self) -> str:
        return f"{self.label} ranks"


class InputIdsRecord(_TableRecord):
    type: Literal["input_ids"] = "input_ids"

    def _table_title(self) -> str:
        return f"{self.label} input_ids & positions"


class TensorComparisonRecord(TensorComparisonInfo, _BaseComparisonRecord):
    model_config = ConfigDict(extra="forbid", defer_build=True)

    type: Literal["comparison"] = "comparison"
    traced_plan: Optional[TracedAlignerPlan] = None
    replicated_checks: list[ReplicatedCheckResult] = Field(default_factory=list)
    raw_bundle_info: Optional[Pair[BundleSideInfo]] = None

    @property
    def category(self) -> str:
        if self.errors:
            return "failed"
        if any(not check.passed for check in self.replicated_checks):
            return "failed"
        return "passed" if self.diff is not None and self.diff.passed else "failed"

    def _format_body(self) -> str:
        body: str = self._format_location_prefix() + format_comparison(self)
        if self.replicated_checks:
            body += "\n" + format_replicated_checks(self.replicated_checks)
        if self.traced_plan is not None:
            body += "\n" + _format_aligner_plan(self.traced_plan)
        return body


class NonTensorComparisonRecord(_BaseComparisonRecord):
    type: Literal["non_tensor"] = "non_tensor"
    name: str
    baseline_value: str
    target_value: str
    baseline_type: str
    target_type: str
    values_equal: bool

    @property
    def category(self) -> str:
        if self.errors:
            return "failed"
        return "passed" if self.values_equal else "failed"

    def _format_body(self) -> str:
        suffix: str = self._format_location_suffix()
        if self.values_equal:
            return f"NonTensor: {self.name}{suffix} = {self.baseline_value} ({self.baseline_type}) [equal]"
        return (
            f"NonTensor: {self.name}{suffix}\n"
            f"  baseline = {self.baseline_value} ({self.baseline_type})\n"
            f"  target   = {self.target_value} ({self.target_type})"
        )


class SummaryRecord(_OutputRecord):
    type: Literal["summary"] = "summary"
    total: int
    passed: int
    failed: int
    skipped: int

    @model_validator(mode="after")
    def _validate_totals(self) -> "SummaryRecord":
        expected: int = self.passed + self.failed + self.skipped
        if self.total != expected:
            raise ValueError(
                f"total={self.total} != passed({self.passed}) + failed({self.failed}) + skipped({self.skipped}) = {expected}"
            )
        return self

    def _format_body(self) -> str:
        return (
            f"Summary: {self.passed} passed, {self.failed} failed, "
            f"{self.skipped} skipped (total {self.total})"
        )


class LogRecord(_OutputRecord):
    type: Literal["log"] = "log"

    def _format_body(self) -> str:
        return ""


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


AnyRecord = Annotated[
    Union[
        ConfigRecord,
        RankInfoRecord,
        InputIdsRecord,
        SkipComparisonRecord,
        TensorComparisonRecord,
        NonTensorComparisonRecord,
        SummaryRecord,
        LogRecord,
    ],
    Discriminator("type"),
]


def _get_any_record_adapter() -> TypeAdapter:
    return TypeAdapter(AnyRecord)


def parse_record_json(json_str: str | bytes) -> AnyRecord:
    return _get_any_record_adapter().validate_json(json_str)
