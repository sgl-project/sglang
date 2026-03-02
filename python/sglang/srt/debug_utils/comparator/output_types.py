from __future__ import annotations

import sys
from abc import abstractmethod
from pathlib import Path
from typing import IO, TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

import polars as pl
from pydantic import ConfigDict, Discriminator, Field, TypeAdapter, model_validator

from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
    format_comparison,
    format_replicated_checks,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
)
from sglang.srt.debug_utils.comparator.utils import _StrictBase

if TYPE_CHECKING:
    from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
        AlignerPlan,
    )


class GeneralWarning(_StrictBase):
    kind: Literal["general"] = "general"
    category: str
    message: str

    def to_text(self) -> str:
        return self.message


# Type alias — currently only GeneralWarning exists.
# When adding new warning types, convert back to Union + Discriminator("kind").
AnyWarning = GeneralWarning


class ReplicatedCheckResult(_StrictBase):
    axis: str
    group_index: int
    compared_index: int
    baseline_index: int
    passed: bool
    atol: float
    diff: DiffInfo


class _OutputRecord(_StrictBase):
    warnings: list[AnyWarning] = Field(default_factory=list)

    @abstractmethod
    def _format_body(self) -> str: ...

    def to_text(self) -> str:
        body = self._format_body()
        if self.warnings:
            body += "\n" + "\n".join(f"  ⚠ {w.to_text()}" for w in self.warnings)
        return body


class ConfigRecord(_OutputRecord):
    type: Literal["config"] = "config"
    config: dict[str, Any]

    @classmethod
    def from_args(cls, args) -> "ConfigRecord":
        """Create ConfigRecord from argparse.Namespace."""
        return cls(config=vars(args))

    def _format_body(self) -> str:
        return f"Config: {self.config}"


class SkipRecord(_OutputRecord):
    type: Literal["skip"] = "skip"
    name: str
    reason: str

    @property
    def category(self) -> str:
        if self.warnings:
            return "failed"
        return "skipped"

    def _format_body(self) -> str:
        return f"Skip: {self.name} ({self.reason})"


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


class ComparisonRecord(TensorComparisonInfo, _OutputRecord):
    model_config = ConfigDict(extra="forbid", defer_build=True)

    type: Literal["comparison"] = "comparison"
    aligner_plan: Optional[AlignerPlan] = None
    replicated_checks: list[ReplicatedCheckResult] = Field(default_factory=list)

    @property
    def category(self) -> str:
        if self.warnings:
            return "failed"
        if any(not check.passed for check in self.replicated_checks):
            return "failed"
        return "passed" if self.diff is not None and self.diff.passed else "failed"

    def _format_body(self) -> str:
        body: str = format_comparison(self)
        if self.replicated_checks:
            body += "\n" + format_replicated_checks(self.replicated_checks)
        if self.aligner_plan is not None:
            body += "\n" + _format_aligner_plan(self.aligner_plan)
        return body


class NonTensorRecord(_OutputRecord):
    type: Literal["non_tensor"] = "non_tensor"
    name: str
    baseline_value: str
    target_value: str
    baseline_type: str
    target_type: str
    values_equal: bool

    @property
    def category(self) -> str:
        if self.warnings:
            return "failed"
        return "passed" if self.values_equal else "failed"

    def _format_body(self) -> str:
        if self.values_equal:
            return f"NonTensor: {self.name} = {self.baseline_value} ({self.baseline_type}) [equal]"
        return (
            f"NonTensor: {self.name}\n"
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


class WarningRecord(_OutputRecord):
    type: Literal["warning"] = "warning"

    def _format_body(self) -> str:
        return ""


def _format_aligner_plan(plan: AlignerPlan) -> str:
    lines: list[str] = ["Aligner Plan:"]

    for side_label, side_plans in [
        ("baseline", plan.per_step_plans.x),
        ("target", plan.per_step_plans.y),
    ]:
        if not side_plans:
            lines.append(f"  {side_label}: (no steps)")
            continue

        step_summaries: list[str] = []
        for step_plan in side_plans:
            sub_strs: list[str] = []
            for sub in step_plan.sub_plans:
                sub_strs.append(f"{sub.type}")
            summary: str = ", ".join(sub_strs) if sub_strs else "passthrough"
            step_summaries.append(f"step={step_plan.step}: {summary}")
        lines.append(f"  {side_label}: [{'; '.join(step_summaries)}]")

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

    return "\n".join(lines)


AnyRecord = Annotated[
    Union[
        ConfigRecord,
        RankInfoRecord,
        InputIdsRecord,
        SkipRecord,
        ComparisonRecord,
        NonTensorRecord,
        SummaryRecord,
        WarningRecord,
    ],
    Discriminator("type"),
]


def _get_any_record_adapter() -> TypeAdapter:
    return TypeAdapter(AnyRecord)


def parse_record_json(json_str: str | bytes) -> AnyRecord:
    return _get_any_record_adapter().validate_json(json_str)


def _print_to_stdout(record: _OutputRecord, *, output_format: str) -> None:
    if output_format == "json":
        print(record.model_dump_json())
    else:
        print(record.to_text())


class ReportSink:
    """Unified entry point for all record output."""

    def __init__(self) -> None:
        self._output_format: str = "text"
        self._report_file: Optional[IO[str]] = None
        self._report_path: Optional[Path] = None

    def configure(
        self,
        *,
        output_format: str = "text",
        report_path: Optional[Path] = None,
    ) -> None:
        self._output_format = output_format

        if report_path is not None:
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                self._report_file = open(report_path, "w", encoding="utf-8")
                self._report_path = report_path
            except OSError as exc:
                print(
                    f"Warning: cannot open report file {report_path}: {exc}",
                    file=sys.stderr,
                )

    def add(self, record: _OutputRecord) -> None:
        _print_to_stdout(record, output_format=self._output_format)

        if self._report_file is not None:
            self._report_file.write(record.model_dump_json())
            self._report_file.write("\n")
            self._report_file.flush()

    def close(self) -> None:
        if self._report_file is not None:
            self._report_file.close()
            self._report_file = None

    @property
    def report_path(self) -> Optional[Path]:
        return self._report_path

    def _reset(self) -> None:
        """Reset state for test isolation."""
        self.close()
        self._output_format = "text"
        self._report_path = None


report_sink = ReportSink()
