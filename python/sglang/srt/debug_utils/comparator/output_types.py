from abc import abstractmethod
from typing import Annotated, Any, Literal, Union

from pydantic import Discriminator, Field, TypeAdapter, model_validator

from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
    format_comparison,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    TensorComparisonInfo,
)
from sglang.srt.debug_utils.comparator.utils import _StrictBase


class ReplicatedMismatchWarning(_StrictBase):
    kind: Literal["replicated_mismatch"] = "replicated_mismatch"
    axis: str
    group_index: int
    differing_index: int
    baseline_index: int
    max_abs_diff: float

    def to_text(self) -> str:
        return (
            f"Replicated along {self.axis}: group {self.group_index}, "
            f"index {self.differing_index} differs from {self.baseline_index} "
            f"(max_abs_diff={self.max_abs_diff:.6e})"
        )


class GeneralWarning(_StrictBase):
    kind: Literal["general"] = "general"
    category: str
    message: str

    def to_text(self) -> str:
        return self.message


AnyWarning = Annotated[
    Union[ReplicatedMismatchWarning, GeneralWarning],
    Discriminator("kind"),
]


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


class ComparisonRecord(TensorComparisonInfo, _OutputRecord):
    type: Literal["comparison"] = "comparison"

    @property
    def category(self) -> str:
        if self.warnings:
            return "failed"
        return "passed" if self.diff is not None and self.diff.passed else "failed"

    def _format_body(self) -> str:
        return format_comparison(self)


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


AnyRecord = Annotated[
    Union[ConfigRecord, SkipRecord, ComparisonRecord, SummaryRecord, WarningRecord],
    Discriminator("type"),
]


_any_record_adapter = TypeAdapter(AnyRecord)


def parse_record_json(json_str: str | bytes) -> AnyRecord:
    return _any_record_adapter.validate_json(json_str)


def print_record(record: _OutputRecord, output_format: str) -> None:
    if output_format == "json":
        print(record.model_dump_json())
    else:
        print(record.to_text())
