from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import Discriminator, TypeAdapter

from sglang.srt.debug_utils.comparator.tensor_comparison.formatter import (
    format_comparison,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.types import (
    TensorComparisonInfo,
)
from sglang.srt.debug_utils.comparator.utils import _StrictBase


class _OutputRecord(_StrictBase):
    @abstractmethod
    def to_text(self) -> str: ...


class ConfigRecord(_OutputRecord):
    type: Literal["config"] = "config"
    baseline_path: str
    target_path: str
    diff_threshold: float
    start_step: int
    end_step: int

    def to_text(self) -> str:
        return (
            f"Config: baseline={self.baseline_path} target={self.target_path}\n"
            f"diff_threshold={self.diff_threshold} "
            f"steps=[{self.start_step}, {self.end_step}]"
        )


class SkipRecord(_OutputRecord):
    type: Literal["skip"] = "skip"
    name: str
    reason: str

    @property
    def category(self):
        return "skipped"

    def to_text(self) -> str:
        return f"Skip: {self.name} ({self.reason})"


class ComparisonRecord(TensorComparisonInfo, _OutputRecord):
    type: Literal["comparison"] = "comparison"

    @property
    def category(self):
        return "passed" if self.diff is not None and self.diff.passed else "failed"

    def to_text(self) -> str:
        return format_comparison(self)


class SummaryRecord(_OutputRecord):
    type: Literal["summary"] = "summary"
    total: int
    passed: int
    failed: int
    skipped: int

    def to_text(self) -> str:
        return (
            f"Summary: {self.passed} passed, {self.failed} failed, "
            f"{self.skipped} skipped (total {self.total})"
        )


AnyRecord = Annotated[
    Union[ConfigRecord, SkipRecord, ComparisonRecord, SummaryRecord],
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
