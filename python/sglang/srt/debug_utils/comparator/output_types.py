from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

from pydantic import ConfigDict, Discriminator, Field, TypeAdapter, model_validator
from rich.console import RenderableType
from rich.markup import escape

from sglang.srt.debug_utils.comparator.output_formatter import (  # noqa: F401 — re-export
    _format_aligner_plan as _format_aligner_plan,
)
from sglang.srt.debug_utils.comparator.output_formatter import (
    _format_config_body,
    _format_config_rich_body,
    _format_error_body,
    _format_error_rich_body,
    _format_log_body,
    _format_non_tensor_body,
    _format_non_tensor_rich_body,
    _format_skip_body,
    _format_skip_rich_body,
    _format_summary_body,
    _format_summary_rich_body,
    _format_table_body,
    _format_table_rich_body,
    _format_tensor_comparison_body,
    _format_tensor_comparison_rich_body,
    _render_record_rich,
    _render_record_text,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
)
from sglang.srt.debug_utils.comparator.utils import Pair, _StrictBase

if TYPE_CHECKING:
    from sglang.srt.debug_utils.comparator.aligner.entrypoint.traced_types import (
        TracedAlignerPlan,
    )
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
        return _render_record_rich(self, verbosity=verbosity)

    def to_text(self) -> str:
        return _render_record_text(self)


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
        return _format_config_body(self)

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return _format_config_rich_body(self, verbosity=verbosity)


class ComparisonSkipRecord(_BaseComparisonRecord):
    type: Literal["comparison_skip"] = "comparison_skip"
    name: str
    reason: str

    @property
    def category(self) -> str:
        if self.errors:
            return "failed"
        return "skipped"

    def _format_body(self) -> str:
        return _format_skip_body(self)

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return _format_skip_rich_body(self, verbosity=verbosity)


class ComparisonErrorRecord(_BaseComparisonRecord):
    type: Literal["comparison_error"] = "comparison_error"
    name: str
    exception_type: str
    traceback_str: str

    @property
    def category(self) -> str:
        return "errored"

    def _format_body(self) -> str:
        return _format_error_body(self)

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return _format_error_rich_body(self, verbosity=verbosity)


class _TableRecord(_OutputRecord):
    label: str
    rows: list[dict[str, Any]]

    @abstractmethod
    def _table_title(self) -> str: ...

    def _format_body(self) -> str:
        return _format_table_body(self)

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return _format_table_rich_body(self, verbosity=verbosity)


class RankInfoRecord(_TableRecord):
    type: Literal["rank_info"] = "rank_info"

    def _table_title(self) -> str:
        return f"{self.label} ranks"


class InputIdsRecord(_TableRecord):
    type: Literal["input_ids"] = "input_ids"

    def _table_title(self) -> str:
        return f"{self.label} input_ids & positions"


class ComparisonTensorRecord(TensorComparisonInfo, _BaseComparisonRecord):
    model_config = ConfigDict(extra="forbid", defer_build=True)

    type: Literal["comparison_tensor"] = "comparison_tensor"
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
        return _format_tensor_comparison_body(self)

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return _format_tensor_comparison_rich_body(self, verbosity=verbosity)


class ComparisonNonTensorRecord(_BaseComparisonRecord):
    type: Literal["comparison_non_tensor"] = "comparison_non_tensor"
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
        return _format_non_tensor_body(self)

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return _format_non_tensor_rich_body(self, verbosity=verbosity)


class SummaryRecord(_OutputRecord):
    type: Literal["summary"] = "summary"
    total: int
    passed: int
    failed: int
    skipped: int
    errored: int = 0

    @model_validator(mode="after")
    def _validate_totals(self) -> "SummaryRecord":
        expected: int = self.passed + self.failed + self.skipped + self.errored
        if self.total != expected:
            raise ValueError(
                f"total={self.total} != passed({self.passed}) + failed({self.failed}) "
                f"+ skipped({self.skipped}) + errored({self.errored}) = {expected}"
            )
        return self

    def _format_body(self) -> str:
        return _format_summary_body(self)

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return _format_summary_rich_body(self, verbosity=verbosity)


class LogRecord(_OutputRecord):
    type: Literal["log"] = "log"

    def _format_body(self) -> str:
        return _format_log_body(self)


AnyRecord = Annotated[
    Union[
        ConfigRecord,
        RankInfoRecord,
        InputIdsRecord,
        ComparisonSkipRecord,
        ComparisonErrorRecord,
        ComparisonTensorRecord,
        ComparisonNonTensorRecord,
        SummaryRecord,
        LogRecord,
    ],
    Discriminator("type"),
]


def _get_any_record_adapter() -> TypeAdapter:
    return TypeAdapter(AnyRecord)


def parse_record_json(json_str: str | bytes) -> AnyRecord:
    return _get_any_record_adapter().validate_json(json_str)
