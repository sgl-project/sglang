from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import msgspec

from sglang.srt.utils.msgspec_utils import msgspec_struct_pydantic_core_schema

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


# ======================================================================
# Shared types
# ======================================================================
class WeightVersionSpan(msgspec.Struct, kw_only=True, array_like=True):
    version: str
    start: int
    end: int

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


WeightVersionSpans = List[WeightVersionSpan]


@dataclasses.dataclass(frozen=True, slots=True)
class WeightVersionEvent:
    old_version: str
    num_output_tokens: int


# ======================================================================
# Scheduler process
# ======================================================================
def record_weight_version_events(reqs: Iterable[Req], old_version: str) -> int:
    num_recorded = 0
    for req in reqs:
        if req.output_ids:
            req.weight_version_events.append(
                WeightVersionEvent(
                    old_version=old_version,
                    num_output_tokens=len(req.output_ids),
                )
            )
            num_recorded += 1
    return num_recorded


def truncate_weight_version_events(
    events: List[WeightVersionEvent], num_kept_tokens: int
) -> List[WeightVersionEvent]:
    truncated = [
        WeightVersionEvent(
            old_version=event.old_version,
            num_output_tokens=min(event.num_output_tokens, num_kept_tokens),
        )
        for event in events
    ]
    return [event for event in truncated if event.num_output_tokens > 0]


def compute_weight_version_spans(
    events: List[WeightVersionEvent],
    current_version: str,
    num_output_tokens: int,
) -> WeightVersionSpans:
    changes = [(event.old_version, event.num_output_tokens) for event in events]
    changes.append((current_version, num_output_tokens))

    spans: WeightVersionSpans = []
    for version, end in changes:
        end = min(end, num_output_tokens)
        if spans and end <= spans[-1].end:
            continue
        if spans and version == spans[-1].version:
            spans[-1].end = end
            continue
        start = spans[-1].end if spans else 0
        spans.append(WeightVersionSpan(version=version, start=start, end=end))
    return spans


# ======================================================================
# TokenizerManager
# ======================================================================
def add_weight_versions_to_meta_info(
    meta_info: Dict[str, Any],
    spans: WeightVersionSpans,
    num_output_tokens: int,
) -> None:
    visible = [
        span for span in spans if span.start < num_output_tokens or span.start == 0
    ]

    meta_info["weight_versions"] = [
        {
            "version": span.version,
            "start": span.start,
            "end": min(span.end, num_output_tokens),
        }
        for span in visible
    ]
    meta_info["weight_version"] = visible[-1].version


# ======================================================================
# OpenAI-compatible endpoints
# ======================================================================
def build_endpoint_weight_version_metadata(meta_info: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {"weight_version": meta_info["weight_version"]}
    if "weight_versions" in meta_info:
        metadata["weight_versions"] = meta_info["weight_versions"]
    return metadata
