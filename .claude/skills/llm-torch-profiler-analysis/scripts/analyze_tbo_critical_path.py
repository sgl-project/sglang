#!/usr/bin/env python3
"""Measure GPU critical-path and communication overlap in rank traces.

The interval metrics in this script are measured directly from Chrome trace
events.  Forward-window boundaries are deliberately reported as an inference,
with confidence and invariants, because profiler traces do not carry a stable
forward-pass boundary marker.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from profile_common import (
    coerce_optional_int,
    discover_trace_files,
    extract_trace_events,
    is_complete_duration_event,
    load_trace_json,
    parse_tp_rank,
    select_heaviest_pid,
)

Interval = Tuple[float, float]
EPSILON_US = 1e-9
COMM_PHASES = ("flydsl_dispatch", "flydsl_combine", "dp_nccl")
MIN_MATCHED_KERNEL_SAMPLES = 20


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    """Return a linearly interpolated percentile, or None for no values."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, quantile))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def descriptive_stats(values: Sequence[float]) -> dict:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "total_us": sum(values),
        "mean_us": statistics.fmean(values) if values else None,
        "p50_us": percentile(values, 0.50),
        "p95_us": percentile(values, 0.95),
        "min_us": min(values) if values else None,
        "max_us": max(values) if values else None,
    }


def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Return the exact union as sorted, disjoint half-open intervals."""
    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    )
    merged: List[Interval] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + EPSILON_US:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged


def interval_union_us(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_intersection_us(
    left: Iterable[Interval], right: Iterable[Interval]
) -> float:
    return interval_union_us(interval_intersections(left, right))


def interval_intersections(
    left: Iterable[Interval], right: Iterable[Interval]
) -> List[Interval]:
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    left_index = right_index = 0
    intersections: List[Interval] = []
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if end > start:
            intersections.append((start, end))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return intersections


def classify_phase(name: object, category: object = "") -> str:
    """Classify only high-confidence communication kernel signals."""
    lowered = f"{name} {category}".lower()
    flydsl_hint = any(
        token in lowered for token in ("flydsl", "ep_", "intranode", "a2a", "alltoall")
    )
    if "dispatch" in lowered and flydsl_hint:
        return "flydsl_dispatch"
    if "combine" in lowered and flydsl_hint:
        return "flydsl_combine"
    if "nccl" in lowered or "rccl" in lowered:
        return "dp_nccl"
    return "non_comm"


def _gpu_event_filter(event: dict) -> bool:
    if not is_complete_duration_event(event):
        return False
    if str(event.get("cat", "")).lower() == "gpu_user_annotation":
        return False
    args = event.get("args")
    return isinstance(args, dict) and (
        args.get("stream") is not None or args.get("cuda_stream") is not None
    )


def _event_interval(event: dict) -> Interval:
    start = float(event["ts"])
    return start, start + float(event["dur"])


def _event_stream(event: dict) -> str:
    args = event.get("args") or {}
    return str(args.get("stream", args.get("cuda_stream", event.get("tid", "unknown"))))


def _multiplicity_durations(interval_groups: Dict[str, List[Interval]]) -> dict:
    """Measure wall time at each number of concurrently busy streams."""
    endpoints: Dict[float, int] = defaultdict(int)
    for intervals in interval_groups.values():
        for start, end in merge_intervals(intervals):
            endpoints[start] += 1
            endpoints[end] -= 1
    by_count: Dict[int, float] = defaultdict(float)
    active = 0
    previous: Optional[float] = None
    for timestamp in sorted(endpoints):
        if previous is not None and timestamp > previous and active > 0:
            by_count[active] += timestamp - previous
        active += endpoints[timestamp]
        previous = timestamp
    return {
        "by_active_stream_count_us": {
            str(count): duration for count, duration in sorted(by_count.items())
        },
        "multi_stream_overlap_us": sum(
            duration for count, duration in by_count.items() if count >= 2
        ),
        "max_concurrent_streams": max(by_count, default=0),
    }


def _overlaps_other_stream(
    interval: Interval, stream: str, merged_by_stream: Dict[str, List[Interval]]
) -> bool:
    start, end = interval
    for other_stream, intervals in merged_by_stream.items():
        if other_stream == stream:
            continue
        candidate_index = bisect_left(intervals, (end, -math.inf)) - 1
        if candidate_index >= 0:
            other_start, other_end = intervals[candidate_index]
            if other_end > start and other_start < end:
                return True
    return False


def canonicalize_kernel_name(name: object) -> str:
    """Normalize whitespace/case while preserving exact kernel identity."""
    return " ".join(str(name).split()).casefold()


def _duration_inflation(
    gpu_events: Sequence[dict],
    minimum_samples_per_state: int = MIN_MATCHED_KERNEL_SAMPLES,
) -> dict:
    by_stream: Dict[str, List[Interval]] = defaultdict(list)
    for event in gpu_events:
        by_stream[_event_stream(event)].append(_event_interval(event))
    merged_by_stream = {
        stream: merge_intervals(intervals) for stream, intervals in by_stream.items()
    }
    durations: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"overlapped": [], "unoverlapped": []}
    )
    display_names: Dict[str, str] = {}
    for event in gpu_events:
        canonical_name = canonicalize_kernel_name(event.get("name", ""))
        display_names.setdefault(canonical_name, str(event.get("name", "")))
        state = (
            "overlapped"
            if _overlaps_other_stream(
                _event_interval(event), _event_stream(event), merged_by_stream
            )
            else "unoverlapped"
        )
        durations[canonical_name][state].append(float(event["dur"]))

    matched = []
    for canonical_name, states in durations.items():
        overlapped = states["overlapped"]
        unoverlapped = states["unoverlapped"]
        if (
            len(overlapped) < minimum_samples_per_state
            or len(unoverlapped) < minimum_samples_per_state
        ):
            continue
        overlap_p50 = percentile(overlapped, 0.50)
        unoverlap_p50 = percentile(unoverlapped, 0.50)
        overlap_mean = statistics.fmean(overlapped)
        unoverlap_mean = statistics.fmean(unoverlapped)
        if not overlap_p50 or not unoverlap_p50:
            continue
        matched.append(
            {
                "canonical_name": canonical_name,
                "name": display_names[canonical_name],
                "overlapped_count": len(overlapped),
                "unoverlapped_count": len(unoverlapped),
                "overlapped_p50_us": overlap_p50,
                "unoverlapped_p50_us": unoverlap_p50,
                "overlapped_mean_us": overlap_mean,
                "unoverlapped_mean_us": unoverlap_mean,
                "p50_ratio": overlap_p50 / unoverlap_p50,
                "mean_ratio": (
                    overlap_mean / unoverlap_mean if unoverlap_mean > 0 else None
                ),
            }
        )
    matched.sort(key=lambda item: item["p50_ratio"], reverse=True)
    ratios = [item["p50_ratio"] for item in matched]
    weights = [
        min(item["overlapped_count"], item["unoverlapped_count"]) for item in matched
    ]
    weighted_geometric_ratio = (
        math.exp(
            sum(weight * math.log(ratio) for weight, ratio in zip(weights, ratios))
            / sum(weights)
        )
        if ratios and sum(weights) > 0 and all(ratio > 0 for ratio in ratios)
        else None
    )
    return {
        "minimum_samples_per_state": minimum_samples_per_state,
        "matched_kernel_count": len(matched),
        "aggregate": {
            "median_p50_ratio": percentile(ratios, 0.50),
            "weighted_geometric_p50_ratio": weighted_geometric_ratio,
            "matched_event_weight": sum(weights),
        },
        "kernels": matched,
        "definition": (
            "Only normalized exact kernel names sampled in both states are "
            "compared. An event is overlapped when its half-open interval "
            "intersects a GPU event on another stream."
        ),
    }


def _stream_idle_gaps(
    intervals_by_stream: Dict[str, List[Interval]],
) -> Tuple[dict, List[float]]:
    result = {}
    all_gaps: List[float] = []
    for stream, intervals in sorted(intervals_by_stream.items()):
        merged = merge_intervals(intervals)
        gaps = [
            current[0] - previous[1]
            for previous, current in zip(merged, merged[1:])
            if current[0] > previous[1]
        ]
        all_gaps.extend(gaps)
        result[stream] = descriptive_stats(gaps)
    return result, all_gaps


def _clip_interval(interval: Interval, window: Interval) -> Optional[Interval]:
    start = max(interval[0], window[0])
    end = min(interval[1], window[1])
    return (start, end) if end > start else None


def _clip_by_stream(
    intervals_by_stream: Dict[str, List[Interval]], window: Optional[Interval]
) -> Dict[str, List[Interval]]:
    if window is None:
        return {
            stream: list(intervals) for stream, intervals in intervals_by_stream.items()
        }
    clipped: Dict[str, List[Interval]] = defaultdict(list)
    for stream, intervals in intervals_by_stream.items():
        for interval in intervals:
            clipped_interval = _clip_interval(interval, window)
            if clipped_interval is not None:
                clipped[stream].append(clipped_interval)
    return dict(clipped)


def hidden_by_other_stream_us(
    comm_by_stream: Dict[str, List[Interval]],
    non_comm_by_stream: Dict[str, List[Interval]],
) -> float:
    """Union of comm time intersecting non-comm work on a different stream."""
    hidden_intervals: List[Interval] = []
    for comm_stream, comm_intervals in comm_by_stream.items():
        other_non_comm = [
            interval
            for non_comm_stream, intervals in non_comm_by_stream.items()
            if non_comm_stream != comm_stream
            for interval in intervals
        ]
        hidden_intervals.extend(interval_intersections(comm_intervals, other_non_comm))
    return interval_union_us(hidden_intervals)


def _communication_metrics(
    phase_intervals_by_stream: Dict[str, Dict[str, List[Interval]]],
    non_comm_by_stream: Dict[str, List[Interval]],
    window: Optional[Interval] = None,
) -> dict:
    clipped_non_comm = _clip_by_stream(non_comm_by_stream, window)
    clipped_phases = {
        phase: _clip_by_stream(phase_intervals_by_stream.get(phase, {}), window)
        for phase in COMM_PHASES
    }
    all_comm_by_stream: Dict[str, List[Interval]] = defaultdict(list)
    for by_stream in clipped_phases.values():
        for stream, intervals in by_stream.items():
            all_comm_by_stream[stream].extend(intervals)
    all_comm = [
        interval for intervals in all_comm_by_stream.values() for interval in intervals
    ]
    comm_union_us = interval_union_us(all_comm)
    comm_hidden_us = hidden_by_other_stream_us(all_comm_by_stream, clipped_non_comm)
    phases = {}
    for phase, by_stream in clipped_phases.items():
        intervals = [
            interval
            for stream_intervals in by_stream.values()
            for interval in stream_intervals
        ]
        union_us = interval_union_us(intervals)
        hidden_us = hidden_by_other_stream_us(by_stream, clipped_non_comm)
        phases[phase] = {
            "event_count": sum(len(items) for items in by_stream.values()),
            "raw_event_duration_us": sum(end - start for start, end in intervals),
            "union_us": union_us,
            "hidden_by_other_stream_non_comm_us": hidden_us,
            "exposed_us": max(0.0, union_us - hidden_us),
            "hidden_ratio": hidden_us / union_us if union_us else None,
        }
    return {
        "union_us": comm_union_us,
        "hidden_by_other_stream_non_comm_us": comm_hidden_us,
        "exposed_us": max(0.0, comm_union_us - comm_hidden_us),
        "hidden_ratio": (comm_hidden_us / comm_union_us if comm_union_us else None),
        "phases": phases,
        "hidden_definition": (
            "Comm interval union intersected only with non-comm intervals on "
            "different streams; same-stream timestamp overlap is excluded."
        ),
    }


OPERATION_SCOPE_NAMES = {
    "dispatch_a",
    "dispatch_b",
    "combine_a",
    "combine_b",
    "gather_a",
    "gather_b",
    "experts",
    "shared_experts",
    "select_experts",
    "moe",
}


def _scope_kind(name: object) -> Optional[str]:
    normalized = str(name).strip().casefold()
    if re.fullmatch(r"[ab]\d+", normalized):
        return "stage"
    if normalized in OPERATION_SCOPE_NAMES:
        return "operation"
    return None


def _extract_named_scopes(events: Sequence[dict], category: str) -> List[dict]:
    scopes = []
    for event in events:
        if (
            not is_complete_duration_event(event)
            or str(event.get("cat", "")) != category
        ):
            continue
        kind = _scope_kind(event.get("name", ""))
        if kind is None:
            continue
        start, end = _event_interval(event)
        scopes.append(
            {
                "scope_id": len(scopes),
                "name": str(event.get("name", "")).strip().casefold(),
                "kind": kind,
                "pid": str(event.get("pid")),
                "tid": str(event.get("tid")),
                "ts": start,
                "end": end,
                "dur": end - start,
                "external_id": coerce_optional_int(
                    (event.get("args") or {}).get("External id")
                ),
            }
        )
    return scopes


def _containing_scopes(
    scopes_by_thread: Dict[Tuple[str, str], List[dict]],
    pid: str,
    tid: str,
    timestamp: float,
) -> Dict[str, dict]:
    containing = [
        scope
        for scope in scopes_by_thread.get((pid, tid), [])
        if scope["ts"] <= timestamp <= scope["end"]
    ]
    output = {}
    for kind in ("stage", "operation"):
        candidates = [scope for scope in containing if scope["kind"] == kind]
        if candidates:
            output[kind] = min(candidates, key=lambda scope: scope["dur"])
    return output


def _scope_gpu_metrics(
    scope: dict,
    mapped_events: Sequence[dict],
    mapping_methods: Counter,
    wait_events: Sequence[dict],
    global_non_comm_by_stream: Dict[str, List[Interval]],
) -> dict:
    intervals_by_stream: Dict[str, List[Interval]] = defaultdict(list)
    phase_by_stream: Dict[str, Dict[str, List[Interval]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in mapped_events:
        stream = _event_stream(event)
        interval = _event_interval(event)
        intervals_by_stream[stream].append(interval)
        phase = classify_phase(event.get("name", ""), event.get("cat", ""))
        if phase in COMM_PHASES:
            phase_by_stream[phase][stream].append(interval)
    intervals = [
        interval
        for stream_intervals in intervals_by_stream.values()
        for interval in stream_intervals
    ]
    gpu_span = (
        max(end for _, end in intervals) - min(start for start, _ in intervals)
        if intervals
        else None
    )
    waits = [
        event
        for event in wait_events
        if str(event.get("pid")) == scope["pid"]
        and str(event.get("tid")) == scope["tid"]
        and scope["ts"] <= float(event["ts"]) <= scope["end"]
    ]
    wait_durations = [
        float(event["dur"]) for event in waits if is_complete_duration_event(event)
    ]
    confidence = "none"
    if mapped_events:
        if mapping_methods and set(mapping_methods) <= {"correlation_ac2g"}:
            confidence = "high"
        elif set(mapping_methods) <= {
            "correlation_ac2g",
            "correlation_launch",
            "external_id",
        }:
            confidence = "medium"
        else:
            confidence = "low"
    return {
        "scope_id": scope["scope_id"],
        "name": scope["name"],
        "kind": scope["kind"],
        "cpu_start_us": scope["ts"],
        "cpu_end_us": scope["end"],
        "cpu_wall_span_us": scope["dur"],
        "mapped_gpu_event_count": len(mapped_events),
        "mapping_methods": dict(sorted(mapping_methods.items())),
        "attribution_confidence": confidence,
        "gpu_wall_span_us": gpu_span,
        "all_stream_busy_union_us": interval_union_us(intervals),
        "stream_busy_union_us": {
            stream: interval_union_us(stream_intervals)
            for stream, stream_intervals in sorted(intervals_by_stream.items())
        },
        "communication": _communication_metrics(
            {phase: dict(phase_by_stream.get(phase, {})) for phase in COMM_PHASES},
            global_non_comm_by_stream,
        ),
        "event_wait_latency": {
            "support": (
                "host_api_duration_only" if waits else "not_represented_in_scope"
            ),
            "count": len(waits),
            "host_api_duration": descriptive_stats(wait_durations),
            "dependency_latency_us": None,
            "limitation": (
                "Trace lacks event-handle identity/record links, so device-side "
                "record-to-wait dependency latency is not supportable."
            ),
        },
    }


def _summarize_scope_metrics(scopes: Sequence[dict], group_key: str) -> dict:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for scope in scopes:
        if group_key == "stage_side":
            key = scope["name"][0] if scope["name"] else "unknown"
        else:
            key = scope["name"]
        grouped[key].append(scope)
    output = {}
    for key, items in sorted(grouped.items()):
        output[key] = {
            "scope_count": len(items),
            "mapped_scope_count": sum(
                item["mapped_gpu_event_count"] > 0 for item in items
            ),
            "cpu_wall_span_us": descriptive_stats(
                [item["cpu_wall_span_us"] for item in items]
            ),
            "gpu_wall_span_us": descriptive_stats(
                [
                    item["gpu_wall_span_us"]
                    for item in items
                    if item["gpu_wall_span_us"] is not None
                ]
            ),
            "all_stream_busy_union_us": descriptive_stats(
                [item["all_stream_busy_union_us"] for item in items]
            ),
            "comm_union_us": descriptive_stats(
                [item["communication"]["union_us"] for item in items]
            ),
            "comm_hidden_us": descriptive_stats(
                [
                    item["communication"]["hidden_by_other_stream_non_comm_us"]
                    for item in items
                ]
            ),
            "comm_exposed_us": descriptive_stats(
                [item["communication"]["exposed_us"] for item in items]
            ),
            "wait_host_api_duration_us": descriptive_stats(
                [
                    item["event_wait_latency"]["host_api_duration"]["total_us"]
                    for item in items
                ]
            ),
        }
    return output


def analyze_cpu_gpu_scopes(
    events: Sequence[dict],
    gpu_events: Sequence[dict],
) -> dict:
    cpu_scopes = _extract_named_scopes(events, "user_annotation")
    gpu_annotations = _extract_named_scopes(events, "gpu_user_annotation")
    scopes_by_thread: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for scope in cpu_scopes:
        scopes_by_thread[(scope["pid"], scope["tid"])].append(scope)

    correlation_external: Dict[int, int] = {}
    launches_by_correlation: Dict[int, List[dict]] = defaultdict(list)
    flow_starts = set()
    flow_finishes = set()
    target_external_ids = {
        external_id
        for event in gpu_events
        for external_id in [
            coerce_optional_int((event.get("args") or {}).get("External id"))
        ]
        if external_id is not None
    }
    for event in events:
        args = event.get("args") or {}
        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if correlation is not None and external_id is not None:
            correlation_external[correlation] = external_id
            target_external_ids.add(external_id)
        if (
            correlation is not None
            and is_complete_duration_event(event)
            and str(event.get("cat", "")).lower() == "cuda_runtime"
        ):
            launches_by_correlation[correlation].append(event)
        if str(event.get("cat", "")) == "ac2g":
            flow_id = coerce_optional_int(event.get("id"))
            if flow_id is not None and event.get("ph") == "s":
                flow_starts.add(flow_id)
            elif flow_id is not None and event.get("ph") == "f":
                flow_finishes.add(flow_id)
    complete_flow_ids = flow_starts & flow_finishes

    cpu_events_by_external: Dict[int, List[dict]] = defaultdict(list)
    for event in events:
        if not is_complete_duration_event(event):
            continue
        external_id = coerce_optional_int((event.get("args") or {}).get("External id"))
        if (
            external_id is not None
            and external_id in target_external_ids
            and str(event.get("cat", "")) in {"cpu_op", "cuda_runtime"}
        ):
            cpu_events_by_external[external_id].append(event)

    mapped_by_scope: Dict[int, List[dict]] = defaultdict(list)
    methods_by_scope: Dict[int, Counter] = defaultdict(Counter)
    stage_mapped = operation_mapped = 0
    method_counts: Counter = Counter()
    for gpu_event in gpu_events:
        args = gpu_event.get("args") or {}
        correlation = coerce_optional_int(args.get("correlation"))
        mapped_scopes: Dict[str, dict] = {}
        method: Optional[str] = None
        if correlation is not None and launches_by_correlation.get(correlation):
            launch = min(
                launches_by_correlation[correlation],
                key=lambda event: abs(
                    float(event.get("ts", 0.0)) - float(gpu_event.get("ts", 0.0))
                ),
            )
            mapped_scopes = _containing_scopes(
                scopes_by_thread,
                str(launch.get("pid")),
                str(launch.get("tid")),
                float(launch["ts"]),
            )
            if mapped_scopes:
                method = (
                    "correlation_ac2g"
                    if correlation in complete_flow_ids
                    else "correlation_launch"
                )
        if not mapped_scopes:
            external_id = coerce_optional_int(args.get("External id"))
            if external_id is None and correlation is not None:
                external_id = correlation_external.get(correlation)
            anchors = (
                cpu_events_by_external.get(external_id, [])
                if external_id is not None
                else []
            )
            if anchors:
                anchor = min(
                    anchors,
                    key=lambda event: abs(
                        float(event.get("ts", 0.0)) - float(gpu_event.get("ts", 0.0))
                    ),
                )
                mapped_scopes = _containing_scopes(
                    scopes_by_thread,
                    str(anchor.get("pid")),
                    str(anchor.get("tid")),
                    float(anchor["ts"]),
                )
                if mapped_scopes:
                    method = "external_id"
        if method is not None:
            method_counts[method] += 1
        if "stage" in mapped_scopes:
            stage_mapped += 1
        if "operation" in mapped_scopes:
            operation_mapped += 1
        for scope in mapped_scopes.values():
            mapped_by_scope[scope["scope_id"]].append(gpu_event)
            methods_by_scope[scope["scope_id"]][method or "unlinked"] += 1

    wait_events = [
        event
        for event in events
        if re.search(
            r"(?:stream)?wait(?:event)?",
            str(event.get("name", "")),
            re.IGNORECASE,
        )
    ]
    global_non_comm_by_stream: Dict[str, List[Interval]] = defaultdict(list)
    for event in gpu_events:
        if classify_phase(event.get("name", ""), event.get("cat", "")) == "non_comm":
            global_non_comm_by_stream[_event_stream(event)].append(
                _event_interval(event)
            )
    scope_metrics = [
        _scope_gpu_metrics(
            scope,
            mapped_by_scope.get(scope["scope_id"], []),
            methods_by_scope.get(scope["scope_id"], Counter()),
            wait_events,
            global_non_comm_by_stream,
        )
        for scope in cpu_scopes
    ]
    stage_scopes = [scope for scope in scope_metrics if scope["kind"] == "stage"]
    operation_scopes = [
        scope for scope in scope_metrics if scope["kind"] == "operation"
    ]
    gpu_count = len(gpu_events)
    flow_linked_count = sum(
        coerce_optional_int((event.get("args") or {}).get("correlation"))
        in complete_flow_ids
        for event in gpu_events
    )
    flow_ratio = flow_linked_count / gpu_count if gpu_count else None
    stage_ratio = stage_mapped / gpu_count if gpu_count else None
    operation_ratio = operation_mapped / gpu_count if gpu_count else None
    link_confidence = (
        "high"
        if flow_ratio is not None and flow_ratio >= 0.95
        else "medium" if flow_linked_count else "low" if gpu_count else "none"
    )
    return {
        "mode": "cpu_gpu_scope_mapping" if cpu_scopes else "gpu_only_fallback",
        "coverage": {
            "gpu_work_event_count": gpu_count,
            "stage_mapped_event_count": stage_mapped,
            "stage_mapping_ratio": stage_ratio,
            "operation_mapped_event_count": operation_mapped,
            "operation_mapping_ratio": operation_ratio,
            "ac2g_flow_linked_event_count": flow_linked_count,
            "ac2g_flow_link_ratio": flow_ratio,
            "mapping_method_event_counts": dict(sorted(method_counts.items())),
            "mapped_event_link_confidence": link_confidence,
            "stage_coverage_confidence": (
                "high"
                if stage_ratio is not None
                and stage_ratio >= 0.95
                and link_confidence == "high"
                else "medium" if stage_mapped else "none"
            ),
            "operation_coverage_confidence": (
                "high"
                if operation_ratio is not None
                and operation_ratio >= 0.95
                and link_confidence == "high"
                else "partial" if operation_mapped else "none"
            ),
            "confidence": (
                "high"
                if gpu_count
                and link_confidence == "high"
                and stage_ratio is not None
                and stage_ratio >= 0.95
                and operation_ratio is not None
                and operation_ratio >= 0.95
                else "medium" if stage_mapped else "none"
            ),
            "limitation": (
                "Coverage is event-count based. Correlation+ac2g validates "
                "launch attribution; External ID and GPU annotation paths are "
                "reported separately and are not promoted to exact flow links."
            ),
        },
        "detected": {
            "stage_scope_count": len(stage_scopes),
            "stage_names": [scope["name"] for scope in stage_scopes],
            "operation_scope_count": len(operation_scopes),
            "operation_names": sorted({scope["name"] for scope in operation_scopes}),
            "gpu_operation_annotation_count": len(gpu_annotations),
            "gpu_operation_annotation_names": sorted(
                {scope["name"] for scope in gpu_annotations}
            ),
        },
        "stage_scopes": stage_scopes,
        "operation_scopes": operation_scopes,
        "stage_summary": _summarize_scope_metrics(stage_scopes, "stage_side"),
        "operation_summary": _summarize_scope_metrics(
            operation_scopes, "operation_name"
        ),
    }


def infer_forward_windows(
    comm_intervals: Sequence[Interval],
    gpu_span: Optional[Interval],
    num_layers: int,
) -> dict:
    """Conservatively split a GPU span only at strongly separated comm markers."""
    invariants = [
        "windows are ordered and non-overlapping",
        "windows stay inside the measured GPU wall span",
        "every communication marker is covered by exactly one window",
    ]
    if gpu_span is None:
        return {
            "windows_us": [],
            "confidence": "none",
            "reason": "no GPU events",
            "invariants": invariants,
            "layer_validation": {
                "num_layers": num_layers,
                "global_marker_count": 0,
                "global_divisible_by_num_layers": True,
                "global_marker_multiplicity": 0.0,
                "plausible_equal_marker_counts": False,
            },
        }
    markers = merge_intervals(comm_intervals)
    if len(markers) < 2:
        return {
            "windows_us": [list(gpu_span)],
            "confidence": "low",
            "reason": "fewer than two disjoint communication markers",
            "invariants": invariants,
            "layer_validation": {
                "num_layers": num_layers,
                "global_marker_count": len(markers),
                "global_divisible_by_num_layers": (len(markers) % num_layers == 0),
                "global_marker_multiplicity": len(markers) / num_layers,
                "plausible_equal_marker_counts": False,
            },
        }

    gaps = [
        (markers[index][1], markers[index + 1][0])
        for index in range(len(markers) - 1)
        if markers[index + 1][0] > markers[index][1]
    ]
    gap_sizes = [end - start for start, end in gaps]
    typical_gap = percentile(gap_sizes, 0.50) or 0.0
    baseline_gaps = sorted(gap_sizes)[:-1] if len(gap_sizes) >= 3 else gap_sizes
    high_tail_gap = percentile(baseline_gaps, 0.99) or typical_gap
    # Repeated layer-level collectives can have gaps well above the median but
    # are not forward boundaries.  Demand separation from both the typical gap
    # and the observed high tail instead of treating every 5 ms gap as a split.
    threshold = max(5000.0, typical_gap * 8.0, high_tail_gap * 5.0)
    candidates = [
        (start, end)
        for start, end in gaps
        if end - start >= threshold
        and sum(marker[1] <= start for marker in markers) >= 2
        and sum(marker[0] >= end for marker in markers) >= 2
    ]
    if not candidates:
        return {
            "windows_us": [list(gpu_span)],
            "confidence": "low",
            "reason": (
                "no comm-marker gap met the conservative threshold and "
                "two-markers-per-side invariant"
            ),
            "comm_marker_count": len(markers),
            "typical_marker_gap_us": typical_gap,
            "p99_marker_gap_us": high_tail_gap,
            "large_gap_threshold_us": threshold,
            "invariants": invariants,
            "layer_validation": {
                "num_layers": num_layers,
                "global_marker_count": len(markers),
                "global_divisible_by_num_layers": (len(markers) % num_layers == 0),
                "global_marker_multiplicity": len(markers) / num_layers,
                "plausible_equal_marker_counts": False,
            },
        }

    boundaries = [(start + end) / 2.0 for start, end in candidates]
    starts = [gpu_span[0], *boundaries]
    ends = [*boundaries, gpu_span[1]]
    windows = [[start, end] for start, end in zip(starts, ends) if end > start]
    marker_counts = [
        sum(
            start <= (marker_start + marker_end) / 2.0 < end
            for marker_start, marker_end in markers
        )
        for start, end in windows
    ]
    divisible = [count > 0 and count % num_layers == 0 for count in marker_counts]
    multiplicities = [
        count / num_layers if num_layers > 0 else None for count in marker_counts
    ]
    plausible_equal_counts = (
        bool(marker_counts) and all(divisible) and len(set(marker_counts)) == 1
    )
    confidence = "medium" if plausible_equal_counts else "low"
    reason = (
        "large gaps plus equal layer-divisible marker counts; capped at medium "
        "because no independent CPU forward spans were found"
        if plausible_equal_counts
        else "large gaps found, but per-window marker counts do not satisfy "
        "equal layer-divisible forward assumptions"
    )
    return {
        "windows_us": windows,
        "confidence": confidence,
        "reason": reason,
        "comm_marker_count": len(markers),
        "typical_marker_gap_us": typical_gap,
        "p99_marker_gap_us": high_tail_gap,
        "large_gap_threshold_us": threshold,
        "selected_gaps_us": [end - start for start, end in candidates],
        "invariants": invariants,
        "layer_validation": {
            "num_layers": num_layers,
            "global_marker_count": len(markers),
            "global_divisible_by_num_layers": len(markers) % num_layers == 0,
            "global_marker_multiplicity": len(markers) / num_layers,
            "marker_counts_per_window": marker_counts,
            "expected_multiplicity_per_window": multiplicities,
            "divisible_by_num_layers_per_window": divisible,
            "plausible_equal_marker_counts": plausible_equal_counts,
            "cpu_forward_span_evidence": False,
        },
    }


def _window_metrics(
    window: Interval,
    all_intervals: Sequence[Interval],
    phase_intervals_by_stream: Dict[str, Dict[str, List[Interval]]],
    non_comm_by_stream: Dict[str, List[Interval]],
) -> dict:
    clipped_busy = [
        clipped
        for interval in all_intervals
        for clipped in [_clip_interval(interval, window)]
        if clipped is not None
    ]
    return {
        "start_us": window[0],
        "end_us": window[1],
        "wall_span_us": window[1] - window[0],
        "busy_union_us": interval_union_us(clipped_busy),
        "communication": _communication_metrics(
            phase_intervals_by_stream, non_comm_by_stream, window
        ),
    }


def analyze_rank_trace(
    path: Path, rank: Optional[int] = None, num_layers: int = 61
) -> dict:
    trace = load_trace_json(path)
    events = extract_trace_events(trace)
    selected_pid = select_heaviest_pid(events, _gpu_event_filter)
    if selected_pid is None:
        raise ValueError(f"No complete GPU stream events found in {path}")
    gpu_events = [
        event
        for event in events
        if _gpu_event_filter(event) and str(event.get("pid")) == selected_pid
    ]
    intervals_by_stream: Dict[str, List[Interval]] = defaultdict(list)
    intervals_by_phase: Dict[str, List[Interval]] = defaultdict(list)
    intervals_by_phase_stream: Dict[str, Dict[str, List[Interval]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in gpu_events:
        interval = _event_interval(event)
        stream = _event_stream(event)
        intervals_by_stream[stream].append(interval)
        phase = classify_phase(event.get("name", ""), event.get("cat", ""))
        intervals_by_phase[phase].append(interval)
        intervals_by_phase_stream[phase][stream].append(interval)

    all_intervals = [_event_interval(event) for event in gpu_events]
    all_busy = merge_intervals(all_intervals)
    gpu_span = (
        (min(start for start, _ in all_intervals), max(end for _, end in all_intervals))
        if all_intervals
        else None
    )
    comm_intervals = [
        interval
        for phase in COMM_PHASES
        for interval in intervals_by_phase.get(phase, [])
    ]
    phase_streams = {
        phase: dict(intervals_by_phase_stream.get(phase, {})) for phase in COMM_PHASES
    }
    non_comm_by_stream = dict(intervals_by_phase_stream.get("non_comm", {}))
    communication_metrics = _communication_metrics(phase_streams, non_comm_by_stream)

    waits: Dict[str, List[float]] = defaultdict(list)
    wait_count_without_duration: Counter = Counter()
    for event in events:
        name = str(event.get("name", ""))
        if not re.search(r"(?:stream)?wait(?:event)?", name, re.IGNORECASE):
            continue
        if is_complete_duration_event(event):
            waits[name].append(float(event["dur"]))
        else:
            wait_count_without_duration[name] += 1

    per_stream_idle, all_idle_gaps = _stream_idle_gaps(intervals_by_stream)
    multiplicity = _multiplicity_durations(intervals_by_stream)
    stream_busy = {
        stream: {
            "event_count": sum(
                1 for event in gpu_events if _event_stream(event) == stream
            ),
            "busy_union_us": interval_union_us(intervals),
        }
        for stream, intervals in sorted(intervals_by_stream.items())
    }
    wall_span_us = gpu_span[1] - gpu_span[0] if gpu_span else 0.0
    busy_union_us = interval_union_us(all_busy)
    forward_inference = infer_forward_windows(comm_intervals, gpu_span, num_layers)
    forward_inference["window_metrics"] = [
        {
            "window_index": index,
            **_window_metrics(
                (window[0], window[1]),
                all_intervals,
                phase_streams,
                non_comm_by_stream,
            ),
        }
        for index, window in enumerate(forward_inference["windows_us"])
    ]
    cpu_gpu_scope_analysis = analyze_cpu_gpu_scopes(events, gpu_events)
    forward_inference["fallback_role"] = (
        "secondary_gpu_only_heuristic"
        if cpu_gpu_scope_analysis["mode"] == "cpu_gpu_scope_mapping"
        else "primary_gpu_only_heuristic"
    )
    comm_union_us = communication_metrics["union_us"]
    comm_hidden_us = communication_metrics["hidden_by_other_stream_non_comm_us"]
    return {
        "rank": rank,
        "trace_path": str(path),
        "selected_gpu_pid": selected_pid,
        "gpu_event_count": len(gpu_events),
        "stream_count": len(intervals_by_stream),
        "gpu_wall_span_us": wall_span_us,
        "all_stream_busy_union_us": busy_union_us,
        "gpu_wall_idle_us": max(0.0, wall_span_us - busy_union_us),
        "all_stream_utilization": (
            busy_union_us / wall_span_us if wall_span_us else None
        ),
        "multi_stream_overlap_us": multiplicity["multi_stream_overlap_us"],
        "max_concurrent_streams": multiplicity["max_concurrent_streams"],
        "busy_by_active_stream_count_us": multiplicity["by_active_stream_count_us"],
        "communication": communication_metrics,
        "streams": stream_busy,
        "stream_idle_gaps": {
            "all_streams": descriptive_stats(all_idle_gaps),
            "per_stream": per_stream_idle,
            "definition": "internal gaps between consecutive merged busy intervals",
        },
        "wait_events": {
            "represented_with_duration": {
                name: descriptive_stats(durations)
                for name, durations in sorted(waits.items())
            },
            "represented_without_complete_duration": dict(
                sorted(wait_count_without_duration.items())
            ),
            "total_count": sum(len(durations) for durations in waits.values())
            + sum(wait_count_without_duration.values()),
            "total_duration_us": sum(sum(durations) for durations in waits.values()),
        },
        "kernel_duration_inflation": _duration_inflation(gpu_events),
        "cpu_gpu_scope_attribution": cpu_gpu_scope_analysis,
        "forward_window_inference": forward_inference,
        "measurement_invariants": {
            "comm_hidden_plus_exposed_equals_union": math.isclose(
                comm_hidden_us + communication_metrics["exposed_us"],
                comm_union_us,
                abs_tol=1e-6,
            ),
            "busy_union_not_greater_than_wall_span": busy_union_us
            <= wall_span_us + 1e-6,
            "multi_stream_overlap_not_greater_than_busy_union": multiplicity[
                "multi_stream_overlap_us"
            ]
            <= busy_union_us + 1e-6,
        },
    }


def discover_rank_traces(path: Path, expected_ranks: int = 8) -> List[Tuple[int, Path]]:
    path = path.expanduser().resolve()
    discovery_root = path.parent if path.is_file() else path
    files = discover_trace_files(discovery_root, recursive=True)
    if path.is_file() and "-TP-" in path.name:
        capture_prefix = path.name.split("-TP-", 1)[0]
        files = [
            trace_path
            for trace_path in files
            if trace_path.name.startswith(f"{capture_prefix}-TP-")
        ]
    if not files:
        raise FileNotFoundError(f"No Chrome trace files found under {path}")
    by_rank: Dict[int, List[Path]] = defaultdict(list)
    unknown: List[Path] = []
    for trace_path in files:
        rank = parse_tp_rank(trace_path)
        if rank is None:
            unknown.append(trace_path)
        else:
            by_rank[rank].append(trace_path)
    if by_rank:
        if len(by_rank) != expected_ranks:
            raise ValueError(
                f"Expected {expected_ranks} ranks under {path}, found "
                f"{sorted(by_rank)}"
            )
        return [
            (rank, max(paths, key=lambda item: item.stat().st_mtime))
            for rank, paths in sorted(by_rank.items())
        ]
    if len(unknown) != expected_ranks:
        raise ValueError(
            f"Expected {expected_ranks} rank traces under {path}, found "
            f"{len(unknown)} files without rank names"
        )
    return list(enumerate(sorted(unknown)))


def _flatten_numeric(value: object, prefix: str = "") -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    if isinstance(value, bool) or value is None:
        return flattened
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        flattened[prefix] = float(value)
        return flattened
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, child_prefix))
    return flattened


def aggregate_ranks(rank_results: Sequence[dict]) -> dict:
    values_by_metric: Dict[str, List[float]] = defaultdict(list)
    excluded_prefixes = (
        "rank",
        "selected_gpu_pid",
        "busy_by_active_stream_count_us.",
        "streams.",
        "stream_idle_gaps.per_stream.",
        "wait_events.represented_",
        "forward_window_inference.",
        "measurement_invariants.",
    )
    for rank_result in rank_results:
        for metric, value in _flatten_numeric(rank_result).items():
            if metric == "rank" or metric.startswith(excluded_prefixes):
                continue
            values_by_metric[metric].append(value)
    return {
        metric: {
            "rank_count": len(values),
            "min": min(values),
            "p50": percentile(values, 0.50),
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
        for metric, values in sorted(values_by_metric.items())
    }


def analyze_label(label: str, path: Path, num_layers: int = 61) -> dict:
    rank_paths = discover_rank_traces(path)
    ranks = [
        analyze_rank_trace(trace_path, rank=rank, num_layers=num_layers)
        for rank, trace_path in rank_paths
    ]
    return {
        "label": label,
        "input_path": str(path.expanduser().resolve()),
        "num_layers": num_layers,
        "rank_count": len(ranks),
        "ranks": ranks,
        "rank_aggregate": aggregate_ranks(ranks),
    }


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict) -> str:
    lines = [
        "# TBO critical-path analysis",
        "",
        "Measured durations are microseconds. Interval unions prevent double "
        "counting. Forward windows are inferred and must be interpreted with "
        "their confidence and invariants.",
        "",
    ]
    selected_metrics = (
        "gpu_wall_span_us",
        "all_stream_busy_union_us",
        "multi_stream_overlap_us",
        "communication.union_us",
        "communication.hidden_by_other_stream_non_comm_us",
        "communication.exposed_us",
        "communication.phases.flydsl_dispatch.union_us",
        "communication.phases.flydsl_combine.union_us",
        "communication.phases.dp_nccl.union_us",
        "kernel_duration_inflation.aggregate.median_p50_ratio",
        "cpu_gpu_scope_attribution.coverage.stage_mapping_ratio",
        "cpu_gpu_scope_attribution.coverage.operation_mapping_ratio",
        "cpu_gpu_scope_attribution.coverage.ac2g_flow_link_ratio",
        "cpu_gpu_scope_attribution.stage_summary.a.gpu_wall_span_us.p50_us",
        "cpu_gpu_scope_attribution.stage_summary.b.gpu_wall_span_us.p50_us",
    )
    for trace_report in report["traces"]:
        lines.extend(
            [
                f"## {trace_report['label']}",
                "",
                f"Input: `{trace_report['input_path']}`  ",
                f"Ranks: {trace_report['rank_count']}",
                "",
                "| Metric | rank min | rank p50/median | rank p95 | rank max |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        aggregate = trace_report["rank_aggregate"]
        for metric in selected_metrics:
            stats = aggregate.get(metric)
            if stats is None:
                continue
            lines.append(
                f"| `{metric}` | {_format_value(stats['min'])} | "
                f"{_format_value(stats['p50'])} | {_format_value(stats['p95'])} | "
                f"{_format_value(stats['max'])} |"
            )
        lines.extend(["", "### Forward-window inference", ""])
        for rank in trace_report["ranks"]:
            inference = rank["forward_window_inference"]
            validation = inference["layer_validation"]
            marker_counts = validation.get("marker_counts_per_window", [])
            lines.append(
                f"- Rank {rank['rank']}: **{inference['confidence']}** confidence, "
                f"{len(inference['windows_us'])} window(s), marker counts "
                f"`{marker_counts}` — {inference['reason']}."
            )
        lines.extend(
            [
                "",
                "Invariants: windows are ordered/non-overlapping, remain inside "
                "the measured GPU span, and cover every communication marker.",
                "",
            ]
        )
        lines.extend(["### CPU/GPU scope attribution", ""])
        for rank in trace_report["ranks"]:
            scope_analysis = rank["cpu_gpu_scope_attribution"]
            detected = scope_analysis["detected"]
            coverage = scope_analysis["coverage"]
            lines.append(
                f"- Rank {rank['rank']}: {detected['stage_scope_count']} stage "
                f"scopes, {detected['operation_scope_count']} operation scopes; "
                f"stage coverage {_format_value(coverage['stage_mapping_ratio'])}, "
                f"operation coverage "
                f"{_format_value(coverage['operation_mapping_ratio'])}, "
                f"ac2g coverage {_format_value(coverage['ac2g_flow_link_ratio'])} "
                f"({coverage['confidence']} confidence)."
            )
        lines.append("")
        inflated = sorted(
            (
                (kernel["p50_ratio"], rank["rank"], kernel)
                for rank in trace_report["ranks"]
                for kernel in rank["kernel_duration_inflation"]["kernels"]
            ),
            reverse=True,
            key=lambda item: item[0],
        )[:10]
        lines.extend(["### Top matched-name duration inflation", ""])
        if not inflated:
            lines.append(
                "No exact normalized kernel name met the minimum sample count "
                "in both overlap states."
            )
        else:
            lines.extend(
                [
                    "| Rank | Kernel | overlapped / unoverlapped samples | "
                    "p50 ratio |",
                    "|---:|---|---:|---:|",
                ]
            )
            for ratio, rank, kernel in inflated:
                name = kernel["name"].replace("|", "\\|")
                if len(name) > 100:
                    name = name[:97] + "..."
                lines.append(
                    f"| {rank} | `{name}` | {kernel['overlapped_count']} / "
                    f"{kernel['unoverlapped_count']} | {ratio:.3f} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_trace_argument(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--trace must be LABEL=PATH_OR_DIR")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--trace must be LABEL=PATH_OR_DIR")
    return label.strip(), Path(raw_path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        type=parse_trace_argument,
        metavar="LABEL=PATH_OR_DIR",
        help="Trace label and a rank-trace file/directory; repeat as needed.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=61,
        help="Expected model layer count for marker-multiplicity validation.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_layers <= 0:
        raise ValueError("--num-layers must be positive")
    labels = [label for label, _ in args.trace]
    if len(labels) != len(set(labels)):
        raise ValueError("--trace labels must be unique")
    report = {
        "schema_version": 3,
        "units": "microseconds",
        "traces": [
            analyze_label(label, path, num_layers=args.num_layers)
            for label, path in args.trace
        ],
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tbo_critical_path.json"
    markdown_path = output_dir / "tbo_critical_path.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
