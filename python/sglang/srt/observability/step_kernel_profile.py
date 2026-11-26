"""Attribute a kernel to the CPU forward that submitted its CUDA launch.

The join is kernel.correlation -> a unique CUDA runtime/driver launch -> an
enclosing step[] CPU annotation on that launch's process and thread. Device
timestamps never choose a step. A graph replay has a separate cudaGraphLaunch
correlation, so graph ids and reused graph nodes do not become step ids.
Missing/ambiguous evidence remains unmatched. Raw profiler loss stays unknown.
"""

from __future__ import annotations

import heapq
import json
from collections import defaultdict
from pathlib import Path

from sglang.srt.observability.kernel_profile import (
    _ns,
    interval_union_ns,
    kernel_intervals,
)


def attribute_step_kernels(payload: dict) -> dict:
    spans = []
    launches = defaultdict(list)
    for event in payload["traceEvents"]:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if not isinstance(name, str):
            continue
        is_step = event.get("cat") == "user_annotation" and name.startswith("step[")
        is_launch = (
            event.get("cat") in {"cuda_runtime", "cuda_driver"} and "Launch" in name
        )
        if not (is_step or is_launch):
            continue
        try:
            start = _ns(event["ts"])
            end = start + _ns(event["dur"])
            pid, tid = event["pid"], event["tid"]
            if (
                end <= start
                or not isinstance(pid, (int, str))
                or not isinstance(tid, (int, str))
            ):
                continue
        except (KeyError, TypeError, ValueError, ArithmeticError):
            continue
        if is_step:
            spans.append(
                {"name": name, "pid": pid, "tid": tid, "start": start, "end": end}
            )
        else:
            args = event.get("args")
            corr = args.get("correlation") if isinstance(args, dict) else None
            if type(corr) is int and corr > 0:
                launches[corr].append((pid, tid, start, end))

    spans.sort(key=lambda s: (s["start"], s["end"], str(s["pid"]), str(s["tid"])))
    by_thread = defaultdict(list)
    for index, span in enumerate(spans):
        by_thread[(span["pid"], span["tid"])].append(index)
    by_launch_thread = defaultdict(list)
    for corr, candidates in launches.items():
        if len(candidates) == 1:
            pid, tid, start, end = candidates[0]
            by_launch_thread[(pid, tid)].append((start, end, corr))
    owners = {}
    for thread, queries in by_launch_thread.items():
        indices = by_thread[thread]
        position = 0
        active = {}
        expiry = []
        for start, end, corr in sorted(queries):
            while (
                position < len(indices) and spans[indices[position]]["start"] <= start
            ):
                index = indices[position]
                active[index] = spans[index]["end"]
                heapq.heappush(expiry, (spans[index]["end"], index))
                position += 1
            while expiry and expiry[0][0] <= start:
                _, index = heapq.heappop(expiry)
                active.pop(index, None)
            enclosing = [index for index, stop in active.items() if end <= stop]
            # Operator spans may nest freely; only overlapping step[]
            # annotations make the CPU launch ownership ambiguous.
            if len(enclosing) == 1:
                owners[corr] = enclosing[0]

    records, skipped = kernel_intervals(payload)
    grouped = defaultdict(list)
    unmatched = 0
    for _, event, device, start, end in records:
        corr = event["args"].get("correlation")
        owner = owners.get(corr) if type(corr) is int else None
        if owner is None:
            unmatched += 1
            continue
        grouped[(owner, device)].append(
            {
                "name": event["name"],
                "start": start,
                "end": end,
                "stream": event["args"]["stream"],
                "correlation": corr,
            }
        )
    steps = []
    for (owner, device), kernels in grouped.items():
        span = spans[owner]
        steps.append(
            {
                "step_index": owner,
                "name": span["name"],
                "source_pid": span["pid"],
                "source_tid": span["tid"],
                "device": device,
                "submit_start_ns": span["start"],
                "submit_end_ns": span["end"],
                "device_start_ns": min(k["start"] for k in kernels),
                "device_end_ns": max(k["end"] for k in kernels),
                "kernel_busy_union_ns": interval_union_ns(
                    (k["start"], k["end"]) for k in kernels
                ),
                "kernels": kernels,
            }
        )
    steps.sort(key=lambda s: (s["device_end_ns"], s["step_index"], s["device"]))
    return {
        "steps": steps,
        "matched_kernels": len(records) - unmatched,
        "unmatched_kernels": unmatched,
        "skipped_kernel_records": skipped,
        "annotated_forwards": len(spans),
        "forwards_without_matched_kernels": len(spans) - len({i for i, _ in grouped}),
    }


def render_recent_steps(
    attribution: dict, *, max_bytes: int, max_steps: int = 64
) -> bytes:
    if not 1 <= max_steps <= 64 or max_bytes <= 0:
        raise ValueError("max_steps must be 1..64 and max_bytes must be positive")
    by_forward = defaultdict(list)
    for step in attribution["steps"]:
        by_forward[step["step_index"]].append(step)
    indices = sorted(
        by_forward,
        key=lambda i: (max(s["device_end_ns"] for s in by_forward[i]), i),
    )
    count_evicted = max(0, len(indices) - max_steps)
    indices = indices[-max_steps:]
    # Keep a fixed time origin while testing successively longer suffixes.
    origin = min(
        (s["device_start_ns"] for i in indices for s in by_forward[i]), default=0
    )

    def events_for(steps):
        for step in steps:
            identity = {
                "step_index": step["step_index"],
                "source_pid": step["source_pid"],
                "source_tid": step["source_tid"],
            }
            yield (
                {
                    "ph": "X",
                    "cat": "step",
                    "name": step["name"],
                    "pid": step["device"],
                    "tid": f"step-{step['step_index']}",
                    "ts": (step["device_start_ns"] - origin) / 1000,
                    "dur": (step["device_end_ns"] - step["device_start_ns"]) / 1000,
                    "args": {
                        **identity,
                        "scope": "observed_kernel_envelope",
                        "kernel_busy_union_ns": step["kernel_busy_union_ns"],
                    },
                }
            )
            for k in step["kernels"]:
                yield (
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": k["name"],
                        "pid": step["device"],
                        "tid": k["stream"],
                        "ts": (k["start"] - origin) / 1000,
                        "dur": (k["end"] - k["start"]) / 1000,
                        "args": {**identity, "launch_correlation": k["correlation"]},
                    }
                )

    def encode(value):
        return json.dumps(
            value, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()

    def trailer(retained):
        metadata = {k: v for k, v in attribution.items() if k != "steps"}
        metadata.update(
            {
                "clock_domain": "chrome_trace_relative_ns",
                "time_origin_ns": origin,
                "collection_mode": "window",
                "dropped_records": None,
                "attribution": "unique_launch_correlation_and_enclosing_cpu_forward",
                "retained_steps": retained,
                "evicted_for_count": count_evicted,
                "evicted_for_bytes": len(indices) - retained,
                "completion_scope": "kernels_observed_at_profile_flush",
            }
        )
        return b'],"metadata":' + encode(metadata) + b"}"

    prefix = b'{"displayTimeUnit":"ms","traceEvents":['
    groups = []
    size = len(prefix)
    for index in reversed(indices):
        tail = trailer(len(groups) + 1)
        group = bytearray()
        for event in events_for(by_forward[index]):
            encoded = encode(event)
            if (
                size
                + len(group)
                + bool(group)
                + len(encoded)
                + bool(groups)
                + len(tail)
                > max_bytes
            ):
                break
            if group:
                group.extend(b",")
            group.extend(encoded)
        else:
            size += len(group) + bool(groups)
            groups.append(group)
            continue
        break
    tail = trailer(len(groups))
    if size + len(tail) > max_bytes:
        raise ValueError("max_bytes cannot hold step trace metadata")
    return prefix + b",".join(reversed(groups)) + tail


def export_step_kernels(trace_path, payload, *, max_bytes: int) -> Path:
    path = Path(trace_path)
    output = path.with_name(
        path.name.removesuffix(".gz").removesuffix(".json") + ".steps.json"
    )
    data = render_recent_steps(attribute_step_kernels(payload), max_bytes=max_bytes)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(output)
    return output
