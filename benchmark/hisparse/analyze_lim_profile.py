"""Analyze raw CANN trace_view.json without treating cumulative time as wall time.

Graph windows use repeated (model, physical stream, task id) anchors and are
accepted only when each contains exactly 78 SFA calls. They include the captured
model graph's other work (for example the LM head); they are not a measured
78-layer-only wall time. Copy direction is source-order attribution for the
current BF16 LIM adapter, accepted only for complete five-copy/SFA groups.
"""

import argparse
import collections
import hashlib
import itertools
import json
import statistics
from decimal import Decimal
from pathlib import Path

TASK_TYPES = {"AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "MIX_AIC", "MIX_AIV"}
COPY_MEANINGS = (
    "new-token D2H",
    "miss DRAM-to-resident H2D",
    "resident-to-selected D2D",
    "short-row H2D (masked on long rows)",
    "visible-tail H2D",
)


def interval_union(intervals):
    """Length of a union, including nested, touching and overlapping spans."""
    result = 0.0
    end = None
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError("Negative interval")
        if end is None or left > end:
            result += right - left
        else:
            result += max(0.0, right - end)
        end = right if end is None else max(end, right)
    return result


def overlap_length(left, right):
    """Intersection of two interval unions; duplicate tasks cannot inflate it."""
    a = interval_union(left)
    b = interval_union(right)
    return max(0.0, a + b - interval_union(left + right))


def normalize(name):
    return name.lower().replace("_", "")


def is_sfa(task):
    return "sparseflashattention" in normalize(task["name"])


def family(task):
    name = normalize(task["name"])
    if "lightningindexer" in name:
        return "LIM" if "manage" in name else "LI"
    if "unidexcopy" in name:
        return "all_unidex_copies"
    if "sparseflashattention" in name:
        return "SFA"
    if "hccl" in name or (
        task["type"] == "AI_CPU"
        and any(
            pattern in name
            for pattern in (
                "allgatheraicpukernel",
                "reducescatteraicpukernel",
                "allreduceaicpukernel",
            )
        )
    ):
        return "HCCL_task_spans_including_waits"
    return "other_tasks"


def spans(tasks):
    return [(r["start"], r["end"]) for r in tasks]


def component_summary(tasks):
    groups = collections.defaultdict(list)
    for task in tasks:
        groups[family(task)].append(task)
    return {
        key: {
            "calls": len(rows),
            "cumulative_ms": sum(r["dur"] for r in rows) / 1000,
            "interval_union_ms": interval_union(spans(rows)) / 1000,
        }
        for key, rows in groups.items()
    }


def graph_windows(tasks, expected_layers):
    models = collections.defaultdict(list)
    for task in tasks:
        if task["model"] not in (None, 4294967295, "4294967295"):
            models[task["model"]].append(task)
    reports = []
    for model, rows in models.items():
        # AIC/AIV graph task ids repeat at replay. AI_CPU communication has
        # separate stream numbering and may contain long waits.
        core = [r for r in rows if r["type"] != "AI_CPU"]
        by_stream = collections.Counter(r["stream"] for r in core)
        if not by_stream:
            continue
        stream = by_stream.most_common(1)[0][0]
        selected = [
            r for r in core if r["stream"] == stream and r["task_id"] is not None
        ]
        if not selected:
            continue
        first_id = min(r["task_id"] for r in selected)
        anchors = [r for r in selected if r["task_id"] == first_id]
        windows = []
        for index, anchor in enumerate(anchors):
            boundary = (
                anchors[index + 1]["start"]
                if index + 1 < len(anchors)
                else float("inf")
            )
            chunk = [r for r in rows if anchor["start"] <= r["start"] < boundary]
            sfa_count = sum(is_sfa(r) for r in chunk)
            if sfa_count != expected_layers:
                windows.append({"accepted": False, "sfa_calls": sfa_count})
                continue
            finish = max(r["end"] for r in chunk)
            selected = [
                r
                for r in chunk
                if family(r) in {"LI", "LIM", "all_unidex_copies", "SFA"}
            ]
            selected_union = interval_union(spans(selected))
            windows.append(
                {
                    "accepted": True,
                    "start_relative_us": anchor["start"],
                    "exclusive_next_anchor_relative_us": (
                        None if boundary == float("inf") else boundary
                    ),
                    "end_relative_us": finish,
                    "device_graph_span_ms": (finish - anchor["start"]) / 1000,
                    "next_anchor_gap_ms": (
                        None
                        if boundary == float("inf")
                        else (boundary - anchor["start"]) / 1000
                    ),
                    "task_interval_union_ms": interval_union(spans(chunk)) / 1000,
                    "selected_components_interval_union_ms": selected_union / 1000,
                    "wall_outside_selected_components_ms": (
                        finish - anchor["start"] - selected_union
                    )
                    / 1000,
                    "sfa_calls": sfa_count,
                    "components": component_summary(chunk),
                }
            )
        reports.append(
            {
                "model_id": model,
                "anchor_stream": stream,
                "anchor_task_id": first_id,
                "windows": windows,
            }
        )
    return reports


def analyze(path, expected_layers=78):
    raw = path.read_bytes()
    payload = json.loads(raw)
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    candidates = [
        r
        for r in events
        if r.get("ph") == "X"
        and r.get("args", {}).get("Task Type") in TASK_TYPES
        and "dur" in r
    ]
    if not candidates:
        raise ValueError("No explicit CANN task events")
    origin = min(Decimal(str(r["ts"])) for r in candidates)
    tasks = []
    for row in candidates:
        info = row["args"]
        start = float(Decimal(str(row["ts"])) - origin)
        duration = float(row["dur"])
        tasks.append(
            {
                "name": row["name"],
                "start": start,
                "end": start + duration,
                "dur": duration,
                "stream": info.get("Physic Stream Id"),
                "type": info["Task Type"],
                "model": info.get("Model Id"),
                "task_id": info.get("Task Id"),
            }
        )
    tasks.sort(key=lambda r: r["start"])
    all_tasks = tasks
    graph = graph_windows(all_tasks, expected_layers)
    accepted = [w for model in graph for w in model["windows"] if w["accepted"]]
    if accepted:
        ranges = [
            (
                model["model_id"],
                window["start_relative_us"],
                window["exclusive_next_anchor_relative_us"],
            )
            for model in graph
            for window in model["windows"]
            if window["accepted"]
        ]
        # Asynchronous start/stop can record the tail of a preceding replay.
        # Keep complete model replays for every kernel/copy total, not just
        # for graph wall spans. Retain untrimmed counts for auditability.
        tasks = [
            task
            for task in all_tasks
            if any(
                task["model"] == model
                and task["start"] >= start
                and (end is None or task["start"] < end)
                for model, start, end in ranges
            )
        ]
    total = sum(r["dur"] for r in tasks)
    by_name = collections.defaultdict(list)
    for row in tasks:
        by_name[row["name"]].append(row)
    kernel = []
    for name, rows in by_name.items():
        duration = sum(r["dur"] for r in rows)
        share = 100 * duration / total
        if share >= 1:
            kernel.append(
                {
                    "name": name,
                    "calls": len(rows),
                    "cumulative_ms": duration / 1000,
                    "cumulative_share_percent": share,
                    "interval_union_ms": interval_union(spans(rows)) / 1000,
                    "streams": sorted({r["stream"] for r in rows}, key=str),
                }
            )
    kernel.sort(key=lambda r: r["cumulative_ms"], reverse=True)
    pending, groups, ambiguous = [], [], []
    for row in tasks:
        if family(row) == "all_unidex_copies":
            pending.append(row)
        elif is_sfa(row):
            if pending:
                if (
                    len(pending) == 5
                    and all(
                        a["end"] <= b["start"] for a, b in itertools.pairwise(pending)
                    )
                    and pending[-1]["end"] <= row["start"]
                ):
                    groups.append((pending, row))
                else:
                    ambiguous.append(len(pending))
            pending = []
    positions = []
    for index, meaning in enumerate(COPY_MEANINGS):
        values = [g[index]["dur"] for g, _ in groups]
        positions.append(
            {
                "position": index,
                "source_order_attribution": meaning,
                "calls": len(values),
                "median_us": statistics.median(values) if values else None,
                "cumulative_ms": sum(values) / 1000,
            }
        )
    copies = [r for r in tasks if family(r) == "all_unidex_copies"]
    sfas = [r for r in tasks if is_sfa(r)]
    return {
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trace": str(path),
        "trace_bytes": len(raw),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "timestamp_origin_us": str(origin),
        "total_cumulative_task_ms": total / 1000,
        "reported_scope": (
            "complete_model_graph_replays"
            if accepted
            else "all_tasks_no_complete_graph_replay"
        ),
        "reported_tasks_interval_union_ms": interval_union(spans(tasks)) / 1000,
        "raw_total_cumulative_task_ms": sum(r["dur"] for r in all_tasks) / 1000,
        "all_tasks_interval_union_ms": interval_union(spans(all_tasks)) / 1000,
        "raw_components": component_summary(all_tasks),
        "excluded_partial_or_outside_graph_task_count": len(all_tasks) - len(tasks),
        "kernel_table": kernel,
        "components": component_summary(tasks),
        "graph_replays": graph,
        "accepted_graph_replays": len(accepted),
        "median_device_graph_span_ms": (
            statistics.median(w["device_graph_span_ms"] for w in accepted)
            if accepted
            else None
        ),
        "cpu_step_annotation_counts": dict(
            collections.Counter(
                r["name"] for r in events if r.get("name", "").startswith("step[")
            )
        ),
        "complete_serial_five_copy_groups": len(groups),
        "ambiguous_copy_group_sizes": dict(collections.Counter(ambiguous)),
        "trailing_unassigned_copies": len(pending),
        "copy_positions": positions,
        "all_copy_streams": sorted({r["stream"] for r in copies}, key=str),
        "sfa_streams": sorted({r["stream"] for r in sfas}, key=str),
        "all_copy_vs_any_sfa_interval_intersection_ms": overlap_length(
            spans(copies), spans(sfas)
        )
        / 1000,
        "limits": [
            "Cumulative task time is not critical-path wall time; HCCL spans may contain waits.",
            "Accepted graph windows require exactly 78 SFA calls; other model configurations need --layers.",
            "When complete graph replays exist, kernel/copy totals exclude partial replays and out-of-graph tasks; raw totals are retained separately.",
            "Device graph span includes captured work outside the transformer layers; it is not end-to-end ITL.",
            "Wall outside selected components is an interval complement, not causal attribution to communication or framework overhead.",
            "Copy direction is inferred from the audited current adapter source, not encoded by CANN.",
            "Hit rate, copied bytes, CopySFA hiding percentage and unprofiled throughput cannot be inferred from task durations.",
            "No C8, MTP or SharedCopy result is implied by a BF16 trace.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=78)
    args = parser.parse_args()
    report = analyze(args.trace, args.layers)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "accepted_graph_replays",
                    "median_device_graph_span_ms",
                    "complete_serial_five_copy_groups",
                    "ambiguous_copy_group_sizes",
                    "cpu_step_annotation_counts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
