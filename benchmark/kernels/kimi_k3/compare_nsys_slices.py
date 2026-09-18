#!/usr/bin/env python3
"""Crop and align matching CUDA-kernel slices from two Nsight SQLite exports.

The output uses the Chrome/Perfetto trace-event format. Independent captures
are placed in separate process tracks and normalized to the selected marker's
start, allowing their GPU timelines to be viewed at the same horizontal scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Kernel:
    start_ns: int
    end_ns: int
    stream_id: int
    short_name: str
    full_name: str


@dataclass(frozen=True)
class Slice:
    step: int
    start_ns: int
    end_ns: int
    marker_starts_ns: tuple[int, ...]
    kernels: tuple[Kernel, ...]

    @property
    def wall_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1000.0

    @property
    def summed_us(self) -> float:
        return sum(k.end_ns - k.start_ns for k in self.kernels) / 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marker", default="sm100_fp8_fp4_mega_moe_impl")
    parser.add_argument("--markers-per-step", type=int, default=92)
    parser.add_argument("--start-marker", type=int, default=83)
    parser.add_argument("--end-marker", type=int, default=87)
    return parser.parse_args()


def load_slices(
    path: Path,
    marker_name: str,
    markers_per_step: int,
    start_marker: int,
    end_marker: int,
) -> list[Slice]:
    connection = sqlite3.connect(path)
    marker_starts = [
        row[0]
        for row in connection.execute(
            """
            SELECT k.start
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.shortName
            WHERE k.deviceId = 0 AND s.value = ?
            ORDER BY k.start
            """,
            (marker_name,),
        )
    ]
    if not marker_starts or len(marker_starts) % markers_per_step:
        raise RuntimeError(
            f"{path}: found {len(marker_starts)} markers; expected a positive "
            f"multiple of {markers_per_step}"
        )
    if not 0 <= start_marker < end_marker < markers_per_step:
        raise ValueError("marker indices must satisfy 0 <= start < end < markers/step")

    slices = []
    for step in range(len(marker_starts) // markers_per_step):
        base = step * markers_per_step
        selected_markers = tuple(
            marker_starts[base + index] for index in range(start_marker, end_marker + 1)
        )
        start_ns, end_ns = selected_markers[0], selected_markers[-1]
        kernels = tuple(
            Kernel(*row)
            for row in connection.execute(
                """
                SELECT k.start, k.end, k.streamId, short.value, full.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                JOIN StringIds AS short ON short.id = k.shortName
                JOIN StringIds AS full ON full.id = k.demangledName
                WHERE k.deviceId = 0 AND k.start >= ? AND k.start < ?
                ORDER BY k.start
                """,
                (start_ns, end_ns),
            )
        )
        slices.append(
            Slice(
                step=step,
                start_ns=start_ns,
                end_ns=end_ns,
                marker_starts_ns=selected_markers,
                kernels=kernels,
            )
        )
    connection.close()
    return slices


def representative(slices: list[Slice]) -> Slice:
    median = statistics.median(item.wall_us for item in slices)
    return min(slices, key=lambda item: (abs(item.wall_us - median), item.step))


def category(name: str) -> str:
    if name.startswith("nvjet") or "splitKreduce" in name:
        return "projection"
    if "mega_moe" in name or "fused_front_epilogue" in name:
        return "moe"
    if "recurrent_kda" in name or "causal_conv1d" in name:
        return "kda"
    if "cutlass_split_kv" in name or "mla" in name:
        return "mla"
    if "attn_res" in name:
        return "residual"
    return "other"


def assign_overlap_lanes(kernels: tuple[Kernel, ...]) -> list[tuple[Kernel, int]]:
    """Color overlapping intervals so each emitted Perfetto track is linear."""
    assignments = []
    lane_ends: dict[int, list[int]] = {}
    for kernel in kernels:
        ends = lane_ends.setdefault(kernel.stream_id, [])
        lane = next(
            (index for index, end_ns in enumerate(ends) if end_ns <= kernel.start_ns),
            len(ends),
        )
        if lane == len(ends):
            ends.append(kernel.end_ns)
        else:
            ends[lane] = kernel.end_ns
        assignments.append((kernel, lane))
    return assignments


def trace_events(label: str, item: Slice, pid: int, start_marker: int) -> list[dict]:
    events: list[dict] = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": pid,
            "tid": 0,
            "args": {"name": f"{label} (capture step {item.step})"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": pid,
            "tid": 0,
            "args": {"name": "Layer intervals"},
        },
    ]

    assignments = assign_overlap_lanes(item.kernels)
    lanes = sorted({(kernel.stream_id, lane) for kernel, lane in assignments})
    lane_tids = {lane: index + 1 for index, lane in enumerate(lanes)}
    for stream_id, lane in lanes:
        suffix = "" if lane == 0 else f" — overlap lane {lane}"
        events.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": pid,
                "tid": lane_tids[(stream_id, lane)],
                "args": {"name": f"CUDA stream {stream_id}{suffix}"},
            }
        )

    for offset, (begin, end) in enumerate(
        zip(item.marker_starts_ns, item.marker_starts_ns[1:])
    ):
        begin_us = (begin - item.start_ns) / 1000.0
        end_us = (end - item.start_ns) / 1000.0
        # Leave a 1 ns visual gap. Otherwise floating-point addition of one
        # interval's ts+dur can round just beyond the next interval's ts and
        # trigger Perfetto's overlapping-complete-event importer warning.
        display_duration_us = max(0.0, end_us - begin_us - 0.001)
        events.append(
            {
                "ph": "X",
                "name": f"marker {start_marker + offset} → {start_marker + offset + 1}",
                "cat": "layer interval",
                "pid": pid,
                "tid": 0,
                "ts": begin_us,
                "dur": display_duration_us,
                "args": {"wall_us": (end - begin) / 1000.0},
            }
        )

    for kernel, lane in assignments:
        events.append(
            {
                "ph": "X",
                "name": kernel.short_name,
                "cat": category(kernel.short_name),
                "pid": pid,
                "tid": lane_tids[(kernel.stream_id, lane)],
                "ts": (kernel.start_ns - item.start_ns) / 1000.0,
                "dur": (kernel.end_ns - kernel.start_ns) / 1000.0,
                "args": {
                    "full_name": kernel.full_name,
                    "cuda_stream": kernel.stream_id,
                    "overlap_lane": lane,
                    "absolute_start_ns": kernel.start_ns,
                    "duration_us": (kernel.end_ns - kernel.start_ns) / 1000.0,
                },
            }
        )
    return events


def write_trace(
    path: Path,
    entries: list[tuple[str, Slice]],
    start_marker: int,
    end_marker: int,
) -> None:
    events = []
    for pid, (label, item) in enumerate(entries, start=1):
        events.extend(trace_events(label, item, pid, start_marker))
    document = {
        "displayTimeUnit": "us",
        "traceEvents": events,
        "otherData": {
            "selection": f"markers {start_marker} through {end_marker}",
            "normalization": "each process track starts at its selected marker",
        },
    }
    path.write_text(json.dumps(document, separators=(",", ":")) + "\n")


def write_samples(path: Path, variants: list[tuple[str, list[Slice]]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["variant", "step", "wall_us", "summed_kernel_us", "kernels"])
        for label, slices in variants:
            for item in slices:
                writer.writerow(
                    [label, item.step, item.wall_us, item.summed_us, len(item.kernels)]
                )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_slices(
        args.baseline,
        args.marker,
        args.markers_per_step,
        args.start_marker,
        args.end_marker,
    )
    candidate = load_slices(
        args.candidate,
        args.marker,
        args.markers_per_step,
        args.start_marker,
        args.end_marker,
    )
    baseline_rep = representative(baseline)
    candidate_rep = representative(candidate)

    write_trace(
        args.output_dir / "four-layer-baseline.perfetto.json",
        [("Baseline", baseline_rep)],
        args.start_marker,
        args.end_marker,
    )
    write_trace(
        args.output_dir / "four-layer-fused.perfetto.json",
        [("Fused", candidate_rep)],
        args.start_marker,
        args.end_marker,
    )
    write_trace(
        args.output_dir / "four-layer-comparison.perfetto.json",
        [("Baseline", baseline_rep), ("Fused", candidate_rep)],
        args.start_marker,
        args.end_marker,
    )
    write_trace(
        args.output_dir / "four-layer-all-steps.perfetto.json",
        [(f"Baseline step {item.step}", item) for item in baseline]
        + [(f"Fused step {item.step}", item) for item in candidate],
        args.start_marker,
        args.end_marker,
    )
    write_samples(
        args.output_dir / "four-layer-samples.csv",
        [("baseline", baseline), ("fused", candidate)],
    )

    for label, slices, selected in (
        ("baseline", baseline, baseline_rep),
        ("fused", candidate, candidate_rep),
    ):
        print(
            f"{label}: median={statistics.median(x.wall_us for x in slices):.3f} us; "
            f"representative step={selected.step}, wall={selected.wall_us:.3f} us, "
            f"kernels={len(selected.kernels)}"
        )
    print(f"Wrote comparison files to {args.output_dir}")


if __name__ == "__main__":
    main()
