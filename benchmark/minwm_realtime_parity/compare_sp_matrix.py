#!/usr/bin/env python3
"""Compare MinWM Ulysses SP lanes and build a local synchronized player."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from pathlib import Path

import numpy as np

from common import load_cases, write_json
from compare_results import metric_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", default=Path(__file__).with_name("cases_720p_compile_smoke.json")
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--degrees", default="1,2,4,8")
    return parser.parse_args()


def parse_degrees(value: str) -> list[int]:
    degrees = [int(item) for item in value.split(",") if item]
    if not degrees or degrees[0] != 1 or len(set(degrees)) != len(degrees):
        raise ValueError("--degrees must be unique and start with 1")
    return degrees


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def format_number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def memory_summary(path: Path, degree: int) -> dict:
    per_gpu: dict[int, list[float]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            match = re.search(r",\s*(\d+),\s*([0-9.]+)\s*$", line)
            if match:
                per_gpu.setdefault(int(match.group(1)), []).append(
                    float(match.group(2))
                )
    peaks = {
        str(index): max(values)
        for index, values in sorted(per_gpu.items())
        if index < degree and values
    }
    return {
        "peak_per_gpu_mb": peaks,
        "peak_max_gpu_mb": max(peaks.values()) if peaks else None,
        "peak_sum_gpu_mb": sum(peaks.values()) if peaks else None,
    }


def lane_performance(run: dict, memory_path: Path, degree: int) -> dict:
    scheduler_ms: list[float] = []
    client_ms: list[float] = []
    steady_frames = 0
    ttff_ms: list[float] = []
    for case in run["cases"]:
        steady_stats = [
            item for item in case["chunk_stats"] if int(item["chunk_index"]) > 0
        ]
        scheduler_ms.extend(
            float(item["scheduler_forward_ms"]) for item in steady_stats
        )
        steady_frames += sum(int(item["num_frames"]) for item in steady_stats)
        client_ms.extend(
            float(value)
            for value in case["client_timing"]["steady_payload_interarrival_ms"]
        )
        ttff_ms.append(
            float(case["client_timing"]["init_send_start_to_first_payload_complete_ms"])
        )
    return {
        "steady_frames": steady_frames,
        "steady_chunks": len(scheduler_ms),
        "scheduler_fps": (
            steady_frames / (sum(scheduler_ms) / 1000) if scheduler_ms else None
        ),
        "client_fps": steady_frames / (sum(client_ms) / 1000) if client_ms else None,
        "scheduler_chunk_p50_ms": (
            statistics.median(scheduler_ms) if scheduler_ms else None
        ),
        "scheduler_chunk_p95_ms": percentile(scheduler_ms, 0.95),
        "client_chunk_p50_ms": statistics.median(client_ms) if client_ms else None,
        "client_chunk_p95_ms": percentile(client_ms, 0.95),
        "ttff_ms": statistics.median(ttff_ms) if ttff_ms else None,
        **memory_summary(memory_path, degree),
    }


def write_player(results: Path, report: dict) -> None:
    template = (
        Path(__file__).with_name("sp_matrix_player.html").read_text(encoding="utf-8")
    )
    placeholder = "__MINWM_SP_MATRIX_REPORT__"
    if template.count(placeholder) != 1:
        raise ValueError("SP player template must contain exactly one placeholder")
    embedded = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    player_dir = results / "player"
    player_dir.mkdir(exist_ok=True)
    (player_dir / "index.html").write_text(
        template.replace(placeholder, embedded), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    degrees = parse_degrees(args.degrees)
    manifest = load_cases(args.cases)
    results = Path(args.results).resolve()
    runtime_path = results.parent / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    gpu_name = runtime.get("gpu", "NVIDIA GPU")

    lane_runs = {}
    lanes = []
    for degree in degrees:
        prefix = f"sp{degree}"
        run = json.loads((results / f"{prefix}_run.json").read_text())
        lane_runs[degree] = run
        lanes.append(
            {
                "degree": degree,
                "prefix": prefix,
                "performance": lane_performance(
                    run, results / f"{prefix}-gpu-memory.csv", degree
                ),
            }
        )

    reference_lane = next(item for item in lanes if item["degree"] == 1)
    reference_scheduler_fps = reference_lane["performance"]["scheduler_fps"]
    reference_client_fps = reference_lane["performance"]["client_fps"]
    for lane in lanes:
        performance = lane["performance"]
        performance["scheduler_speedup_vs_sp1"] = (
            performance["scheduler_fps"] / reference_scheduler_fps
            if reference_scheduler_fps and performance["scheduler_fps"]
            else None
        )
        performance["client_speedup_vs_sp1"] = (
            performance["client_fps"] / reference_client_fps
            if reference_client_fps and performance["client_fps"]
            else None
        )

    case_reports = []
    all_bitwise = True
    for case in manifest["cases"]:
        case_dir = results / "cases" / case["id"]
        reference = np.load(case_dir / "sp1.npy", allow_pickle=False)
        comparisons = {}
        for degree in degrees[1:]:
            candidate = np.load(case_dir / f"sp{degree}.npy", allow_pickle=False)
            metrics = metric_block(reference[1:], candidate[1:])
            comparisons[f"sp{degree}"] = metrics
            all_bitwise = all_bitwise and metrics["bitwise_equal"]
        case_reports.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "action_label": case["action_label"],
                "keys": case["keys"],
                "comparisons": comparisons,
            }
        )

    report = {
        "schema_version": 1,
        "title": "MinWM 720p causal Ulysses SP matrix",
        "contract": {
            **manifest["contract"],
            "hardware": f"single-node {max(degrees)}x {gpu_name} Spot",
            "location": os.environ.get("MINWM_SP_LOCATION"),
            "attention": "packed deterministic causal attention",
            "reference_lane": "sp1",
            "parity_scope": "generated uint8 frames",
        },
        "summary": {
            "degrees": degrees,
            "case_count": len(case_reports),
            "all_generated_bitwise": all_bitwise,
        },
        "lanes": lanes,
        "cases": case_reports,
    }
    write_json(results / "sp_matrix_report.json", report)

    markdown = [
        "# MinWM 720p causal Ulysses SP matrix",
        "",
        f"Generated-frame parity: **{'bitwise exact' if all_bitwise else 'drift detected'}**",
        "",
        "| SP | scheduler FPS | client FPS | scheduler speedup | client speedup | peak/GPU MiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in lanes:
        perf = lane["performance"]
        markdown.append(
            f"| {lane['degree']} | {format_number(perf['scheduler_fps'])} | "
            f"{format_number(perf['client_fps'])} | "
            f"{format_number(perf['scheduler_speedup_vs_sp1'])}× | "
            f"{format_number(perf['client_speedup_vs_sp1'])}× | "
            f"{format_number(perf['peak_max_gpu_mb'], 0)} |"
        )
    (results / "sp_matrix_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    write_player(results, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if not all_bitwise:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
