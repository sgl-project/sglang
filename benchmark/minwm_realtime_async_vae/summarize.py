#!/usr/bin/env python3
"""Summarize MinWM realtime concurrency runs and synchronous/async A/B data."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def latency_summary(values: Iterable[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        return {}
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "p50": statistics.median(samples),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "max": max(samples),
    }


def run_meets_slo(
    run: dict,
    *,
    action_p95_slo_ms: float = 1000.0,
    min_fps: float = 16.0,
    max_error_rate: float = 0.0,
) -> bool:
    e2e = run.get("action_to_first_frame_ms") or run.get("chunk_total_ms") or {}
    p95 = e2e.get("p95")
    min_session_fps = run.get("min_session_fps")
    if min_session_fps is None:
        min_session_fps = run.get("aggregate_fps") or 0.0
    return bool(
        p95 is not None
        and float(p95) < action_p95_slo_ms
        and float(min_session_fps) >= min_fps
        and float(run.get("error_rate") or 0.0) <= max_error_rate
    )


def summarize_runs(
    runs: list[dict],
    *,
    action_p95_slo_ms: float = 1000.0,
    min_fps: float = 16.0,
) -> dict:
    ordered = sorted(runs, key=lambda item: int(item["concurrency"]))
    supported = [
        run
        for run in ordered
        if run_meets_slo(
            run,
            action_p95_slo_ms=action_p95_slo_ms,
            min_fps=min_fps,
        )
    ]
    return {
        "max_supported_concurrency": (
            int(supported[-1]["concurrency"]) if supported else 0
        ),
        "action_p95_slo_ms": action_p95_slo_ms,
        "min_fps": min_fps,
        "runs": ordered,
    }


def compare_profiles(baseline: dict, asynchronous: dict) -> dict:
    baseline_runs = {
        int(run["concurrency"]): run for run in baseline.get("runs", [])
    }
    async_runs = {
        int(run["concurrency"]): run for run in asynchronous.get("runs", [])
    }
    common = sorted(set(baseline_runs) & set(async_runs))
    if not common:
        raise ValueError("baseline and async profiles have no common concurrency")
    comparison_concurrency = 1 if 1 in common else common[0]
    base = baseline_runs[comparison_concurrency]
    current = async_runs[comparison_concurrency]
    base_e2e = base.get("action_to_first_frame_ms") or base["chunk_total_ms"]
    async_e2e = current.get("action_to_first_frame_ms") or current["chunk_total_ms"]
    base_p95 = float(base_e2e["p95"])
    async_p95 = float(async_e2e["p95"])
    improvement = (base_p95 - async_p95) / base_p95 * 100.0 if base_p95 else 0.0
    return {
        "comparison_concurrency": comparison_concurrency,
        "baseline_p95_ms": base_p95,
        "async_p95_ms": async_p95,
        "async_improvement_pct": improvement,
        "baseline_fps": float(base.get("aggregate_fps") or 0.0),
        "async_fps": float(current.get("aggregate_fps") or 0.0),
    }


def build_report(baseline: dict, asynchronous: dict) -> dict:
    return {
        "schema_version": "minwm-async-vae-report/v1",
        "baseline": summarize_runs(baseline["runs"]),
        "async": summarize_runs(asynchronous["runs"]),
        "comparison": compare_profiles(baseline, asynchronous),
        "hardware": {
            "baseline": baseline.get("hardware", {}),
            "async": asynchronous.get("hardware", {}),
        },
    }


def render_markdown(report: dict) -> str:
    comparison = report["comparison"]
    lines = [
        "# MinWM 异步 VAE 端到端测试报告",
        "",
        "## 结论",
        "",
        f"- 同步基线最高稳定并发：{report['baseline']['max_supported_concurrency']}",
        f"- 异步 VAE 最高稳定并发：{report['async']['max_supported_concurrency']}",
        f"- 并发 {comparison['comparison_concurrency']} 下 P95："
        f"{comparison['baseline_p95_ms']:.1f} ms → {comparison['async_p95_ms']:.1f} ms",
        f"- 端到端 P95 改善：{comparison['async_improvement_pct']:.2f}%",
        "",
        "## 并发压测",
        "",
        "| 模式 | 并发 | P95 action→首帧 (ms) | 最低单会话 FPS | 集群 FPS | 错误率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile_name in ("baseline", "async"):
        for run in report[profile_name]["runs"]:
            e2e = run.get("action_to_first_frame_ms") or run.get("chunk_total_ms", {})
            lines.append(
                f"| {profile_name} | {run['concurrency']} | "
                f"{float(e2e.get('p95') or 0):.1f} | "
                f"{float(run.get('min_session_fps') or 0):.2f} | "
                f"{float(run.get('aggregate_fps') or 0):.2f} | "
                f"{float(run.get('error_rate') or 0) * 100:.2f}% |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--async-profile", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    asynchronous = json.loads(args.async_profile.read_text())
    report = build_report(baseline, asynchronous)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(report))


if __name__ == "__main__":
    main()
