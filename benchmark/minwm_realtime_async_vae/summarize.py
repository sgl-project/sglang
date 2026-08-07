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
    by_concurrency = []
    for concurrency in common:
        baseline_run = baseline_runs[concurrency]
        async_run = async_runs[concurrency]
        baseline_action = float(
            (baseline_run.get("action_to_first_frame_ms") or {})["p95"]
        )
        async_action = float(
            (async_run.get("action_to_first_frame_ms") or {})["p95"]
        )
        baseline_chunk = float(baseline_run["chunk_total_ms"]["p95"])
        async_chunk = float(async_run["chunk_total_ms"]["p95"])
        baseline_throughput = float(baseline_run.get("aggregate_fps") or 0.0)
        async_throughput = float(async_run.get("aggregate_fps") or 0.0)
        by_concurrency.append(
            {
                "concurrency": concurrency,
                "baseline_action_p95_ms": baseline_action,
                "async_action_p95_ms": async_action,
                "action_improvement_pct": _improvement_pct(
                    baseline_action, async_action
                ),
                "baseline_chunk_p95_ms": baseline_chunk,
                "async_chunk_p95_ms": async_chunk,
                "chunk_improvement_pct": _improvement_pct(
                    baseline_chunk, async_chunk
                ),
                "baseline_fps": baseline_throughput,
                "async_fps": async_throughput,
                "throughput_improvement_pct": _increase_pct(
                    baseline_throughput, async_throughput
                ),
            }
        )
    return {
        "comparison_concurrency": comparison_concurrency,
        "baseline_p95_ms": base_p95,
        "async_p95_ms": async_p95,
        "async_improvement_pct": improvement,
        "baseline_fps": float(base.get("aggregate_fps") or 0.0),
        "async_fps": float(current.get("aggregate_fps") or 0.0),
        "by_concurrency": by_concurrency,
    }


def _improvement_pct(baseline: float, current: float) -> float:
    return (baseline - current) / baseline * 100.0 if baseline else 0.0


def _increase_pct(baseline: float, current: float) -> float:
    return (current - baseline) / baseline * 100.0 if baseline else 0.0


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
    baseline_hardware = report["hardware"]["baseline"]
    async_hardware = report["hardware"]["async"]
    async_first = report["async"]["runs"][0]
    stage_ms = async_first.get("stage_ms", {})

    def p95(name: str) -> float:
        return float((stage_ms.get(name) or {}).get("p95") or 0.0)

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
        "## 硬件与部署",
        "",
        "| 模式 | Denoiser | VAE | 实例/容量 | GPU 使用数 |",
        "|---|---|---|---|---:|",
        _hardware_row("baseline", baseline_hardware),
        _hardware_row("async", async_hardware),
        "",
        "## 并发压测",
        "",
        "| 模式 | 并发 | P95 action→首帧 (ms) | P95 chunk (ms) | "
        "最低单会话 FPS | 集群 FPS | 错误率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile_name in ("baseline", "async"):
        for run in report[profile_name]["runs"]:
            e2e = run.get("action_to_first_frame_ms") or run.get("chunk_total_ms", {})
            lines.append(
                f"| {profile_name} | {run['concurrency']} | "
                f"{float(e2e.get('p95') or 0):.1f} | "
                f"{float((run.get('chunk_total_ms') or {}).get('p95') or 0):.1f} | "
                f"{float(run.get('min_session_fps') or 0):.2f} | "
                f"{float(run.get('aggregate_fps') or 0):.2f} | "
                f"{float(run.get('error_rate') or 0) * 100:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## 异步收益",
            "",
            "| 并发 | action P95 降低 | chunk P95 降低 | 集群吞吐提升 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["by_concurrency"]:
        lines.append(
            f"| {row['concurrency']} | {row['action_improvement_pct']:.2f}% | "
            f"{row['chunk_improvement_pct']:.2f}% | "
            f"{row['throughput_improvement_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 单用户关键阶段 P95",
            "",
            f"- Denoising：{p95('denoise_ms'):.1f} ms",
            f"- 远端 TAEHV decode：{p95('vae_decode_ms'):.1f} ms",
            f"- WebP encode：{p95('frame_encode_ms'):.1f} ms",
            f"- Latent send：{p95('latent_send_ms'):.3f} ms",
            f"- VAE queue wait：{p95('vae_queue_wait_ms'):.3f} ms",
            f"- 与下一 chunk denoising overlap：{p95('overlap_with_next_denoise_ms'):.1f} ms",
        ]
    )
    return "\n".join(lines) + "\n"


def _hardware_row(profile: str, hardware: dict) -> str:
    denoiser = hardware.get("denoiser", {})
    vae = hardware.get("vae", {})
    instance = hardware.get("instance_type", "-")
    capacity = hardware.get("capacity_type", "-")
    gpu_count = int(denoiser.get("gpu_count") or 0) + int(
        vae.get("gpu_count") or 0
    )
    return (
        f"| {profile} | {denoiser.get('gpu_type', '-')} | "
        f"{vae.get('backend', '-')} / {vae.get('placement', vae.get('gpu_type', '-'))} | "
        f"{instance} / {capacity} | {gpu_count} |"
    )


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
