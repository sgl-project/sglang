#!/usr/bin/env python3
"""Compare baseline, bitwise SGLang, and optimized SGLang artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_cases, sha256_file, write_json
from compare_results import metric_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--results", required=True)
    parser.add_argument("--bitwise-prefix", default="sglang_bitwise")
    parser.add_argument("--optimized-prefix", default="sglang_optimized")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def index_cases(run: dict) -> dict[str, dict]:
    return {item["id"]: item for item in run["cases"]}


def baseline_performance(record: dict) -> dict:
    elapsed_s = float(record["elapsed_s"])
    frames = int(record["frames"])
    return {
        "elapsed_s": elapsed_s,
        "output_frames": frames,
        "whole_clip_output_fps": frames / elapsed_s,
        "warmup_runs": int(record.get("warmup_runs", 0)),
        "note": "Includes one reference frame, VAE encode/decode, and the full clip.",
    }


def sglang_performance(record: dict) -> dict:
    chunk_stats = sorted(record["chunk_stats"], key=lambda item: item["chunk_index"])
    steady_stats = [item for item in chunk_stats if int(item["chunk_index"]) > 0]
    steady_frames = sum(int(item["num_frames"]) for item in steady_stats)
    steady_scheduler_ms = sum(
        float(item["scheduler_forward_ms"]) for item in steady_stats
    )
    interarrival_ms = [
        float(value)
        for value in record["client_timing"]["steady_payload_interarrival_ms"]
    ]
    total_client_ms = float(
        record["client_timing"]["init_send_start_to_first_payload_complete_ms"]
    ) + sum(interarrival_ms)
    total_frames = sum(int(item["num_frames"]) for item in chunk_stats)
    return {
        "ttff_ms": float(
            record["client_timing"]["init_send_start_to_first_payload_complete_ms"]
        ),
        "steady_frames": steady_frames,
        "steady_chunks": len(steady_stats),
        "steady_scheduler_fps": (
            1000.0 * steady_frames / steady_scheduler_ms
            if steady_scheduler_ms
            else None
        ),
        "steady_client_fps": (
            1000.0 * steady_frames / sum(interarrival_ms)
            if interarrival_ms and sum(interarrival_ms)
            else None
        ),
        "whole_clip_client_fps": (
            1000.0 * total_frames / total_client_ms if total_client_ms else None
        ),
        "warmup_runs": int(record.get("warmup_runs", 0)),
        "note": "Steady metrics exclude chunk 0; whole-clip metric includes TTFF.",
    }


def compare_pair(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    baseline_path: Path,
    candidate_path: Path,
) -> dict:
    return {
        "all_frames": metric_block(baseline, candidate),
        "reference_frame": metric_block(baseline[:1], candidate[:1]),
        "generated_frames": metric_block(baseline[1:], candidate[1:]),
        "baseline_frames_sha256": sha256_file(baseline_path),
        "candidate_frames_sha256": sha256_file(candidate_path),
    }


def write_player(results: Path, report: dict) -> None:
    template = (
        Path(__file__).with_name("triptych_player.html").read_text(encoding="utf-8")
    )
    placeholder = "__MINWM_EMBEDDED_TRIPTYCH_REPORT__"
    if template.count(placeholder) != 1:
        raise ValueError("triptych template must contain exactly one placeholder")
    embedded = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    player_dir = results / "player"
    player_dir.mkdir(exist_ok=True)
    (player_dir / "index.html").write_text(
        template.replace(placeholder, embedded), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    manifest = load_cases(args.cases)
    results = Path(args.results).resolve()
    baseline_run = load_json(results / "baseline_run.json")
    bitwise_run = load_json(results / f"{args.bitwise_prefix}_run.json")
    optimized_run = load_json(results / f"{args.optimized_prefix}_run.json")
    baseline_cases = index_cases(baseline_run)
    bitwise_cases = index_cases(bitwise_run)
    optimized_cases = index_cases(optimized_run)

    records = []
    for case in manifest["cases"]:
        case_id = case["id"]
        case_dir = results / "cases" / case_id
        baseline_path = case_dir / "baseline.npy"
        bitwise_path = case_dir / f"{args.bitwise_prefix}.npy"
        optimized_path = case_dir / f"{args.optimized_prefix}.npy"
        baseline = np.load(baseline_path, allow_pickle=False, mmap_mode="r")
        bitwise = np.load(bitwise_path, allow_pickle=False, mmap_mode="r")
        optimized = np.load(optimized_path, allow_pickle=False, mmap_mode="r")
        bitwise_metrics = compare_pair(
            baseline,
            bitwise,
            baseline_path=baseline_path,
            candidate_path=bitwise_path,
        )
        optimized_metrics = compare_pair(
            baseline,
            optimized,
            baseline_path=baseline_path,
            candidate_path=optimized_path,
        )
        performance = {
            "baseline": baseline_performance(baseline_cases[case_id]),
            "bitwise": sglang_performance(bitwise_cases[case_id]),
            "optimized": sglang_performance(optimized_cases[case_id]),
        }
        bitwise_fps = performance["bitwise"]["steady_client_fps"]
        optimized_fps = performance["optimized"]["steady_client_fps"]
        performance["optimized_speedup_over_bitwise"] = (
            optimized_fps / bitwise_fps
            if bitwise_fps is not None
            and optimized_fps is not None
            and bitwise_fps != 0
            else None
        )
        records.append(
            {
                "id": case_id,
                "prompt": case["prompt"],
                "action_label": case["action_label"],
                "action_weights": case.get("action_weights"),
                "keys": case["keys"],
                "videos": {
                    "baseline": f"../cases/{case_id}/baseline.mp4",
                    "bitwise": (f"../cases/{case_id}/{args.bitwise_prefix}.mp4"),
                    "optimized": (f"../cases/{case_id}/{args.optimized_prefix}.mp4"),
                },
                "bitwise_comparison": bitwise_metrics,
                "optimized_comparison": optimized_metrics,
                "performance": performance,
            }
        )

    bitwise_exact_count = sum(
        item["bitwise_comparison"]["generated_frames"]["bitwise_equal"]
        for item in records
    )
    optimized_exact_count = sum(
        item["optimized_comparison"]["generated_frames"]["bitwise_equal"]
        for item in records
    )
    speedups = [
        item["performance"]["optimized_speedup_over_bitwise"]
        for item in records
        if item["performance"]["optimized_speedup_over_bitwise"] is not None
    ]
    report = {
        "schema_version": 1,
        "contract": manifest["contract"],
        "profiles": {
            "baseline": {
                "engine": baseline_run["engine"],
                "description": "minWM main/V3",
            },
            "bitwise": {
                "engine": bitwise_run["engine"],
                "description": "SGLang parity path: packed deterministic attention",
            },
            "optimized": {
                "engine": optimized_run["engine"],
                "description": (
                    "SGLang speed path: dense attention, SGLang components, "
                    "whole-DiT torch.compile"
                ),
            },
        },
        "summary": {
            "case_count": len(records),
            "bitwise_exact_count": bitwise_exact_count,
            "optimized_exact_count": optimized_exact_count,
            "mean_optimized_speedup_over_bitwise": (
                sum(speedups) / len(speedups) if speedups else None
            ),
        },
        "cases": records,
    }
    write_json(results / "manifest.resolved.json", manifest)
    write_json(results / "triptych_report.json", report)
    markdown = [
        "# MinWM baseline / bitwise / optimized report",
        "",
        (
            f"Bitwise path exact: **{bitwise_exact_count}/{len(records)}**; "
            f"optimized path exact: **{optimized_exact_count}/{len(records)}**."
        ),
        "",
        "| case | baseline clip FPS | bitwise steady client FPS | optimized steady client FPS | speedup | optimized RMSE | optimized SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in records:
        performance = item["performance"]
        optimized_metric = item["optimized_comparison"]["generated_frames"]
        speedup = performance["optimized_speedup_over_bitwise"]
        ssim = optimized_metric["ssim"]
        markdown.append(
            f"| {item['id']} | "
            f"{performance['baseline']['whole_clip_output_fps']:.3f} | "
            f"{performance['bitwise']['steady_client_fps']:.3f} | "
            f"{performance['optimized']['steady_client_fps']:.3f} | "
            f"{speedup:.3f}× | {optimized_metric['rmse']:.4f} | "
            f"{'n/a' if ssim is None else f'{ssim:.6f}'} |"
        )
    (results / "triptych_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    write_player(results, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
