#!/usr/bin/env python3
"""Compare lossless baseline/API frames and publish a machine-readable report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import load_cases, sha256_file, write_json


def write_player(results: Path, report: dict) -> None:
    """Write a self-contained report manifest into the synchronized player."""
    template = Path(__file__).with_name("player.html").read_text(encoding="utf-8")
    placeholder = "__MINWM_EMBEDDED_REPORT__"
    if template.count(placeholder) != 1:
        raise ValueError("player template must contain exactly one report placeholder")
    embedded_report = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    player_dir = results / "player"
    player_dir.mkdir(exist_ok=True)
    (player_dir / "index.html").write_text(
        template.replace(placeholder, embedded_report), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--thresholds", default=Path(__file__).with_name("thresholds.json")
    )
    parser.add_argument("--profile", default="bitwise")
    return parser.parse_args()


def metric_block(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    if np.array_equal(reference, candidate):
        # Long 720p regressions contain hundreds of millions of values.  Once
        # exact equality is established, float64 cosine and per-frame SSIM are
        # mathematical identities; recomputing them only multiplies memory and
        # CPU cost without adding evidence.
        return {
            "bitwise_equal": True,
            "max_abs": 0,
            "mean_abs": 0.0,
            "rmse": 0.0,
            "psnr_db": 999.0,
            "cosine_similarity": 1.0,
            "ssim": 1.0,
            "min_frame_ssim": 1.0,
            "changed_value_fraction": 0.0,
        }
    reference_i = reference.astype(np.int16)
    candidate_i = candidate.astype(np.int16)
    abs_error = np.abs(reference_i - candidate_i)
    error = reference_i.astype(np.float64) - candidate_i.astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(error))))
    ref_flat = reference.astype(np.float64).reshape(-1)
    candidate_flat = candidate.astype(np.float64).reshape(-1)
    denominator = float(np.linalg.norm(ref_flat) * np.linalg.norm(candidate_flat))
    cosine = (
        1.0
        if denominator == 0
        else float(np.dot(ref_flat, candidate_flat) / denominator)
    )
    try:
        from skimage.metrics import structural_similarity

        ssim_values = [
            float(
                structural_similarity(
                    ref_frame,
                    candidate_frame,
                    channel_axis=2,
                    data_range=255,
                )
            )
            for ref_frame, candidate_frame in zip(reference, candidate)
        ]
        ssim = float(np.mean(ssim_values)) if ssim_values else None
        min_frame_ssim = float(min(ssim_values)) if ssim_values else None
    except ImportError:
        ssim = None
        min_frame_ssim = None
    return {
        "bitwise_equal": False,
        "max_abs": int(abs_error.max(initial=0)),
        "mean_abs": float(abs_error.mean()),
        "rmse": rmse,
        "psnr_db": 999.0 if rmse == 0 else float(20 * math.log10(255.0 / rmse)),
        "cosine_similarity": cosine,
        "ssim": ssim,
        "min_frame_ssim": min_frame_ssim,
        "changed_value_fraction": float(np.count_nonzero(abs_error) / abs_error.size),
    }


def evaluate(metrics: dict, profile: dict) -> tuple[bool, list[str]]:
    failures = []
    generated = metrics["generated_frames"]
    if profile.get("require_bitwise") and not generated["bitwise_equal"]:
        failures.append("generated frames are not bitwise equal")
    if "max_abs_lte" in profile and generated["max_abs"] > profile["max_abs_lte"]:
        failures.append(f"max_abs={generated['max_abs']} > {profile['max_abs_lte']}")
    if "rmse_lte" in profile and generated["rmse"] > profile["rmse_lte"]:
        failures.append(f"rmse={generated['rmse']:.6g} > {profile['rmse_lte']}")
    if "ssim_gte" in profile:
        if generated["ssim"] is None:
            failures.append("SSIM unavailable (install scikit-image)")
        elif generated["ssim"] < profile["ssim_gte"]:
            failures.append(f"ssim={generated['ssim']:.9g} < {profile['ssim_gte']}")
    return not failures, failures


def main() -> None:
    args = parse_args()
    manifest = load_cases(args.cases)
    with Path(args.thresholds).open(encoding="utf-8") as source:
        threshold_manifest = json.load(source)
    try:
        profile = threshold_manifest["profiles"][args.profile]
    except KeyError as exc:
        raise ValueError(f"unknown threshold profile: {args.profile}") from exc

    results = Path(args.results).resolve()
    records = []
    for case in manifest["cases"]:
        case_dir = results / "cases" / case["id"]
        baseline_path = case_dir / "baseline.npy"
        sglang_path = case_dir / "sglang.npy"
        baseline = np.load(baseline_path, allow_pickle=False)
        sglang = np.load(sglang_path, allow_pickle=False)
        metrics = {
            "all_frames": metric_block(baseline, sglang),
            "reference_frame": metric_block(baseline[:1], sglang[:1]),
            "generated_frames": metric_block(baseline[1:], sglang[1:]),
            "baseline_frames_sha256": sha256_file(baseline_path),
            "sglang_frames_sha256": sha256_file(sglang_path),
        }
        passed, failures = evaluate(metrics, profile)
        record = {
            "id": case["id"],
            "prompt": case["prompt"],
            "action_label": case["action_label"],
            "keys": case["keys"],
            "passed": passed,
            "failures": failures,
            "metrics": metrics,
        }
        write_json(case_dir / "metrics.json", record)
        records.append(record)

    observed_ssim = [
        record["metrics"]["generated_frames"]["ssim"]
        for record in records
        if record["metrics"]["generated_frames"]["ssim"] is not None
    ]
    summary = {
        "case_count": len(records),
        "passed": sum(record["passed"] for record in records),
        "failed": sum(not record["passed"] for record in records),
        "all_passed": all(record["passed"] for record in records),
        "all_generated_bitwise": all(
            record["metrics"]["generated_frames"]["bitwise_equal"] for record in records
        ),
        "observed_generated_max_abs": max(
            record["metrics"]["generated_frames"]["max_abs"] for record in records
        ),
        "observed_generated_max_rmse": max(
            record["metrics"]["generated_frames"]["rmse"] for record in records
        ),
        "observed_generated_min_ssim": min(observed_ssim) if observed_ssim else None,
    }
    report = {
        "schema_version": 1,
        "contract": manifest["contract"],
        "threshold_profile_name": args.profile,
        "threshold_profile": profile,
        "summary": summary,
        "cases": records,
    }
    # Preserve the exact evaluated manifest beside the report. This avoids a
    # later source-tree edit changing the apparent inputs of an archived run.
    write_json(results / "manifest.resolved.json", manifest)
    write_json(results / "report.json", report)
    markdown = [
        "# MinWM realtime parity report",
        "",
        f"Profile: `{args.profile}`",
        "",
        f"Result: **{summary['passed']}/{summary['case_count']} passed**",
        "",
        "| case | action | bitwise | max abs | RMSE | SSIM | pass |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        metric = record["metrics"]["generated_frames"]
        ssim = "n/a" if metric["ssim"] is None else f"{metric['ssim']:.8f}"
        markdown.append(
            f"| {record['id']} | {record['action_label']} | {metric['bitwise_equal']} | "
            f"{metric['max_abs']} | {metric['rmse']:.6f} | {ssim} | {record['passed']} |"
        )
    (results / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_player(results, report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
