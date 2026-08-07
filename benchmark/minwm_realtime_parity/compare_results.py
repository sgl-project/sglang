#!/usr/bin/env python3
"""Compare lossless baseline/API frames and publish a machine-readable report."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import (
    load_cases,
    prompt_switch_boundary,
    resolve_case_contract,
    sha256_file,
    write_json,
)


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
    parser.add_argument("--case", action="append", dest="selected_cases")
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
    # Process one frame at a time. A 129-frame 720p uint8 clip is ~325 MiB;
    # promoting two full clips to float64 would need several GiB just for a
    # report and can OOM the benchmark after inference has succeeded.
    value_count = 0
    changed_count = 0
    max_abs = 0
    absolute_sum = 0.0
    squared_error_sum = 0.0
    dot_sum = 0.0
    reference_squared_sum = 0.0
    candidate_squared_sum = 0.0
    for ref_frame, candidate_frame in zip(reference, candidate):
        difference = ref_frame.astype(np.int16) - candidate_frame.astype(np.int16)
        absolute = np.abs(difference)
        difference_float = difference.astype(np.float64)
        ref_float = ref_frame.astype(np.float64)
        candidate_float = candidate_frame.astype(np.float64)
        value_count += difference.size
        changed_count += int(np.count_nonzero(difference))
        max_abs = max(max_abs, int(absolute.max(initial=0)))
        absolute_sum += float(absolute.sum(dtype=np.float64))
        squared_error_sum += float(np.square(difference_float).sum(dtype=np.float64))
        dot_sum += float(np.multiply(ref_float, candidate_float).sum(dtype=np.float64))
        reference_squared_sum += float(np.square(ref_float).sum(dtype=np.float64))
        candidate_squared_sum += float(np.square(candidate_float).sum(dtype=np.float64))
    rmse = float(math.sqrt(squared_error_sum / value_count)) if value_count else 0.0
    denominator = math.sqrt(reference_squared_sum * candidate_squared_sum)
    cosine = 1.0 if denominator == 0 else dot_sum / denominator
    try:
        from skimage.metrics import structural_similarity

        def frame_ssim(frame_pair) -> float:
            ref_frame, candidate_frame = frame_pair
            return float(
                structural_similarity(
                    ref_frame, candidate_frame, channel_axis=2, data_range=255
                )
            )

        worker_count = max(1, int(os.environ.get("MINWM_SSIM_WORKERS", "8")))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            ssim_values = list(pool.map(frame_ssim, zip(reference, candidate)))
        ssim = float(np.mean(ssim_values)) if ssim_values else None
        min_frame_ssim = float(min(ssim_values)) if ssim_values else None
    except ImportError:
        ssim = None
        min_frame_ssim = None
    return {
        "bitwise_equal": False,
        "max_abs": max_abs,
        "mean_abs": absolute_sum / value_count if value_count else 0.0,
        "rmse": rmse,
        "psnr_db": 999.0 if rmse == 0 else float(20 * math.log10(255.0 / rmse)),
        "cosine_similarity": cosine,
        "ssim": ssim,
        "min_frame_ssim": min_frame_ssim,
        "changed_value_fraction": (changed_count / value_count if value_count else 0.0),
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
    selected = set(args.selected_cases or [])
    cases = [
        case for case in manifest["cases"] if not selected or case["id"] in selected
    ]
    unknown = selected - {case["id"] for case in manifest["cases"]}
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")

    results = Path(args.results).resolve()
    baseline_by_case = {
        case["id"]: np.load(
            results / "cases" / case["id"] / "baseline.npy",
            allow_pickle=False,
        )
        for case in cases
    }
    records = []
    for case in cases:
        case_contract = resolve_case_contract(case, manifest["contract"])
        reference_frames = int(case_contract["reference_pixel_frames"])
        case_dir = results / "cases" / case["id"]
        baseline_path = case_dir / "baseline.npy"
        sglang_path = case_dir / "sglang.npy"
        baseline = baseline_by_case[case["id"]]
        sglang = np.load(sglang_path, allow_pickle=False)
        all_frames = metric_block(baseline, sglang)
        metrics = {
            "all_frames": all_frames,
            "reference_frame": (
                metric_block(
                    baseline[:reference_frames],
                    sglang[:reference_frames],
                )
                if reference_frames
                else None
            ),
            "generated_frames": (
                metric_block(
                    baseline[reference_frames:],
                    sglang[reference_frames:],
                )
                if reference_frames
                else dict(all_frames)
            ),
            "baseline_frames_sha256": sha256_file(baseline_path),
            "sglang_frames_sha256": sha256_file(sglang_path),
        }
        passed, failures = evaluate(metrics, profile)
        prompt_switch = None
        switch_boundary = prompt_switch_boundary(case, manifest["contract"])
        if switch_boundary is not None:
            with (case_dir / "sglang.json").open(encoding="utf-8") as source:
                sglang_record = json.load(source)
            observation = sglang_record.get("prompt_switch")
            target_chunk = int(case["prompt_switch"]["target_chunk"])
            event_id = int(case["prompt_switch"]["event_id"])
            event_hit_target = bool(
                observation
                and observation.get("first_stats_chunk_with_event") == target_chunk
                and observation.get("first_frame_chunk_with_event") == target_chunk
                and observation.get("stats_event_id_at_target") == event_id
                and observation.get("frame_event_id_at_target") == event_id
            )
            if not event_hit_target:
                failures.append(
                    f"prompt event {event_id} did not first affect chunk {target_chunk}"
                )

            effect = None
            control_case_id = case["prompt_switch"].get("control_case_id")
            if control_case_id is not None:
                control = baseline_by_case[control_case_id]
                prefix_metrics = metric_block(
                    control[:switch_boundary], baseline[:switch_boundary]
                )
                switched_tail_metrics = metric_block(
                    control[switch_boundary:], baseline[switch_boundary:]
                )
                effect = {
                    "control_case_id": control_case_id,
                    "prefix_before_switch": prefix_metrics,
                    "tail_after_switch": switched_tail_metrics,
                    "prefix_bitwise_equal": prefix_metrics["bitwise_equal"],
                    "tail_changed": not switched_tail_metrics["bitwise_equal"],
                }
                if not effect["prefix_bitwise_equal"]:
                    failures.append(
                        "prompt-switch case diverged from control before cutover"
                    )
                if not effect["tail_changed"]:
                    failures.append(
                        "prompt switch produced no pixel change after cutover"
                    )
            prompt_switch = {
                **case["prompt_switch"],
                "pixel_frame_boundary": switch_boundary,
                "switch_time_s": switch_boundary / float(manifest["contract"]["fps"]),
                "event_hit_target_chunk": event_hit_target,
                "observation": observation,
                "effect_vs_control": effect,
            }
            passed = passed and not failures
        record = {
            "id": case["id"],
            "prompt": case["prompt"],
            "action_label": case.get("action_label"),
            "keys": case.get("keys"),
            "trajectory": case.get("trajectory"),
            "contract": case_contract,
            "action_schedule": case.get("action_schedule"),
            "prompt_switch": prompt_switch,
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
        "prompt_switch_case_count": sum(
            record["prompt_switch"] is not None for record in records
        ),
        "all_prompt_events_hit_target": all(
            record["prompt_switch"]["event_hit_target_chunk"]
            for record in records
            if record["prompt_switch"] is not None
        ),
        "all_prompt_switch_prefixes_bitwise": all(
            record["prompt_switch"]["effect_vs_control"]["prefix_bitwise_equal"]
            for record in records
            if record["prompt_switch"] is not None
            and record["prompt_switch"]["effect_vs_control"] is not None
        ),
        "all_prompt_switch_tails_changed": all(
            record["prompt_switch"]["effect_vs_control"]["tail_changed"]
            for record in records
            if record["prompt_switch"] is not None
            and record["prompt_switch"]["effect_vs_control"] is not None
        ),
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
    resolved_manifest = {
        **manifest,
        "contract": {**manifest["contract"], "case_count": len(cases)},
        "cases": cases,
    }
    write_json(results / "manifest.resolved.json", resolved_manifest)
    write_json(results / "report.json", report)
    markdown = [
        "# MinWM realtime parity report",
        "",
        f"Profile: `{args.profile}`",
        "",
        f"Result: **{summary['passed']}/{summary['case_count']} passed**",
        "",
        "| case | action | prompt switch | event chunk | bitwise | max abs | RMSE | SSIM | pass |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        metric = record["metrics"]["generated_frames"]
        ssim = "n/a" if metric["ssim"] is None else f"{metric['ssim']:.8f}"
        prompt_switch = record["prompt_switch"]
        markdown.append(
            f"| {record['id']} | "
            f"{record['action_label'] if record['action_label'] is not None else record['trajectory']} | "
            f"{'yes' if prompt_switch else 'no'} | "
            f"{prompt_switch['target_chunk'] if prompt_switch else 'n/a'} | "
            f"{metric['bitwise_equal']} | "
            f"{metric['max_abs']} | {metric['rmse']:.6f} | {ssim} | {record['passed']} |"
        )
    (results / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_player(results, report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
