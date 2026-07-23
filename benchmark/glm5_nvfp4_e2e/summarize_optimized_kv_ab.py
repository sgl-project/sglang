#!/usr/bin/env python3
"""Summarize the one-round GLM-5 NVFP4 KV-cache E2E A/B."""

import argparse
import csv
import json
import math
from pathlib import Path


CASES = (
    (
        "prefill_8k_b1",
        "prefill",
        "median_ttft_ms",
        "latency",
        8192,
        32,
        3,
        (("median_ttft_ms", "latency"),),
    ),
    (
        "prefill_32k_b1",
        "prefill",
        "median_ttft_ms",
        "latency",
        32768,
        32,
        3,
        (("median_ttft_ms", "latency"),),
    ),
    (
        "prefill_32k_c12",
        "prefill",
        "input_throughput",
        "throughput",
        32768,
        32,
        12,
        (("input_throughput", "throughput"), ("median_ttft_ms", "latency")),
    ),
    (
        "decode_32k_b1",
        "decode",
        "median_tpot_ms",
        "latency",
        32768,
        512,
        3,
        (("median_tpot_ms", "latency"), ("p95_itl_ms", "latency")),
    ),
    (
        "decode_32k_c12_long",
        "decode",
        "output_throughput",
        "throughput",
        32768,
        2048,
        12,
        (("output_throughput", "throughput"), ("p95_itl_ms", "latency")),
    ),
    (
        "decode_8k_c32",
        "decode",
        "output_throughput",
        "throughput",
        8192,
        512,
        32,
        (("output_throughput", "throughput"), ("p95_itl_ms", "latency")),
    ),
)


def last_json(path: Path) -> dict:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"empty benchmark result: {path}")
    return json.loads(lines[-1])


def pct(candidate: float, baseline: float, kind: str) -> float:
    del kind
    return (candidate / baseline - 1.0) * 100.0


def decision(delta: float, kind: str) -> str:
    regression = delta > 2.0 if kind == "latency" else delta < -2.0
    if regression:
        return "REGRESSION"
    improvement = delta < -3.0 if kind == "latency" else delta > 3.0
    return "IMPROVEMENT" if improvement else "TIE"


def metric_value(result: dict, metric: str, context: str) -> float:
    if metric not in result:
        raise RuntimeError(f"{context}: missing metric {metric}")
    value = float(result[metric])
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{context}: {metric} must be finite and positive, got {value}")
    return value


def metric_delta(
    fp8: dict, nvfp4: dict, metric: str, kind: str, case: str = "unknown"
) -> float:
    return pct(
        metric_value(nvfp4, metric, f"nvfp4 {case}"),
        metric_value(fp8, metric, f"fp8 {case}"),
        kind,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--expected-capacity", type=int, required=True)
    args = parser.parse_args()
    raw = args.result_dir / "raw"
    rows = []
    for (
        case,
        phase,
        primary_metric,
        kind,
        input_len,
        output_len,
        prompts,
        gates,
    ) in CASES:
        fp8 = last_json(raw / f"fp8__{case}.jsonl")
        nvfp4 = last_json(raw / f"nvfp4__{case}.jsonl")
        for variant, result in (("fp8", fp8), ("nvfp4", nvfp4)):
            if result.get("completed") != prompts:
                raise RuntimeError(
                    f"{variant} {case}: completed={result.get('completed')} != {prompts}"
                )
            if result.get("total_input_tokens") != input_len * prompts:
                raise RuntimeError(f"{variant} {case}: unexpected total input tokens")
            if result.get("total_output_tokens") != output_len * prompts:
                raise RuntimeError(f"{variant} {case}: unexpected total output tokens")
            if any(result.get("errors", [])):
                raise RuntimeError(f"{variant} {case}: request error present")
            capacity = (result.get("server_info") or {}).get("max_total_num_tokens")
            if capacity != args.expected_capacity:
                raise RuntimeError(
                    f"{variant} {case}: max_total_num_tokens={capacity} "
                    f"!= {args.expected_capacity}"
                )
            for metric in (
                "median_ttft_ms",
                "median_tpot_ms",
                "p95_itl_ms",
                "input_throughput",
                "output_throughput",
                "total_throughput",
            ):
                metric_value(result, metric, f"{variant} {case}")
        fp8_value = metric_value(fp8, primary_metric, f"fp8 {case}")
        nvfp4_value = metric_value(nvfp4, primary_metric, f"nvfp4 {case}")
        delta = pct(nvfp4_value, fp8_value, kind)
        failed_gates = [
            metric
            for metric, metric_kind in gates
            if decision(
                metric_delta(fp8, nvfp4, metric, metric_kind, case), metric_kind
            )
            == "REGRESSION"
        ]
        rows.append(
            {
                "case": case,
                "phase": phase,
                "primary_metric": primary_metric,
                "direction": kind,
                "fp8": fp8_value,
                "nvfp4": nvfp4_value,
                "nvfp4_vs_fp8_pct": delta,
                "decision": "REGRESSION" if failed_gates else decision(delta, kind),
                "failed_gates": ";".join(failed_gates),
                "fp8_completed": fp8["completed"],
                "nvfp4_completed": nvfp4["completed"],
                "fp8_median_ttft_ms": fp8.get("median_ttft_ms"),
                "nvfp4_median_ttft_ms": nvfp4.get("median_ttft_ms"),
                "fp8_median_tpot_ms": fp8.get("median_tpot_ms"),
                "nvfp4_median_tpot_ms": nvfp4.get("median_tpot_ms"),
                "fp8_p95_itl_ms": fp8.get("p95_itl_ms"),
                "nvfp4_p95_itl_ms": nvfp4.get("p95_itl_ms"),
                "fp8_output_throughput": fp8.get("output_throughput"),
                "nvfp4_output_throughput": nvfp4.get("output_throughput"),
                "ttft_delta_pct": metric_delta(
                    fp8, nvfp4, "median_ttft_ms", "latency", case
                ),
                "tpot_delta_pct": metric_delta(
                    fp8, nvfp4, "median_tpot_ms", "latency", case
                ),
                "p95_itl_delta_pct": metric_delta(
                    fp8, nvfp4, "p95_itl_ms", "latency", case
                ),
                "input_throughput_delta_pct": metric_delta(
                    fp8, nvfp4, "input_throughput", "throughput", case
                ),
                "output_throughput_delta_pct": metric_delta(
                    fp8, nvfp4, "output_throughput", "throughput", case
                ),
                "total_throughput_delta_pct": metric_delta(
                    fp8, nvfp4, "total_throughput", "throughput", case
                ),
            }
        )

    csv_path = args.result_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    regressions = [row for row in rows if row["decision"] == "REGRESSION"]
    md = [
        "# GLM-5 optimized NVFP4 KV-cache E2E A/B",
        "",
        "Positive latency delta is slower; negative throughput delta is slower. ",
        "The one-round noise band is +/-2%; an improvement label requires >3%.",
        "",
        "| Case | Phase | Primary metric | FP8 | NVFP4 | Delta | P95 ITL delta | Decision |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['case']} | {row['phase']} | {row['primary_metric']} | "
            f"{row['fp8']:.3f} | {row['nvfp4']:.3f} | "
            f"{row['nvfp4_vs_fp8_pct']:+.2f}% | "
            f"{row['p95_itl_delta_pct']:+.2f}% | {row['decision']} |"
        )
    md.extend(
        [
            "",
            f"Overall gate: **{'FAIL' if regressions else 'PASS'}**",
            "",
        ]
    )
    if regressions:
        md.append(
            "Regressions requiring another optimization pass: "
            + ", ".join(
                f"{row['case']} ({row['failed_gates']})" for row in regressions
            )
            + "."
        )
    else:
        md.append(
            "No measured prefill or decode case regressed outside the +/-2% noise band."
        )
    (args.result_dir / "summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print("\n".join(md))
    return 2 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
