"""Aggregate per-(mc, variant) results into a CSV.

Handles three schemas, auto-detected by file content:

  bench_serving (sharegpt)  — flat dict with `output_throughput`, `median_ttft_ms`, ...
      filename: mc{mc}_{variant}_n{N}.jsonl

  bench_serving (openai-mode; supergpqa / ifbench / livecodebench_v6 / …)
      — flat dict with `generated_texts`, `input_lens`, `output_lens`, plus
        perf fields.  Scoring is offline; scored metrics live in a sibling
        `mc{mc}_{variant}.scores.json` file and are merged in here when
        present.
      filename: mc{mc}_{variant}.jsonl (+ optional .scores.json sidecar)

  bench_eval (gsm8k / mmlu / ...) — nested {accuracy, perf, n_samples, run}
      filename: mc{mc}_{variant}.json  (JSONL-append; first record is used)

Output is a wide CSV with a stable leading column set
(mc, variant, schema, task) plus the union of every metric that appears
across the scanned rows.

Usage (paths relative to experiments/):
    python pipeline/collect_result/collect_results.py \
        --results_dir data/results/sharegpt --out_csv data/results/sharegpt/summary.csv
    python pipeline/collect_result/collect_results.py \
        --results_dir data/results/gsm8k --out_csv data/results/gsm8k/summary.csv
    python pipeline/collect_result/collect_results.py \
        --results_dir data/results/supergpqa --out_csv data/results/supergpqa/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

MC_VARIANT_RE = re.compile(
    r"^mc(?P<mc>\d+)_(?P<variant>(hot\d+|thr\d+|hess\d+))"
    r"(?:_n\d+)?\.(jsonl|json)$"
)

PERF_FIELDS = [
    "completed",
    "duration",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "input_throughput",
    "output_throughput", "total_throughput",
    "mean_e2e_latency_ms", "median_e2e_latency_ms", "p99_e2e_latency_ms",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_itl_ms", "median_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "concurrency", "max_output_tokens_per_s",
]

LEADING = ["mc", "variant", "schema", "task"]


def _variant_sort_key(v: str) -> tuple:
    if v.startswith("hot"):
        return (0, int(v[3:]))
    if v.startswith("thr"):
        return (1, int(v[3:]))
    if v.startswith("hess"):
        return (2, int(v[4:]))
    return (3, 0)


def _read_first_record(path: Path) -> Dict[str, Any] | None:
    """Return the first JSON object in the file.

    Handles both JSONL (one record per line) and pretty-printed JSON
    (a single multi-line object), using raw_decode to consume the first
    top-level value regardless of newlines.
    """
    with open(path) as f:
        text = f.read().lstrip()
    if not text:
        return None
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


def _row_from_record(
    mc: int, variant: str, rec: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize bench_serving flat dict OR bench_eval nested dict into a row."""
    if "accuracy" in rec and "perf" in rec:
        row: Dict[str, Any] = {
            "mc": mc, "variant": variant, "schema": "bench_eval",
            "task": rec.get("task", ""),
        }
        for k, v in (rec.get("accuracy") or {}).items():
            row[f"acc_{k.replace(',', '__')}"] = v
        for k in PERF_FIELDS:
            if k in rec.get("perf", {}):
                row[k] = rec["perf"][k]
        ns = rec.get("n_samples") or {}
        if isinstance(ns, dict):
            for k, v in ns.items():
                row[f"n_{k}"] = v
        return row
    schema = "bench_serving_openai" if "generated_texts" in rec else "bench_serving"
    row = {
        "mc": mc, "variant": variant, "schema": schema,
        "task": "sharegpt" if schema == "bench_serving" else "",
    }
    for k in PERF_FIELDS:
        if k in rec:
            row[k] = rec[k]
    return row


_SCORE_KEYS_TO_FLATTEN = {
    "accuracy", "pass@1",
    "prompt_level_strict_acc", "prompt_level_loose_acc",
    "inst_level_strict_acc", "inst_level_loose_acc",
    "n_total", "n_correct", "n_failed", "n_unparsed", "n_pass",
    "n_failed_generation", "n_failed_execution",
}


def _merge_scores_sidecar(row: Dict[str, Any], trace_path: Path) -> None:
    sidecar = trace_path.with_suffix(".scores.json")
    if not sidecar.exists():
        return
    try:
        with open(sidecar) as f:
            scored = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if scored.get("task"):
        row["task"] = scored["task"]
    for k in _SCORE_KEYS_TO_FLATTEN:
        if k in scored:
            row[f"score_{k}"] = scored[k]
    for breakdown_key in (
        "accuracy_by_discipline", "accuracy_by_difficulty",
        "accuracy_by_is_calculation", "pass@1_by_platform",
        "pass@1_by_difficulty",
    ):
        bd = scored.get(breakdown_key)
        if isinstance(bd, dict):
            for sub, stats in bd.items():
                if isinstance(stats, dict) and "acc" in stats:
                    row[f"score_{breakdown_key}__{sub}"] = stats["acc"]
                elif isinstance(stats, dict) and "pass@1" in stats:
                    row[f"score_{breakdown_key}__{sub}"] = stats["pass@1"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True,
                    help="Directory containing mc*_*.{json,jsonl} files.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--round", type=int, default=4,
                    help="Decimals to round floats to (default 4).")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("mc*_*.json*")):
        if path.name.endswith(".scores.json"):
            continue
        m = MC_VARIANT_RE.match(path.name)
        if not m:
            continue
        rec = _read_first_record(path)
        if rec is None:
            continue
        row = _row_from_record(int(m.group("mc")), m.group("variant"), rec)
        _merge_scores_sidecar(row, path)
        rows.append(row)

    rows.sort(key=lambda r: (r["mc"], _variant_sort_key(r["variant"])))

    # Round floats for CSV readability.
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, float):
                r[k] = round(v, args.round)

    # Build column order: leading fixed, then perf order-preserved, then
    # remaining keys (accuracy & n_* & anything else) sorted.
    present_perf = [k for k in PERF_FIELDS if any(k in r for r in rows)]
    other_keys: set = set()
    for r in rows:
        for k in r:
            if k in LEADING or k in present_perf:
                continue
            other_keys.add(k)
    columns = LEADING + present_perf + sorted(other_keys)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows, {len(columns)} cols → {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
