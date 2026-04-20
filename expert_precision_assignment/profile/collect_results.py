"""Aggregate per-(mc, variant) results into a CSV.

Handles two schemas, auto-detected by file content:

  bench_serving (sharegpt)  — flat dict with `output_throughput`, `median_ttft_ms`, ...
      filename: mc{mc}_{variant}_n{N}.jsonl

  bench_eval (gsm8k / mmlu / ...) — nested {accuracy, perf, n_samples, run}
      filename: mc{mc}_{variant}.json  (JSONL-append; first record is used)

Output is a wide CSV with a stable leading column set
(mc, variant, schema) plus the union of every metric that appears
across the scanned rows.

Usage:
    python collect_results.py --results_dir results/sharegpt --out_csv results/sharegpt/summary.csv
    python collect_results.py --results_dir results/gsm8k    --out_csv results/gsm8k/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

MC_VARIANT_RE = re.compile(
    r"^mc(?P<mc>\d+)_(?P<variant>(hot\d+|thr\d+))"
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
    return (0, int(v[3:])) if v.startswith("hot") else (1, int(v[3:]))


def _read_first_record(path: Path) -> Dict[str, Any] | None:
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None
    return json.loads(line)


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
    # bench_serving: flat top-level perf metrics.
    row = {
        "mc": mc, "variant": variant, "schema": "bench_serving",
        "task": "sharegpt",
    }
    for k in PERF_FIELDS:
        if k in rec:
            row[k] = rec[k]
    return row


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
        m = MC_VARIANT_RE.match(path.name)
        if not m:
            continue
        rec = _read_first_record(path)
        if rec is None:
            continue
        rows.append(_row_from_record(int(m.group("mc")), m.group("variant"), rec))

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
