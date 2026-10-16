"""Rows of results.tsv from evalscope's per-rung output.

usage: summarize_rung.py header
       summarize_rung.py row RUNG_DIR CC HICACHE_GB STATUS BOOT_S LADDER_S ATTEMPTS CONN_SPLIT
       summarize_rung.py table RESULTS_TSV
"""

import glob
import json
import os
import sys

COLUMNS = (
    "cc",
    "hicache_gb",
    "requests",
    "failed",
    "status",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "interactivity_p50",
    "out_tok_s",
    "total_tok_s",
    "total_tok_s_per_gpu",
    "evalscope_cache_hit_pct",
    "accept_len",
    "conn_split",
    "attempts",
    "boot_s",
    "ladder_s",
)
N_GPUS = 8


def load_json(path):
    with open(path) as f:
        return json.load(f)


def percentile(rows, which, key):
    for row in rows:
        if row.get("Percentiles") == which:
            return row.get(key, "")
    return ""


def accept_length(rung_dir):
    path = os.path.join(rung_dir, "server-info-after.json")
    if not os.path.exists(path):
        return ""
    states = load_json(path).get("internal_states") or [{}]
    value = states[0].get("avg_spec_accept_length")
    return round(value, 4) if value else ""


def row(rung_dir, cc, hicache_gb, status, boot_s, ladder_s, attempts, conn_split):
    summaries = glob.glob(os.path.join(rung_dir, f"c{cc}", "parallel_*", "benchmark_summary.json"))
    summary, percentiles = {}, []
    if len(summaries) == 1:
        summary = load_json(summaries[0])
        pct_path = os.path.join(os.path.dirname(summaries[0]), "benchmark_percentile.json")
        percentiles = load_json(pct_path) if os.path.exists(pct_path) else []
    elif status in ("OK", "OK_UNBALANCED"):
        status = "NO_SUMMARY"
    tpot_p50 = percentile(rows=percentiles, which="50%", key="TPOT (ms)")
    total = summary.get("Total Throughput (tok/s)", "")
    values = (
        cc,
        hicache_gb,
        summary.get("Total Requests", ""),
        summary.get("Failed Requests", ""),
        status,
        percentile(rows=percentiles, which="50%", key="TTFT (ms)"),
        percentile(rows=percentiles, which="90%", key="TTFT (ms)"),
        tpot_p50,
        percentile(rows=percentiles, which="90%", key="TPOT (ms)"),
        round(1000 / tpot_p50, 2) if tpot_p50 else "",
        summary.get("Output Throughput (tok/s)", ""),
        total,
        round(total / N_GPUS, 1) if total else "",
        summary.get("KV Cache Hit Rate (%)", ""),
        accept_length(rung_dir),
        conn_split,
        attempts,
        boot_s,
        ladder_s,
    )
    return "\t".join(str(v) for v in values)


def table(tsv_path):
    lines = [line.rstrip("\n").split("\t") for line in open(tsv_path) if line.strip()]
    head, body = lines[0], lines[1:]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for cells in body:
        print("| " + " | ".join(cells) + " |")


def main():
    mode = sys.argv[1]
    if mode == "header":
        print("\t".join(COLUMNS))
    elif mode == "row":
        rung_dir, cc, hicache_gb, status, boot_s, ladder_s, attempts, conn_split = sys.argv[2:10]
        print(
            row(
                rung_dir=rung_dir,
                cc=int(cc),
                hicache_gb=hicache_gb,
                status=status,
                boot_s=boot_s,
                ladder_s=ladder_s,
                attempts=attempts,
                conn_split=conn_split,
            )
        )
    elif mode == "table":
        table(tsv_path=sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
