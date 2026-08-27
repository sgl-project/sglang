"""Print a markdown summary table from processed benchmark results.

Usage:
    python3 summarize.py <results_dir>

Reads all agg_*.json files recursively from <results_dir> and prints a
markdown table to stdout (redirect to $GITHUB_STEP_SUMMARY to publish).
"""

import json
import sys
from pathlib import Path

from tabulate import tabulate

HEADERS = [
    "Model",
    "Served Model",
    "Hardware",
    "Framework",
    "Precision",
    "ISL",
    "OSL",
    "Prefill TP",
    "Prefill EP",
    "Prefill DP Attn",
    "Prefill Workers",
    "Prefill GPUs",
    "Decode TP",
    "Decode EP",
    "Decode DP Attn",
    "Decode Workers",
    "Decode GPUs",
    "Conc",
    "TTFT (ms)",
    "TPOT (ms)",
    "Interactivity (tok/s/user)",
    "E2EL (s)",
    "TPUT per GPU",
    "Output TPUT per GPU",
    "Input TPUT per GPU",
]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 summarize.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    results = [
        r
        for path in results_dir.rglob("agg_*.json")
        if (r := load_json(path)) and "is_multinode" in r
    ]

    if not results:
        print("No processed result files found.")
        return

    results.sort(
        key=lambda r: (
            r["infmax_model_prefix"],
            r["hw"],
            r["framework"],
            r["precision"],
            r["isl"],
            r["osl"],
            r["prefill_tp"],
            r["prefill_ep"],
            r["decode_tp"],
            r["decode_ep"],
            r["conc"],
        )
    )

    rows = [
        [
            r["infmax_model_prefix"],
            r["model"],
            r["hw"].upper(),
            r["framework"].upper(),
            r["precision"].upper(),
            r["isl"],
            r["osl"],
            r["prefill_tp"],
            r["prefill_ep"],
            r["prefill_dp_attention"],
            r["prefill_num_workers"],
            r["num_prefill_gpu"],
            r["decode_tp"],
            r["decode_ep"],
            r["decode_dp_attention"],
            r["decode_num_workers"],
            r["num_decode_gpu"],
            r["conc"],
            f"{r['median_ttft'] * 1000:.4f}",
            f"{r['median_tpot'] * 1000:.4f}",
            f"{r['median_intvty']:.4f}",
            f"{r['median_e2el']:.4f}",
            f"{r['tput_per_gpu']:.4f}",
            f"{r['output_tput_per_gpu']:.4f}",
            f"{r['input_tput_per_gpu']:.4f}",
        ]
        for r in results
    ]

    print("## GB200 Nightly Benchmark Results\n")
    print(tabulate(rows, headers=HEADERS, tablefmt="github"))
    print()


if __name__ == "__main__":
    main()
