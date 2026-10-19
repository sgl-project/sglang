#!/usr/bin/env python3
"""Read saved serving results without importing a GPU runtime or running a server.

Compare explicit baseline/candidate globs within one experiment. Matching JSON
metadata is necessary, but cannot establish matching code, clocks or warmups.
"""

import argparse
import glob
import json
import math
from pathlib import Path


def load(pattern):
    rows = {}
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No results match {pattern}")
    for name in paths:
        data = json.loads(Path(name).read_text())
        for key in (
            "max_concurrency",
            "num_prompts",
            "completed",
            "duration",
            "total_output_tokens",
            "output_throughput",
            "median_tpot_ms",
            "median_ttft_ms",
            "median_e2el_ms",
        ):
            value = data.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid/incomplete {name}: {key}={value!r}")
        if data["completed"] != data["num_prompts"]:
            raise ValueError(f"Incomplete requests in {name}")
        concurrency = data["max_concurrency"]
        if concurrency in rows:
            raise ValueError(f"Ambiguous conc={concurrency}; narrow glob {pattern}")
        rows[concurrency] = (name, data)
    return rows


def compare(baseline, candidate):
    if baseline.keys() != candidate.keys():
        raise ValueError("Concurrency sets differ; narrow both globs to matching runs")
    results = []
    for concurrency in sorted(baseline):
        bp, b = baseline[concurrency]
        cp, c = candidate[concurrency]
        for key in (
            "model_id",
            "tokenizer_id",
            "num_prompts",
            "request_rate",
            "burstiness",
        ):
            if key not in b or key not in c or b[key] != c[key]:
                raise ValueError(f"conc={concurrency}: unmatched {key}")
        results.append(
            {
                "concurrency": concurrency,
                "baseline_path": bp,
                "candidate_path": cp,
                "baseline_date": b.get("date"),
                "candidate_date": c.get("date"),
                "model": b["model_id"],
                "num_prompts": b["num_prompts"],
                "baseline_output_tok_s": b["output_throughput"],
                "candidate_output_tok_s": c["output_throughput"],
                "throughput_change_pct": 100
                * (c["output_throughput"] / b["output_throughput"] - 1),
                "baseline_tpot_ms": b["median_tpot_ms"],
                "candidate_tpot_ms": c["median_tpot_ms"],
                "tpot_change_pct": 100
                * (c["median_tpot_ms"] / b["median_tpot_ms"] - 1),
                "baseline_output_tokens": b["total_output_tokens"],
                "candidate_output_tokens": c["total_output_tokens"],
            }
        )
    return {
        "scope": "Saved results only; no new measurements",
        "unverified": [
            "code/installed kernel hashes",
            "warmup protocol",
            "GPU isolation and clocks",
            "identical input requests",
            "statistical significance",
        ],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Quoted JSON glob")
    parser.add_argument("--candidate", required=True, help="Quoted JSON glob")
    args = parser.parse_args()
    try:
        result = compare(load(args.baseline), load(args.candidate))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
