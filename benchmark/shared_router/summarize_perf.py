# SPDX-License-Identifier: Apache-2.0
"""Validate three repetitions; report repetition statistics, not pooled percentiles."""

import argparse
import json
import math
import statistics
from pathlib import Path


def counter(path, name):
    return sum(
        float(line.split()[-1])
        for line in path.read_text().splitlines()
        if line.startswith(name + "{")
    )


def summarize(root):
    rows, requests, acceptance = [], [], []
    for repeat in (1, 2, 3):
        folder = root / f"repeat-{repeat:02d}"
        row = json.loads((folder / "benchmark.json").read_text())
        manifest = json.loads((folder / "requests.json").read_text())["requests"]
        n = len(manifest)
        assert n in (160, 320) and row["completed"] == row["num_prompts"] == n
        assert len(row["errors"]) == n and not any(row["errors"])
        assert row["output_lens"] == [r["requested_output_tokens"] for r in manifest]
        assert row["input_lens"] == [r["prompt_tokens"] for r in manifest]
        assert row["total_output_tokens"] == sum(row["output_lens"])

        def delta(name):
            return counter(folder / "metrics.after.prom", name) - counter(
                folder / "metrics.before.prom", name
            )

        calls = delta("sglang:spec_verify_calls_total")
        tokens = delta("sglang:generation_tokens_total")
        assert calls > 0 and tokens > 0
        effective = tokens / calls
        assert abs(effective - 3.51) < 0.15, "Synthetic acceptance not evidenced"
        acceptance.append(effective)
        rows.append(row)
        requests.append(manifest)
    assert requests[0] == requests[1] == requests[2], "Repeated workloads differ"
    metrics = {}
    for name in (
        "median_tpot_ms",
        "median_itl_ms",
        "median_ttft_ms",
        "output_throughput",
    ):
        values = [row[name] for row in rows]
        assert all(
            isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in values
        )
        metrics[name] = dict(
            samples=values, median=statistics.median(values), minimum=min(values)
        )
    return dict(
        status="valid_perf_only",
        accuracy_evaluated=False,
        aggregation="median and minimum of per-repetition statistics; not pooled percentiles",
        effective_tokens_per_verify_including_client_warmup=acceptance,
        metrics=metrics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = summarize(args.run)
    (args.run / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
