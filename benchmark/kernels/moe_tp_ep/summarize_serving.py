"""Summarize paired independent service runs written by bench_serving.py."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory):
    groups = {}
    for path in sorted(directory.glob("*_r*_in*.json")):
        backend, repeat, _ = path.stem.split("_")
        run = json.loads(path.read_text())
        if backend not in ("none", "deepep"):
            raise ValueError(f"unexpected backend in {path}")
        if len(run["per_request"]) != run["requests"] or any(
            r["output_tokens"] != run["output_len"] for r in run["per_request"]
        ):
            raise ValueError(f"incomplete requests: {path}")
        expected = run["requests"] * run["output_len"]
        if (
            run["output_tokens"] != expected
            or abs(run["throughput"] - expected / run["wall_seconds"]) > 1e-6
        ):
            raise ValueError(f"invalid throughput: {path}")
        groups.setdefault(run["input_len"], {}).setdefault(backend, {})[repeat] = run
    results = []
    for length, configs in sorted(groups.items()):
        if set(configs) != {"none", "deepep"}:
            raise ValueError(f"missing backend: input length {length}")
        if set(configs["none"]) != set(configs["deepep"]):
            raise ValueError(f"unpaired repetitions: input length {length}")
        for rep, tp in configs["none"].items():
            ep = configs["deepep"][rep]
            for key in (
                "input_sha256",
                "seed",
                "requests",
                "concurrency",
                "output_len",
                "warmup_requests",
                "script_sha256",
            ):
                if tp[key] != ep[key]:
                    raise ValueError(f"mismatched {key}: {length} {rep}")
        report = dict(input_len=length, independent_runs=len(configs["none"]))
        report["configurations"] = {}
        for backend, runs in configs.items():
            metrics = {}
            for key in (
                "throughput",
                "ttft_ms_median",
                "tpot_ms_median",
                "total_ms_median",
            ):
                values = [runs[rep][key] for rep in sorted(runs)]
                median = statistics.median(values)
                metrics[key] = dict(
                    median=median,
                    run_values=values,
                    relative_range=(max(values) - min(values)) / median,
                )
            report["configurations"][backend] = metrics
        results.append(report)
    if not results:
        raise ValueError("no serving results found")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.write_text(json.dumps(summarize(args.directory), indent=2) + "\n")
