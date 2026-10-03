"""Validate paired global workloads and summarize independent process medians."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory):
    groups = {}
    for path in sorted(directory.glob("*.json")):
        run = json.loads(path.read_text())
        if run.get("schema_version") != 2 or "results" not in run:
            continue
        for point in run["results"]:
            key = run["phase"], point["global_tokens"]
            samples = point["samples_per_rank_ms"]
            assert len(samples) == run["tp_size"]
            assert all(len(rank) == run["iters"] for rank in samples)
            maxima = [max(rank[i][0] for rank in samples) for i in range(run["iters"])]
            assert abs(statistics.median(maxima) - point["median_ms"]) < 1e-6
            expected_rows = point["global_tokens"]
            if run["backend"] == "deepep":
                assert sum(point["tokens_per_rank"]) == expected_rows
            else:
                assert point["tokens_per_rank"] == [expected_rows] * run["tp_size"]
            groups.setdefault(key, {}).setdefault(run["backend"], []).append(
                dict(
                    repeat_id=run["repeat_id"],
                    median_ms=point["median_ms"],
                    input_hash=point["input_sha256"],
                    weights_hash=run["weights_sha256"],
                    input_seed=run["input_seed"],
                    environment=run["metadata"]["environment"],
                    source_commit=run["metadata"]["sglang_commit"],
                    timing_script=run["metadata"]["scripts_sha256"][
                        "bench_moe_tp_ep.py"
                    ],
                    file=path.name,
                )
            )
    results = []
    for (phase, n), configs in sorted(groups.items()):
        if set(configs) != {"none", "deepep"}:
            raise ValueError(f"missing paired configuration: {phase} {n}")
        tp, ep = configs["none"], configs["deepep"]
        ids_tp = [x["repeat_id"] for x in tp]
        ids_ep = [x["repeat_id"] for x in ep]
        if len(set(ids_tp)) != len(ids_tp) or sorted(ids_tp) != sorted(ids_ep):
            raise ValueError("unpaired or duplicate independent runs")
        for key in (
            "input_hash",
            "weights_hash",
            "input_seed",
            "source_commit",
            "timing_script",
        ):
            if len({x[key] for x in tp + ep}) != 1:
                raise ValueError(f"mismatched {key}: {phase} {n}")
        if any(x["environment"] != tp[0]["environment"] for x in tp + ep):
            raise ValueError("mismatched experiment environment")
        values = {}
        for name, runs in configs.items():
            medians = [x["median_ms"] for x in runs]
            mid = statistics.median(medians)
            values[name] = dict(
                median_ms=mid,
                run_medians_ms=medians,
                relative_range=(max(medians) - min(medians)) / mid,
            )
        results.append(
            dict(
                phase=phase,
                global_tokens=n,
                independent_runs=len(tp),
                configurations=values,
                ep_over_tp=values["deepep"]["median_ms"] / values["none"]["median_ms"],
            )
        )
    if not results:
        raise ValueError("no schema-version-2 benchmark results found")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.directory), indent=2) + "\n")
