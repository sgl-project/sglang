"""Summarize DSV4 IndexCache torch-profiler Chrome traces."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

PREFIX = "dsv4_indexcache."


def open_trace(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with path.open() as f:
        return json.load(f)


def event_name(event: dict) -> str:
    return str(event.get("name", ""))


def event_duration_us(event: dict) -> float:
    dur = event.get("dur", 0)
    return float(dur or 0)


def event_category(name: str) -> str:
    suffix = name.removeprefix(PREFIX)
    if suffix.startswith("cuda_graph."):
        return "cuda_graph"
    if suffix.startswith("csa_indexer"):
        return "csa_indexer"
    if suffix.startswith("raw_to_page_translation"):
        return "raw_to_page_translation"
    if suffix.startswith("core_attention"):
        return suffix.split(".layer_", maxsplit=1)[0]
    if suffix.startswith("ffn_moe"):
        return "ffn_moe"
    return suffix.split(".layer_", maxsplit=1)[0]


def summarize_cuda_graph_paths(graph_paths: dict[str, int]) -> dict:
    outcomes = defaultdict(int)
    modes = defaultdict(lambda: defaultdict(int))
    variants = defaultdict(lambda: defaultdict(int))

    for path, count in graph_paths.items():
        parts = path.split(".")
        if not parts:
            continue
        outcome = parts[-1]
        mode = parts[0]
        variant = ".".join(parts[1:-1]) or "default"
        outcomes[outcome] += count
        modes[mode][outcome] += count
        variants[variant][outcome] += count

    total = sum(outcomes.values())
    replay = outcomes.get("replay", 0)
    fallback = outcomes.get("fallback", 0)
    return {
        "total": total,
        "replay": replay,
        "fallback": fallback,
        "replay_rate": replay / total if total else 0.0,
        "fallback_rate": fallback / total if total else 0.0,
        "by_mode": {
            mode: dict(sorted(counts.items())) for mode, counts in sorted(modes.items())
        },
        "by_variant": {
            variant: dict(sorted(counts.items()))
            for variant, counts in sorted(variants.items())
        },
    }


def summarize_objective_buckets(categories: dict[str, dict]) -> dict:
    bucket_categories = {
        "csa_indexer": ["csa_indexer"],
        "raw_to_page_translation": ["raw_to_page_translation"],
        "sparse_core_attention": [
            category for category in categories if category.startswith("core_attention")
        ],
        "ffn_moe": ["ffn_moe"],
        "cuda_graph": ["cuda_graph"],
    }
    buckets = {}
    for bucket, names in bucket_categories.items():
        present = [name for name in names if name in categories]
        total_ms = sum(categories[name]["total_ms"] for name in present)
        count = sum(categories[name]["count"] for name in present)
        buckets[bucket] = {
            "categories": present,
            "count": count,
            "total_ms": total_ms,
        }
    return buckets


def summarize_trace(path: Path) -> dict:
    trace = open_trace(path)
    totals = defaultdict(float)
    counts = defaultdict(int)
    graph_paths = defaultdict(int)
    by_layer = defaultdict(lambda: defaultdict(float))

    for event in trace.get("traceEvents", []):
        name = event_name(event)
        if not name.startswith(PREFIX):
            continue
        dur = event_duration_us(event)
        category = event_category(name)
        totals[category] += dur
        counts[category] += 1
        suffix = name.removeprefix(PREFIX)
        if suffix.startswith("cuda_graph."):
            graph_paths[suffix.removeprefix("cuda_graph.")] += 1
        if ".layer_" in name:
            layer_id = name.rsplit(".layer_", maxsplit=1)[1]
            by_layer[layer_id][category] += dur

    total_us = sum(totals.values())
    categories = {
        category: {
            "count": counts[category],
            "total_ms": total / 1000.0,
            "pct": (total / total_us * 100.0) if total_us else 0.0,
        }
        for category, total in sorted(totals.items())
    }
    return {
        "path": str(path),
        "total_ms": total_us / 1000.0,
        "categories": categories,
        "objective_buckets": summarize_objective_buckets(categories),
        "cuda_graph_paths": dict(sorted(graph_paths.items())),
        "cuda_graph_summary": summarize_cuda_graph_paths(graph_paths),
        "layers": {
            layer: {
                category: total / 1000.0
                for category, total in sorted(categories.items())
            }
            for layer, categories in sorted(by_layer.items(), key=lambda x: int(x[0]))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {"traces": [summarize_trace(path) for path in args.trace]}
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
