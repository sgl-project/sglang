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
    if suffix.startswith("csa_indexer"):
        return "csa_indexer"
    if suffix.startswith("raw_to_page_translation"):
        return "raw_to_page_translation"
    if suffix.startswith("core_attention"):
        return suffix.split(".layer_", maxsplit=1)[0]
    if suffix.startswith("ffn_moe"):
        return "ffn_moe"
    return suffix.split(".layer_", maxsplit=1)[0]


def summarize_trace(path: Path) -> dict:
    trace = open_trace(path)
    totals = defaultdict(float)
    counts = defaultdict(int)
    by_layer = defaultdict(lambda: defaultdict(float))

    for event in trace.get("traceEvents", []):
        name = event_name(event)
        if not name.startswith(PREFIX):
            continue
        dur = event_duration_us(event)
        category = event_category(name)
        totals[category] += dur
        counts[category] += 1
        if ".layer_" in name:
            layer_id = name.rsplit(".layer_", maxsplit=1)[1]
            by_layer[layer_id][category] += dur

    total_us = sum(totals.values())
    return {
        "path": str(path),
        "total_ms": total_us / 1000.0,
        "categories": {
            category: {
                "count": counts[category],
                "total_ms": total / 1000.0,
                "pct": (total / total_us * 100.0) if total_us else 0.0,
            }
            for category, total in sorted(totals.items())
        },
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
