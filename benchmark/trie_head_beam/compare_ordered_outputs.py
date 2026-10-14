#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def load_reference(path: Path) -> list[list[list[int]]]:
    if path.suffix == ".gz":
        return load_candidate(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(data["results"], key=lambda row: row["request_index"])
    return [row["sid_tokens"] for row in rows]


def load_candidate(path: Path) -> list[list[list[int]]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "sid_tokens" in row:
                rows.append(row)
    rows.sort(key=lambda row: row["request_index"])
    return [row["sid_tokens"] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ordered trie-head SID outputs request by request."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    candidate = load_candidate(args.candidate)
    if len(reference) != len(candidate):
        raise ValueError(
            f"request count mismatch: reference={len(reference)}, candidate={len(candidate)}"
        )

    exact_order = 0
    exact_sets = 0
    intersections = []
    jaccards = []
    top_overlaps = {10: [], 100: [], 500: [], 1000: []}
    per_request = []
    for index, (ref_rows, cand_rows) in enumerate(zip(reference, candidate)):
        ref = [tuple(int(token) for token in row) for row in ref_rows]
        cand = [tuple(int(token) for token in row) for row in cand_rows]
        ref_set = set(ref)
        cand_set = set(cand)
        intersection = len(ref_set & cand_set)
        union = len(ref_set | cand_set)
        ordered_equal = ref == cand
        set_equal = ref_set == cand_set
        exact_order += ordered_equal
        exact_sets += set_equal
        intersections.append(intersection)
        jaccards.append(intersection / union if union else 1.0)
        row_overlaps = {}
        for cutoff in top_overlaps:
            overlap = len(set(ref[:cutoff]) & set(cand[:cutoff]))
            top_overlaps[cutoff].append(overlap)
            row_overlaps[str(cutoff)] = overlap
        per_request.append(
            {
                "request_index": index,
                "ordered_equal": ordered_equal,
                "set_equal": set_equal,
                "intersection": intersection,
                "jaccard": intersection / union if union else 1.0,
                "top_overlap": row_overlaps,
            }
        )

    summary = {
        "requests": len(reference),
        "beams": len(reference[0]) if reference else 0,
        "ordered_equal_requests": exact_order,
        "set_equal_requests": exact_sets,
        "mean_intersection": float(np.mean(intersections)),
        "minimum_intersection": int(min(intersections)),
        "mean_jaccard": float(np.mean(jaccards)),
        "minimum_jaccard": float(min(jaccards)),
        "top_overlap": {
            str(cutoff): {
                "mean": float(np.mean(values)),
                "minimum": int(min(values)),
            }
            for cutoff, values in top_overlaps.items()
        },
        "reference": str(args.reference),
        "candidate": str(args.candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "per_request": per_request}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
