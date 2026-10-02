"""Attribute GPU kernels and D2H events using explicit capture CPU scopes.

CUDA correlation IDs connect device activity to a runtime/driver call inside
the named CPU range. Summed device durations are work, not serving wall time.
"""

import argparse
import bisect
import gzip
import hashlib
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path


def summarize(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    scopes = defaultdict(list)
    groups = {}
    for event in events:
        name = event.get("name", "")
        if (
            event.get("ph") == "X"
            and event.get("cat") == "user_annotation"
            and name.startswith("training_capture.")
        ):
            scopes[event["pid"], event["tid"]].append(event)
            group = groups.setdefault(
                name,
                {
                    "cpu_calls": 0,
                    "cpu_scope_us": 0.0,
                    "gpu_kernel_calls": 0,
                    "gpu_kernel_us": 0.0,
                    "d2h_calls": 0,
                    "d2h_us": 0.0,
                    "d2h_bytes": 0,
                    "d2h_events_missing_bytes": 0,
                    "d2d_calls": 0,
                    "d2d_us": 0.0,
                    "d2d_bytes": 0,
                    "d2d_events_missing_bytes": 0,
                    "kernels": {},
                },
            )
            group["cpu_calls"] += 1
            group["cpu_scope_us"] += event["dur"]
    indices = {}
    for thread, ranges in scopes.items():
        ranges.sort(key=lambda event: event["ts"])
        if any(
            left["ts"] + left["dur"] > right["ts"] for left, right in pairwise(ranges)
        ):
            raise ValueError("Capture scopes must be disjoint on each CPU thread")
        indices[thread] = ([event["ts"] for event in ranges], ranges)
    correlations = {}
    for event in events:
        if event.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        index = indices.get((event["pid"], event["tid"]))
        if index is None:
            continue
        position = bisect.bisect_right(index[0], event["ts"]) - 1
        if position < 0:
            continue
        scope = index[1][position]
        if event["ts"] + event.get("dur", 0) > scope["ts"] + scope["dur"]:
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            previous = correlations.setdefault(correlation, scope["name"])
            if previous != scope["name"]:
                raise ValueError("CUDA correlation ID maps to different capture scopes")
    for event in events:
        if event.get("ph") != "X":
            continue
        group_name = correlations.get(event.get("args", {}).get("correlation"))
        if group_name is None:
            continue
        group = groups[group_name]
        if event.get("cat") == "kernel":
            group["gpu_kernel_calls"] += 1
            group["gpu_kernel_us"] += event["dur"]
            kernel = group["kernels"].setdefault(event["name"], {"calls": 0, "us": 0.0})
            kernel["calls"] += 1
            kernel["us"] += event["dur"]
        elif event.get("cat") == "gpu_memcpy" and event["name"].startswith(
            ("Memcpy DtoH", "Memcpy DtoD")
        ):
            kind = "d2h" if event["name"].startswith("Memcpy DtoH") else "d2d"
            group[kind + "_calls"] += 1
            group[kind + "_us"] += event["dur"]
            size = event["args"].get("bytes")
            if not isinstance(size, (int, float)) or size < 0:
                group[kind + "_events_missing_bytes"] += 1
            else:
                group[kind + "_bytes"] += size
    for group in groups.values():
        for kind in ("d2h", "d2d"):
            if group[kind + "_events_missing_bytes"]:
                group[kind + "_bytes"] = None
    return {
        "trace": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "attribution": "CUDA correlation -> runtime/driver call contained in disjoint capture CPU scope",
        "groups": groups,
        "all_kernel_us": sum(
            event["dur"]
            for event in events
            if event.get("cat") == "kernel" and event.get("ph") == "X"
        ),
        "all_d2h_us": sum(
            event["dur"]
            for event in events
            if event.get("cat") == "gpu_memcpy"
            and event.get("ph") == "X"
            and event["name"].startswith("Memcpy DtoH")
        ),
        "all_d2d_us": sum(
            event["dur"]
            for event in events
            if event.get("cat") == "gpu_memcpy"
            and event.get("ph") == "X"
            and event["name"].startswith("Memcpy DtoD")
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-capture", action="store_true")
    args = parser.parse_args()
    results = [summarize(path) for path in args.traces]
    if args.require_capture:
        for result in results:
            groups = result["groups"]
            if (
                not groups
                or not any(group["gpu_kernel_calls"] for group in groups.values())
                or not any(group["d2h_calls"] for group in groups.values())
            ):
                raise RuntimeError(
                    f"Missing attributed capture activity: {result['trace']}"
                )
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
