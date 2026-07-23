#!/usr/bin/env python3
"""Measure uncached short-extend latency after a shared token-ID prefix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def generate(
    *,
    url: str,
    input_ids: list[int],
    timeout: float,
    rid: str,
) -> float:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0,
        },
        "stream": False,
        "rid": rid,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    return (time.perf_counter() - started) * 1000.0


def parse_lengths(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Standalone SGLang /generate URL")
    parser.add_argument("--prefix-length", type=int, default=8192)
    parser.add_argument("--suffix-lengths", default="1,4,8,16,32,64")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--base-token-id", type=int, default=100)
    parser.add_argument("--unique-token-base", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = [args.base_token_id] * args.prefix_length

    # Populate only the common prefix. Every measured suffix starts with a
    # distinct token so previous trials cannot turn the suffix into a cache hit.
    generate(
        url=args.url,
        input_ids=prefix,
        timeout=args.timeout,
        rid="token-handoff-prefix-warmup",
    )

    results = []
    unique_offset = 0
    for suffix_length in parse_lengths(args.suffix_lengths):
        latencies = []
        for trial in range(args.trials):
            first_suffix_token = args.unique_token_base + unique_offset
            unique_offset += 1
            suffix = [first_suffix_token] + [args.base_token_id + 1] * max(
                suffix_length - 1, 0
            )
            latency_ms = generate(
                url=args.url,
                input_ids=prefix + suffix,
                timeout=args.timeout,
                rid=f"token-handoff-extend-{suffix_length}-{trial}",
            )
            latencies.append(latency_ms)
        results.append(
            {
                "suffix_tokens": suffix_length,
                "latency_ms": latencies,
                "median_ms": statistics.median(latencies),
                "p95_ms": percentile(latencies, 0.95),
            }
        )

    report = {
        "config": vars(args),
        "results": results,
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output)
            file.write("\n")


if __name__ == "__main__":
    main()
