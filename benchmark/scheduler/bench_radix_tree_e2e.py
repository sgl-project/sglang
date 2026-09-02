"""Benchmark end-to-end latency through a deliberately deep radix-tree path.

The benchmark first caches one long token sequence, then inserts one-token
branches at regular prefix boundaries.  Those branches split the compressed
path into ``--depth`` nodes without requiring repeated long prefills.  Measured
requests reuse the full path, so they exercise the scheduler's radix traversal
alongside a real model decode.

Example:
    python benchmark/scheduler/bench_radix_tree_e2e.py \
        --base-url http://127.0.0.1:30000 --total-tokens 65536 --depth 128 \
        --branch-tokens 128
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import requests


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _request(
    base_url: str,
    input_ids: list[int],
    timeout: float,
    *,
    max_new_tokens: int = 1,
) -> tuple[float, float, dict]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
    }
    start = time.perf_counter()
    response = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
    client_latency_ms = (time.perf_counter() - start) * 1_000
    response.raise_for_status()
    body = response.json()
    meta_info = body["meta_info"]
    server_latency_ms = float(meta_info["e2e_latency"]) * 1_000
    return server_latency_ms, client_latency_ms, meta_info


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--total-tokens", type=int, default=65_536)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument(
        "--measurement-output-len",
        type=int,
        default=1,
        help="Generated tokens per measured request; use 0 to isolate matching.",
    )
    parser.add_argument("--base-token-id", type=int, default=1_000)
    parser.add_argument("--branch-token-id", type=int, default=1_001)
    parser.add_argument(
        "--branch-tokens",
        type=int,
        default=1,
        help=(
            "Tokens appended per branch. Because SGLang retains the trailing "
            "cache page, use at least twice the server page size."
        ),
    )
    parser.add_argument("--timeout", type=float, default=1_200)
    args = parser.parse_args()

    if args.total_tokens <= 0 or args.depth <= 0:
        parser.error("--total-tokens and --depth must be positive")
    if args.total_tokens % args.depth:
        parser.error("--total-tokens must be divisible by --depth")
    if args.base_token_id == args.branch_token_id:
        parser.error("base and branch token ids must differ")
    if args.branch_tokens <= 0:
        parser.error("--branch-tokens must be positive")
    if args.measurement_output_len < 0:
        parser.error("--measurement-output-len must be non-negative")

    base_url = args.base_url.rstrip("/")
    response = requests.post(f"{base_url}/flush_cache", timeout=args.timeout)
    response.raise_for_status()

    base_tokens = [args.base_token_id] * args.total_tokens
    _request(base_url, base_tokens, args.timeout)

    segment_tokens = args.total_tokens // args.depth
    for boundary in range(segment_tokens, args.total_tokens, segment_tokens):
        branch = base_tokens[:boundary] + [args.branch_token_id] * args.branch_tokens
        _request(base_url, branch, args.timeout)

    for _ in range(args.warmup_requests):
        _request(
            base_url,
            base_tokens,
            args.timeout,
            max_new_tokens=args.measurement_output_len,
        )

    server_samples = []
    client_samples = []
    last_meta_info = {}
    for _ in range(args.num_requests):
        server_ms, client_ms, last_meta_info = _request(
            base_url,
            base_tokens,
            args.timeout,
            max_new_tokens=args.measurement_output_len,
        )
        server_samples.append(server_ms)
        client_samples.append(client_ms)

    result = {
        "total_tokens": args.total_tokens,
        "depth": args.depth,
        "segment_tokens": segment_tokens,
        "branch_tokens": args.branch_tokens,
        "warmup_requests": args.warmup_requests,
        "num_requests": args.num_requests,
        "measurement_output_len": args.measurement_output_len,
        "server_e2e": _summarize(server_samples),
        "client_e2e": _summarize(client_samples),
        "last_prompt_tokens": last_meta_info.get("prompt_tokens"),
        "last_cached_tokens": last_meta_info.get("cached_tokens"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
