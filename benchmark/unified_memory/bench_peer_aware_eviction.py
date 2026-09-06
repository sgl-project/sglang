"""Reproduce and measure peer-aware eviction in the unified memory pool.

The workload deliberately fills the shared FULL/Mamba pool with many small,
reusable prefixes.  It then submits one long pressure request and probes every
prefix again.  Keeping more probe prefixes cached demonstrates that allocator
capacity gained from a peer component stopped radix eviction early.

Example:

    python benchmark/unified_memory/bench_peer_aware_eviction.py \
        --label proposed --output /tmp/proposed.json

Concurrent fan-out from a prefix retained only by the proposed path:

    python benchmark/unified_memory/bench_peer_aware_eviction.py \
        --model-path Qwen/Qwen3.5-4B \
        --probe-output-len 32 --probe-concurrency 6 \
        --burst-target-index 14 --burst-requests 30 \
        --label proposed-concurrent --output /tmp/proposed-concurrent.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return values[round((len(values) - 1) * fraction)]


def generate(base_url: str, text: str, max_new_tokens: int) -> dict:
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": text,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    body = response.json()
    meta = body["meta_info"]
    prefill_finished_time = meta.get("prefill_finished_time")
    forward_entry_time = meta.get("forward_entry_time")
    return {
        "prompt_tokens": meta["prompt_tokens"],
        "cached_tokens": meta["cached_tokens"],
        "e2e_latency_s": meta["e2e_latency"],
        "client_latency_s": elapsed,
        "prefill_latency_s": (
            prefill_finished_time - forward_entry_time
            if prefill_finished_time is not None and forward_entry_time is not None
            else None
        ),
        "completion_tokens": meta["completion_tokens"],
        "num_retractions": meta["num_retractions"],
        "output_ids": body["output_ids"],
    }


def text_with_target_tokens(tokenizer, seed: str, target: int) -> str:
    """Create deterministic text whose tokenized length is close to ``target``."""
    repeated = (seed + " ") * target
    token_ids = tokenizer.encode(repeated, add_special_tokens=False)[:target]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def summarize_probe(probes: list[dict], batch_wall_latency_s: float) -> dict[str, Any]:
    cached = [item["cached_tokens"] for item in probes]
    e2e = [item["e2e_latency_s"] for item in probes]
    client = [item["client_latency_s"] for item in probes]
    prefill = [
        item["prefill_latency_s"]
        for item in probes
        if item["prefill_latency_s"] is not None
    ]
    completion_tokens = sum(item["completion_tokens"] for item in probes)
    return {
        "cached_prefixes": sum(value > 0 for value in cached),
        "cache_survival_rate": sum(value > 0 for value in cached) / len(cached),
        "total_cached_tokens": sum(cached),
        "mean_cached_tokens": statistics.mean(cached),
        "cached_tokens": cached,
        "mean_e2e_latency_s": statistics.mean(e2e),
        "p95_e2e_latency_s": percentile(e2e, 0.95),
        "mean_client_latency_s": statistics.mean(client),
        "p95_client_latency_s": percentile(client, 0.95),
        "mean_prefill_latency_s": statistics.mean(prefill) if prefill else None,
        "p95_prefill_latency_s": percentile(prefill, 0.95) if prefill else None,
        "batch_wall_latency_s": batch_wall_latency_s,
        "request_throughput_rps": len(probes) / batch_wall_latency_s,
        "output_throughput_tps": completion_tokens / batch_wall_latency_s,
        "completion_tokens": completion_tokens,
        "total_retractions": sum(item["num_retractions"] for item in probes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-prefixes", type=int, default=28)
    parser.add_argument("--prefix-len", type=int, default=400)
    parser.add_argument("--pressure-len", type=int, default=7000)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--probe-output-len", type=int)
    parser.add_argument("--probe-concurrency", type=int, default=1)
    parser.add_argument("--burst-target-index", type=int)
    parser.add_argument("--burst-requests", type=int, default=30)
    args = parser.parse_args()
    if args.probe_concurrency < 1:
        parser.error("--probe-concurrency must be at least 1")
    if args.burst_requests < 1:
        parser.error("--burst-requests must be at least 1")
    if args.burst_target_index is not None and not (
        0 <= args.burst_target_index < args.warm_prefixes
    ):
        parser.error("--burst-target-index must select a warm prefix")
    probe_output_len = args.probe_output_len or args.output_len
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    prompt_pairs = []
    prompt_bases = []
    for index in range(args.warm_prefixes):
        base = text_with_target_tokens(
            tokenizer,
            f"stable reusable unified cache prefix group {index}",
            args.prefix_len - 16,
        )
        prompt_bases.append(base)
        prompt_pairs.append(
            (
                base + f" warm suffix for group {index}",
                base + f" replay suffix for group {index}",
            )
        )
    pressure_text = text_with_target_tokens(
        tokenizer, "distinct long allocation pressure payload", args.pressure_len
    )

    requests.post(f"{args.base_url}/flush_cache", timeout=60).raise_for_status()

    # Exclude server startup and first-request kernel initialization.
    generate(args.base_url, "server kernel warmup " * 32, args.output_len)
    requests.post(f"{args.base_url}/flush_cache", timeout=60).raise_for_status()

    warm = []
    for warm_text, _ in prompt_pairs:
        warm.append(
            generate(
                args.base_url,
                warm_text,
                args.output_len,
            )
        )

    # A distinct long request forces the FULL side toward the Mamba frontier.
    pressure = generate(
        args.base_url,
        pressure_text,
        args.output_len,
    )

    if args.burst_target_index is None:
        probe_indices = list(reversed(range(len(prompt_pairs))))
        probe_texts = [prompt_pairs[index][1] for index in probe_indices]
    else:
        probe_indices = [args.burst_target_index] * args.burst_requests
        target_base = prompt_bases[args.burst_target_index]
        probe_texts = [
            target_base + f" concurrent burst replay suffix request {ordinal}"
            for ordinal in range(args.burst_requests)
        ]

    def run_probe(replay_text: str) -> dict:
        return generate(
            args.base_url,
            replay_text,
            probe_output_len,
        )

    probe_start = time.perf_counter()
    if args.probe_concurrency == 1:
        probes = [run_probe(replay_text) for replay_text in probe_texts]
    else:
        with ThreadPoolExecutor(max_workers=args.probe_concurrency) as executor:
            probes = list(executor.map(run_probe, probe_texts))
    probe_wall_latency_s = time.perf_counter() - probe_start

    result = {
        "label": args.label,
        "config": {
            "warm_prefixes": args.warm_prefixes,
            "prefix_len": args.prefix_len,
            "pressure_len": args.pressure_len,
            "output_len": args.output_len,
            "probe_output_len": probe_output_len,
            "probe_concurrency": args.probe_concurrency,
            "burst_target_index": args.burst_target_index,
            "burst_requests": (
                args.burst_requests if args.burst_target_index is not None else None
            ),
            "actual_warm_prompt_tokens": [item["prompt_tokens"] for item in warm],
        },
        "warm": {
            "total_cached_tokens": sum(item["cached_tokens"] for item in warm),
            "total_retractions": sum(item["num_retractions"] for item in warm),
        },
        "warm_requests": warm,
        "pressure": pressure,
        "probe": summarize_probe(probes, probe_wall_latency_s),
        "probe_requests": probes,
        "probe_indices": probe_indices,
        "output_ids": {
            "warm": [item["output_ids"] for item in warm],
            "pressure": pressure["output_ids"],
            "probe": [item["output_ids"] for item in probes],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["probe"], indent=2))


if __name__ == "__main__":
    main()
