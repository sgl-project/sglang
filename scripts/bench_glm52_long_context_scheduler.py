#!/usr/bin/env python3
"""Benchmark for cost-aware chunked-prefill scheduler.

Replays deterministic synthetic traces to validate controller mathematics.
All cost constants are SYNTHETIC — they do not represent GLM-5.2 measurements.

Usage:
    python scripts/bench_glm52_long_context_scheduler.py
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Add the python source to path
sys.path.insert(0, "python")

from sglang.srt.managers.scheduler_components.iteration_cost_estimator import (
    IterationCostEstimator,
)


@dataclass
class SyntheticRequest:
    req_id: int
    prefill_tokens: int
    prefix_cache_hit: int
    max_new_tokens: int
    arrival_iteration: int = 0


@dataclass
class IterationResult:
    iteration: int
    is_decode: bool
    decode_bs: int
    num_prefill_tokens: int
    chunk_size_used: int
    iteration_ms: float
    prefill_wait_count: int


@dataclass
class BenchmarkConfig:
    base_chunk_size: int = 2048
    max_prefill_tokens: int = 16384
    page_size: int = 1
    max_iterations: int = 500
    # SYNTHETIC cost model — NOT calibrated to any real hardware.
    # Used only to exercise controller mathematics.
    prefill_ms_per_token: float = 0.05
    decode_ms_per_token: float = 0.15
    decode_base_ms: float = 2.0


def generate_traces() -> dict:
    traces = {}
    traces["one_150k_prefill"] = [
        SyntheticRequest(0, 150000, 0, 256, 0),
    ]
    traces["150k_prefill_plus_decode"] = [
        SyntheticRequest(0, 150000, 0, 1024, 0),
        SyntheticRequest(1, 1, 0, 1024, 5),
    ]
    traces["four_long_plus_16_decode"] = [
        SyntheticRequest(i, 128000, 0, 256, 0) for i in range(4)
    ] + [
        SyntheticRequest(i + 4, 1, 0, 256, 10 + i * 2) for i in range(16)
    ]
    traces["prefix_hit_plus_decode"] = [
        SyntheticRequest(0, 50000, 45000, 256, 0),
        SyntheticRequest(1, 1, 0, 256, 3),
        SyntheticRequest(2, 1, 0, 256, 3),
        SyntheticRequest(3, 1, 0, 256, 3),
    ]
    traces["continuous_long_prompts"] = [
        SyntheticRequest(i, 64000 + i * 1000, 0, 256, i * 8)
        for i in range(20)
    ]
    return traces


def simulate_iteration(
    config: BenchmarkConfig,
    chunk_size: int,
    decode_bs: int,
    prefill_tokens_remaining: int,
    cached_ctx_len: int = 0,
) -> Tuple[float, int]:
    """Simulate one iteration. Returns (wall_ms, actual_prefill).

    NOTE: iteration_ms is the wall-clock latency of the iteration.
    TPOT (per-request inter-token latency) ~= iteration_ms for decode,
    since each decode request produces one token per iteration.
    Do NOT divide iteration_ms by decode_bs to get TPOT.
    """
    actual_prefill = min(chunk_size, prefill_tokens_remaining)
    prefill_cost = actual_prefill * config.prefill_ms_per_token
    if cached_ctx_len > 0:
        prefill_cost += actual_prefill * 0.001 * math.sqrt(cached_ctx_len)

    if decode_bs > 0:
        decode_cost = config.decode_base_ms + decode_bs * config.decode_ms_per_token
        if actual_prefill > 0:
            # Mixed batch: decode waits for prefill (serial in practice)
            wall_ms = prefill_cost + decode_cost
        else:
            wall_ms = decode_cost
    else:
        wall_ms = prefill_cost
    return wall_ms, actual_prefill


def run_trace(
    trace_name: str,
    requests: List[SyntheticRequest],
    config: BenchmarkConfig,
    use_cost_aware: bool,
) -> List[IterationResult]:
    if use_cost_aware:
        estimator = IterationCostEstimator(
            max_slowdown_ratio=1.5,
            min_chunk_ratio=0.25,
            max_prefill_wait_iters=64,
        )
        estimator.enable()
    else:
        estimator = None

    results = []
    iteration = 0
    decode_active = []
    prefill_remaining = {}
    prefill_total = {}

    for req in requests:
        prefill_remaining[req.req_id] = req.prefill_tokens
        prefill_total[req.req_id] = req.prefill_tokens
        if req.prefill_tokens <= 1:
            decode_active.append(req.req_id)
            prefill_remaining[req.req_id] = 0

    pending = list(requests)
    arrived = []

    while iteration < config.max_iterations:
        newly_arrived = [r for r in pending if r.arrival_iteration <= iteration]
        for r in newly_arrived:
            arrived.append(r)
            pending.remove(r)
            if r.prefill_tokens <= 1:
                if r.req_id not in decode_active:
                    decode_active.append(r.req_id)
                prefill_remaining[r.req_id] = 0

        decode_bs = len(decode_active)
        base_chunk = config.base_chunk_size

        has_decode_work = decode_bs > 0

        if estimator and estimator.enabled and base_chunk is not None:
            chunk_size = estimator.choose_prefill_chunk_size(
                base_chunk_size=base_chunk,
                has_decode_work=has_decode_work,
                max_chunk_size=config.max_prefill_tokens,
                alignment=config.page_size,
            )
        else:
            chunk_size = base_chunk

        prefill_req = None
        for r in arrived:
            if prefill_remaining.get(r.req_id, 0) > 0:
                prefill_req = r
                break

        if prefill_req is None and decode_bs == 0:
            break

        prefill_tokens_this_iter = 0
        if prefill_req is not None:
            cached_ctx = prefill_total[prefill_req.req_id] - prefill_remaining[prefill_req.req_id]
            wall_ms, actual_prefill = simulate_iteration(
                config, chunk_size, decode_bs,
                prefill_remaining[prefill_req.req_id], cached_ctx,
            )
            prefill_remaining[prefill_req.req_id] -= actual_prefill
            prefill_tokens_this_iter = actual_prefill
            if prefill_remaining[prefill_req.req_id] <= 0:
                if prefill_req.req_id not in decode_active:
                    decode_active.append(prefill_req.req_id)
        else:
            wall_ms, _ = simulate_iteration(
                config, 0, decode_bs, 0,
            )

        if estimator and estimator.enabled:
            if prefill_tokens_this_iter > 0 and decode_bs > 0:
                batch_type = "mixed"
            elif decode_bs > 0:
                batch_type = "decode"
            else:
                batch_type = "prefill"

            estimator.update_observation(
                batch_type=batch_type,
                iteration_ms=wall_ms,
                num_prefill_tokens=prefill_tokens_this_iter,
            )

        results.append(IterationResult(
            iteration=iteration,
            is_decode=(prefill_req is None and decode_bs > 0),
            decode_bs=decode_bs,
            num_prefill_tokens=prefill_tokens_this_iter,
            chunk_size_used=chunk_size,
            iteration_ms=wall_ms,
            prefill_wait_count=estimator._prefill_wait_count if estimator else 0,
        ))
        iteration += 1

    return results


def analyze_results(trace_name, baseline, optimized):
    print(f"\n{'='*80}")
    print(f"Trace: {trace_name}")
    print(f"{'='*80}")

    baseline_decode = [r for r in baseline if r.is_decode]
    opt_decode = [r for r in optimized if r.is_decode]

    if baseline_decode and opt_decode:
        baseline_avg_iter = sum(r.iteration_ms for r in baseline_decode) / len(baseline_decode)
        opt_avg_iter = sum(r.iteration_ms for r in opt_decode) / len(opt_decode)
        baseline_max_iter = max(r.iteration_ms for r in baseline_decode)
        opt_max_iter = max(r.iteration_ms for r in opt_decode)
    else:
        baseline_avg_iter = opt_avg_iter = 0
        baseline_max_iter = opt_max_iter = 0

    baseline_total_iters = len(baseline)
    opt_total_iters = len(optimized)
    baseline_max_wait = max((r.prefill_wait_count for r in baseline), default=0)
    opt_max_wait = max((r.prefill_wait_count for r in optimized), default=0)

    print(f"  Total iterations:          baseline={baseline_total_iters}, optimized={opt_total_iters}")
    print(f"  Decode iterations:         baseline={len(baseline_decode)}, optimized={len(opt_decode)}")
    print(f"  Avg decode iter ms:        baseline={baseline_avg_iter:.2f}, optimized={opt_avg_iter:.2f}")
    if baseline_avg_iter > 0:
        print(f"  Iter latency improvement:  {(1 - opt_avg_iter/baseline_avg_iter)*100:.1f}%")
    print(f"  Max decode iter ms:        baseline={baseline_max_iter:.2f}, optimized={opt_max_iter:.2f}")
    print(f"  Max prefill wait count:    baseline={baseline_max_wait}, optimized={opt_max_wait}")

    if optimized:
        chunk_sizes = set(r.chunk_size_used for r in optimized if not r.is_decode)
        print(f"  Chunk sizes used (opt):    {chunk_sizes}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark cost-aware chunked-prefill scheduler")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Base chunked prefill size")
    args = parser.parse_args()

    config = BenchmarkConfig(base_chunk_size=args.chunk_size)

    traces = generate_traces()
    print(f"Cost-Aware Chunked-Prefill Scheduler Benchmark (SYNTHETIC cost model)")
    print(f"Base chunk size: {config.base_chunk_size}")

    for trace_name, requests in traces.items():
        baseline = run_trace(trace_name, requests, config, use_cost_aware=False)
        optimized = run_trace(trace_name, requests, config, use_cost_aware=True)
        analyze_results(trace_name, baseline, optimized)

    print(f"\n{'='*80}")
    print("Benchmark complete. NOTE: All costs are SYNTHETIC, not GLM-5.2 measured.")


if __name__ == "__main__":
    main()
