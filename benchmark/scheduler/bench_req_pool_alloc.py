"""Microbenchmark: `ReqToTokenPool.alloc()` head-pop (old) vs tail-pop (new)
slot-selection cost.

The old implementation selected the head of `free_slots` and rebuilt the
list from the remainder (`self.free_slots = self.free_slots[need_size:]`),
which is O(len(free_slots) - need_size) -- proportional to how many free
slots *remain*, not to how many are being allocated (`need_size`). The new
implementation pops from the tail (`del self.free_slots[-need_size:]`),
which only touches the `need_size` removed elements.

Deviation from the original proposal's per-cell recipe: pre-filling each
cell so that `free_slots` holds exactly `batch_size` entries (as literally
described in the proposal doc) makes the old code's remainder-copy cost
~0 in every cell, since the remainder after removing `need_size` from a
`need_size`-long list is empty -- that setup can't actually surface the
bug. The regime that does (and that matches the doc's own problem
analysis in section 1) is a *large* free-slot cushion with a *small*
admission batch: `--max-running-requests` sized well above steady-state
concurrency, so most steps see thousands of free slots and admit only a
handful of new requests. This script sweeps `free_ratio` (free slots as a
fraction of pool size) independently of `batch_size` to reproduce that.

Usage:
    python benchmark/scheduler/bench_req_pool_alloc.py
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Callable

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.utils import get_device

POOL_SIZES = (256, 1024, 4096, 16384)
FREE_RATIOS = (0.1, 0.5, 0.9)
BATCH_SIZES = (1, 8, 32, 128)
ITERATIONS = 500
WARMUP = 50


def _mk_req() -> SimpleNamespace:
    return SimpleNamespace(
        req_pool_idx=None, inflight_middle_chunks=0, kv_committed_len=0
    )


def _alloc_head_pop(pool: ReqToTokenPool, reqs: list) -> list[int] | None:
    """Pre-fix behavior: pop from the front of `free_slots`."""
    reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
    need_size = len(reqs) - len(reusing)
    if need_size > len(pool.free_slots):
        return None
    select_index = pool.free_slots[:need_size]
    pool.free_slots = pool.free_slots[need_size:]
    offset = 0
    for r in reqs:
        if r.req_pool_idx is None:
            r.req_pool_idx = select_index[offset]
            pool.req_generation[r.req_pool_idx] += 1
            offset += 1
    return [r.req_pool_idx for r in reqs]


def _alloc_tail_pop(pool: ReqToTokenPool, reqs: list) -> list[int] | None:
    """Current (fixed) behavior: pool.alloc() itself."""
    return pool.alloc(reqs)


def _percentile(sorted_vals: list[float], p: float) -> float:
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def bench_one(
    pool_size: int,
    free_count: int,
    batch_size: int,
    alloc_fn: Callable,
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    """Steady state: `free_count` slots stay free across iterations --
    each iteration allocates `batch_size` reqs then immediately frees them
    back, so pool occupancy is constant and only `alloc()` is timed."""
    pool = ReqToTokenPool(
        size=pool_size,
        max_context_len=8,
        device=get_device(),
        enable_memory_saver=False,
    )
    filler = [_mk_req() for _ in range(pool_size - free_count)]
    alloc_fn(pool, filler)  # leaves exactly `free_count` slots free

    def run_once() -> float:
        reqs = [_mk_req() for _ in range(batch_size)]
        t0 = time.perf_counter_ns()
        alloc_fn(pool, reqs)
        elapsed = time.perf_counter_ns() - t0
        for r in reqs:
            pool.free(r)
        return elapsed

    for _ in range(warmup):
        run_once()
    samples = [run_once() for _ in range(iterations)]
    samples.sort()
    return {
        "mean_ns": sum(samples) / len(samples),
        "median_ns": _percentile(samples, 0.5),
        "p99_ns": _percentile(samples, 0.99),
    }


def print_row(label: str, old: dict[str, float], new: dict[str, float]) -> None:
    speedup = old["mean_ns"] / new["mean_ns"] if new["mean_ns"] > 0 else float("inf")
    print(
        f"{label:<32s}  "
        f"{old['mean_ns']:>10.0f} {old['median_ns']:>10.0f} {old['p99_ns']:>10.0f}   "
        f"{new['mean_ns']:>10.0f} {new['median_ns']:>10.0f} {new['p99_ns']:>10.0f}   "
        f"{speedup:>9.1f}x"
    )


def print_header() -> None:
    print(
        f"{'cell':<32s}  "
        f"{'old mean':>10s} {'old p50':>10s} {'old p99':>10s}   "
        f"{'new mean':>10s} {'new p50':>10s} {'new p99':>10s}   "
        f"{'speedup':>10s}"
    )
    print("-" * 110)


def main() -> None:
    print("All times in ns per alloc() call. 'old' = head-pop, 'new' = tail-pop.\n")

    print_header()
    for pool_size in POOL_SIZES:
        for free_ratio in FREE_RATIOS:
            free_count = max(1, int(pool_size * free_ratio))
            for batch_size in BATCH_SIZES:
                if batch_size >= free_count:
                    continue
                old = bench_one(
                    pool_size,
                    free_count,
                    batch_size,
                    _alloc_head_pop,
                    ITERATIONS,
                    WARMUP,
                )
                new = bench_one(
                    pool_size,
                    free_count,
                    batch_size,
                    _alloc_tail_pop,
                    ITERATIONS,
                    WARMUP,
                )
                label = f"pool={pool_size} free={free_count} batch={batch_size}"
                print_row(label, old, new)
    print()

    print("Pathological case: large pool, high free-slot cushion, tiny trickle batch")
    print_header()
    pool_size, free_ratio, batch_size = 16384, 0.9, 4
    free_count = int(pool_size * free_ratio)
    old = bench_one(
        pool_size, free_count, batch_size, _alloc_head_pop, ITERATIONS, WARMUP
    )
    new = bench_one(
        pool_size, free_count, batch_size, _alloc_tail_pop, ITERATIONS, WARMUP
    )
    label = f"pool={pool_size} free={free_count} batch={batch_size}"
    print_row(label, old, new)

    delta_ns = old["mean_ns"] - new["mean_ns"]
    cpu_seconds_per_million_steps = delta_ns * 1_000_000 / 1e9
    print(
        f"\nExtrapolated: {delta_ns:.0f} ns saved per alloc() call in this cell"
        f" -> {cpu_seconds_per_million_steps:.2f} CPU-seconds saved per million scheduler steps."
    )


if __name__ == "__main__":
    main()
