"""Benchmark DecodeReqToTokenPool.alloc(): head-pop vs tail-pop free_slots.

Companion to the ReqToTokenPool.alloc() fix (python/sglang/srt/mem_cache/memory_pool.py):
DecodeReqToTokenPool.alloc() (python/sglang/srt/disaggregation/decode.py) has
the identical free_slots bookkeeping shape, used on the PD decode side to hand
out pre-allocated + transferring + running request slots. This compares the
old head-slicing implementation (`free_slots[:n]` / `free_slots[n:]`, O(len(free_slots)))
against the current tail-pop implementation (O(need_size)) across a sweep of
pool sizes and free-slot ratios, independent of batch size.

Usage:
    python benchmark/scheduler/bench_decode_req_pool_alloc.py
"""

from __future__ import annotations

import time
from statistics import mean, quantiles

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool


class _FakeReq:
    def __init__(self):
        self.req_pool_idx = None
        self.inflight_middle_chunks = 0
        self.kv_committed_len = 0


def _alloc_head_pop(pool: DecodeReqToTokenPool, need_size: int):
    """Pre-fix behavior: O(len(free_slots)) regardless of need_size."""
    select_index = pool.free_slots[:need_size]
    pool.free_slots = pool.free_slots[need_size:]
    return select_index


def _alloc_tail_pop(pool: DecodeReqToTokenPool, need_size: int):
    """Current behavior: O(need_size)."""
    if need_size == 0:
        return []
    select_index = pool.free_slots[-need_size:]
    del pool.free_slots[-need_size:]
    return select_index


def _make_pool(pool_size: int, n_free: int) -> DecodeReqToTokenPool:
    pool = DecodeReqToTokenPool(
        size=pool_size,
        max_context_len=8,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=0,
    )
    # free_slots is 1..pool_size; trim to leave exactly n_free free.
    pool.free_slots = pool.free_slots[:n_free]
    return pool


def _time_alloc_calls(
    alloc_fn, pool_size: int, n_free: int, batch: int, iterations: int
) -> list[float]:
    """Time `iterations` calls to alloc_fn, each on a freshly reset pool
    with exactly n_free free slots, allocating `batch` slots. Returns
    per-call latencies in nanoseconds.
    """
    latencies = []
    for _ in range(iterations):
        pool = _make_pool(pool_size, n_free)
        t0 = time.perf_counter_ns()
        alloc_fn(pool, batch)
        latencies.append(time.perf_counter_ns() - t0)
    return latencies


def _stats(latencies_ns: list[float]) -> tuple[float, float, float]:
    qs = quantiles(latencies_ns, n=100)
    return mean(latencies_ns), qs[49], qs[98]


def main() -> None:
    pool_sizes = (256, 1024, 4096, 16384)
    free_ratios = (0.1, 0.5, 0.9)
    batches = (1, 4)
    iterations = 2000

    header = (
        f"{'cell':<38s}{'old mean':>10s}{'old p50':>10s}{'old p99':>10s}"
        f"{'new mean':>12s}{'new p50':>10s}{'new p99':>10s}{'speedup':>10s}"
    )
    print(header)
    print("-" * len(header))

    for pool_size in pool_sizes:
        for ratio in free_ratios:
            n_free = int(pool_size * ratio)
            if n_free == 0:
                continue
            for batch in batches:
                if batch > n_free:
                    continue
                old = _stats(
                    _time_alloc_calls(
                        _alloc_head_pop, pool_size, n_free, batch, iterations
                    )
                )
                new = _stats(
                    _time_alloc_calls(
                        _alloc_tail_pop, pool_size, n_free, batch, iterations
                    )
                )
                cell = f"pool={pool_size} free={n_free} batch={batch}"
                speedup = old[0] / new[0] if new[0] > 0 else float("inf")
                print(
                    f"{cell:<38s}{old[0]:>10.0f}{old[1]:>10.0f}{old[2]:>10.0f}"
                    f"{new[0]:>12.0f}{new[1]:>10.0f}{new[2]:>10.0f}{speedup:>9.1f}x"
                )


if __name__ == "__main__":
    main()
