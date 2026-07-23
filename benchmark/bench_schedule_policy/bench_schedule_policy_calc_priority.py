"""Microbenchmark for `SchedulePolicy.calc_priority` host-side cost.

Measures the full scheduler-tick cost of cache-aware (LPM) priority
computation as the waiting queue depth and prompt length vary. The
optimization in `_compute_prefix_matches` removes an unconditional
`origin_input_ids + output_ids` list copy when `output_ids` is empty.

This bench drives the public `SchedulePolicy.calc_priority` API end to end
so it captures every Python overhead a real scheduler tick pays:

- per-req `_prefix_match_token_ids` (the optimization site);
- per-req `tree_cache.match_prefix(...)`;
- per-req in-batch radix tree match + insert;
- final `_sort_by_longest_prefix`.

GPU is not needed: `RadixCache.create_simulated()` provides a CPU-only
prefix tree and the priority computation never touches a model.

Run:

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=python \\
        python3 benchmark/bench_schedule_policy/bench_schedule_policy_calc_priority.py
"""

from __future__ import annotations

import statistics
import time
from typing import List

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams


def make_waiting_queue(
    queue_size: int, prompt_len: int, retracted_fraction: float = 0.0
) -> List[Req]:
    """Build a synthetic waiting queue with the requested shape.

    `retracted_fraction` of the requests carry a non-empty `output_ids`,
    simulating retracted re-admissions (which still pay the concat cost).
    """
    reqs = []
    n_retracted = int(round(queue_size * retracted_fraction))
    for i in range(queue_size):
        # Use distinct token ranges per req so radix tree inserts diverge.
        prompt = list(range(i * prompt_len, (i + 1) * prompt_len))
        req = Req(i, "x", prompt, SamplingParams())
        if i < n_retracted:
            # Pretend this req has generated 4 tokens before being retracted.
            req.output_ids = [99000 + k for k in range(4)]
        reqs.append(req)
    return reqs


def run_one_cycle(policy: SchedulePolicy, queue: List[Req]) -> None:
    """One scheduler-tick worth of cache-aware priority computation."""
    policy.calc_priority(queue)


def time_cycles(
    queue_size: int,
    prompt_len: int,
    retracted_fraction: float,
    repeats: int = 50,
) -> List[float]:
    """Time `repeats` independent scheduling cycles. Returns elapsed ms."""
    samples: List[float] = []
    for _ in range(repeats):
        # Fresh tree + queue each iteration: real schedulers see new work
        # every tick and the in-batch tree is reset at the top of
        # `_compute_prefix_matches` anyway.
        tree_cache = RadixCache.create_simulated()
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        queue = make_waiting_queue(queue_size, prompt_len, retracted_fraction)
        t0 = time.perf_counter()
        run_one_cycle(policy, queue)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def fmt_row(label: str, samples: List[float]) -> str:
    median = statistics.median(samples)
    p90 = sorted(samples)[int(0.9 * (len(samples) - 1))]
    return f"  {label:<32} median={median:7.3f} ms   p90={p90:7.3f} ms"


def bench_grid() -> None:
    grids = [
        (50, 1024, 0.0),
        (50, 4096, 0.0),
        (50, 16384, 0.0),
        (50, 32768, 0.0),
        (100, 8192, 0.0),
        (100, 32768, 0.0),
        # Retracted-mix: half the queue carries output_ids, so the helper
        # falls back to the original concat for those reqs. Expected to be
        # at parity with the old code.
        (50, 16384, 0.5),
    ]
    print("`SchedulePolicy.calc_priority` (LPM) host-side cost")
    print("=" * 70)
    for queue_size, prompt_len, retract_frac in grids:
        label = (
            f"queue={queue_size}, prompt={prompt_len:>5}, "
            f"retracted={int(retract_frac * 100):>3}%"
        )
        samples = time_cycles(queue_size, prompt_len, retract_frac)
        print(fmt_row(label, samples))


if __name__ == "__main__":
    bench_grid()
