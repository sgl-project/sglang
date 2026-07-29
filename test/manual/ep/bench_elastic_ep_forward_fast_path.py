"""Microbenchmark for the Elastic-EP forward fast-path sync removal.

Measures the wall-clock cost of the fast-path check in
`maybe_recover_ep_ranks`, comparing the old two-tensor check (which forced a
host-device sync via `active_ranks.all()`) against the new CPU-only check.

Usage (single GPU is enough, tested on H20):

    python test/manual/ep/bench_elastic_ep_forward_fast_path.py \\
        --iters 5000 --warmup 500 --queue-depth 32 --world-size 8
"""

from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

import torch


def _queue_background_work(device: torch.device, matrix_dim: int, iters: int):
    """Fill the compute stream with pending work so a d2h sync is expensive."""
    x = torch.randn(matrix_dim, matrix_dim, device=device)
    for _ in range(iters):
        x = x @ x
    return x  # return so the compiler cannot elide it


def _old_check(tp_group) -> bool:
    """Reproduces the pre-optimization check: BOTH tensors, GPU `.all()` forces sync."""
    return bool(tp_group.active_ranks.all() and tp_group.active_ranks_cpu.all())


def _new_check(tp_group) -> bool:
    """Post-optimization check: CPU tensor only."""
    return bool(tp_group.active_ranks_cpu.all())


def _bench(fn, tp_group, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn(tp_group)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn(tp_group)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e3)  # microseconds
    return samples


def _summarize(name: str, samples: list[float]) -> None:
    samples_sorted = sorted(samples)
    mean = statistics.fmean(samples)
    median = samples_sorted[len(samples_sorted) // 2]
    p95 = samples_sorted[int(0.95 * len(samples_sorted))]
    p99 = samples_sorted[int(0.99 * len(samples_sorted))]
    peak = samples_sorted[-1]
    print(
        f"{name:<12s}  n={len(samples):>6d}  "
        f"mean={mean:8.2f}us  median={median:8.2f}us  "
        f"p95={p95:8.2f}us  p99={p99:8.2f}us  peak={peak:9.2f}us"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument(
        "--queue-depth",
        type=int,
        default=32,
        help="Number of matmul iters queued on the compute stream between checks.",
    )
    p.add_argument(
        "--matrix-dim",
        type=int,
        default=4096,
        help="Matrix dim for the background matmul; larger => longer sync stall.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    tp_group = SimpleNamespace(
        active_ranks=torch.ones(args.world_size, dtype=torch.int32, device=device),
        active_ranks_cpu=torch.ones(args.world_size, dtype=torch.int32),
    )

    print(
        f"[cfg] world_size={args.world_size} iters={args.iters} "
        f"warmup={args.warmup} queue_depth={args.queue_depth} "
        f"matrix_dim={args.matrix_dim}"
    )

    def _run_bench(fn, label):
        # Reset the stream state and enqueue background work each call so the
        # measured op has something to sync against.
        samples = []
        for _ in range(args.warmup):
            _queue_background_work(device, args.matrix_dim, args.queue_depth)
            fn(tp_group)
        for _ in range(args.iters):
            _queue_background_work(device, args.matrix_dim, args.queue_depth)
            t0 = time.perf_counter_ns()
            fn(tp_group)
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) / 1e3)
        _summarize(label, samples)

    torch.cuda.synchronize()
    _run_bench(_old_check, "old (sync)")
    torch.cuda.synchronize()
    _run_bench(_new_check, "new (cpu)")


if __name__ == "__main__":
    main()
