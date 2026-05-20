"""Cycle-sweep benchmark for the kv-canary spin-wait jitter kernel.

Two axes: ``cycles`` (the spin target) and ``slots`` (how many sequential single-slot launches per
sample). ``slots=4`` matches the canary runner's per-step jitter cost (4 fixed in-graph slots).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, run_benchmark
from sglang.jit_kernel.kv_canary.jitter import spin_wait_step
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)


_CYCLE_VALUES: Tuple[int, ...] = (
    0,
    1_000,
    10_000,
    100_000,
    1_000_000,
)
_SLOT_COUNTS: Tuple[int, ...] = (1, 4)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["cycles", "slots"],
        x_vals=[(c, s) for c in _CYCLE_VALUES for s in _SLOT_COUNTS],
        line_arg="provider",
        line_vals=["spin_wait"],
        line_names=["spin_wait_step"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="kv-canary-jitter-perf",
        args={},
    )
)
def benchmark(cycles: int, slots: int, provider: str) -> Tuple[float, float, float]:
    device = torch.device(DEFAULT_DEVICE)
    cycle_bufs = [
        torch.tensor([cycles], dtype=torch.int64, device=device) for _ in range(slots)
    ]

    def fn() -> None:
        for buf in cycle_bufs:
            spin_wait_step(cycles=buf)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
