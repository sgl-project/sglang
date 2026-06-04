"""Shared timing helpers for the lora_moe_triton_prep kernel testbeds.

Goal: measure each kernel's steady-state device time with **cold L2 / true HBM**
reads, which is what the e2e sees (every kernel reads freshly-written HBM, and
intervening kernels evict L2 between invocations).

How: capture many back-to-back calls in ONE CUDA graph and divide by the count
(amortizes the per-replay launch overhead to ~0), while ROTATING the calls over
many INDEPENDENT buffer sets whose combined footprint vastly exceeds the L2. Each
call then reads data the intervening calls already evicted -> cold.

The rotation count is chosen to FILL a memory budget (default 16 GB), so it
auto-grows when a future optimization shrinks the per-call working set -- you
never hand-tune it up. `bench_kernel` prints the chosen n_sets + footprint and
flags whether the footprint actually exceeds L2 (for tiny, sub-L2, latency-bound
kernels it cannot, and the number is then L2-state-independent anyway).
"""

from __future__ import annotations

from typing import Callable

import torch
import triton.testing


def l2_bytes() -> int:
    return int(torch.cuda.get_device_properties(0).L2_cache_size)


def set_bytes(one_set: dict) -> int:
    return sum(
        t.numel() * t.element_size() for t in one_set.values() if torch.is_tensor(t)
    )


def pick_n_sets(per_set_bytes: int, budget_gb: float = 16.0, requested: int = 0) -> int:
    """Number of rotation sets. requested>0 forces it; else fill the budget (so it
    auto-grows when per_set shrinks). Bounded to never OOM and capped at 2048."""
    free = torch.cuda.mem_get_info()[0]
    budget = min(int(budget_gb * 1024**3), int(0.5 * free))
    per = max(1, per_set_bytes)
    n = requested if requested and requested > 0 else max(8, budget // per)
    return int(max(2, min(n, int(0.6 * free / per), 2048)))


def bench_kernel(call: Callable[[int], None], n_sets: int, rep: int = 100) -> float:
    """Per-call ms. `call(i)` runs the kernel (or pipeline) on buffer set i. We capture
    one full sweep over all n_sets into a CUDA graph via triton.testing.do_bench_cudagraph
    and divide by n_sets: the graph replays all n_sets back-to-back (amortizes launch
    overhead), and because the n_sets footprint exceeds L2, each set is read L2-cold."""
    sweep = lambda: [call(i) for i in range(n_sets)]
    ms = triton.testing.do_bench_cudagraph(sweep, rep=rep)
    return float(ms) / n_sets


def report_sets(per_set_bytes: int, n_sets: int) -> str:
    foot = n_sets * per_set_bytes
    l2 = l2_bytes()
    tag = (
        "L2-COLD"
        if foot > l2
        else "WARN: footprint<L2 (sub-L2 / latency-bound; L2 state immaterial)"
    )
    return f"n_sets={n_sets} footprint={foot/1e9:.2f}GB L2={l2/1e6:.0f}MB -> {tag}"
