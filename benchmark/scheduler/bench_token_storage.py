"""Benchmark `list[int]` vs `array.array('q')` storage for
`Req.origin_input_ids` / `Req.output_ids` over one request lifecycle.

Simulated steps (per batch):
    1. ingest        -- tokenizer list[int] -> storage container.
    2. prefix_match  -- scheduler radix-tree lookup; RadixKey.match()
                        zip+!= walk. Exposes the per-element PyLong-boxing
                        cost array.array introduces (list[int] iterates
                        existing PyLongs and pays nothing).
    3. prefill       -- (a) fill_ids = origin + output,
                        (b) per-req slice fill_ids[prefix_len:],
                        (c) cross-req flatten + pinned cuda tensor build.
    4. decode        -- per-step output.append(next_token) for n_decode steps.
    5. finish        -- cache_finished_req:
                        (a) concat (origin + output)[:kv_committed_len]
                            for the radix-tree insert.
                        (b) RadixKey.match() zip+!= walk during insert's
                            tree traversal — second PyLong-boxing hotspot
                            on the array.array path.

Usage:
    python benchmark/scheduler/bench_token_storage.py
"""

from __future__ import annotations

import time
from array import array
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from typing import Any, Callable, Iterator

import numpy as np
import torch

# Per-req stages accumulate across reqs in a batch; batch_torch_tensor
# is the single cross-req prepare_for_extend tensor build.
STAGES = (
    "ingest",
    "prefix_match",
    "prefill_concat",
    "prefill_perreq_slice",
    "batch_torch_tensor",
    "decode_append",
    "finish_concat",
    "cache_finished_req",
)


def _ingest_list(seed: list[int]) -> list[int]:
    return seed


def _ingest_pyarray(seed: list[int]) -> array:
    return array("q", seed)


def _empty_list() -> list[int]:
    return []


def _empty_pyarray() -> array:
    return array("q")


def _zip_iterate(t0: Any, t1: Any) -> int:
    """Simulate zip iteration which surface PyLong boxing cost in array scenario"""
    i = 0
    for a, b in zip(t0, t1):
        if a != b:
            break
        i += 1
    return i


def _batch_tensor_from_lists(parts: list[list[int]]) -> torch.Tensor:
    flat = list(chain.from_iterable(parts))
    return torch.tensor(flat, dtype=torch.int64, pin_memory=True).to(
        "cuda", non_blocking=True
    )


def _batch_tensor_from_pyarrays(parts: list[array]) -> torch.Tensor:
    # np.frombuffer gives a zero-copy view; np.concatenate is one C-level
    # memcpy. This bypasses the per-element PyLong->int64 walk that
    # torch.tensor(array('q')) would otherwise do.
    views = [np.frombuffer(p, dtype=np.int64) for p in parts]
    combined = np.concatenate(views) if len(views) > 1 else views[0]
    return torch.from_numpy(combined).pin_memory().to("cuda", non_blocking=True)


LIST_KIT = {
    "ingest_fn": _ingest_list,
    "empty_fn": _empty_list,
    "batch_torch_fn": _batch_tensor_from_lists,
}

PYARRAY_KIT = {
    "ingest_fn": _ingest_pyarray,
    "empty_fn": _empty_pyarray,
    "batch_torch_fn": _batch_tensor_from_pyarrays,
}


@contextmanager
def timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    t0 = time.monotonic_ns()
    try:
        yield
    finally:
        timings[stage] += time.monotonic_ns() - t0


def simulate(
    seeds: list[list[int]],
    n_decode: int,
    *,
    ingest_fn: Callable[[list[int]], Any],
    empty_fn: Callable[[], Any],
    batch_torch_fn: Callable[[list[Any]], torch.Tensor],
) -> dict[str, float]:
    """One scheduling-round lifecycle. Returns per-stage cumulative ns."""
    timings: dict[str, float] = defaultdict(float)
    n_reqs = len(seeds)
    n_origins = [len(s) for s in seeds]
    origins: list[Any] = [None] * n_reqs
    outputs: list[Any] = [None] * n_reqs

    # 1. ingest
    for i, seed in enumerate(seeds):
        with timed(timings, "ingest"):
            origins[i] = ingest_fn(seed)
        outputs[i] = empty_fn()

    # 2. prefix_match: simulating the worse scenario of PyLong-boxing overhead during prefix_match
    for i in range(n_reqs):
        with timed(timings, "prefix_match"):
            _ = _zip_iterate(origins[i], origins[i])

    # 3. prefill
    per_req_slices: list[Any] = [None] * n_reqs
    for i in range(n_reqs):
        # 3a. fill_ids = origin_input_ids + output_ids
        with timed(timings, "prefill_concat"):
            fill_ids = origins[i] + outputs[i]
        # 3b. input_ids = fill_ids[len(prefix_indices):]; prefix_len=0 here.
        with timed(timings, "prefill_perreq_slice"):
            per_req_slices[i] = fill_ids[0:]
    # 3c. prepare_for_extend tensor build: flatten per-req slices, then
    #     build the pinned GPU tensor (kit-specific path).
    with timed(timings, "batch_torch_tensor"):
        _ = batch_torch_fn(per_req_slices)

    # 4. decode
    for i in range(n_reqs):
        with timed(timings, "decode_append"):
            for j in range(n_decode):
                outputs[i].append(j)

    # 5. finish: cache_finished_req -> insert -> _insert_helper tree walk.
    for i in range(n_reqs):
        # 5a. (origin + output)[:kv_committed_len] for the radix-tree insert.
        with timed(timings, "finish_concat"):
            committed = (origins[i] + outputs[i])[: n_origins[i] + n_decode]
        # 5b. simulating the worse scenario of PyLong-boxing overhead during cache_finished_req
        with timed(timings, "cache_finished_req"):
            _ = _zip_iterate(committed, committed)

    return timings


def bench_lifecycle(
    seeds: list[list[int]],
    n_decode: int,
    iterations: int,
    *,
    ingest_fn: Callable[[list[int]], Any],
    empty_fn: Callable[[], Any],
    batch_torch_fn: Callable[[list[Any]], torch.Tensor],
    warmup: int = 5,
) -> dict[str, float]:
    """Run simulate() N times, return mean per-stage us per batch.

    GPU sync is excluded from per-iteration timing: production issues
    `to(device, non_blocking=True)` and continues, so we measure issue
    cost rather than H2D completion.
    """
    kit = {
        "ingest_fn": ingest_fn,
        "empty_fn": empty_fn,
        "batch_torch_fn": batch_torch_fn,
    }
    torch.cuda.synchronize()
    for _ in range(warmup):
        simulate(seeds, n_decode, **kit)
    torch.cuda.synchronize()
    accum: dict[str, float] = defaultdict(float)
    for _ in range(iterations):
        t = simulate(seeds, n_decode, **kit)
        for k, v in t.items():
            accum[k] += v
    torch.cuda.synchronize()
    return {k: accum[k] / iterations / 1000.0 for k in STAGES}  # ns -> us


def print_breakdown(title: str, results: dict[str, dict[str, float]]) -> None:
    """Print per-stage timings with delta us vs the first (baseline) column."""
    labels = list(results.keys())
    baseline_label = labels[0]
    baseline = results[baseline_label]

    width = max(len(s) for s in STAGES)

    header_cells = [f"{baseline_label + ' us':>11s}"]
    for lbl in labels[1:]:
        header_cells.append(f"{lbl + ' us':>11s}")
        header_cells.append(f"{'delta':>10s}")

    print(f"=== {title} ===")
    print(f"{'Stage':<{width}s}  " + "  ".join(header_cells))
    print("-" * (width + 2 + sum(len(c) + 2 for c in header_cells)))

    for s in STAGES:
        cells = [f"{baseline[s]:>11.3f}"]
        for lbl in labels[1:]:
            v = results[lbl][s]
            cells.append(f"{v:>11.3f}")
            d = v - baseline[s]
            cells.append(f"{d:>+10.3f}")
        print(f"{s:<{width}s}  " + "  ".join(cells))

    print("-" * (width + 2 + sum(len(c) + 2 for c in header_cells)))

    base_total = sum(baseline.values())
    total_cells = [f"{base_total:>11.3f}"]
    for lbl in labels[1:]:
        v = sum(results[lbl].values())
        total_cells.append(f"{v:>11.3f}")
        d = v - base_total
        total_cells.append(f"{d:>+10.3f}")
    print(f"{'TOTAL':<{width}s}  " + "  ".join(total_cells))

    print()
    for lbl in labels[1:]:
        v = sum(results[lbl].values())
        d = v - base_total
        speedup = base_total / v if v > 0 else 0.0
        verdict = "LOSES" if d > 0 else "WINS"
        print(
            f"  {lbl:<14s} vs {baseline_label}: {verdict} by {abs(d):>8.2f} us  ({speedup:.2f}x)"
        )
    print()


def microbench_torch_tensor_paths(
    sizes: tuple[int, ...] = (1_000, 10_000, 100_000)
) -> None:
    """Compare three CPU-buffer -> pinned cuda tensor paths.

    A. torch.tensor(list,        pin) -> cuda
    B. torch.tensor(array('q'),  pin) -> cuda
    C. torch.from_numpy(np.frombuffer(array('q'))).pin() -> cuda
    """

    def t(fn, iterations: int) -> float:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.monotonic_ns()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        return (time.monotonic_ns() - t0) / iterations / 1000.0

    print("=== microbench: CPU-buffer -> pinned cuda tensor (us/op) ===\n")
    width = 56
    print(f"{'Path':<{width}s} " + "  ".join(f"{f'N={n}':>10s}" for n in sizes))
    print("-" * (width + 2 + 12 * len(sizes)))

    for label, build in [
        (
            "(A) torch.tensor(list,        pin) -> cuda",
            lambda x: torch.tensor(x, dtype=torch.int64, pin_memory=True).to(
                "cuda", non_blocking=True
            ),
        ),
        (
            "(B) torch.tensor(array('q'),  pin) -> cuda  (naive)",
            lambda x: torch.tensor(x, dtype=torch.int64, pin_memory=True).to(
                "cuda", non_blocking=True
            ),
        ),
        (
            "(C) from_numpy(frombuf(array('q'))).pin() -> cuda",
            lambda x: torch.from_numpy(np.frombuffer(x, dtype=np.int64))
            .pin_memory()
            .to("cuda", non_blocking=True),
        ),
    ]:
        cells = []
        for n in sizes:
            iters = max(50, 200_000 // max(n, 1))
            if "(A)" in label:
                src = list(range(n))
            else:
                src = array("q", range(n))
            us = t(lambda src=src, build=build: build(src), iters)
            cells.append(f"{us:>10.2f}")
        print(f"{label:<{width}s} " + "  ".join(cells))
    print()


def main() -> None:
    microbench_torch_tensor_paths()

    n_reqs = 2
    cases = [
        ("short  prompt  N_origin=1K    N_decode=1K", 1_000, 1_000, 1_000),
        ("medium prompt  N_origin=10K   N_decode=1K", 10_000, 1_000, 200),
        ("long   prompt  N_origin=100K  N_decode=1K", 100_000, 1_000, 30),
    ]
    print(f"Batch size = {n_reqs} reqs/batch (per-req stages accumulate)\n")
    for label, n_origin, n_decode, iters in cases:
        seeds = [list(range(n_origin)) for _ in range(n_reqs)]
        results = {
            "list": bench_lifecycle(seeds, n_decode, iters, **LIST_KIT),
            "pyarray": bench_lifecycle(seeds, n_decode, iters, **PYARRAY_KIT),
        }
        print_breakdown(label, results)


if __name__ == "__main__":
    main()
