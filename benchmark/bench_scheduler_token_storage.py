"""End-to-end lifecycle benchmark: `list[int]` vs `array.array('q')`
storage for `Req.origin_input_ids` and `Req.output_ids` across one
typical request's prefill -> decode -> finish path.

Scope: OSS production caches only -- `radix_cache.py`,
`swa_radix_cache.py`, etc. NOT the Meta Rust-coordinator wrapper
(which would add an extra `list_to_numpy_int64_1d` bridge step). In
OSS the radix cache is pure Python and consumes `token_ids` directly
into `RadixKey(token_ids, ...)` -- no Python-level conversion.

Models the OSS default-mode lifecycle (no `--enable-mixed-chunk`,
no chunked prefill, single-chunk single-request).

Each timed stage links to the OSS code site that motivates it. Stage
labels match the constants in STAGES below.

Tokenizer-output assumption: tokenizer keeps producing `list[int]`
(no migration). Pyarray storage pays a `list -> array('q')` ingest
cost at request creation, via the CPython-native `array('q', list)`
constructor -- pure stdlib, no extra dependency. Faster ingest paths
exist (e.g. via a Rust pyo3 helper that returns numpy + frombytes
buffer-protocol bridging), but the methodology here is "ship the
naive approach first, optimize bottlenecks only if they show up."

Implementation: one `simulate` function parameterized by a "kit"
(ingest_fn, empty_fn, torch_fn). The list and pyarray paths only
differ in those three; concat / slice / append flow naturally on
either container via Python's `+` / `[a:b]` / `.append`.

Usage (from the repo root):
    PYTHONPATH=python python benchmark/bench_scheduler_token_storage.py
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

# Stage names in execution order. Per-req stages accumulate across
# all reqs in the batch; batch_torch_tensor is one cross-req op per
# batch (the actual prepare_for_extend tensor build).
STAGES = (
    "ingest",  # list -> array('q') (pyarray only)        [per req]
    "prefill_concat",  # fill_ids = origin + output               [per req; schedule_batch.py:989]
    "prefill_perreq_slice",  # fill_ids[len(prefix_indices):]           [per req; schedule_batch.py:1688]
    "batch_torch_tensor",  # flatten per-req slices + torch.tensor    [PER BATCH; schedule_batch.py:1715-1717]
    "decode_append",  # output.append(tok)                       [per req; scheduler_output_processor_mixin]
    "finish_concat",  # (origin + output)[:kv_committed_len]     [per req; radix_cache.py:452]
)


# --- Per-container kits (the only places list and pyarray differ) ---


def _ingest_list(seed: list[int]) -> list[int]:
    # Identity: tokenizer hands us list[int] already.
    return seed


def _ingest_pyarray(seed: list[int]) -> array:
    # CPython-native ingest: array('q', list) iterates per-element
    # through PyLong->int64. Pure stdlib, no Rust dependency.
    return array("q", seed)


def _empty_list() -> list[int]:
    return []


def _empty_pyarray() -> array:
    return array("q")


# Bench requires CUDA -- run on a GPU host. Mirrors the production
# `torch.tensor(..., pin_memory=_pin).to(self.device, non_blocking=True)`
# path at schedule_batch.py:1715-1717 (with `_pin = is_pin_memory_available(device)`
# at utils/common.py:616-621, which returns True on any GPU server).


def _batch_tensor_from_lists(parts: list[list[int]]) -> torch.Tensor:
    # Today's path at schedule_batch.py:1715-1717:
    #   torch.tensor(list(chain.from_iterable(input_ids)),
    #                dtype=torch.int64, pin_memory=_pin).to(device, non_blocking=True)
    # `chain.from_iterable + list(...)` flattens the per-req
    # list[int] slices into one flat list[int]; torch.tensor then
    # walks each element through PyLong->int64 into a pinned buffer.
    flat = list(chain.from_iterable(parts))
    return torch.tensor(flat, dtype=torch.int64, pin_memory=True).to(
        "cuda", non_blocking=True
    )


def _batch_tensor_from_pyarrays(parts: list[array]) -> torch.Tensor:
    # Post-migration path:
    #   1. np.frombuffer per part        -- zero-copy numpy views over array bytes.
    #   2. np.concatenate                -- one C-level memcpy into a fresh numpy buffer.
    #   3. torch.from_numpy              -- zero-copy torch CPU view.
    #   4. .pin_memory()                 -- copies into pinned CPU buffer.
    #   5. .to(cuda, non_blocking=True)  -- async H2D copy.
    # No per-element PyLong dispatch anywhere; the only CPU work is
    # the concat memcpy + the pin memcpy.
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
    """Accumulate the wall-clock duration of the wrapped block into
    `timings[stage]` (nanoseconds).
    """
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
    """One scheduling-round lifecycle for a batch of `len(seeds)` reqs.

    Per-req stages (ingest / prefill_concat / prefill_perreq_slice /
    decode_append / finish_concat) accumulate across all reqs in the
    batch. `batch_torch_tensor` is a single cross-req op modeling the
    actual prepare_for_extend tensor build that runs once per batch.

    Returns per-stage cumulative time in nanoseconds. Code links
    cited inline -- see STAGES tuple for the canonical OSS site.
    """
    timings: dict[str, float] = defaultdict(float)
    n_reqs = len(seeds)
    n_origins = [len(s) for s in seeds]
    origins: list[Any] = [None] * n_reqs
    outputs: list[Any] = [None] * n_reqs

    # ----- Per-req: ingest -----
    # (ingest) list -> storage container.
    #  * list path: identity (tokenizer-supplied list reused as-is).
    #  * pyarray path: list -> Rust numpy -> array('q') frombytes.
    for i, seed in enumerate(seeds):
        with timed(timings, "ingest"):
            origins[i] = ingest_fn(seed)
        outputs[i] = empty_fn()

    # ----- Per-req: prefill (concat + slice) -----
    per_req_slices: list[Any] = [None] * n_reqs
    for i in range(n_reqs):
        # (prefill_concat) init_next_round_input rebuilds fill_ids each
        # scheduling round. Source: schedule_batch.py:989
        #     self.fill_ids = self.origin_input_ids + self.output_ids
        with timed(timings, "prefill_concat"):
            fill_ids = origins[i] + outputs[i]

        # (prefill_perreq_slice) prepare_for_extend's per-req slice.
        # Source: schedule_batch.py:1688
        #     input_ids = [r.fill_ids[len(r.prefix_indices):] for r in reqs]
        # prefix_len = 0 (first prefill, no cache hit).
        with timed(timings, "prefill_perreq_slice"):
            per_req_slices[i] = fill_ids[0:]

    # ----- Cross-req: batch_torch_tensor -----
    # (batch_torch_tensor) prepare_for_extend's cross-batch input_ids
    # tensor build. Source: schedule_batch.py:1715-1717
    #     torch.tensor(list(chain.from_iterable(input_ids)),
    #                  dtype=torch.int64, pin_memory=_pin
    #     ).to(self.device, non_blocking=True)
    # Modeled exactly: flatten per-req slices, then build the pinned
    # GPU tensor. The kit decides the flatten/build implementation
    # (list path: chain.from_iterable + torch.tensor; pyarray path:
    # np.concatenate views + torch.from_numpy + pin).
    with timed(timings, "batch_torch_tensor"):
        _ = batch_torch_fn(per_req_slices)

    # ----- Per-req: decode -----
    # (decode_append) per-step output token append from sampling.
    # Source: scheduler_output_processor_mixin.py and similar:
    #     req.output_ids.append(next_token_id)
    for i in range(n_reqs):
        with timed(timings, "decode_append"):
            for j in range(n_decode):
                outputs[i].append(j)

    # ----- Per-req: finish -----
    # (finish_concat) cache_finished_req builds token_ids from origin +
    # output (NOT from fill_ids). Source: radix_cache.py:452
    #     token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
    for i in range(n_reqs):
        with timed(timings, "finish_concat"):
            _ = (origins[i] + outputs[i])[: n_origins[i] + n_decode]

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
    """Run simulate() `iterations` times, accumulate per-stage ns,
    return mean per-stage us per BATCH (not per req).

    GPU sync notes: we sync once before warmup and once before/after
    timing, but NOT inside the timed simulate(). The scheduler in
    production issues `to(device, non_blocking=True)` and continues;
    we measure that issue cost, not the H2D transfer completion.
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
    """Multi-column per-stage breakdown.

    `results` maps a column label (e.g. "list", "pyarr-rust") to the
    per-stage timings dict from bench_lifecycle. The first column is
    the baseline; subsequent columns show their `delta us` (col -
    baseline) right after the timing, so you can read each variant's
    overhead/savings against the baseline at a glance.
    """
    labels = list(results.keys())
    baseline_label = labels[0]
    baseline = results[baseline_label]

    width = max(len(s) for s in STAGES)

    # Header: stage | <label1 us> | <label2 us> | <label2 delta> | ...
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

    # TOTAL row
    base_total = sum(baseline.values())
    total_cells = [f"{base_total:>11.3f}"]
    for lbl in labels[1:]:
        v = sum(results[lbl].values())
        total_cells.append(f"{v:>11.3f}")
        d = v - base_total
        total_cells.append(f"{d:>+10.3f}")
    print(f"{'TOTAL':<{width}s}  " + "  ".join(total_cells))

    # Verdict line: who wins vs the baseline?
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
    """Three-way comparison of CPU-token-buffer -> pinned cuda tensor:

      A. torch.tensor(list,         dtype=int64, pin) -> cuda     (current OSS path)
      B. torch.tensor(array('q'),   dtype=int64, pin) -> cuda     (naive pyarray drop-in)
      C. torch.from_numpy(np.frombuffer(array('q'))).pin() -> cuda (current pyarray path)

    Justifies the np.frombuffer intermediate in _batch_tensor_from_pyarrays:
    (B) is actually slightly SLOWER than (A) because torch.tensor falls into
    its iterable-init code path for array.array (no buffer-protocol
    dispatch), so element-by-element PyLong->int64 still happens. (C) is
    the only path that triggers the buffer-protocol fast path -- ~5000x
    faster on CPU work at N=100K (eliminates the per-element walk; only
    the pin memcpy remains).
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
    # Microbench first -- justifies why _batch_tensor_from_pyarrays uses
    # the np.frombuffer intermediate instead of torch.tensor(arr) directly.
    microbench_torch_tensor_paths()

    # Typical agentic / serving shapes (no chunked prefill, no
    # mixed-chunk). BS=2 to model the per-batch flatten cost in
    # prepare_for_extend (chain.from_iterable across reqs). All reqs
    # in a batch use the same prompt length for simplicity.
    n_reqs = 2
    cases = [
        # (label, n_origin per req, n_decode per req, iterations)
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
