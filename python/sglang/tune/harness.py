"""Timing harness — do_bench semantics, plus a deterministic mock latency model.

Real path (guarded) adopts ``triton.testing.do_bench`` semantics exactly, because they
are what home-grown benchmarks get wrong:

  1. warmup/rep are millisecond BUDGETS, not counts (divide by a 5-iter estimate).
  2. CUDA events, never wall-clock (async launches → wall-clock times only the launch).
  3. enqueue all timed iters, then a SINGLE torch.cuda.synchronize().
  4. flush a 256 MiB L2 buffer (zeroed) before each timed run — else decode kernels
     report fantasy latencies off resident KV.
  5. report median (steady state) and min (intrinsic cost); emit p20/p50/p80.
  6. CUDA-graph mode for tiny decode kernels (launch-overhead-bound).

Mock path: a deterministic, GPU-free latency model with a realistic crossover, so the
whole selection pipeline is testable and the picks are assertable.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

from .shapes import AttnProfile, DecodeShape, PrefillShape


@dataclasses.dataclass
class BenchResult:
    median_us: float
    min_us: float
    p20_us: float
    p80_us: float

    @classmethod
    def constant(cls, us: float) -> BenchResult:
        return cls(us, us, us, us)


L2_FLUSH_BYTES = (
    256 * 1024 * 1024
)  # Triton do_bench convention (a knob; NVBench queries actual L2)


def do_bench(
    fn: Callable[[], None],
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    use_cuda_graph: bool = False,
) -> BenchResult:  # pragma: no cover - real GPU only
    """do_bench-faithful timing. Only runs on a real CUDA device."""
    import torch

    if use_cuda_graph:
        return _do_bench_cudagraph(fn, rep_ms)

    flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int, device="cuda")

    # 5-iter estimate to convert ms budgets into iteration counts.
    start = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    for i in range(5):
        flush.zero_()
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    est = max(1e-3, sum(s.elapsed_time(e) for s, e in zip(start, end)) / 5)
    n_warmup = max(1, int(warmup_ms / est))
    n_rep = max(1, int(rep_ms / est))

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    for i in range(n_rep):
        flush.zero_()  # cold inputs each run
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()  # single trailing sync
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))  # -> us
    return _quantiles(times)


def _quantiles(times_us: list) -> BenchResult:
    n = len(times_us)
    pick = lambda q: times_us[min(n - 1, int(q * n))]
    return BenchResult(
        median_us=pick(0.5), min_us=times_us[0], p20_us=pick(0.2), p80_us=pick(0.8)
    )


GRAPH_ITERS = 32  # unrolled calls captured per graph — amortizes the replay-launch cost
GRAPH_SANITY_FLOOR_US = (
    0.5  # below this per-iter median, graph timing is fantasy — fall back
)


def _do_bench_cudagraph(
    fn: Callable[[], None], rep_ms: float = 100.0
) -> BenchResult:  # pragma: no cover - real GPU only
    """do_bench_cudagraph-style timing for launch-overhead-bound kernels (tiny decode).

    Captures GRAPH_ITERS unrolled calls into one CUDA graph, then times whole-graph
    replays with CUDA events and reports per-iteration quantiles — so the measurement
    excludes the Python launch path the eager loop would otherwise time. NOTE: capture/
    replay cannot interleave the L2 flush, so inputs stay cache-resident; use this ONLY
    for the launch-bound small-batch decode regime and cross-check against plain do_bench.
    """
    import torch

    fn()  # warm: JIT/backend selection settles pre-capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(GRAPH_ITERS):
            fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    est = max(1e-3, start.elapsed_time(end))  # ms per replay
    n_rep = max(1, int(rep_ms / est))

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    for i in range(n_rep):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()  # single trailing sync
    times = sorted(
        s.elapsed_time(e) * 1000.0 / GRAPH_ITERS for s, e in zip(starts, ends)
    )  # -> us per iteration
    result = _quantiles(times)
    if result.median_us < GRAPH_SANITY_FLOOR_US:
        # Physically implausible: no attention kernel completes in sub-half-microsecond.
        # Seen on virtualized/intercepted CUDA stacks (GPU-over-TCP) where graph replay
        # returns without observably executing the captured work. Discard and cross-check
        # with the eager path rather than report fantasy numbers.
        print(
            f"[attune] WARNING: graph-mode timing implausible "
            f"({result.median_us:.4f} us/iter < {GRAPH_SANITY_FLOOR_US}); "
            "falling back to eager do_bench for this shape."
        )
        return do_bench(fn, use_cuda_graph=False)
    return result


# --- mock latency model -------------------------------------------------------
# Encodes the qualitative facts the corpus established, so the selection logic has a
# real crossover to resolve:
#   * decode is memory-bound; on bandwidth-divergent SKUs (H20-like, sm==90 with a low
#     "bandwidth" score) FA3's 128-row-tile rounding wastes work at small batch, so
#     FlashInfer wins low-batch decode and FA3 only catches up at large batch.
#   * prefill is compute-bound; FA3 wins on Hopper, FlashInfer wins very-long-seq.
#   * triton is a correct-but-slow floor.
_BASE = {
    "fa3": 40.0,
    "fa4": 38.0,
    "flashinfer": 44.0,
    "trtllm_mha": 39.0,
    "flashmla": 41.0,
    "cutlass_mla": 43.0,
    "trtllm_mla": 40.0,
    "triton": 90.0,
    "torch_native": 300.0,
}


def mock_decode_latency(
    backend: str, shape: DecodeShape, profile: AttnProfile, bandwidth_divergent: bool
) -> float:
    base = _BASE.get(backend, 100.0)
    us = base + 0.12 * shape.batch + 0.0009 * shape.ctx_len
    # Compute-tile kernels (FA3/FA4) round a decode query up to a 128-row tile, wasting work
    # at small batch. On a bandwidth-divergent SKU (H20) with a low FLOPS/bandwidth ratio the
    # waste dominates, so FlashInfer's matrix-vector paged decode wins low-batch — the exact
    # #5630 crossover that the SM heuristic is blind to. On a fully-powered flagship the
    # penalty is mild and the tile kernel still wins.
    if backend in ("fa3", "fa4"):
        tile_waste = max(0.0, (128 - shape.batch)) / 128.0
        us *= 1.0 + (1.10 if bandwidth_divergent else 0.06) * tile_waste
    if backend == "flashinfer":
        us *= 0.95  # strong small-batch paged decode
    return us


def mock_prefill_latency(
    backend: str, shape: PrefillShape, profile: AttnProfile
) -> float:
    base = _BASE.get(backend, 100.0)
    us = base + 0.02 * shape.seq_len * shape.batch
    if backend == "fa3":
        us *= 0.90  # compute-bound Hopper win
        if shape.seq_len >= 32768:
            us *= 1.25  # loses at very long seq
    if backend == "flashinfer" and shape.seq_len >= 32768:
        us *= 0.85
    return us
