"""Timing and conservative paired selection shared by the LoRA tuners."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence


def _count(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def measure(
    fn: Callable,
    *,
    mode: str,
    warmup: int,
    repeats: int,
    iterations: int,
) -> list[float]:
    """Return whole-call microseconds; graph mode captures and actually replays.

    Eager includes synchronized host dispatch and device execution. Graph timing
    measures replay on the device, not graph construction or CPU route planning.
    The caller owns correctness, inputs, interleaving and graph eligibility.
    """
    if mode not in ("eager", "graph"):
        raise ValueError("mode must be eager or graph")
    _count(warmup, "warmup", 0)
    _count(repeats, "repeats", 1)
    _count(iterations, "iterations", 1)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LoRA tuning")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if mode == "graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(max(warmup, 1)):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = fn()
        graph.replay()
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        if mode == "graph":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) * 1000
        else:
            start_ns = time.perf_counter_ns()
            for _ in range(iterations):
                fn()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - start_ns) / 1000
        samples.append(elapsed / iterations)
    _samples(samples)
    if mode == "graph":
        # Keep graph-owned output storage live until every replay has completed.
        del captured_output
    return samples


def _samples(values: Sequence[float]) -> None:
    if not values or any(
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or not math.isfinite(x)
        or x <= 0
        for x in values
    ):
        raise ValueError("timings must be nonempty, finite, positive numbers")


def compare(
    baseline_us: Sequence[float],
    candidate_us: Sequence[float],
    *,
    min_gain: float = 0.02,
    max_regression: float = 0.02,
) -> dict:
    """Compare corresponding independent repeats; never discard a bad sample.

    Gain is baseline/candidate - 1 (lower latency is better). WIN/LOSS need
    unanimous signs and a median magnitude exceeding both the configured gain
    floor and baseline (max-min)/median spread. Otherwise a negative tail beyond
    max_regression is INCONCLUSIVE; remaining comparisons are TIE, not wins.
    These are tuning selection rules, not serving-performance acceptance gates.
    """
    _samples(baseline_us)
    _samples(candidate_us)
    if len(baseline_us) != len(candidate_us) or len(baseline_us) < 3:
        raise ValueError("paired comparison requires at least three complete pairs")
    for value in (min_gain, max_regression):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value < 1
        ):
            raise ValueError("gain and regression bounds must be finite in [0, 1)")
    gains = [b / c - 1 for b, c in zip(baseline_us, candidate_us)]
    if any(not math.isfinite(gain) or gain <= -1 for gain in gains):
        raise ValueError("paired ratios must be finite and positive")
    median = statistics.median(gains)
    spread = (max(baseline_us) - min(baseline_us)) / statistics.median(baseline_us)
    if (
        not math.isfinite(median)
        or not math.isfinite(statistics.median(baseline_us))
        or not math.isfinite(spread)
    ):
        raise ValueError("derived timing statistics must be finite")
    bound = max(min_gain, spread)
    if min(gains) > 0 and median >= bound:
        status = "WIN"
    elif max(gains) < 0 and -median >= bound:
        status = "LOSS"
    elif min(gains) < -max_regression:
        status = "INCONCLUSIVE"
    else:
        status = "TIE"
    return {
        "status": status,
        "median_gain": median,
        "worst_gain": min(gains),
        "baseline_spread": spread,
        "pairs": len(gains),
    }


def choose(
    baseline_us: Sequence[float],
    candidates: Mapping[str, Sequence[float]],
    *,
    min_gain: float = 0.02,
    max_regression: float = 0.02,
) -> dict:
    """Return a winning name or None (retain baseline); keep every result."""
    compare(baseline_us, baseline_us, min_gain=min_gain, max_regression=max_regression)
    results = {
        name: compare(
            baseline_us, samples, min_gain=min_gain, max_regression=max_regression
        )
        for name, samples in candidates.items()
    }
    wins = [name for name, result in results.items() if result["status"] == "WIN"]
    winner = (
        min(wins, key=lambda name: (-results[name]["median_gain"], name))
        if wins
        else None
    )
    return {"winner": winner, "results": results}
