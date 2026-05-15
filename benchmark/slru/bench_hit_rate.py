"""SLRU hit-rate simulator.

Pure-Python, no GPU needed. Replays a scripted traffic pattern through a
simulated ``RadixCache`` under both legacy and optimized SLRU, then emits a
CSV of hit-rate samples over time. The output is a lightweight stand-in for
full ``bench_serving`` runs.

Two scenarios are supported:

* ``mixed`` (default) — adversarial burst workload that stresses the
  **Burst Distortion** failure mode (debounce fix).
* ``shifting_zipfian`` — rotating-popularity Zipfian workload that
  stresses the **Stagnation** failure mode (lazy-decay + cap fix).

Example:

    python benchmark/slru/bench_hit_rate.py \\
        --output /tmp/slru_hit_rate.csv \\
        --scenario mixed --duration-sec 300

    python benchmark/slru/bench_hit_rate.py \\
        --output /tmp/slru_shifting.csv \\
        --scenario shifting_zipfian --duration-sec 600

The CSV columns are: ``scenario, policy, t, hits, misses, hit_rate``.

This is *not* a TTFT / throughput benchmark — it operates purely on the
cache-hit dimension. Combine with the real ``bench_serving`` run to get
the full picture.
"""

from __future__ import annotations

import argparse
import csv
import importlib.abc
import importlib.machinery
import random
import sys
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch


def _maybe_stub_sgl_kernel_for_cpu_sim() -> None:
    """Keep the CPU-only simulator usable when local CUDA kernels are absent."""
    try:
        import sgl_kernel  # noqa: F401

        return
    except (ImportError, OSError):
        pass

    from unittest.mock import MagicMock

    class _SglKernelLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__getattr__ = lambda name: MagicMock()

    class _SglKernelFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
                return importlib.machinery.ModuleSpec(
                    fullname,
                    _SglKernelLoader(),
                    is_package=True,
                )
            return None

    sys.meta_path.insert(0, _SglKernelFinder())


_maybe_stub_sgl_kernel_for_cpu_sim()

from sglang.srt.environ import envs
from sglang.srt.mem_cache import evict_policy as _ep_module
from sglang.srt.mem_cache import radix_cache as _rc_module
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.evict_policy import SLRUStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class _MockAllocator:
    def __init__(self):
        self.device = "cpu"

    def free(self, value):
        pass


class _FakeClock:
    def __init__(self, start: float = 0.0):
        self._t = start

    def advance(self, seconds: float):
        self._t += seconds

    def now(self) -> float:
        return self._t


def _patch_clock(clock: _FakeClock):
    rc_orig = _rc_module.time.monotonic
    ep_orig = _ep_module.time.monotonic
    _rc_module.time.monotonic = clock.now
    _ep_module.time.monotonic = clock.now

    def restore():
        _rc_module.time.monotonic = rc_orig
        _ep_module.time.monotonic = ep_orig

    return restore


@dataclass
class Event:
    """A single traffic event.

    ``kind == "access"`` → cache lookup; if miss, also insert.
    ``kind == "advance"`` → advance the fake clock by ``seconds``.
    """

    kind: str
    token_ids: Tuple[int, ...] = ()
    seconds: float = 0.0


def scenario_mixed(rng: random.Random, duration_sec: float = 300.0) -> Iterator[Event]:
    """Adversarial burst-distortion workload.

    Hot prompts receive steady reuse, while throwaway prefixes arrive in
    near-simultaneous bursts. Legacy SLRU may promote those bursts into the
    protected tier; optimized SLRU should keep them probationary.
    """
    hot_prompts = [tuple(range(10 + i * 32, 10 + i * 32 + 16)) for i in range(40)]
    next_unique = 100000
    t = 0.0
    last_burst_at = -5.0  # allow a burst right away

    while t < duration_sec:
        r = rng.random()
        if r < 0.60:
            # 60% hot access.
            yield Event("access", rng.choice(hot_prompts))
            yield Event("advance", seconds=0.01)
            t += 0.01
        elif t - last_burst_at >= 5.0 and r < 0.70:
            # 10% burst when cooldown allows.
            burst = tuple(range(next_unique, next_unique + 16))
            next_unique += 16
            for _ in range(50):
                yield Event("access", burst)
                yield Event("advance", seconds=0.0002)
            t += 0.01
            last_burst_at = t
        else:
            # 30-40% unique user prompts.
            uniq = tuple(range(next_unique, next_unique + 16))
            next_unique += 16
            yield Event("access", uniq)
            yield Event("advance", seconds=0.05)
            t += 0.05


def scenario_shifting_zipfian(
    rng: random.Random,
    duration_sec: float = 600.0,
    pool_size: int = 100,
    hot_size: int = 10,
    rotate_sec: float = 30.0,
    zipf_alpha: float = 1.2,
    rate_rps: float = 50.0,
    shift_stride: int = 5,
) -> Iterator[Event]:
    """Rotating-popularity Zipfian workload for lazy decay.

    A pool of ``pool_size`` distinct prefixes. At any moment, a
    contiguous "hot set" of ``hot_size`` prefixes receives
    ~all the traffic, with intra-set popularity skewed via Zipf(alpha).
    Every ``rotate_sec`` virtual seconds the hot set slides forward by
    ``shift_stride`` positions: prefixes that were hot become cold, and
    new prefixes enter the hot set.

    Once a prefix leaves the hot set, optimized SLRU's hit-count cap and lazy
    decay let it yield protected capacity to newly hot prefixes.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if hot_size < 1:
        raise ValueError(f"hot_size must be >= 1, got {hot_size}")
    if hot_size > pool_size:
        raise ValueError(f"hot_size ({hot_size}) must be <= pool_size ({pool_size})")
    if rotate_sec <= 0:
        raise ValueError(f"rotate_sec must be > 0, got {rotate_sec}")
    if rate_rps <= 0:
        raise ValueError(f"rate_rps must be > 0, got {rate_rps}")
    if zipf_alpha < 0:
        raise ValueError(f"zipf_alpha must be >= 0, got {zipf_alpha}")
    if shift_stride < 0:
        raise ValueError(f"shift_stride must be >= 0, got {shift_stride}")

    # Distinct, non-overlapping prefix IDs (16-token width, matching
    # ``scenario_mixed``'s convention).
    prefixes = [tuple(range(10 + i * 32, 10 + i * 32 + 16)) for i in range(pool_size)]

    # Pre-compute Zipfian CDF within the hot set: weights[i] ∝
    # 1/(i+1)^alpha. The CDF lets us sample in O(hot_size) per draw
    # without importing numpy — the simulator is deliberately
    # numpy-free elsewhere.
    raw = [1.0 / ((i + 1) ** zipf_alpha) for i in range(hot_size)]
    total_w = sum(raw)
    cdf: List[float] = []
    cum = 0.0
    for w in raw:
        cum += w / total_w
        cdf.append(cum)
    # Fix trailing float drift so the last bucket is reachable even if
    # rng.random() returns 0.999...9.
    cdf[-1] = 1.0

    dt = 1.0 / rate_rps
    t = 0.0

    while t < duration_sec:
        rotation_idx = int(t / rotate_sec)
        # Hot set is ``hot_size`` consecutive prefixes from ``center``,
        # wrapping around the pool. shift_stride=0 degenerates to a
        # fixed hot set (static Zipfian) — intentionally allowed as a
        # controlled baseline; pass --shifting-stride 0 to compare.
        center = (rotation_idx * shift_stride) % pool_size
        hot_set = [prefixes[(center + i) % pool_size] for i in range(hot_size)]

        rotation_end = min((rotation_idx + 1) * rotate_sec, duration_sec)

        while t < rotation_end:
            # Inverse-CDF sample the Zipfian position in the hot set.
            u = rng.random()
            idx = 0
            while idx < hot_size - 1 and u > cdf[idx]:
                idx += 1
            yield Event("access", hot_set[idx])
            yield Event("advance", seconds=dt)
            t += dt


def _run_simulation(
    events: List[Event],
    optimization_enabled: bool,
    cache_capacity_tokens: int,
    sample_every_sec: float,
) -> List[Tuple[float, int, int, float]]:
    """Drive the events through a fresh SLRU-equipped ``RadixCache``.
    Returns a list of (t, hits, misses, hit_rate) tuples sampled
    roughly every ``sample_every_sec`` seconds of virtual time."""
    clock = _FakeClock()
    restore = _patch_clock(clock)
    samples: List[Tuple[float, int, int, float]] = []
    try:
        with envs.SGLANG_ENABLE_SLRU_OPTIMIZATION.override(optimization_enabled):
            cache = RadixCache.create_simulated(
                mock_allocator=_MockAllocator(), page_size=1
            )
            cache.eviction_strategy = SLRUStrategy(
                protected_threshold=2,
                debounce_sec=0.1,
                decay_sec=60.0,
            )

            hits = 0
            misses = 0
            window_hits = 0
            window_misses = 0
            next_sample_at = sample_every_sec

            for ev in events:
                if ev.kind == "advance":
                    clock.advance(ev.seconds)
                    if clock.now() >= next_sample_at:
                        window_total = window_hits + window_misses
                        rate = window_hits / window_total if window_total else 0.0
                        samples.append(
                            (
                                clock.now(),
                                window_hits,
                                window_misses,
                                rate,
                            )
                        )
                        window_hits = 0
                        window_misses = 0
                        next_sample_at += sample_every_sec
                    continue

                key = RadixKey(token_ids=list(ev.token_ids), extra_key=None)
                # Mirror real serving behavior: every request runs
                # match_prefix (for the scheduler's prefix lookup) AND
                # insert (cache_{un,}finished_req). ``insert`` walks
                # the tree and bumps hit_count on every matching node
                # along the path — without it, the cache would never
                # promote anything to Protected.
                result = cache.match_prefix(MatchPrefixParams(key=key))
                if int(result.device_indices.numel()) == len(ev.token_ids):
                    hits += 1
                    window_hits += 1
                else:
                    misses += 1
                    window_misses += 1
                value = torch.tensor(list(ev.token_ids), dtype=torch.int64)
                cache.insert(InsertParams(key=key, value=value))

                # Evict on capacity — keeps the working set under a
                # steady ceiling so eviction policy matters.
                if cache.evictable_size_ > cache_capacity_tokens:
                    overflow = cache.evictable_size_ - cache_capacity_tokens
                    cache.evict(EvictParams(num_tokens=overflow))
    finally:
        restore()

    return samples


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the CSV. Columns: scenario, policy, t, "
        "hits, misses, hit_rate.",
    )
    parser.add_argument(
        "--scenario",
        default="mixed",
        choices=["mixed", "shifting_zipfian"],
        help="Traffic pattern to replay. ``mixed`` targets burst "
        "distortion; ``shifting_zipfian`` targets stagnation.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=300.0,
        help="Virtual seconds of traffic to replay.",
    )
    parser.add_argument(
        "--cache-capacity-tokens",
        type=int,
        default=2048,
        help="Target evictable footprint — eviction kicks in when "
        "the cache exceeds this.",
    )
    parser.add_argument(
        "--sample-every-sec",
        type=float,
        default=5.0,
        help="Emit one hit-rate sample every N virtual seconds.",
    )
    parser.add_argument("--seed", type=int, default=0xC0FFEE, help="RNG seed.")
    # shifting_zipfian-specific knobs. Ignored unless
    # --scenario shifting_zipfian is selected.
    parser.add_argument(
        "--shifting-pool-size",
        type=int,
        default=100,
        help="shifting_zipfian: total distinct prefixes in the pool.",
    )
    parser.add_argument(
        "--shifting-hot-size",
        type=int,
        default=10,
        help="shifting_zipfian: number of prefixes in the hot set at " "any moment.",
    )
    parser.add_argument(
        "--shifting-rotate-sec",
        type=float,
        default=30.0,
        help="shifting_zipfian: slide the hot set every N virtual " "seconds.",
    )
    parser.add_argument(
        "--shifting-zipf-alpha",
        type=float,
        default=1.2,
        help="shifting_zipfian: Zipf exponent for within-hot-set "
        "popularity skew. Higher = more skewed toward the top.",
    )
    parser.add_argument(
        "--shifting-rate-rps",
        type=float,
        default=50.0,
        help="shifting_zipfian: virtual requests per second.",
    )
    parser.add_argument(
        "--shifting-stride",
        type=int,
        default=5,
        help="shifting_zipfian: positions to slide the hot set per "
        "rotation. 0 degenerates to a static Zipfian baseline.",
    )
    args = parser.parse_args(argv)

    # Materialize events once so both policies see identical traffic.
    rng = random.Random(args.seed)
    if args.scenario == "mixed":
        events = list(scenario_mixed(rng, duration_sec=args.duration_sec))
    elif args.scenario == "shifting_zipfian":
        events = list(
            scenario_shifting_zipfian(
                rng,
                duration_sec=args.duration_sec,
                pool_size=args.shifting_pool_size,
                hot_size=args.shifting_hot_size,
                rotate_sec=args.shifting_rotate_sec,
                zipf_alpha=args.shifting_zipf_alpha,
                rate_rps=args.shifting_rate_rps,
                shift_stride=args.shifting_stride,
            )
        )
    else:
        # argparse ``choices`` already rejects anything else, but guard
        # in case that list is extended without updating this dispatch.
        print(f"unsupported scenario: {args.scenario}", file=sys.stderr)
        return 2

    print(
        f"[bench_hit_rate] scenario={args.scenario} "
        f"events={len(events)} capacity={args.cache_capacity_tokens}"
    )

    rows = []
    for policy, enabled in (("naive", False), ("optimized", True)):
        samples = _run_simulation(
            events,
            optimization_enabled=enabled,
            cache_capacity_tokens=args.cache_capacity_tokens,
            sample_every_sec=args.sample_every_sec,
        )
        total_hits = sum(h for _, h, _, _ in samples)
        total_misses = sum(m for _, _, m, _ in samples)
        overall = (
            total_hits / (total_hits + total_misses)
            if (total_hits + total_misses)
            else 0.0
        )
        print(
            f"[bench_hit_rate] policy={policy} "
            f"samples={len(samples)} total_hits={total_hits} "
            f"total_misses={total_misses} hit_rate={overall:.4f}"
        )
        for t, h, m, rate in samples:
            rows.append((args.scenario, policy, t, h, m, rate))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "policy", "t", "hits", "misses", "hit_rate"])
        writer.writerows(rows)

    print(f"[bench_hit_rate] wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
