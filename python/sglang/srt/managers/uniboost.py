"""UniBoost schedule policy.

Y. Li et al. "Beyond Prediction: Tail-Aware Scheduling for LLM Inference", ICML 2026.
https://yl3469.github.io/uniboost-icml26/ (arXiv:2606.18431).

A waiting request r is served in ascending

    phi(r) = A(r) - b_gamma(q(u(r)))

where A(r) is its first-queue-entry stamp, u(r) its uncached prefill work,
q() a geometric quantization at bin size k and b_gamma(w) =
-ln(1 - exp(-gamma * w)) / gamma a boost that is large for small work and
vanishes for large work. b_gamma(k) is the largest boost any request can get,
so a newcomer can overtake only requests that arrived less than b_gamma(k)
earlier: large gamma converges to FCFS, small gamma to shortest-uncached-work.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Sequence, Set

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class UniBoostClock(Enum):
    """Axis of A(r) and u(r); see --uniboost-clock."""

    # Prefill tokens processed by this scheduler (the paper's clock).
    TOKENS = "tokens"
    # Wall seconds; work is converted at the measured token-clock rate.
    WALL = "wall"


# Unset --uniboost-gamma. Tokens: the paper's per-token scale. Wall: geometric
# center of the gamma-Ada fit's fixed points measured at concurrency 4-16 on
# agentic traces (0.017-0.62 1/s).
DEFAULT_GAMMA = {UniBoostClock.TOKENS: 1e-4, UniBoostClock.WALL: 0.1}


def boost(work: float, gamma: float) -> float:
    """b_gamma(w) = -ln(1 - exp(-gamma * w)) / gamma (paper Eq. 2)."""
    x = gamma * work
    if x <= 0:
        return float("inf")
    v = -math.expm1(-x)
    if v >= 1.0:
        # 1 - exp(-x) rounds to 1.0 for large x; the boost underflows to 0.
        return 0.0
    return -math.log(v) / gamma


def geometric_bin(work: int, bin_size: int) -> int:
    """q(u) = k * 2^floor(log2(max(u, k) / k)) (paper Eq. 8)."""
    return bin_size * 2 ** math.floor(math.log2(max(work, bin_size) / bin_size))


def uncached_work(req: Req) -> int:
    """Tokens left to prefill: input plus generated (re-prefilled after a
    retraction) minus the matched prefix (device and host hits)."""
    return max(
        1,
        len(req.origin_input_ids) + len(req.output_ids) - req.num_matched_prefix_tokens,
    )


class TokenClockRate:
    """EMA of the token clock's rate (counted prefill tokens per wall second),
    used by the wall clock to turn uncached tokens into seconds of service."""

    BETA = 0.2

    def __init__(self) -> None:
        self._last_tokens: Optional[int] = None
        self._last_t: Optional[float] = None
        self._tokens_per_s: Optional[float] = None

    @property
    def tokens_per_s(self) -> Optional[float]:
        return self._tokens_per_s

    def observe(self, processed_tokens: int, now_s: float) -> Optional[float]:
        if self._last_tokens is None:
            self._last_tokens = processed_tokens
            self._last_t = now_s
            return self._tokens_per_s
        delta_tokens = processed_tokens - self._last_tokens
        delta_s = now_s - self._last_t
        # An idle or decode-only stretch takes no sample and keeps the
        # baseline, so it folds into the next sample's window.
        if delta_tokens > 0 and delta_s > 0:
            sample = delta_tokens / delta_s
            self._last_tokens = processed_tokens
            self._last_t = now_s
            if self._tokens_per_s is None:
                self._tokens_per_s = sample
            else:
                self._tokens_per_s = (
                    1.0 - self.BETA
                ) * self._tokens_per_s + self.BETA * sample
        return self._tokens_per_s


def stamp_arrivals(
    waiting_queue: Sequence[Req], *, processed_tokens: int, now_s: float
) -> None:
    """Stamps A(r) the first time the policy sees r; a retracted request that
    re-enters the queue keeps its original stamps."""
    for r in waiting_queue:
        if r.uniboost_arrival_tokens is None:
            r.uniboost_arrival_tokens = processed_tokens
            r.uniboost_arrival_s = now_s


def sort_waiting_queue(
    waiting_queue: List[Req],
    *,
    gamma: float,
    bin_size: int,
    clock: UniBoostClock,
    rate_tokens_per_s: Optional[float],
    temporary_deprioritized: Set[str],
    enable_priority_scheduling: bool,
    priority_sign: int,
) -> None:
    """Sorts by ascending phi(r); see the module docstring.

    Wall clock: A(r) is in seconds and work becomes u(r) / rate seconds, so
    gamma is in 1/s; until the rate is measured the boost is 0 (FCFS).

    In-batch prefix-dedup followers sort last, as under the other cache-aware
    policies. Ties break by A(r), then rid.
    """
    wall = clock is UniBoostClock.WALL

    def arrival(r: Req) -> float:
        stamp = r.uniboost_arrival_s if wall else r.uniboost_arrival_tokens
        return stamp if stamp is not None else 0.0

    def phi(r: Req) -> float:
        w = geometric_bin(uncached_work(r), bin_size)
        if not wall:
            return arrival(r) - boost(w, gamma)
        if rate_tokens_per_s is None:
            return arrival(r)
        return arrival(r) - boost(w / rate_tokens_per_s, gamma)

    def key(r: Req):
        group, rank = 0, phi(r)
        if r.rid in temporary_deprioritized:
            group, rank = 1, 0.0
        if enable_priority_scheduling:
            return (r.priority * priority_sign, group, rank, arrival(r), r.rid)
        return (group, rank, arrival(r), r.rid)

    waiting_queue.sort(key=key)
