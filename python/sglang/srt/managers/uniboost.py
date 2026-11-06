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
gamma can be fixed or adapted online from the response-time tail (gamma-Ada).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Commit-gate frontier: tokens compared just past a waiting request's current
# match point; an earlier extender that shares them will extend that match.
COMMIT_GATE_FRONTIER_TOKENS = 256
# Commit-gate cap on the cumulative hold of one request, in wall seconds. One
# extender's prefill takes well under a minute at 256k context on 8xH200, so a
# longer hold means the extender is not committing (e.g. retraction churn).
COMMIT_GATE_MAX_HOLD_S = 120.0


class UniBoostClock(Enum):
    """Axis of A(r) and u(r); see --uniboost-clock."""

    # Prefill tokens processed by this scheduler (the paper's clock).
    TOKENS = "tokens"
    # Wall seconds; work is converted at the measured token-clock rate.
    WALL = "wall"


# Unset --uniboost-gamma. Tokens: the paper's per-token scale. Wall: geometric
# center of the adaptive fit's fixed points measured at concurrency 4-16
# (0.017-0.62 1/s).
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


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (k = (n - 1) * p / 100)."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


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


class AdaptiveGamma:
    """gamma-Ada (paper Eq. 9): gamma_hat = ln(5) / (t99 - t95) over the last
    WINDOW completed-request response times (wall seconds), EMA-smoothed.

    Two additions to the paper's estimator:
    - The low clip is a max-jump bound: gamma is floored so the largest boost,
      b_gamma(bin_size / rate), stays within max_jump_s wall seconds. A raw
      gamma floor would bound starvation by an amount that scales with the
      service rate; a jump bound does not.
    - Size-aware fit: when the requests above median work have a fatter tail
      than those below it (the policy itself starving large requests under
      overload), fitting the overall tail drives gamma toward 0 and starves
      them further. The fit then uses the small-request tail instead.
    """

    WINDOW = 128
    MIN_SAMPLES = 32
    BETA = 0.2
    # Outer clip in 1/s; the jump floor usually binds first on the low side.
    GAMMA_MIN_S = 1e-3
    GAMMA_MAX_S = 15.0
    # The paper's main text derives ln(CCDF(t95) / CCDF(t99)) = ln 5; its
    # Algorithm 1 computes ln(99 / 95) instead. The main-text form is used.
    LN5 = math.log(5.0)
    # A collapsed band (t99 == t95) is a thin tail, i.e. gamma -> GAMMA_MAX_S.
    TAIL_EPS_S = 1e-6
    LOG_MIN_UPDATES = 64
    LOG_REL_CHANGE = 0.25

    def __init__(
        self,
        *,
        gamma0: float,
        bin_size: int,
        max_jump_s: float,
        window: int = WINDOW,
        min_samples: int = MIN_SAMPLES,
    ) -> None:
        self._samples: deque = deque(maxlen=window)
        self._min_samples = min_samples
        self._bin_size = bin_size
        self._max_jump_s = max_jump_s
        self._gamma = min(max(gamma0, self.GAMMA_MIN_S), self.GAMMA_MAX_S)
        self._updates_since_log = 0
        self._last_logged_gamma = self._gamma
        self._updates = 0
        self._pinned_updates = 0

    @property
    def gamma(self) -> float:
        return self._gamma

    def record(
        self,
        *,
        latencies_s: Iterable[float],
        works: Iterable[int],
        rate_tokens_per_s: Optional[float],
    ) -> Optional[float]:
        """Folds completed requests in; returns the new gamma, or None while
        fewer than min_samples have been seen."""
        for lat, work in zip(latencies_s, works):
            self._samples.append((max(0.0, float(lat)), work))
        if len(self._samples) < self._min_samples:
            return None
        gamma_floor = self._gamma_floor(rate_tokens_per_s)
        raw_fit, fit_src = self._tail_fit()
        gamma_hat = min(max(raw_fit, gamma_floor), self.GAMMA_MAX_S)
        gamma = min(
            max((1.0 - self.BETA) * self._gamma + self.BETA * gamma_hat, gamma_floor),
            self.GAMMA_MAX_S,
        )
        self._updates += 1
        # A fit at or below the floor means the jump bound, not the tail, set gamma.
        if raw_fit <= gamma_floor:
            self._pinned_updates += 1
        self._maybe_log(gamma, raw_fit, fit_src, gamma_floor)
        self._gamma = gamma
        return gamma

    def _tail_fit(self) -> tuple[float, str]:
        lats = [lat for lat, _ in self._samples]
        works = [work for _, work in self._samples]
        overall = self._tail_gamma_hat(lats)
        if len(works) < 2 * self._min_samples or min(works) == max(works):
            return overall, "overall"
        median_w = sorted(works)[(len(works) - 1) // 2]
        small = [lat for lat, work in self._samples if work <= median_w]
        large = [lat for lat, work in self._samples if work > median_w]
        if min(len(small), len(large)) < self._min_samples // 2:
            return overall, "overall"
        gamma_small = self._tail_gamma_hat(small)
        if self._tail_gamma_hat(large) < gamma_small:
            return gamma_small, "small-work"
        return overall, "overall"

    @classmethod
    def _tail_gamma_hat(cls, lats: List[float]) -> float:
        ordered = sorted(lats)
        width = max(_percentile(ordered, 99) - _percentile(ordered, 95), cls.TAIL_EPS_S)
        return cls.LN5 / width

    def _gamma_floor(self, rate_tokens_per_s: Optional[float]) -> float:
        """Smallest gamma with b_gamma(bin_size / rate) <= max_jump_s.

        b_gamma(tau) = tau * g(gamma * tau) with g(u) = -ln(1 - exp(-u)) / u
        strictly decreasing, so the floor is the root of g(u) = max_jump_s /
        tau, found by bisection. Without a measured rate the sort applies no
        boost at all, so there is no jump to bound.
        """
        if rate_tokens_per_s is None or rate_tokens_per_s <= 0.0:
            return self.GAMMA_MIN_S
        tau_s = self._bin_size / rate_tokens_per_s
        target = self._max_jump_s / tau_s
        lo, hi = 1e-9, 1e3  # g(1e-9) ~ 2e10 exceeds any target; g(1e3) ~ 0.
        for _ in range(64):
            mid = math.sqrt(lo * hi)
            if -math.log(-math.expm1(-mid)) / mid > target:
                lo = mid
            else:
                hi = mid
        return hi / tau_s

    def _maybe_log(
        self, gamma: float, raw_fit: float, fit_src: str, gamma_floor: float
    ) -> None:
        self._updates_since_log += 1
        rel = abs(gamma - self._last_logged_gamma) / self._last_logged_gamma
        if self._updates_since_log < self.LOG_MIN_UPDATES and rel < self.LOG_REL_CHANGE:
            return
        logger.info(
            "uniboost adaptive gamma: gamma=%.3e (fit=%.3e from %s tail, "
            "floor=%.3e, n=%d, pinned=%.0f%%)",
            gamma,
            raw_fit,
            fit_src,
            gamma_floor,
            len(self._samples),
            100.0 * self._pinned_updates / self._updates,
        )
        self._updates_since_log = 0
        self._last_logged_gamma = gamma


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
    leader_of: Dict[str, str],
    enable_priority_scheduling: bool,
    priority_sign: int,
) -> None:
    """Sorts by ascending phi(r); see the module docstring.

    Wall clock: A(r) is in seconds and work becomes u(r) / rate seconds, so
    gamma is in 1/s; until the rate is measured the boost is 0 (FCFS).

    In-batch prefix-dedup followers sort just behind their leader (the queued
    request whose admission populates the shared prefix), not behind every
    other request: a follower behind the whole queue also sits behind every
    later arrival and has no starvation bound. A follower whose leader left
    the queue goes last. Ties break by A(r), then rid.
    """
    req_by_rid = {r.rid: r for r in waiting_queue} if leader_of else {}
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

    # Smaller than any gap between two unrelated requests' phi in either unit.
    eps = 1e-3 if wall else 1.0

    def key(r: Req):
        group, rank = 0, phi(r)
        if r.rid in temporary_deprioritized:
            leader = req_by_rid.get(leader_of.get(r.rid))
            if leader is not None:
                rank = phi(leader) + eps
            else:
                group, rank = 1, 0.0
        if enable_priority_scheduling:
            return (r.priority * priority_sign, group, rank, arrival(r), r.rid)
        return (group, rank, arrival(r), r.rid)

    waiting_queue.sort(key=key)


def mark_commit_holds(
    waiting_queue: Sequence[Req], *, chunked_req: Optional[Req], now_s: float
) -> None:
    """Holds a request out of this admission pass while an earlier extender
    will extend its matched prefix by at least one frontier page.

    The extender is the in-flight chunked prefill or a waiting request that
    sorts (and so admits) before it. Admitting first would prefill, uncached,
    the tokens the extender is about to commit to the radix tree. The hold
    ends with that one extender; COMMIT_GATE_MAX_HOLD_S bounds the total.
    waiting_queue must already be in admission order.
    """
    frontier = COMMIT_GATE_FRONTIER_TOKENS

    def extends_match(extender: Optional[Req], k: int, ids) -> bool:
        if extender is None:
            return False
        e_ids = extender.origin_input_ids
        return (
            len(e_ids) >= k + frontier
            and e_ids[k : k + frontier] == ids[k : k + frontier]
        )

    for i, r in enumerate(waiting_queue):
        r.uniboost_commit_hold = False
        k = r.num_matched_prefix_tokens
        if k <= 0:
            continue
        ids = r.origin_input_ids
        if not (
            extends_match(chunked_req, k, ids)
            or any(extends_match(w, k, ids) for w in waiting_queue[:i])
        ):
            continue
        if r.uniboost_commit_hold_first_s is None:
            r.uniboost_commit_hold_first_s = now_s
        if now_s - r.uniboost_commit_hold_first_s <= COMMIT_GATE_MAX_HOLD_S:
            r.uniboost_commit_hold = True


def completed_response_times(reqs: Sequence[Req]) -> tuple[List[float], List[int]]:
    """(scheduler-recv -> completion seconds, total tokens) of the finished
    requests in reqs; requests missing either stamp are skipped."""
    latencies, works = [], []
    for r in reqs:
        ts = r.time_stats
        if (
            not r.finished()
            or ts.completion_time <= 0.0
            or ts.scheduler_recv_time <= 0.0
        ):
            continue
        latencies.append(ts.completion_time - ts.scheduler_recv_time)
        works.append(len(r.origin_input_ids) + len(r.output_ids))
    return latencies, works
