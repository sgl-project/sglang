"""Skip-join multi-level feedback queue (FastServe) with cache-aware placement.

A request joins the lowest queue level whose quantum covers its predicted first
iteration (its prefill), instead of always starting at the top. While it waits
and has not run yet, it is re-priced every ordering against the current cache
and may only move to a better level, so a request whose shared prefix another
request has just prefilled is not stuck at the level it was given before.
After each forward pass it is charged the pass's time and drops one level once
it has used up its level's quantum; a request left waiting too long is promoted
back to level 0. The level is stored as ``req.priority = -level``, so SGLang's
priority preemption and the priority retraction policy follow MLFQ order.

Every attention-parallel rank runs its own scheduler and must place, demote and
promote identically, so the policy never reads the wall clock: ``charge``
receives each batch's elapsed time and end timestamp as measured on one rank
and broadcast to the others (``batch_result_processor._charge_mlfq``), and
keeps that timestamp as its clock for level entry times and the starvation
check.

The placement cost is either FastServe's (a fresh prefill of the whole prompt)
or cache-aware (the uncached suffix prefilled on top of the cached prefix). It
comes from a seed cost shape whose compute and attention scales are fitted
online to the prefill batches the scheduler runs, so no hardware profile is
needed.

The prefill gate (``prefill_gate_level``) is FastServe's iteration-level
preemption in SGLang's step model: a new prefill starts only when the head of
the queue is at least as urgent as the most urgent running request; otherwise
the step is a decode step.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import TYPE_CHECKING, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)

# FastServe's settings (Wu et al., arXiv:2305.05920): a 10 ms base quantum that
# doubles per level, and promotion after 3 s without service, checked every
# 1000 scheduling steps. The base quantum is a server argument because it
# should track the hardware's decode step time.
DEFAULT_BASE_QUANTUM_S = 0.010
QUANTUM_MULTIPLIER = 2.0
STARVATION_S = 3.0
STARVATION_CHECK_INTERVAL = 1000

COST_MODES = ("cache-aware", "prompt")

# Seed for the online cost model, in seconds: per uncached token, per (uncached x
# cached) token pair, per cached token. Fitted to Qwen3-8B on an A100; levels are
# logarithmic, so a factor-two error elsewhere costs one level until the fit corrects.
DEFAULT_COST_SEED = (7.2e-5, 2.8e-9, 6e-7)


class OnlinePrefillCostModel:
    """Prefill time as compute * a*u + attention * (b*u*c + d*c), fitted online.

    (a, b, d) is a seed shape: dense compute per uncached token u, attention of new
    tokens over the cached prefix c, and the per-chunk read of the prefix KV. The two
    scale factors are fitted by ridge least squares with exponential forgetting from
    the prefill batches the scheduler runs. Two factors, not three free coefficients,
    because u*c and c are nearly collinear in real batches and a free fit drifts; a
    wrong shape costs at most about one level.
    """

    FORGETTING = 0.98
    # Weight of the seed factors (1, 1): about one typical batch (x^2 ~ 0.1-0.4 s^2).
    RIDGE = 0.2
    FACTOR_RANGE = (0.1, 10.0)

    def __init__(self, seed: Tuple[float, float, float] = DEFAULT_COST_SEED):
        self._seed = seed
        self._factors = [1.0, 1.0]
        self._ata = [[self.RIDGE, 0.0], [0.0, self.RIDGE]]
        self._atb = [self.RIDGE, self.RIDGE]
        self.observations = 0

    def _features(self, rows: Sequence[Tuple[int, int]]) -> Tuple[float, float]:
        a, b, d = self._seed
        compute = attention = 0.0
        for u, c in rows:
            compute += a * u
            attention += b * u * c + d * c
        return compute, attention

    def observe(self, rows: Sequence[Tuple[int, int]], elapsed_s: float) -> None:
        """One forward pass over ``rows`` = [(uncached, cached), ...] that took ``elapsed_s``."""
        if not rows or not 0 < elapsed_s < 60:
            return
        x = self._features(rows)
        lam = self.FORGETTING
        for i in range(2):
            for j in range(2):
                self._ata[i][j] = lam * self._ata[i][j] + x[i] * x[j]
            # The seed anchor is topped up so forgetting never erodes it.
            self._ata[i][i] += (1 - lam) * self.RIDGE
            self._atb[i] = (
                lam * self._atb[i] + x[i] * elapsed_s + (1 - lam) * self.RIDGE
            )
        self.observations += 1
        (p, q), (r, t) = self._ata
        det = p * t - q * r
        if abs(det) < 1e-12:
            return
        lo, hi = self.FACTOR_RANGE
        f0 = (t * self._atb[0] - q * self._atb[1]) / det
        f1 = (p * self._atb[1] - r * self._atb[0]) / det
        self._factors = [min(max(f0, lo), hi), min(max(f1, lo), hi)]

    def predict_s(self, *, uncached: int, cached: int) -> float:
        compute, attention = self._features([(uncached, cached)])
        return max(0.0, self._factors[0] * compute + self._factors[1] * attention)


class SkipJoinMlfq:
    """Queue levels, demotion and starvation promotion for ``--schedule-policy sj-mlfq``."""

    def __init__(
        self,
        *,
        cost_mode: str,
        base_quantum_s: float = DEFAULT_BASE_QUANTUM_S,
    ):
        if cost_mode not in COST_MODES:
            raise ValueError(
                f"--sj-mlfq-cost must be one of {COST_MODES}, got {cost_mode!r}"
            )
        if base_quantum_s <= 0:
            raise ValueError(
                f"--sj-mlfq-base-quantum-ms must be positive, got {base_quantum_s * 1000}"
            )
        self._base_quantum_s = base_quantum_s
        self._cache_aware = cost_mode == "cache-aware"
        self._cost_model = OnlinePrefillCostModel()
        self._steps = 0
        # Timestamp of the last charged batch result, the same on every rank.
        self._clock = 0.0
        self._placed_per_level: Counter = Counter()
        self._demotions = 0
        self._promotions = 0
        self._repricings = 0

    def order(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Optional[Set[str]] = None,
    ) -> None:
        """Place new requests, promote starved ones, then sort by (level, level entry time).

        ``temporary_deprioritized`` is SGLang's in-batch prefix-caching set: waiting
        requests whose uncached prefix another waiting request is about to prefill.
        They go to the back for this ordering so the prefix is computed once.
        """
        now = self._clock
        for req in waiting_queue:
            if req.mlfq_level is None:
                self._place(req, now)
            elif req.mlfq_runs == 0:
                self._reprice(req)
        self._steps += 1
        if self._steps % STARVATION_CHECK_INTERVAL == 0:
            for req in waiting_queue:
                if (
                    req.mlfq_level > 0
                    and now - req.mlfq_last_service_time >= STARVATION_S
                ):
                    # Keeps its entry time, so it lands ahead of newer level-0 requests.
                    self._set_level(req, 0)
                    self._promotions += 1
            logger.info(
                "sj-mlfq: placed per level %s, demotions %d, starvation promotions %d, "
                "repricings %d",
                dict(sorted(self._placed_per_level.items())),
                self._demotions,
                self._promotions,
                self._repricings,
            )
        deprioritized = temporary_deprioritized or ()
        waiting_queue.sort(
            key=lambda r: (
                r.rid in deprioritized,
                r.mlfq_level,
                r.mlfq_level_entry_time,
            )
        )

    def charge(self, reqs: List[Req], elapsed_s: float, *, now: float) -> None:
        """Charge one forward pass to each request in it; demote those past their quantum.

        ``elapsed_s`` and ``now`` (the pass's end timestamp) must be the same on every
        rank; they advance the policy's clock.
        """
        self._clock = now = max(self._clock, now)
        for req in reqs:
            if req.mlfq_level is None:
                continue
            req.mlfq_runs += 1
            req.mlfq_service_s += elapsed_s
            req.mlfq_last_service_time = now
            if req.mlfq_service_s > self.quantum_s(req.mlfq_level):
                self._set_level(req, req.mlfq_level + 1)
                req.mlfq_level_entry_time = now
                self._demotions += 1

    def observe_batch(self, batch: ScheduleBatch, elapsed_s: float) -> None:
        """Feed a finished prefill batch's rows and time to the online fit."""
        if batch.forward_mode.is_extend() and batch.extend_lens is not None:
            self._cost_model.observe(
                list(zip(batch.extend_lens, batch.prefix_lens)), elapsed_s
            )

    @staticmethod
    def prefill_gate_level(running_reqs: List[Req]) -> Optional[int]:
        """The worst level a waiting request may have to be prefilled now, or None.

        SGLang runs a waiting request's prefill before the running batch's next
        decode step, so a long, low-priority prefill stalls every running request;
        FastServe runs the highest-priority work each iteration instead. Only
        requests at least as urgent as the most urgent running request pass;
        with nothing running, everything does.
        """
        levels = [r.mlfq_level for r in running_reqs if r.mlfq_level is not None]
        return min(levels) if levels else None

    def quantum_s(self, level: int) -> float:
        return self._base_quantum_s * QUANTUM_MULTIPLIER**level

    def _placement_level(self, req: Req) -> int:
        prompt_len = len(req.origin_input_ids)
        if self._cache_aware:
            # Counts both GPU-resident and CPU-resident (HiCache) prefix hits.
            cached = req.num_matched_prefix_tokens
            cost_s = self._cost_model.predict_s(
                uncached=prompt_len - cached, cached=cached
            )
        else:
            cost_s = self._cost_model.predict_s(uncached=prompt_len, cached=0)
        if cost_s <= self._base_quantum_s:
            return 0
        return math.ceil(math.log(cost_s / self._base_quantum_s, QUANTUM_MULTIPLIER))

    def _place(self, req: Req, now: float) -> None:
        level = self._placement_level(req)
        self._set_level(req, level)
        req.mlfq_level_entry_time = now
        req.mlfq_last_service_time = now
        self._placed_per_level[level] += 1

    def _reprice(self, req: Req) -> None:
        # Only toward a better level: a promoted request keeps level 0, and the cost
        # never rises while HiCache counts host-resident prefixes as cached. The level
        # entry time stays, so the request keeps its place among its new peers.
        level = self._placement_level(req)
        if level < req.mlfq_level:
            self._set_level(req, level)
            self._repricings += 1

    @staticmethod
    def _set_level(req: Req, level: int) -> None:
        req.mlfq_level = level
        req.priority = -level
        req.mlfq_service_s = 0.0
