"""UniBoost: tail-aware admission boosting."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policies.base import (
    BaseRanker,
    HeadWindowMatch,
    prefill_work,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policies.uniboost_gamma_ada import (
        GammaAdaController,
    )
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)

# UniBoost re-matches this many requests at the head of the sorted queue each pass.
UNIBOOST_MATCH_WINDOW_SIZE = 64


class UniBoostRanker(BaseRanker):
    ranks_by_prefill_work = True

    def __init__(
        self,
        tree_cache: BasePrefixCache,
        gamma: float,
        gamma_ada: Optional[GammaAdaController],
        rank_invariant: bool,
    ):
        """``gamma_ada``, when given, supplies the current gamma in place of
        ``gamma``; the ranker only reads it."""
        self.tree_cache = tree_cache
        self.gamma = gamma
        self.alpha_s_per_token = envs.SGLANG_UNIBOOST_ALPHA_US_PER_TOKEN.get() / 1e6
        self.rank_invariant = rank_invariant
        self.gamma_ada = gamma_ada
        # A shared-prefix group's leader is its highest-priority member.
        self.match_plan = HeadWindowMatch(
            window=UNIBOOST_MATCH_WINDOW_SIZE, leader_key=self._rank_key
        )
        alpha = self.alpha_s_per_token
        if not (math.isfinite(gamma) and gamma > 0):
            raise ValueError(f"uniboost_gamma must be finite and > 0, got {gamma}")
        if not (math.isfinite(alpha) and alpha > 0):
            raise ValueError(
                "SGLANG_UNIBOOST_ALPHA_US_PER_TOKEN must be finite and > 0, got "
                f"{alpha * 1e6}"
            )
        # The largest boost, b(alpha) ~= -ln(gamma * alpha) / gamma, must be finite.
        if not (gamma * alpha > 0 and math.isfinite(math.log(gamma * alpha) / gamma)):
            raise ValueError(f"uniboost_gamma={gamma} is too small for alpha={alpha}")
        logger.info(
            "UniBoost enabled: gamma=%.4g /s, alpha=%.1f us/token, gamma-Ada=%s, "
            "arrival clock=%s",
            self._current_gamma(),
            alpha * 1e6,
            "on" if gamma_ada is not None else "off",
            "processed tokens" if rank_invariant else "wall",
        )

    def prefill_rank(self, r: Req, work: int, processed_tokens: int) -> float:
        """``Phi = arrival - b(alpha * work)`` in seconds; lower is admitted first.

        ``b(w) = ln(1 / (1 - e^(-gamma * w))) / gamma`` decays to 0 for long work,
        so a long request keeps its FCFS place and is never starved.
        """
        if self.rank_invariant:
            # Processed prefill tokens advance identically on every rank.
            arrival = self.alpha_s_per_token * r.arrival_processed_tokens
        else:
            arrival = r.time_stats.wait_queue_entry_time
        gamma = self._current_gamma()
        # A resumed chunk may have no work left; b(0) is unbounded.
        gamma_w = gamma * self.alpha_s_per_token * max(work, 1)
        return arrival + math.log(-math.expm1(-gamma_w)) / gamma

    def _current_gamma(self) -> float:
        if self.gamma_ada is not None:
            return self.gamma_ada.gamma
        return self.gamma

    def _rank_key(self, r: Req) -> float:
        return self.prefill_rank(r, prefill_work(self.tree_cache, r), 0)

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        waiting_queue.sort(key=self._rank_key)
