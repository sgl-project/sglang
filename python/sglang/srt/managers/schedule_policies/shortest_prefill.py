"""Shortest prefill first: the least uncached prefill work goes first."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import (
    BaseRanker,
    FullMatch,
    prefill_work,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


class ShortestPrefillFirstRanker(BaseRanker):
    ranks_by_prefill_work = True

    def __init__(self, tree_cache: BasePrefixCache, aging_factor: float):
        self.tree_cache = tree_cache
        self.aging_factor = aging_factor
        self._sort_calls = 0
        # The first of a shared-prefix group runs: make it the shortest, or a
        # short request waits behind a long one that SPF defers.
        self.match_plan = FullMatch(leader_key=partial(prefill_work, tree_cache))

    def prefill_rank(self, r: Req, work: int, processed_tokens: int) -> float:
        # Aging lowers the rank by the prefill work done while the request
        # waited, bounding its wait to about work / factor processed tokens.
        waited_tokens = max(0, processed_tokens - r.arrival_processed_tokens)
        return work - self.aging_factor * waited_tokens

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        # Prioritize short uncached prefills while deferring duplicate prefixes.
        waiting_queue.sort(
            key=lambda r: (
                r.rid in temporary_deprioritized,
                self.prefill_rank(
                    r, prefill_work(self.tree_cache, r), processed_tokens
                ),
                r.time_stats.wait_queue_entry_time,
            )
        )
        self._sort_calls += 1
        if waiting_queue and self._sort_calls % 128 == 1:
            logger.info(
                "Experimental shortest-prefill-first: queue=%d shortest_uncached=%d",
                len(waiting_queue),
                prefill_work(self.tree_cache, waiting_queue[0]),
            )
