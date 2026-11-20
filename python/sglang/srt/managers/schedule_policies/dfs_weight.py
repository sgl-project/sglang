"""Depth-first search weighting over the prefix tree."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import BaseRanker, FullMatch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class DfsWeightRanker(BaseRanker):
    match_plan = FullMatch()

    def __init__(self, tree_cache: BasePrefixCache):
        self.tree_cache = tree_cache

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        order = self.tree_cache.dfs_weight_order(
            [req.last_node for req in waiting_queue]
        )
        waiting_queue[:] = [waiting_queue[index] for index in order]
