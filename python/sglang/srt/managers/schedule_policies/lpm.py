"""Longest prefix match."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import BaseRanker, FullMatch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class LpmRanker(BaseRanker):
    match_plan = FullMatch()
    falls_back_to_fcfs = True

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -r.num_matched_prefix_tokens
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )
