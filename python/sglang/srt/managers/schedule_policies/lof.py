"""Longest output first."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import BaseRanker

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class LofRanker(BaseRanker):
    def __init__(self, enable_priority_scheduling: bool, priority_sign: int):
        self.enable_priority_scheduling = enable_priority_scheduling
        self.priority_sign = priority_sign

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens). If using priority scheduling, sort by priority first."""
        if self.enable_priority_scheduling:
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * self.priority_sign,
                    -x.sampling_params.max_new_tokens,
                )
            )
        else:
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)
