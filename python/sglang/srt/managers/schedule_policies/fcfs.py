"""First come first serve, optionally ordered by request priority first."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import BaseRanker

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FcfsRanker(BaseRanker):
    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Keeps the queue in arrival order."""


class PriorityFcfsRanker(BaseRanker):
    def __init__(self, priority_sign: int):
        self.priority_sign = priority_sign

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp."""
        waiting_queue.sort(
            key=lambda x: (
                x.priority * self.priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )
