"""Random order."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import BaseRanker

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RandomRanker(BaseRanker):
    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)
