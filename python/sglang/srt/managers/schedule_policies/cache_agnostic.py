"""The cache-agnostic rankers: they order the queue without prefix matches."""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import BaseRanker
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")
logger = logging.getLogger(__name__)


class FcfsRanker(BaseRanker):
    """First come first serve."""

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
    """First come first serve, ordered by request priority first."""

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
        """Sorts the waiting queue based on the request priority then received timestamp."""
        waiting_queue.sort(
            key=lambda x: (
                x.priority * self.priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )


class LofRanker(BaseRanker):
    """Longest output first."""

    def __init__(self, priority_sign: Optional[int]):
        """``priority_sign`` orders by request priority first; None skips it."""
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
        if self.priority_sign is not None:
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * self.priority_sign,
                    -x.sampling_params.max_new_tokens,
                )
            )
        else:
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)


class RandomRanker(BaseRanker):
    """Random order."""

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


class RoutingKeyRanker(BaseRanker):
    """Routing key: requests whose key is frequent in the running batch go first."""

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch."""
        if running_reqs is None:
            return
        routing_key_counts = Counter(
            r.routing_key for r in running_reqs if r.routing_key
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_before = [r.routing_key for r in waiting_queue]
            logger.info(
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        if not routing_key_counts:
            return

        def sort_key(req: Req):
            key = req.routing_key
            if key and key in routing_key_counts:
                count = routing_key_counts[key]
                return (0, -count, key)
            else:
                return (1, 0, key or "")

        waiting_queue.sort(key=sort_key)

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_after = [r.routing_key for r in waiting_queue]
            logger.info(f"waiting_keys_after={waiting_keys_after}")
