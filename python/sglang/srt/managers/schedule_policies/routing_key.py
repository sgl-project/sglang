"""Routing key: requests whose key is frequent in the running batch go first."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import BaseRanker
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")
logger = logging.getLogger(__name__)


class RoutingKeyRanker(BaseRanker):
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
