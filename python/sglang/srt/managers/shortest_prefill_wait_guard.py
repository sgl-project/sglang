"""Occasionally admit an overtaken request ahead of the SPF queue head."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from sglang.srt.arg_groups.fields.schedule import Schedule
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import ParallelContext


class ShortestPrefillWaitGuard:
    def __init__(
        self, wait_seconds: float, age_gap_seconds: float, cooldown_seconds: float
    ):
        self.wait_seconds = wait_seconds
        self.age_gap_seconds = age_gap_seconds
        self.cooldown_seconds = cooldown_seconds
        self.next_promotion_time = 0.0

    @classmethod
    def from_config(
        cls, schedule: Schedule, parallel: ParallelContext, *, is_pd_prefill: bool
    ) -> Optional[ShortestPrefillWaitGuard]:
        wait = schedule.shortest_prefill_wait_guard_seconds
        if not math.isfinite(wait) or wait < 0:
            raise ValueError(
                "--shortest-prefill-wait-guard-seconds must be finite and >= 0"
            )
        if wait == 0:
            return None
        if not is_pd_prefill or schedule.schedule_policy != "shortest-prefill-first":
            raise ValueError(
                "The SPF wait guard requires PD prefill and "
                "--schedule-policy shortest-prefill-first"
            )
        # The wall clock belongs to the sole scheduler of each attention DP queue.
        # Replicated scheduling would require a shared decision at the thresholds.
        if not (
            parallel.attn_tp_size == parallel.attn_cp_size == parallel.pp_size == 1
        ):
            raise ValueError(
                "The SPF wait guard requires attention TP=1, attention CP=1 and PP=1"
            )
        gap = schedule.shortest_prefill_wait_guard_age_gap_seconds
        cooldown = schedule.shortest_prefill_wait_guard_cooldown_seconds
        if not math.isfinite(gap) or gap <= 0:
            raise ValueError(
                "--shortest-prefill-wait-guard-age-gap-seconds must be finite and > 0"
            )
        if not math.isfinite(cooldown) or cooldown <= 0:
            raise ValueError(
                "--shortest-prefill-wait-guard-cooldown-seconds must be finite and > 0"
            )
        return cls(wait, gap, cooldown)

    @staticmethod
    def _arrival(req: Req) -> float:
        stats = req.time_stats
        return (
            stats.scheduler_recv_time
            or stats.prefill_bootstrap_queue_entry_time
            or stats.wait_queue_entry_time
        )

    def pick(
        self,
        waiting_queue: Sequence[Req],
        *,
        now: float,
        has_active_chunk: bool,
        priority_scheduling: bool = False,
    ) -> Optional[Req]:
        if has_active_chunk or len(waiting_queue) < 2 or now < self.next_promotion_time:
            return None

        head = waiting_queue[0]
        head_arrival = self._arrival(head)
        if not head_arrival or head.finished():
            return None
        cutoff = min(now - self.wait_seconds, head_arrival - self.age_gap_seconds)
        oldest = None
        oldest_arrival = math.inf
        for req in waiting_queue:
            arrival = self._arrival(req)
            if (
                0 < arrival <= cutoff
                and arrival < oldest_arrival
                and not req.finished()
                and (not priority_scheduling or req.priority == head.priority)
            ):
                oldest, oldest_arrival = req, arrival
        return oldest

    def record_admission(self, now: float) -> None:
        self.next_promotion_time = now + self.cooldown_seconds
