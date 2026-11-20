"""Longest prefix match, and its bounded form for long queues."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple

from sglang.srt.managers.schedule_policies.base import (
    WAITING_QUEUE_PREFIX_MATCH_MAX,
    BaseRanker,
    FullMatch,
)
from sglang.srt.managers.schedule_policies.lpm_opportunity import (
    RECHECK_PASSES,
    competition_opportunity,
)
from sglang.srt.mem_cache.radix_cache import RadixCache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

# Bound matching work and cross-batch overtaking without a new server argument.
BOUNDED_LPM_WINDOW_SIZE = 64


class LpmRanker(BaseRanker):
    match_plan = FullMatch()
    fcfs_fallback_queue_len = WAITING_QUEUE_PREFIX_MATCH_MAX

    def __init__(
        self,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        bounded_enabled: bool,
    ):
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self._gate_calls = 0
        self._bounded_enabled = bounded_enabled
        self._bounded_active = False
        self._bounded_members: List[Req] = []
        self._bounded_next_members: List[Req] = []
        self._bounded_next_selected = False
        self._bounded_skipped: set[int] = set()

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

    @property
    def bounded_active(self) -> bool:
        return self._bounded_active

    def bounded_engaged(self, num_waiting: int) -> bool:
        return self._bounded_active or (
            num_waiting > WAITING_QUEUE_PREFIX_MATCH_MAX and self._bounded_enabled
        )

    def refresh_bounded_members(self, waiting_queue: List[Req]) -> bool:
        """Prune departed members and pick the next batch; returns whether
        bounded LPM engaged or released."""
        was_active = self._bounded_active
        if was_active:
            live = {id(req) for req in waiting_queue}
            self._bounded_members = [r for r in self._bounded_members if id(r) in live]
            self._bounded_next_members = [
                r for r in self._bounded_next_members if id(r) in live
            ]
        if not self._bounded_members:
            self._bounded_members = self._bounded_next_members
            self._bounded_next_members = []
            self._bounded_next_selected = False
            if not self._bounded_members:
                self._bounded_members = (
                    waiting_queue[:BOUNDED_LPM_WINDOW_SIZE]
                    if len(waiting_queue) > WAITING_QUEUE_PREFIX_MATCH_MAX
                    and self._should_start_bounded_batch(waiting_queue)
                    else []
                )
        # Finish committed batches even after the queue drops below 128.
        self._bounded_active = bool(self._bounded_members)
        return self._bounded_active != was_active

    def _should_start_bounded_batch(self, waiting_queue: List[Req]) -> bool:
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        # Ranks sharing a queue execute the same eligible scheduling passes.
        # Wall-clock deadlines can disagree across TP ranks and split admission.
        self._gate_calls += 1
        if (self._gate_calls - 1) % RECHECK_PASSES:
            return False
        self._gate_competition = None
        cache = self.tree_cache
        # Fail closed for cache modes not represented in the observation study.
        if type(cache) not in (RadixCache, UnifiedRadixCache):
            return False
        if (
            cache.disable
            or self.enable_hierarchical_cache
            or cache.cache_controller is not None
            or getattr(cache, "is_swa_enabled", False)
            or getattr(cache, "is_mamba_enabled", False)
            or getattr(cache, "enable_session_radix_cache", False)
        ):
            return False
        allocator = cache.token_to_kv_pool_allocator
        capacity = getattr(allocator, "size", 0)
        if not isinstance(capacity, int) or capacity <= 0:
            return False
        candidates = waiting_queue[:BOUNDED_LPM_WINDOW_SIZE]
        self._gate_competition = competition_opportunity(
            candidates, capacity, include_pairs=False
        )
        return self._gate_competition["allowed"]

    def split_waiting(self, waiting_queue: List[Req]) -> Tuple[List[Req], List[Req]]:
        """The current batch, and the waiting requests outside it."""
        self._bounded_skipped.clear()
        members = {id(req) for req in self._bounded_members}
        tail = [r for r in waiting_queue if id(r) not in members]
        return self._bounded_members, tail

    def bounded_order(
        self, temporary_deprioritized: Set[int], tail: List[Req]
    ) -> List[Req]:
        """The current batch ranked by LPM, then the fallback batch, then the rest."""
        self.sort(self._bounded_members, temporary_deprioritized)
        next_ids = {id(r) for r in self._bounded_next_members}
        return (
            self._bounded_members
            + self._bounded_next_members
            + [r for r in tail if id(r) not in next_ids]
        )

    def admission_members(self) -> List[Req]:
        self._bounded_skipped.clear()
        return self._bounded_members

    def fallback_members(self, waiting: Iterable[Req]) -> List[Req]:
        """At most one fixed fallback batch, once a current member was skipped."""
        if not self._bounded_skipped:
            return []
        if not self._bounded_next_selected:
            members = {id(r) for r in self._bounded_members}
            self._bounded_next_members = [r for r in waiting if id(r) not in members][
                :BOUNDED_LPM_WINDOW_SIZE
            ]
            self._bounded_next_selected = True
        return self._bounded_next_members

    def retire_waiting_members(self, reqs: Iterable[Req]) -> None:
        """Retire on departure; never refill the next batch within an epoch."""
        if self._bounded_members or self._bounded_next_members:
            leaving = {id(req) for req in reqs}
            self._bounded_members = [
                req for req in self._bounded_members if id(req) not in leaving
            ]
            self._bounded_next_members = [
                req for req in self._bounded_next_members if id(req) not in leaving
            ]
            self._bounded_skipped.difference_update(leaving)

    def mark_admission_skipped(self, req: Req) -> None:
        """Record only a current member skipped for a temporary prerequisite."""
        if self._bounded_active and any(
            req is member for member in self._bounded_members
        ):
            self._bounded_skipped.add(id(req))

    def allows_admission(self, req: Req) -> bool:
        if not self._bounded_active:
            return True
        return any(req is member for member in self._bounded_members) or (
            bool(self._bounded_skipped)
            and any(req is member for member in self._bounded_next_members)
        )
