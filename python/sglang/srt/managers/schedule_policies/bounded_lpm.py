"""Bounded LPM: on a long queue, LPM admits from a fixed batch at its head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from sglang.srt.managers.schedule_policies.base import WAITING_QUEUE_PREFIX_MATCH_MAX
from sglang.srt.managers.schedule_policies.lpm_opportunity import (
    RECHECK_PASSES,
    competition_opportunity,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

# Bound matching work and cross-batch overtaking without a new server argument.
BOUNDED_LPM_WINDOW_SIZE = 64


class BoundedLpmAdmission:
    """Admission on a long LPM queue: the fixed batch it admits from, and the
    one fallback batch behind it once a member is skipped."""

    def __init__(self, tree_cache: BasePrefixCache, enable_hierarchical_cache: bool):
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self._gate_calls = 0
        self._gate_competition: Optional[dict] = None
        self.active = False
        self.members: List[Req] = []
        self._next_members: List[Req] = []
        self._next_selected = False
        self._skipped: set[int] = set()

    def refresh(self, waiting_queue: List[Req]) -> bool:
        """Prune departed members and pick the next batch; returns whether
        bounded LPM engaged or released."""
        was_active = self.active
        if not was_active and len(waiting_queue) <= WAITING_QUEUE_PREFIX_MATCH_MAX:
            return False
        if was_active:
            live = {id(req) for req in waiting_queue}
            self.members = [r for r in self.members if id(r) in live]
            self._next_members = [r for r in self._next_members if id(r) in live]
        if not self.members:
            self.members = self._next_members
            self._next_members = []
            self._next_selected = False
            if not self.members:
                self.members = (
                    waiting_queue[:BOUNDED_LPM_WINDOW_SIZE]
                    if len(waiting_queue) > WAITING_QUEUE_PREFIX_MATCH_MAX
                    and self._should_start_batch(waiting_queue)
                    else []
                )
        # Finish committed batches even after the queue shrinks to
        # WAITING_QUEUE_PREFIX_MATCH_MAX or fewer.
        self.active = bool(self.members)
        return self.active != was_active

    def _should_start_batch(self, waiting_queue: List[Req]) -> bool:
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

    def split(self, waiting_queue: List[Req]) -> Tuple[List[Req], List[Req]]:
        """The current batch, and the waiting requests outside it. The batch is
        returned as is: sorting it in place sets the order it admits in."""
        members = {id(req) for req in self.members}
        tail = [r for r in waiting_queue if id(r) not in members]
        return self.members, tail

    def order(self, tail: List[Req]) -> List[Req]:
        """The current batch, then the fallback batch, then the rest."""
        next_ids = {id(r) for r in self._next_members}
        return (
            self.members
            + self._next_members
            + [r for r in tail if id(r) not in next_ids]
        )

    def reset_skipped(self) -> None:
        """Start a pass with no current member skipped."""
        self._skipped.clear()

    def fallback_members(self, waiting: Iterable[Req]) -> List[Req]:
        """At most one fixed fallback batch, once a current member was skipped.
        The batch is returned as is: sorting it in place sets its order."""
        if not self._skipped:
            return []
        if not self._next_selected:
            members = {id(r) for r in self.members}
            self._next_members = [r for r in waiting if id(r) not in members][
                :BOUNDED_LPM_WINDOW_SIZE
            ]
            self._next_selected = True
        return self._next_members

    def retire(self, reqs: Iterable[Req]) -> None:
        """Retire on departure; never refill the next batch within an epoch."""
        if self.members or self._next_members:
            leaving = {id(req) for req in reqs}
            self.members = [req for req in self.members if id(req) not in leaving]
            self._next_members = [
                req for req in self._next_members if id(req) not in leaving
            ]
            self._skipped.difference_update(leaving)

    def mark_skipped(self, req: Req) -> None:
        """Record only a current member skipped for a temporary prerequisite."""
        if self.active and any(req is member for member in self.members):
            self._skipped.add(id(req))

    def allows_admission(self, req: Req) -> bool:
        if not self.active:
            return True
        return any(req is member for member in self.members) or (
            bool(self._skipped) and any(req is member for member in self._next_members)
        )
