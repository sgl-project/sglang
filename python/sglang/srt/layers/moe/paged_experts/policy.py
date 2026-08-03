"""Residency (eviction) policy for the eager keep-warm decision.

When a decode step routes to an expert that is not resident, the pager must evict one of the slots whose
expert is *not needed this step* to make room. *Which* one is the policy — and that is the only thing
these classes decide. The pager (``pager.py``) owns the slot<->expert maps, the page-in plan, and the
per-step mechanism; a ``ResidencyPolicy`` only tracks its own recency/frequency bookkeeping and answers
``pick_victim``. Two strategies ship:

* ``LRUPolicy`` — evict the least-recently-used non-needed slot (the default; matches the original
  keep-warm behavior).
* ``LFUPolicy`` — evict the slot whose expert has the lowest use count, least-recently-used as a
  tiebreak. Per-expert frequency persists across eviction/re-paging, so a hot expert that briefly leaves
  is favored back in — better than LRU for skewed expert routing.

The on-device captured path mirrors this with a kernel flag (the ``decide`` kernel's LRU/LFU selector);
``--paged-experts-eviction`` drives both. New strategies (predictive, pinned-hot-set) are additional
``ResidencyPolicy`` subclasses with no change to the pager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set


class ResidencyPolicy(ABC):
    """Picks which non-needed resident slot to evict for a miss (eager keep-warm).

    ``lastuse[slot]`` is the step a slot was last used (a monotonic per-call clock); subclasses may keep
    further state (e.g. per-expert frequency). ``begin_step`` advances the clock once per decode step;
    ``record_use(expert, slot)`` is called for each resident hit and for each fresh assignment.
    """

    name = "?"

    def __init__(self, num_resident_K: int, num_experts_E: int):
        self.K = num_resident_K
        self.E = num_experts_E
        self.step = 0
        self.lastuse: List[int] = [0] * num_resident_K

    def begin_step(self) -> None:
        self.step += 1

    def record_use(self, expert: int, slot: int) -> None:
        self.lastuse[slot] = self.step

    @abstractmethod
    def pick_victim(self, slot_expert: List[int], needed: Set[int]) -> int:
        """Return a non-needed resident slot to evict, or -1 if every slot holds a needed expert."""


class LRUPolicy(ResidencyPolicy):
    """Evict the least-recently-used slot whose expert is not needed this step (lowest index on ties)."""

    name = "lru"

    def pick_victim(self, slot_expert: List[int], needed: Set[int]) -> int:
        victim, best = -1, None
        for s in range(self.K):
            if slot_expert[s] in needed:
                continue
            if best is None or self.lastuse[s] < best:
                best, victim = self.lastuse[s], s
        return victim


class LFUPolicy(ResidencyPolicy):
    """Evict the non-needed slot whose expert has the lowest use count, LRU as the tiebreak. Per-expert
    frequency persists across eviction (it indexes experts, not slots), so frequently-routed experts are
    favored to stay resident — better than LRU when expert routing is skewed."""

    name = "lfu"

    def __init__(self, num_resident_K: int, num_experts_E: int):
        super().__init__(num_resident_K, num_experts_E)
        self.freq: List[int] = [0] * num_experts_E

    def record_use(self, expert: int, slot: int) -> None:
        super().record_use(expert, slot)
        if expert >= 0:
            self.freq[expert] += 1

    def pick_victim(self, slot_expert: List[int], needed: Set[int]) -> int:
        victim, best = -1, None
        for s in range(self.K):
            e = slot_expert[s]
            if e in needed:
                continue
            key = (self.freq[e] if e >= 0 else -1, self.lastuse[s])  # LFU, LRU tiebreak
            if best is None or key < best:
                best, victim = key, s
        return victim


_POLICIES = {LRUPolicy.name: LRUPolicy, LFUPolicy.name: LFUPolicy}


def make_residency_policy(
    name: str, num_resident_K: int, num_experts_E: int
) -> ResidencyPolicy:
    """Build the eager residency policy named by ``--paged-experts-eviction`` (``lru`` | ``lfu``)."""
    try:
        cls = _POLICIES[name]
    except KeyError:
        raise ValueError(
            f"[paged-experts] unknown eviction policy {name!r}; choices: {sorted(_POLICIES)}"
        )
    return cls(num_resident_K, num_experts_E)
