"""Stable physical-slot plans for same-capacity expert-set replacement."""

from __future__ import annotations

import json
from dataclasses import dataclass

from sglang.srt.eplb.expert_profile import load_expert_profile


@dataclass(frozen=True)
class SlotChange:
    layer: int
    slot: int
    old_expert: int
    new_expert: int


@dataclass(frozen=True)
class ExpertSwapPlan:
    physical_to_logical: tuple[tuple[int, ...], ...]
    changes: tuple[SlotChange, ...]


def plan_expert_swap(current, requested, num_experts: int) -> ExpertSwapPlan:
    """Retain intersection IDs in their current slots; fill only evicted slots.

    Requested ordering determines the order of incoming IDs. The returned mapping,
    rather than requested ordering, describes the actual physical weight layout.
    """
    old = load_expert_profile(json.dumps({"num_experts": num_experts, "keep": current}))
    new = load_expert_profile(
        json.dumps({"num_experts": num_experts, "keep": requested})
    )
    new.validate_model(len(old.keep), num_experts, 1)
    if old.num_kept != new.num_kept:
        raise ValueError("Hot expert replacement requires the same physical capacity")
    mapping, changes = [], []
    for layer, (before, target) in enumerate(zip(old.keep, new.keep, strict=True)):
        retained, wanted = set(before), set(target)
        evicted = [slot for slot, expert in enumerate(before) if expert not in wanted]
        incoming = [expert for expert in target if expert not in retained]
        actual = list(before)
        for slot, expert in zip(evicted, incoming, strict=True):
            changes.append(SlotChange(layer, slot, before[slot], expert))
            actual[slot] = expert
        mapping.append(tuple(actual))
    return ExpertSwapPlan(tuple(mapping), tuple(changes))
