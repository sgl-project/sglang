"""Which expert sits in which GPU slot, and how a step's experts are split into waves.

Pure bookkeeping: no expert moves here. The executor pages ``Wave.loads`` into place before it
runs the wave. With CUDA graphs, decode steps decide on the device (``DeviceResidency``), and
host-planned steps load that state into the policy and store it back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple

import msgspec
import torch


class Wave(msgspec.Struct, frozen=True):
    experts: List[int]  # logical expert ids served by this wave
    slots: List[int]  # the GPU slot of each expert, same order
    loads: List[Tuple[int, int]]  # (expert, slot) pairs to page in before the wave runs


class ResidencyPolicy(ABC):
    """Chooses the slot of each expert of a wave, evicting experts outside the wave."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots

    @abstractmethod
    def place(self, experts: List[int]) -> Wave:
        """Make ``experts`` (at most ``num_slots``) resident and return where they are."""


class LRUPolicy(ResidencyPolicy):
    """Evicts the least recently used experts. Slots start holding experts 0..K-1."""

    def __init__(self, num_slots: int):
        super().__init__(num_slots)
        # Logical expert -> slot for the resident experts, least recently used first.
        self.resident = OrderedDict((e, e) for e in range(num_slots))

    def place(self, experts: List[int]) -> Wave:
        for e in experts:
            if e in self.resident:
                self.resident.move_to_end(e)
        loads = []
        for e in experts:
            if e not in self.resident:
                # Every expert of this wave sits at the LRU tail, so the head is never one of them.
                _, slot = self.resident.popitem(last=False)
                self.resident[e] = slot
                loads.append((e, slot))
        return Wave(
            experts=experts, slots=[self.resident[e] for e in experts], loads=loads
        )


def plan_waves(policy: ResidencyPolicy, distinct: List[int]) -> List[Wave]:
    """Split a step's distinct experts into waves of at most ``num_slots`` and place each.

    A step with no routed expert (all padding) still gets one empty wave, so the layer runs.
    """
    k = policy.num_slots
    groups = [distinct[i : i + k] for i in range(0, len(distinct), k)] or [[]]
    return [policy.place(group) for group in groups]


class DeviceResidency:
    """The LRU residency state on the GPU, which the decide kernel updates in place, plus the
    buffers of its page-in plan. Slots start holding experts 0..K-1, like ``LRUPolicy``."""

    def __init__(self, num_experts: int, num_slots: int, device):
        i32 = dict(dtype=torch.int32, device=device)
        self.step = torch.zeros(1, **i32)
        self.slot_expert = torch.arange(num_slots, **i32)
        self.expert_slot = torch.full((num_experts,), -1, **i32)
        self.expert_slot[:num_slots] = self.slot_expert
        self.slot_lastuse = torch.zeros(num_slots, **i32)
        self.src = torch.zeros(num_slots, **i32)
        self.dst = torch.zeros(num_slots, **i32)
        self.count = torch.zeros(1, **i32)

    def load_into(self, policy: LRUPolicy) -> None:
        """Make ``policy`` hold this state: the same residents, least recently used first."""
        experts = self.slot_expert.tolist()
        lastuse = self.slot_lastuse.tolist()
        order = sorted(range(len(experts)), key=lambda s: (lastuse[s], s))
        policy.resident = OrderedDict((experts[s], s) for s in order)

    def store_from(self, policy: LRUPolicy) -> None:
        """Make this state hold ``policy``'s, its recency order as the last steps."""
        num_slots = self.slot_expert.numel()
        base = int(self.step.item())
        slot_expert = [0] * num_slots
        lastuse = [0] * num_slots
        for rank, (expert, slot) in enumerate(policy.resident.items(), start=1):
            slot_expert[slot] = expert
            lastuse[slot] = base + rank
        self.slot_expert.copy_(torch.tensor(slot_expert, dtype=torch.int32))
        self.slot_lastuse.copy_(torch.tensor(lastuse, dtype=torch.int32))
        self.expert_slot.fill_(-1)
        self.expert_slot[self.slot_expert.long()] = torch.arange(
            num_slots, dtype=torch.int32, device=self.expert_slot.device
        )
        self.step.fill_(base + num_slots)
