"""Which expert sits in which GPU slot, and how a step's experts are split into waves.

Pure bookkeeping: no expert moves here. The executor pages ``Wave.loads`` into place before it
runs the wave. With CUDA graphs, decode steps decide on the device (``DeviceResidency``), and
host-planned steps load that state into the policy and store it back.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import msgspec
import torch


class Wave(msgspec.Struct, frozen=True):
    experts: List[int]  # logical expert ids served by this wave
    slots: List[int]  # the GPU slot of each expert, same order
    loads: List[Tuple[int, int]]  # (expert, slot) pairs to page in before the wave runs


class LRUPolicy:
    """Chooses the slot of each expert of a wave, evicting the least recently used experts
    outside it. Slots start holding experts 0..K-1."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        # Logical expert -> slot for the resident experts, least recently used first.
        self.resident = OrderedDict((e, e) for e in range(num_slots))

    def place(self, experts: List[int]) -> Wave:
        """Make ``experts`` (at most ``num_slots``) resident and return where they are."""
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


def plan_waves(policy: LRUPolicy, distinct: List[int]) -> List[Wave]:
    """Split a step's distinct experts into waves of at most ``num_slots`` and place each.

    A step with no routed expert (all padding) still gets one empty wave, so the layer runs.
    """
    k = policy.num_slots
    ordered = [e for e in distinct if e in policy.resident] + [
        e for e in distinct if e not in policy.resident
    ]
    groups = [ordered[i : i + k] for i in range(0, len(ordered), k)] or [[]]
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
        self._step = 0  # the step the last ``load_into`` read

    def load_into(self, policy: LRUPolicy, counts: torch.Tensor) -> List[int]:
        """Make ``policy`` hold this state: the same residents, least recently used first. The
        state is read with ``counts`` (a step's routed entries per expert) in one copy, and
        ``counts`` is returned."""
        k = self.slot_expert.numel()
        host = torch.cat(
            [self.slot_expert, self.slot_lastuse, self.step, counts]
        ).tolist()
        experts, lastuse, self._step = host[:k], host[k : 2 * k], host[2 * k]
        order = sorted(range(k), key=lambda s: (lastuse[s], s))
        policy.resident = OrderedDict((experts[s], s) for s in order)
        return host[2 * k + 1 :]

    def store_from(self, policy: LRUPolicy) -> None:
        """Make this state hold ``policy``'s, its recency order as the last steps, in one
        upload that does not wait for the GPU."""
        k, e, base = self.slot_expert.numel(), self.expert_slot.numel(), self._step
        slot_expert, lastuse, expert_slot = [0] * k, [0] * k, [-1] * e
        for rank, (expert, slot) in enumerate(policy.resident.items(), start=1):
            slot_expert[slot] = expert
            lastuse[slot] = base + rank
            expert_slot[expert] = slot
        host = torch.tensor(
            slot_expert + lastuse + [base + k] + expert_slot, dtype=torch.int32
        )
        if self.step.is_cuda:
            host = host.pin_memory()
        host = host.to(self.step.device, non_blocking=True)
        states = (self.slot_expert, self.slot_lastuse, self.step, self.expert_slot)
        for state, new in zip(states, host.split([k, k, 1, e])):
            state.copy_(new)
