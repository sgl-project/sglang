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
    buffers of its page-in plan. Slots start holding experts 0..K-1, like ``LRUPolicy``.

    The state is one packed buffer, ``[slot_expert K | slot_lastuse K | step 1 | counts E+1 |
    expert_slot E]``, so a host-planned step reads it with the step's per-expert routing counts
    in one copy (``readback``) and writes it back in one (``store_from``)."""

    def __init__(self, num_experts: int, num_slots: int, device):
        i32 = dict(dtype=torch.int32, device=device)
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.state = torch.zeros(2 * num_slots + 1 + 2 * num_experts + 1, **i32)
        (
            self.slot_expert,
            self.slot_lastuse,
            self.step,
            self.counts,
            self.expert_slot,
        ) = self.state.split([num_slots, num_slots, 1, num_experts + 1, num_experts])
        self.slot_expert.copy_(torch.arange(num_slots))
        self.expert_slot.fill_(-1)
        self.expert_slot[:num_slots] = self.slot_expert
        self.src = torch.zeros(num_slots, **i32)
        self.dst = torch.zeros(num_slots, **i32)
        self.count = torch.zeros(1, **i32)
        self._step = 0  # the step the last ``load_into`` read

    @property
    def readback(self) -> torch.Tensor:
        """What a host-planned step reads: the LRU state and ``counts``."""
        return self.state[: 2 * self.num_slots + 2 + self.num_experts]

    def load_into(self, policy: LRUPolicy, host: torch.Tensor) -> None:
        """Make ``policy`` hold this state, from ``host``, a host copy of ``readback``: the same
        residents, least recently used first."""
        k = self.num_slots
        experts = host[:k].tolist()
        lastuse = host[k : 2 * k].tolist()
        self._step = int(host[2 * k])
        order = sorted(range(k), key=lambda s: (lastuse[s], s))
        policy.resident = OrderedDict((experts[s], s) for s in order)

    def store_from(self, policy: LRUPolicy) -> None:
        """Make this state hold ``policy``'s, its recency order as the last steps, in one
        upload that does not wait for the GPU."""
        k = self.num_slots
        slot_expert, lastuse = [0] * k, [0] * k
        expert_slot = [-1] * self.num_experts
        for rank, (expert, slot) in enumerate(policy.resident.items(), start=1):
            slot_expert[slot] = expert
            lastuse[slot] = self._step + rank
            expert_slot[expert] = slot
        state = slot_expert + lastuse + [self._step + k]
        state += [0] * (self.num_experts + 1) + expert_slot
        host = torch.tensor(state, dtype=torch.int32)
        if self.state.is_cuda:
            host = host.pin_memory()
        self.state.copy_(host, non_blocking=True)
