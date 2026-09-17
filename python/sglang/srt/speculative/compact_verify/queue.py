"""Queue kernels and bounded eager warmup for the graph verifier.

The serving graph repairs every queued row before returning acceptance.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.speculative.compact_verify.core import Verify


@triton.jit
def _queue(Flags, Descriptor, R: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    flags = tl.load(Flags + row, row < R, other=0).to(tl.int32)
    slot = tl.cumsum(flags) - 1
    tl.store(Descriptor + slot, row, (flags != 0) & (slot < C))
    tl.store(Descriptor + C, tl.sum(flags))


@triton.jit
def _patch(P, Repaired, Descriptor, C: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.arange(0, BLOCK)
    row = tl.load(Descriptor + slot, slot < C, other=-1)
    p = tl.load(Repaired + slot, slot < C, other=0)
    tl.store(P + row, p, (slot < C) & (row >= 0))


class FixedQueueVerify(Verify):
    def __init__(self, local, q, candidates, coins, final_coins, capacity=16):
        super().__init__(local, q, candidates, coins, final_coins)
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.force_flags = torch.zeros(self.r, dtype=torch.bool, device=local.device)

    def uncertainty_flags(self, p):
        raise NotImplementedError("a qualified probability enclosure is required")

    def fixed(self):
        p = self.selected_probabilities()
        flags = self.uncertainty_flags(p)
        flags = flags | self.force_flags
        descriptor = torch.full(
            (self.capacity + 1,), -1, dtype=torch.int32, device=p.device
        )
        if self.rank == 0:
            _queue[(1,)](
                flags, descriptor, self.r, self.capacity, triton.next_power_of_2(self.r)
            )
        dist.broadcast(descriptor, 0)
        rows = descriptor[: self.capacity].long().clamp(min=0)
        local = self.local.view(self.r, self.lv)[rows]
        local = torch.where((descriptor[: self.capacity] >= 0)[:, None], local, 0)
        repaired = torch.softmax(self.gather(local).float(), -1)
        selected = repaired.gather(1, self.targets[rows, None]).flatten()
        _patch[(1,)](
            p,
            selected,
            descriptor,
            self.capacity,
            triton.next_power_of_2(self.capacity),
        )
        output = self.finish(p)
        overflow = descriptor[-1] > self.capacity
        # Never expose plausible token/count/index outputs for overflow batches.
        masked = tuple(torch.where(overflow, -1, value) for value in output[:3])
        return (*masked, p, overflow, descriptor[-1])

    def compact(self):
        # Conservative experimental dispatch: 1024 is the smallest tested
        # winning R for this fixed queue, not a measured optimal crossover.
        if self.r < 1024:
            return self.baseline()
        return self.fixed()[:4]
