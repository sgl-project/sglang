"""Coordinate execution mode, not bucket size, on the existing EP CPU group."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class NcclEpGraphDecision:
    can_run: bool
    capture_hidden_mode: int
    recapture: bool


class NcclEpGraphAdmission:
    """One host control exchange before any runtime Graph/eager divergence.

    All EP ranks must call this for every scheduled forward, including prefill
    and IDLE. Capture callbacks do not call it. Native eager and Graph resources
    use different groups, so local-only eligibility is not sufficient.
    """

    def __init__(self, cpu_group):
        self.cpu_group = cpu_group
        self.local = torch.empty(3, dtype=torch.int64, device="cpu")
        self.peers = [
            torch.empty_like(self.local) for _ in range(dist.get_world_size(cpu_group))
        ]

    def decide(self, *, eligible: bool, required_mode: int, captured_mode: int):
        self.local[0] = int(eligible)
        self.local[1] = int(required_mode)
        self.local[2] = int(captured_mode)
        dist.all_gather(self.peers, self.local, group=self.cpu_group)
        values = [peer.tolist() for peer in self.peers]
        can_run = all(value[0] for value in values)
        target = max(value[1] for value in values)
        return NcclEpGraphDecision(
            can_run=can_run,
            capture_hidden_mode=target,
            recapture=can_run and any(value[2] != target for value in values),
        )
