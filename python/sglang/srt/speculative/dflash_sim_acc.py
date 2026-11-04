"""Simulated DFLASH acceptance lengths (SGLANG_SIMULATE_ACC_LEN) for a device reader.

Draws stay on the host CPU RNG, so the device and eager paths see one sequence.
"""

from __future__ import annotations

import torch


class SimAccLenRing:
    """Stages host-drawn commit lengths into ``device_len`` (int32 [1]).

    A pinned slot is rewritten only after its previous non-blocking copy completes."""

    SLOTS = 4

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        cuda = self.device.type == "cuda"
        self.host = torch.zeros(self.SLOTS, dtype=torch.int32, pin_memory=cuda)
        self.device_len = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._copied: list = [
            None
        ] * self.SLOTS  # each slot's last copy-done event (CUDA)
        self.staged = 0

    def stage(self, length: int) -> torch.Tensor:
        """Queue ``length`` into ``device_len`` on the current stream and return it."""
        slot = self.staged % self.SLOTS
        done = self._copied[slot]
        if done is not None:
            done.synchronize()  # issued SLOTS rounds ago: complete in practice
        self.host[slot] = length
        self.device_len.copy_(self.host[slot : slot + 1], non_blocking=True)
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record()
            self._copied[slot] = event
        self.staged += 1
        return self.device_len
