from __future__ import annotations

import torch

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES


def alloc_canary_buf(
    *,
    num_slots: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
