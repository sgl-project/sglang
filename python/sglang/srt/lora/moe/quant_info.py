from __future__ import annotations

import msgspec
import torch


class MoeLoraBf16QuantInfo(msgspec.Struct, kw_only=True):
    """W13: [E, S*I, H], gate first when S=2. W2: [E, H, I]."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
