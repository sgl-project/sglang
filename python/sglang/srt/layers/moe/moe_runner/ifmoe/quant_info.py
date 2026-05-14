"""MoeQuantInfo subclass for IFMoe (FP8 blockwise scaled)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo


@dataclass
class IFMoeQuantInfo(MoeQuantInfo):
    """FP8 blockwise-scaled weight payload for IFMoe kernel.

    Weight layouts must match the kernel's expectations:
      w13_weight:       (num_local_experts, 2*intermediate_size, hidden_size) fp8_e4m3
      w2_weight:        (num_local_experts, hidden_size, intermediate_size)  fp8_e4m3
      w13_weight_scale: (num_local_experts, 2*I/128, H/128) float32
      w2_weight_scale:  (num_local_experts, H/128, I/128) float32
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    routing_bias: Optional[torch.Tensor] = None
    local_expert_offset: int = 0
