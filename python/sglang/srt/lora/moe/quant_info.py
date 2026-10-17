from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


class MoeLoraBf16QuantInfo(msgspec.Struct, kw_only=True):
    """W13: [E, S*I, H], gate first when S=2. W2: [E, H, I]."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int

    @classmethod
    def from_layer(cls, base_layer: FusedMoE) -> MoeLoraBf16QuantInfo:
        return cls(
            w13_weight=base_layer.w13_weight,
            w2_weight=base_layer.w2_weight,
            num_local_experts=int(base_layer.num_local_experts),
            intermediate_size=int(base_layer.w2_weight.shape[2]),
            hidden_size=int(base_layer.w2_weight.shape[1]),
        )


# Marlin's packed weights do not use the standard row-domain layout.
StandardLayoutQuantInfo = MoeLoraBf16QuantInfo
