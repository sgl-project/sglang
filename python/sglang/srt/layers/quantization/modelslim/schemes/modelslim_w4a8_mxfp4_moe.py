from __future__ import annotations

from typing import Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A8Mxfp4MoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

__all__ = ["ModelSlimW4A8Mxfp4MoE"]

MXFP4_BLOCK_SIZE = 32
MXFP4_PACK_FACTOR = 2


class ModelSlimW4A8Mxfp4MoE(ModelSlimMoEScheme):
    """ModelSlim MoE scheme with packed MXFP4 weights and MXFP8 activations."""

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,
        group_size: int = 0,
    ) -> None:
        if group_size == 0:
            group_size = quant_config.quant_description.get(
                "group_size", MXFP4_BLOCK_SIZE
            )
        if group_size != MXFP4_BLOCK_SIZE:
            raise ValueError(
                "W4A8_MXFP MoE requires group_size=32, "
                f"but received group_size={group_size}"
            )
        if weight_prefix not in ("w13", "w2"):
            raise ValueError(
                f"weight_prefix must be 'w13' or 'w2', got '{weight_prefix}'"
            )

        self.quant_config = quant_config
        self.weight_prefix = weight_prefix
        self.group_size = group_size
        self.kernel = NPUW4A8Mxfp4MoEMethod(group_size=group_size)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if self.weight_prefix == "w13":
            out_features = 2 * intermediate_size_per_partition
            in_features = hidden_size
        else:
            out_features = hidden_size
            in_features = intermediate_size_per_partition

        if in_features % (self.group_size * MXFP4_PACK_FACTOR) != 0:
            raise ValueError(
                f"{self.weight_prefix} input size ({in_features}) must be divisible "
                f"by {self.group_size * MXFP4_PACK_FACTOR} for W4A8_MXFP"
            )

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                out_features,
                in_features // MXFP4_PACK_FACTOR,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                out_features,
                in_features // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

        layer.register_parameter(f"{self.weight_prefix}_weight", weight)
        layer.register_parameter(f"{self.weight_prefix}_weight_scale", weight_scale)
        set_weight_attrs(weight, extra_weight_attrs)
        set_weight_attrs(weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
