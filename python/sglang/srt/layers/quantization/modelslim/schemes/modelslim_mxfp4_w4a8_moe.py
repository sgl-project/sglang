"""ModelSlim W4A8_MXFP MoE scheme for Ascend NPU."""

from __future__ import annotations

from typing import Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP4W4A8MoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

MXFP4_W4A8_BLOCK_SIZE = 32
MXFP4_W4A8_PACK_FACTOR = 2

__all__ = ["ModelSlimMXFP4W4A8MoE"]


class ModelSlimMXFP4W4A8MoE(ModelSlimMoEScheme):
    """Create one ModelSlim W4A8 MXFP expert-weight group (w13 or w2)."""

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,
    ) -> None:
        if weight_prefix not in ("w13", "w2"):
            raise ValueError(
                f"weight_prefix must be 'w13' or 'w2', got '{weight_prefix}'"
            )
        self.quant_config = quant_config
        self.weight_prefix = weight_prefix
        self.kernel = NPUMXFP4W4A8MoEMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        if self.weight_prefix == "w13":
            output_size = 2 * intermediate_size_per_partition
            input_size = hidden_size
        else:
            output_size = hidden_size
            input_size = intermediate_size_per_partition

        weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                output_size,
                input_size // MXFP4_W4A8_PACK_FACTOR,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter(f"{self.weight_prefix}_weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                output_size,
                (input_size + MXFP4_W4A8_BLOCK_SIZE - 1)
                // MXFP4_W4A8_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter(f"{self.weight_prefix}_weight_scale", weight_scale)
        set_weight_attrs(weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
