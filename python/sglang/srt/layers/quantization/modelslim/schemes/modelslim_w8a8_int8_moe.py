from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimW8A8Int8MoE",
]


class ModelSlimW8A8Int8MoE(ModelSlimMoEScheme):
    """
    W8A8 integer MoE scheme that creates weights for either the
    w13 (gate+up) or w2 (down) projection group.

    Two instances of this class are used per MoE layer:
      - weight_prefix="w13"   → handles the fused gate_proj + up_proj weights
      - weight_prefix="w2"    → handles the down_proj weights
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,  # "w13" or "w2"
    ) -> None:
        self.quant_config = quant_config
        self.kernel = NPUW8A8Int8MoEMethod()
        self.weight_prefix = weight_prefix
        if weight_prefix not in ("w13", "w2"):
            raise ValueError(
                f"weight_prefix must be 'w13' or 'w2', got '{weight_prefix}'"
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        # Determine shape based on weight group
        if self.weight_prefix == "w13":
            a_dim = 2 * intermediate_size_per_partition
            b_dim = hidden_size
        else:  # w2
            a_dim = hidden_size
            b_dim = intermediate_size_per_partition

        prefix = self.weight_prefix

        # Create and register weight
        weight_name = f"{prefix}_weight"
        weight = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, b_dim, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter(weight_name, weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # Create and register scale
        scale_name = f"{prefix}_weight_scale"
        scale = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(scale, extra_weight_attrs)

        # Create and register offset
        offset_name = f"{prefix}_weight_offset"
        offset = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(offset_name, offset)
        set_weight_attrs(offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Delegate weight processing to the NPU kernel for the fixed weight group.
        """
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
