from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A4Int4DynamicMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimW4A4Int4MoE",
]


class ModelSlimW4A4Int4MoE(ModelSlimMoEScheme):
    """
    W4A4 integer MoE scheme that creates weights for either the
    w13 (gate+up) or w2 (down) projection group.

    Two instances of this class are used per MoE layer:
      - weight_prefix="w13"   → handles the fused gate_proj + up_proj weights
      - weight_prefix="w2"    → handles the down_proj weights
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,  # "w13" or "w2"
    ):
        self.quant_config = quant_config
        self.weight_prefix = weight_prefix
        self.kernel = NPUW4A4Int4DynamicMoEMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        if self.weight_prefix == "w13":
            prefix = "w13"
            a_dim = 2 * intermediate_size_per_partition
            b_dim = hidden_size
        elif self.weight_prefix == "w2":
            prefix = "w2"
            a_dim = hidden_size
            b_dim = intermediate_size_per_partition
        else:
            raise ValueError(
                f"Unknown weight_prefix: {self.weight_prefix}. Use 'w13' or 'w2'."
            )

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

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix
    ) -> None:
        self.kernel.process_weights_after_loading(layer, weight_prefix)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer,
        hidden_states,
        expert_tokens,
        pertoken_scale,
        original_dtype,
        weight_prefix,
    ):
        return self.kernel.apply(
            layer,
            hidden_states,
            expert_tokens,
            pertoken_scale,
            original_dtype,
            weight_prefix,
        )

    def quant_activations(
        self,
        hidden_states,
    ):
        return self.kernel.quant_activations(hidden_states)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        logger.warning_once(
            "Warning: Performance may be reduced, because DeepEP Dispatcher does not support 4-bit quantization, "
            "switching to the bf16 dispatcher, quantization will be performed separately..."
        )
        return self.kernel.apply_without_routing_weights(
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )
