from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A8MxFpDynamicMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimW4A8MxFpMoE",
]


class ModelSlimW4A8MxFpMoE(ModelSlimMoEScheme):

    def __init__(
        self,
        quant_config: Dict[str, Any],
        prefix: str = None,
    ):
        self.quant_config = quant_config
        self.kernel = NPUW4A8MxFpDynamicMoEMethod()

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
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        # scale
        block_k = 32
        scale_dtype = torch.uint8
        w13_weight_scale = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    (hidden_size + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        return self.kernel.apply(layer, dispatch_output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        return self.kernel.apply_without_routing_weights(
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )