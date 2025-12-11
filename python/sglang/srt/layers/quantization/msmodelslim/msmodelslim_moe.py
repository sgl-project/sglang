# Adapted from https://github.com/vllm-project/vllm/tree/v0.8.2/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
import logging
from enum import Enum
from typing import Callable, Optional, TYPE_CHECKING
from typing import Any, Dict, List

import torch

from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.msmodelslim.schemes import (
    ModelSlimScheme,
)
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW8A8Int8DynamicMoEMethod,
)

from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.msmodelslim.msmodelslim import (
        ModelSlimConfig,
    )

logger = logging.getLogger(__name__)


__all__ = [
    "ModelSlimMoEMethod",
    "ModelSlimW8A8Int8MoE",
]


class ModelSlimMoEMethod(FusedMoEMethodBase):
    def __new__(cls, *args, **kwargs):
        if cls is ModelSlimMoEMethod:
            return super().__new__(cls)
        return super().__new__(cls)

    @staticmethod
    def get_moe_method(
        quant_config: ModelSlimConfig,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "ModelSlimMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.

        return ModelSlimW8A8Int8MoE(quant_config)
        # weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        # input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        # is_moe_w4_dynamic = quant_config.is_dynamic_token_w4(weight_quant, input_quant)
        # is_moe_input_quant = input_quant

        # if (
        #     is_moe_w4_dynamic and is_moe_input_quant is not None
        # ) or quant_config._is_moe_w4a8_dynamic(prefix, weight_quant, input_quant):
        #     return NPUW4A8Int4DynamicMoEMethod(quant_config)
        # elif is_moe_w4_dynamic and is_moe_input_quant is None:
        #     return NPUW4A16Int4DynamicMoEMethod(quant_config)
        # else:
        #     return NPUW8A8Int8DynamicMoEMethod(quant_config)
        # else:
        #     raise RuntimeError(
        #         f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
        #     )


class ModelSlimW8A8Int8MoE(ModelSlimMoEMethod):

    def __init__(
            self, quant_config: Dict[str, Any], prefix: str = None,
    ):
        self.quant_config = quant_config

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

        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        # scale
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)
        w2_weight_offset = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        NPUW8A8Int8DynamicMoEMethod.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config
    
    def apply(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        return NPUW8A8Int8DynamicMoEMethod.apply(layer, dispatch_output)
    

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        return NPUW8A8Int8DynamicMoEMethod.apply_without_routing_weights(layer,
                                                                        hidden_states,
                                                                        hidden_states_scale,
                                                                        group_list_type,
                                                                        group_list,
                                                                        output_dtype,)