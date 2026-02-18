from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A4Int4DynamicMoEMethod,
    NPUW4A8Int8DynamicMoEMethod,
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
<<<<<<< HEAD:python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py
    "ModelSlimMoEMethod",
    "ModelSlimW4A4Int4MoE",
=======
>>>>>>> sglang_orig:python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a8_int8_moe.py
    "ModelSlimW4A8Int8MoE",
]


<<<<<<< HEAD:python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py
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

        prefix_in_quant_config = prefix + ".0.gate_proj.weight"
        is_moe_w4a4_dynamic = (
            quant_config.quant_description.get(prefix_in_quant_config, "STATIC")
            == "W4A4_DYNAMIC"
        )
        is_moe_w4a8_dynamic = (
            quant_config.quant_description.get(prefix_in_quant_config, "STATIC")
            == "W4A8_DYNAMIC"
        )
        is_moe_w8a8_dynamic = (
            quant_config.quant_description.get(prefix_in_quant_config, "STATIC")
            == "W8A8_DYNAMIC"
        )
        if is_moe_w4a4_dynamic:
            logger.info_once("Using ModelSlimW4A4Int4MoE")
            return ModelSlimW4A4Int4MoE(quant_config)
        elif is_moe_w4a8_dynamic:
            logger.info_once("Using ModelSlimW4A8Int8MoE")
            return ModelSlimW4A8Int8MoE(quant_config)
        elif is_moe_w8a8_dynamic:
            logger.info_once("Using ModelSlimW8A8Int8MoE")
            return ModelSlimW8A8Int8MoE(quant_config)
        else:
            logger.warning(f"Unsupported FusedMoe modelslim scheme: \
                    {quant_config.quant_description.get(prefix_in_quant_config.strip())} \
                    in layer: {prefix}")
            return None


class ModelSlimW4A4Int4MoE(ModelSlimMoEMethod):

    def __init__(
        self,
        quant_config: Dict[str, Any],
        prefix: str = None,
    ):
        self.quant_config = quant_config
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
        self.kernel.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
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


class ModelSlimW4A8Int8MoE(ModelSlimMoEMethod):
=======
class ModelSlimW4A8Int8MoE(ModelSlimMoEScheme):
>>>>>>> sglang_orig:python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a8_int8_moe.py

    def __init__(
        self,
        quant_config: Dict[str, Any],
        prefix: str = None,
    ):
        self.quant_config = quant_config
        self.group_size = 0
        self.is_per_channel_weight = self.group_size == 0
        self.tp_size = 1
        self.activation_use_clip = False
        self.kernel = NPUW4A8Int8DynamicMoEMethod()

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

        self.is_per_channel_weight = self.group_size == 0
        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        # >> weight
        w13_output_size = intermediate_size_per_partition
        w2_output_size = hidden_size // 2
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_output_size, hidden_size, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w2_output_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # >> scale
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

        # >> offset
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

        # >>> special param for w4a8
        if not self.is_per_channel_weight:
            w13_weight_scale_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_second", w13_weight_scale_second)
            set_weight_attrs(w13_weight_scale_second, extra_weight_attrs)
            w13_weight_offset_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter(
                "w13_weight_offset_second", w13_weight_offset_second
            )
            set_weight_attrs(w13_weight_offset_second, extra_weight_attrs)

            w2_weight_scale_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale_second", w2_weight_scale_second)
            set_weight_attrs(w2_weight_scale_second, extra_weight_attrs)

            w2_weight_offset_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_offset_second", w2_weight_offset_second)
            set_weight_attrs(w2_weight_offset_second, extra_weight_attrs)

        w13_scale_bias = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale_bias", w13_scale_bias)
        set_weight_attrs(w13_scale_bias, extra_weight_attrs)

        w2_scale_bias = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, 16 // self.tp_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale_bias", w2_scale_bias)
        set_weight_attrs(w2_scale_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(
            layer, self.is_per_channel_weight, self.activation_use_clip
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        # FIXME W4A8 without EP can give 0 accuracy
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
