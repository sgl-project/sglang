from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A8Int8DynamicMoEMethod,
)
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["NPUCompressedTensorsW4A8Int8DynamicMoE"]


logger = logging.getLogger(__name__)


class NPUCompressedTensorsW4A8Int8DynamicMoE(CompressedTensorsMoEScheme):

    ### TODO: Get rid of code duplication with python/sglang/srt/modelslim/modelslim_moe.py @OrangeRedeng @TamirBaydasov
    def __init__(self, quantization_config) -> None:
        self.group_size = 0
        self.is_per_channel_weight = self.group_size == 0
        self.tp_size = 1
        self.activation_use_clip = (
            quantization_config.get("config_groups", {})
            .get("group_1", {})
            .get("activation_use_clip", False)
        )
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
        weight_scale_dtype = torch.int64 if self.activation_use_clip else torch.float32
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=weight_scale_dtype),
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
        if self.activation_use_clip:
            self._init_activation_clip_params(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                extra_weight_attrs,
            )
        else:
            self._init_extra_scale_params(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                extra_weight_attrs,
            )

    def _init_activation_clip_params(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        extra_weight_attrs: dict,
    ) -> None:
        """
        Initializes bias and alpha parameters for quantization schemes that use activation clipping.

        This helper registers `w13_bias`, `w2_bias`, and `w2_alpha`, which are required to
        shift and scale the activations or outputs to compensate for the precision loss
        introduced by clamping activations.
        """
        w13_bias = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, dtype=torch.float),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

        w2_alpha = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float), requires_grad=False
        )
        layer.register_parameter("w2_alpha", w2_alpha)
        set_weight_attrs(w2_alpha, extra_weight_attrs)

    def _init_extra_scale_params(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        extra_weight_attrs: dict,
    ) -> None:
        """
        Initializes additional scaling, offset, and bias parameters for quantization schemes without activation clipping.

        This method registers the following parameters:
        1. Scale Biases: `w13_scale_bias` and `w2_scale_bias`.
        2. Secondary Quantization Params (initialized only for grouped quantization):
            `w13_weight_scale_second`, `w13_weight_offset_second`,
            `w2_weight_scale_second`, and `w2_weight_offset_second`.
        """
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
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.kernel.apply(layer, dispatch_output)

    def apply_weights_with_router_logits(
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
