# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_hip,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

_is_shuffle_moe_mxfp4 = is_gfx95_supported()

__all__ = ["QuarkW4A4MXFp4MoE"]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A4MXFp4MoE(QuarkMoEScheme):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group" and input_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}"
            )

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        self.with_bias = False

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=params_dtype,
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
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # WEIGHT_BIAS (zeros, matching standard mxfp4 path for CUDA graph compat)
        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        set_weight_attrs(w2_weight_bias, extra_weight_attrs)

        self.hidden_pad = 0
        self.intermediate_pad = 0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _use_aiter:
            return

        # Convert bias to float32 (matching standard path)
        if layer.w13_weight_bias is not None:
            layer.w13_weight_bias.data = layer.w13_weight_bias.data.to(torch.float32)
        if layer.w2_weight_bias is not None:
            layer.w2_weight_bias.data = layer.w2_weight_bias.data.to(torch.float32)

        # Interleave w1 (gate) and w3 (up) halves in w13 for Swiglu activation
        e, n, k = layer.w13_weight.shape
        layer.w13_weight.view(torch.uint8).copy_(
            layer.w13_weight.data.view(torch.uint8)
            .view(e, n // 2, 2, k)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(e, n, k)
        )
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.data.view(e, n // 2, 2, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(e, n, -1)
        )

        # Interleave bias halves too
        layer.w13_weight_bias.data = (
            layer.w13_weight_bias.data.view(-1, n // 2, 2)
            .permute(0, 2, 1)
            .contiguous()
            .view(-1, n)
        )

        # Shuffle weights using CUDA-graph-compatible a16w4 functions
        layer.w13_weight.data = shuffle_weight_a16w4(layer.w13_weight, 16, True)
        shuffled_w13_scale = shuffle_scale_a16w4(
            layer.w13_weight_scale.view(-1, layer.w13_weight_scale.shape[-1]),
            e,
            True,
        )

        layer.w2_weight.data = shuffle_weight_a16w4(layer.w2_weight, 16, False)
        shuffled_w2_scale = shuffle_scale_a16w4(
            layer.w2_weight_scale.view(-1, layer.w2_weight_scale.shape[-1]),
            e,
            False,
        )

        layer.w13_weight_scale = torch.nn.Parameter(
            shuffled_w13_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            shuffled_w2_scale, requires_grad=False
        )

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config
        topk_weights, topk_ids, _ = topk_output
        if _is_hip:
            topk_weights = topk_weights.to(torch.float32)

        if hasattr(torch, "float4_e2m1fn_x2"):
            w13_weight = layer.w13_weight.view(torch.float4_e2m1fn_x2)
            w2_weight = layer.w2_weight.view(torch.float4_e2m1fn_x2)
        else:
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

        output = fused_moe(
            x,
            w13_weight,
            w2_weight,
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_1x32,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation=ActivationType.Swiglu,
            doweight_stage1=moe_runner_config.apply_router_weight_on_input,
            expert_mask=layer.expert_mask_gpu,
            hidden_pad=self.hidden_pad,
            intermediate_pad=self.intermediate_pad,
            bias1=layer.w13_weight_bias,
            bias2=layer.w2_weight_bias,
        )
        return StandardCombineInput(hidden_states=output)
