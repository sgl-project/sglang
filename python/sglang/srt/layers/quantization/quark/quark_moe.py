# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import torch
import logging

from sglang.srt.utils import set_weight_attrs
from sglang.srt.utils import get_bool_env_var, supports_mx

from aiter import ActivationType, QuantType, biased_grouped_topk
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.fused_moe import fused_moe

logger = logging.getLogger(__name__)

__all__ = ["QuarkMoEMethod", "QuarkW4A4MXFp4MoEMethod"]

use_aiter_moe = get_bool_env_var("SGLANG_USE_AITER")

OCP_MX_BLOCK_SIZE = 32

class QuarkMoEMethod():
    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    @staticmethod
    def get_moe_method(
            quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
            module: torch.nn.Module,
            layer_name: str) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(
            layer_name, module)

        if (layer_quant_config.get("output_tensors")
                or layer_quant_config.get("bias")):
            raise NotImplementedError("Currently, Quark models with "
                                      "output_tensors and bias "
                                      "quantized are not supported")
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if quant_config._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFp4MoEMethod(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")


class QuarkW4A4MXFp4MoEMethod(QuarkMoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str,
                                                                         Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group"
                and input_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}")  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        self.emulate = not supports_mx()

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
            dtype=params_dtype),
                                       requires_grad=False)
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        float_dtype = torch.get_default_dtype()

        if use_aiter_moe:
            # Pre-shuffle weight scales
            s0, s1, _ = layer.w13_weight_scale.shape
            w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)
            w13_weight_scale = e8m0_shuffle(w13_weight_scale)
            layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)

            s0, s1, _ = layer.w2_weight_scale.shape
            w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)
            w2_weight_scale = e8m0_shuffle(w2_weight_scale)
            layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        # Expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        if use_aiter_moe:
            # if SGLANG_AITER_MOE=1 then we use CK MoE
            return fused_moe(
               x,
               layer.w13_weight,
               layer.w2_weight,
               topk_weights,
               topk_ids,
               w1_scale=layer.w13_weight_scale.view(-1, layer.w13_weight_scale.shape[-1]),
               w2_scale=layer.w2_weight_scale.view(-1, layer.w2_weight_scale.shape[-1]),
               quant_type=QuantType.per_1x32,
               activation=ActivationType.Silu,
               doweight_stage1=False,
            )
             
        else:
            return fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=False,#inplace and not no_combine,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                use_fp8_w8a8=False,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                no_combine=no_combine,
                use_mxf4_w4a4=True,
            )
