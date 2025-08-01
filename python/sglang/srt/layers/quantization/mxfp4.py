from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
# from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.utils import set_weight_attrs
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.quantization.mxfp4_moe import (
    fused_experts_mxfp4_oai,
    shuffle_for_activation_kernel,
    quantize_to_mxfp4,
    get_swizzle_type,
    swizzle_weight_and_scale,
    pad_weight_and_scale_on_hopper,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)


class Mxfp4Config(QuantizationConfig):
    """Config class for MXFP4."""

    def __init__(
        self,
        is_checkpoint_mxfp4_serialized: bool = True,
        moe_activation_scheme: str = "static",
    ) -> None:
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        if moe_activation_scheme not in ["dynamic"]:
            raise ValueError(f"Unsupported activation scheme {moe_activation_scheme}")
        self.moe_activation_scheme = moe_activation_scheme

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Mxfp4Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method
        moe_activation_scheme = "dynamic"
        return cls(
            is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized,
            moe_activation_scheme=moe_activation_scheme,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            return Mxfp4MoEMethod(self)
        elif isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Mxfp4MoEMethod(FusedMoEMethodBase, CustomOp):
    def __init__(self,
        quant_config: Mxfp4Config,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
        bias: bool = False,
        activation_dtype: torch.dtype = torch.bfloat16,
        shuffle_weight: bool = True,
    ):
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = quant_config.is_checkpoint_mxfp4_serialized
        self.moe_activation_scheme = quant_config.moe_activation_scheme

        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.bias = bias
        self.activation_dtype = activation_dtype
        self.shuffle_weight = shuffle_weight
        self.swizzle_value, self.swizzle_scale = get_swizzle_type(activation_dtype)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if self.is_checkpoint_mxfp4_serialized:
            from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoeWeightScaleSupported
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size, hidden_size // 2, dtype=torch.uint8
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            w13_scale = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size, hidden_size // 32, dtype=torch.uint8
                ),
                requires_grad=False,
            )
            w13_scale.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
            layer.register_parameter("w13_scale", w13_scale)
            set_weight_attrs(w13_scale, extra_weight_attrs)
            
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size // 2, dtype=torch.uint8
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)

            w2_scale = torch.nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size // 32, dtype=torch.uint8
                ),
                requires_grad=False,
            )
            w2_scale.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
            layer.register_parameter("w2_scale", w2_scale)
            set_weight_attrs(w2_scale, extra_weight_attrs)
        else:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float8_e5m2
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size, dtype=torch.float8_e5m2
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)

            layer.register_parameter("w13_scale", None)
            layer.register_parameter("w2_scale", None)

        if self.bias:
            w13_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size, dtype=params_dtype
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, dtype=params_dtype
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.register_parameter("w13_bias", None)
            layer.register_parameter("w2_bias", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self.is_checkpoint_mxfp4_serialized:
            w1_weight_fp4, w1_weight_scale = quantize_to_mxfp4(layer.w13_weight.data[:, :self.intermediate_size, :])
            w3_weight_fp4, w3_weight_scale = quantize_to_mxfp4(layer.w13_weight.data[:, self.intermediate_size:, :])
            w2_weight_fp4, w2_weight_scale = quantize_to_mxfp4(layer.w2_weight.data)
            w13_weight_fp4 = torch.cat([w1_weight_fp4, w3_weight_fp4], dim=1)
            w13_weight_scale = torch.cat([w1_weight_scale, w3_weight_scale], dim=1)
        else:
            w13_weight_fp4 = layer.w13_weight.data
            w13_weight_scale = layer.w13_scale.data
            w2_weight_fp4 = layer.w2_weight.data
            w2_weight_scale = layer.w2_scale.data

        # (num_experts, 2 * intermediate_size, hidden_size // 2)
        w13_weight_fp4 = torch.transpose(w13_weight_fp4, 1, 2) # (num_experts, hidden_size // 2, 2 * intermediate_size)
        if self.shuffle_weight:
            w13_weight_fp4 = shuffle_for_activation_kernel(w13_weight_fp4)

        w13_weight_scale = torch.transpose(w13_weight_scale, 1, 2)
        if self.shuffle_weight:
            w13_weight_scale = shuffle_for_activation_kernel(w13_weight_scale)

        w2_weight_fp4 = torch.transpose(w2_weight_fp4, 1, 2)
        w2_weight_scale = torch.transpose(w2_weight_scale, 1, 2)

        w13_weight_fp4, w13_weight_scale = pad_weight_and_scale_on_hopper(
            w13_weight_fp4, w13_weight_scale, self.swizzle_scale)
        w2_weight_fp4, w2_weight_scale = pad_weight_and_scale_on_hopper(
            w2_weight_fp4, w2_weight_scale, self.swizzle_scale)

        w13_weight_fp4, w13_weight_scale, actual_w13_scale_shape = swizzle_weight_and_scale(
            w13_weight_fp4, w13_weight_scale, self.swizzle_value, self.swizzle_scale)
        w2_weight_fp4, w2_weight_scale, actual_w2_scale_shape = swizzle_weight_and_scale(
            w2_weight_fp4, w2_weight_scale, self.swizzle_value, self.swizzle_scale)

        self.actual_w13_weight_shape = actual_w13_scale_shape
        self.actual_w2_weight_shape = actual_w2_scale_shape

        layer.w13_weight.data = w13_weight_fp4
        torch.cuda.empty_cache()
        layer.w2_weight.data = w2_weight_fp4
        torch.cuda.empty_cache()
        if self.bias:
            w13_bias = layer.w13_bias.data.to(torch.float32)
            w2_bias = layer.w2_bias.data.to(torch.float32)
            if self.shuffle_weight:
                w13_bias = shuffle_for_activation_kernel(w13_bias)
            layer.w13_bias.data = w13_bias
            torch.cuda.empty_cache()
            layer.w2_bias.data = w2_bias
            torch.cuda.empty_cache()
        layer.w13_weight_scale = w13_weight_scale
        layer.w2_weight_scale = w2_weight_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output,
        top_k: int,
        # renormalize: bool,
        # use_grouped_topk: bool,
        # topk_group: Optional[int] = None,
        # num_expert_group: Optional[int] = None,
        # num_fused_shared_experts: int = 0,
        # custom_routing_function: Optional[Callable] = None,
        # correction_bias: Optional[torch.Tensor] = None,
        activation: str = "swiglu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        return self.forward(
            x=x,
            layer=layer,
            expert_logits=topk_output.router_logits,
            top_k=top_k,
            # renormalize=renormalize,
            # use_grouped_topk=use_grouped_topk,
            # topk_group=topk_group,
            # num_expert_group=num_expert_group,
            # num_fused_shared_experts=num_fused_shared_experts,
            # custom_routing_function=custom_routing_function,
            # correction_bias=correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        top_k: int,
        # use_grouped_topk: bool,
        # renormalize: bool,
        # topk_group: Optional[int] = None,
        # num_expert_group: Optional[int] = None,
        # num_fused_shared_experts: int = 0,
        # custom_routing_function: Optional[Callable] = None,
        # correction_bias: Optional[torch.Tensor] = None,
        activation: str = "swiglu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        w2_bias = None
        if get_tensor_model_parallel_rank() == 0:
            w2_bias = getattr(layer, 'w2_bias', None)
        return fused_experts_mxfp4_oai(
            hidden_states=x,
            w13=layer.w13_weight.data,
            w2=layer.w2_weight.data,
            expert_logits=expert_logits,
            top_k=top_k,
            fc31_input_dequant=getattr(layer, 'fc31_input_dequant', None),
            fc2_input_dequant=getattr(layer, 'fc2_input_dequant', None),
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation=activation,
            w1_bias=getattr(layer, 'w13_bias', None),
            w2_bias=w2_bias,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            dtype=x.dtype,
            activation_dtype=self.activation_dtype,
            swizzle_value=self.swizzle_value,
            swizzle_scale=self.swizzle_scale,
            actual_w13_scale_shape=self.actual_w13_weight_shape,
            actual_w2_scale_shape=self.actual_w2_weight_shape,
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
            clamp_limit=7.0,
        )

    def forward_cpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("CPU is not supported for OpenAI MoE")

    def forward_tpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("TPU is not supported for OpenAI MoE")

    forward_native = forward_cpu
