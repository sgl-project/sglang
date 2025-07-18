# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py

from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_oai import (
    fused_experts_oai,
    fused_experts_mxfp4_oai,
    shuffle_for_activation_kernel,
    quantize_to_mxfp4,
    get_swizzle_type,
    swizzle_weight_and_scale,
    pad_weight_and_scale_on_hopper,
)
if torch.cuda.is_available():
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
else:
    fused_experts = None  # type: ignore

import logging

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.fused_moe_bf16_asm import ck_moe_2stages
    from aiter.ops.shuffle import shuffle_weight

logger = logging.getLogger(__name__)


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        bias: bool = False,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
    ) -> torch.Tensor:
        raise NotImplementedError


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        bias: bool = False,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Create bias parameters only if bias=True
        if bias:
            # Add bias for gate_up_proj
            w13_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size, dtype=params_dtype
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            # Add bias for down_proj
            w2_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, dtype=params_dtype
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            # Register as None when bias is disabled
            layer.register_parameter("w13_bias", None)
            layer.register_parameter("w2_bias", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _use_aiter:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
        return

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
        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
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
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
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

        if _use_aiter:
            assert not no_combine, "unsupported"
            if apply_router_weight_on_input:
                assert (
                    topk_weights.dim() == 2
                ), "`topk_weights` should be in shape (num_tokens, topk)"
                _, topk = topk_weights.shape
                assert (
                    topk == 1
                ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(
                    topk_weights, dtype=torch.float32
                )  # topk_weights must be FP32 (float32)

            return fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                activation=(
                    ActivationType.Silu if activation == "silu" else ActivationType.Gelu
                ),
            )
        else:
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=inplace and not no_combine,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                no_combine=no_combine,
                routed_scaling_factor=routed_scaling_factor,
                w1_bias=getattr(layer, 'w13_bias', None),
                w2_bias=getattr(layer, 'w2_bias', None),
            )

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
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
        return moe_forward_native(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            num_fused_shared_experts,
            custom_routing_function,
            correction_bias,
        )

    def forward_tpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu

class UnquantizedFusedMoEMethodOpenAI(FusedMoEMethodBase, CustomOp):
    def __init__(self, 
        swiglu_alpha: Optional[float] = 1.702, 
        swiglu_beta: Optional[float] = 1.0,
        bias: bool = True,
        shuffle_weight: bool = True,
    ):
        super().__init__()
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.bias = bias
        self.shuffle_weight = shuffle_weight
        if not bias:
            raise ValueError("bias is required for OpenAI MoE")

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):        
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.bias:
            # Add bias for gate_up_proj
            w13_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size, dtype=params_dtype
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            # Add bias for down_proj
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
        layer.w13_weight.data = torch.transpose(layer.w13_weight.data, 1, 2)
        if self.shuffle_weight:
            layer.w13_weight.data = shuffle_for_activation_kernel(layer.w13_weight.data)
            torch.cuda.empty_cache()
        layer.w2_weight.data = torch.transpose(layer.w2_weight.data, 1, 2)
        torch.cuda.empty_cache()
        if self.bias:
            layer.w13_bias.data = layer.w13_bias.data.to(torch.float32)
            if self.shuffle_weight:
                layer.w13_bias.data = shuffle_for_activation_kernel(layer.w13_bias.data)
                torch.cuda.empty_cache()
            layer.w2_bias.data = layer.w2_bias.data.to(torch.float32)
            torch.cuda.empty_cache()
        return
    
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
        activation: str = "swiglu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
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
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "swiglu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        if _use_aiter:
            raise NotImplementedError("Aiter is not supported for OpenAI MoE")
        else:
            w2_bias = None
            if get_tensor_model_parallel_rank() == 0:
                w2_bias = getattr(layer, 'w2_bias', None)
            return fused_experts_oai(
                hidden_states=x,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
                expert_logits=router_logits,
                top_k=top_k,
                activation=activation,
                w1_bias=getattr(layer, 'w13_bias', None),
                w2_bias=w2_bias,
                swiglu_alpha=self.swiglu_alpha,
                swiglu_beta=self.swiglu_beta,
                dtype=x.dtype,
            )
    
    def forward_cpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("CPU is not supported for OpenAI MoE")
    
    def forward_tpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("TPU is not supported for OpenAI MoE")
    
    forward_native = forward_cpu

class MXFP4FusedMoEMethodOpenAI(FusedMoEMethodBase, CustomOp):
    def __init__(self,
        swiglu_alpha: Optional[float] = 1.702, 
        swiglu_beta: Optional[float] = 1.0,
        bias: bool = True,
        activation_dtype: torch.dtype = torch.float8_e4m3fn,
        shuffle_weight: bool = True,
    ):
        super().__init__()
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.bias = bias
        self.activation_dtype = activation_dtype
        self.shuffle_weight = shuffle_weight
        self.swizzle_value, self.swizzle_scale = get_swizzle_type(activation_dtype)
        if not bias:
            raise ValueError("bias is required for OpenAI MoE")

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

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

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
        w1_weight_fp4, w1_weight_scale = quantize_to_mxfp4(layer.w13_weight.data[:, :self.intermediate_size, :])
        w3_weight_fp4, w3_weight_scale = quantize_to_mxfp4(layer.w13_weight.data[:, self.intermediate_size:, :])
        w2_weight_fp4, w2_weight_scale = quantize_to_mxfp4(layer.w2_weight.data)

        tmp_w13_weight = torch.cat([w1_weight_fp4, w3_weight_fp4], dim=1)  # (num_experts, 2 * intermediate_size, hidden_size // 2)
        tmp_w13_weight = torch.transpose(tmp_w13_weight, 1, 2).contiguous()  # (num_experts, hidden_size // 2, 2 * intermediate_size)
        if self.shuffle_weight:
            tmp_w13_weight = shuffle_for_activation_kernel(tmp_w13_weight)

        tmp_w13_scale = torch.cat([w1_weight_scale, w3_weight_scale], dim=1)
        tmp_w13_scale = torch.transpose(tmp_w13_scale, 1, 2).contiguous()
        if self.shuffle_weight:
            tmp_w13_scale = shuffle_for_activation_kernel(tmp_w13_scale)
        
        tmp_w2_weight = torch.transpose(w2_weight_fp4, 1, 2).contiguous()
        tmp_w2_scale = torch.transpose(w2_weight_scale, 1, 2).contiguous()

        tmp_w13_weight, tmp_w13_scale = pad_weight_and_scale_on_hopper(
            tmp_w13_weight, tmp_w13_scale, self.swizzle_scale)
        tmp_w2_weight, tmp_w2_scale = pad_weight_and_scale_on_hopper(
            tmp_w2_weight, tmp_w2_scale, self.swizzle_scale)

        tmp_w13_weight, tmp_w13_scale, tmp_w13_scale_shape = swizzle_weight_and_scale(
            tmp_w13_weight, tmp_w13_scale, self.swizzle_value, self.swizzle_scale)
        tmp_w2_weight, tmp_w2_scale, tmp_w2_scale_shape = swizzle_weight_and_scale(
            tmp_w2_weight, tmp_w2_scale, self.swizzle_value, self.swizzle_scale)
        
        self.actual_w13_weight_shape = tmp_w13_scale_shape
        self.actual_w2_weight_shape = tmp_w2_scale_shape

        layer.w13_weight.data = tmp_w13_weight
        torch.cuda.empty_cache()
        layer.w2_weight.data = tmp_w2_weight
        torch.cuda.empty_cache()
        if self.bias:
            tmp_w13_bias = layer.w13_bias.data.to(torch.float32)
            tmp_w2_bias = layer.w2_bias.data.to(torch.float32)
            if self.shuffle_weight:
                tmp_w13_bias = shuffle_for_activation_kernel(tmp_w13_bias)
            layer.w13_bias.data = tmp_w13_bias
            torch.cuda.empty_cache()
            layer.w2_bias.data = tmp_w2_bias
            torch.cuda.empty_cache()
        layer.w13_weight_scale = tmp_w13_scale
        layer.w2_weight_scale = tmp_w2_scale
        return
    
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
        activation: str = "swiglu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
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
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
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
            expert_logits=router_logits,
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
        )

    def forward_cpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("CPU is not supported for OpenAI MoE")
    
    def forward_tpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("TPU is not supported for OpenAI MoE")
    
    forward_native = forward_cpu


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        inplace: suggestion to compute inplace (modify input activation).
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
        enable_flashinfer_moe: Optional[bool] = False,
        enable_ep_moe: Optional[bool] = False,
        enable_mxfp4_moe: Optional[bool] = False,
        enable_fp8_activation: Optional[bool] = False,
        bias: bool = False,
        is_openai_moe: Optional[bool] = False,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
        shuffle_weight: bool = True,
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.hidden_size = hidden_size
        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_experts = num_experts
        self.expert_map = None

        if enable_flashinfer_moe and quant_config is None:
            logger.warning("Disable flashinfer MoE when quantization config is None.")
            enable_flashinfer_moe = False
            enable_ep_moe = False

        self.enable_flashinfer_moe = enable_flashinfer_moe
        if enable_ep_moe:
            assert (
                self.enable_flashinfer_moe
            ), "FusedMoE only supports EP with --enable-flashinfer-moe"
            self.ep_size = self.tp_size
            self.ep_rank = self.tp_rank
            self.tp_size = 1
            self.tp_rank = 0
            # Create a tensor of size num_experts filled with -1
            self.expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32)
            # Create a expert map for the local experts
            assert num_experts % self.ep_size == 0
            self.local_num_experts = num_experts // self.ep_size
            self.expert_map[
                self.ep_rank
                * self.local_num_experts : (self.ep_rank + 1)
                * self.local_num_experts
            ] = torch.arange(0, self.local_num_experts, dtype=torch.int32, device="cpu")
        else:
            self.ep_size = 1
            self.ep_rank = 0
            self.local_num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.top_k = top_k
        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.num_fused_shared_experts = num_fused_shared_experts
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.correction_bias = correction_bias
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.use_presharded_weights = use_presharded_weights
        self.inplace = inplace
        self.no_combine = no_combine
        self.bias = bias

        if is_openai_moe:
            if self.ep_size > 1:
                raise ValueError("OpenAI FusedMoE only supports ep_size=1")
        
        if quant_config is None:
            if not is_openai_moe:
                self.quant_method: Optional[QuantizeMethodBase] = (
                    UnquantizedFusedMoEMethod()
                )
            else:
                if self.activation != "swiglu":
                    raise ValueError("OpenAI FusedMoE only supports swiglu activation")
                if not enable_mxfp4_moe and not enable_fp8_activation:
                    logger.info("use unquantized fused moe method")
                    self.quant_method = UnquantizedFusedMoEMethodOpenAI(
                        swiglu_alpha=swiglu_alpha or 1.0,
                        swiglu_beta=swiglu_beta or 0.0,
                        bias=bias,
                        shuffle_weight=shuffle_weight,
                    )
                elif enable_mxfp4_moe:
                    activation_dtype = torch.bfloat16 if not enable_fp8_activation else torch.float8_e4m3fn
                    logger.info("use mxfp4 fused moe method, activation_dtype: %s", activation_dtype)
                    self.quant_method = MXFP4FusedMoEMethodOpenAI(
                        swiglu_alpha=swiglu_alpha or 1.0,
                        swiglu_beta=swiglu_beta or 0.0,
                        bias=bias,
                        activation_dtype=activation_dtype,
                        shuffle_weight=shuffle_weight,
                    )
                else:
                    raise ValueError("Invalid quantization method")
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
            if self.quant_method.__class__.__name__ == "ModelOptNvFp4FusedMoEMethod":
                self.quant_method.enable_flashinfer_moe = self.enable_flashinfer_moe
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.local_num_experts,
            hidden_size=hidden_size,
            # FIXME: figure out which intermediate_size to use
            intermediate_size=self.intermediate_size_per_partition,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
            bias=self.bias,
        )

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):
        # Load grouped weight scales for group quantization
        # or model weights
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2

        if not self.use_presharded_weights:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        # w3, up_proj: Load into second logical weight of w13.
        # trtllm cutlass kernel assumes differently
        assert shard_id in ("w1", "w3")
        switch_w13 = getattr(self.quant_method, "load_up_proj_weight_first", False)
        if (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3"):
            start = shard_size
        else:
            start = 0
        expert_data = expert_data.narrow(shard_dim, start, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]

        if not self.use_presharded_weights:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )

        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):

        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3", "w13")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        checkpoint_weights_transposed: bool = False,
    ) -> None:
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return

        # TP rank is set to 0 if EP is enabled
        tp_rank = 0 if self.ep_size > 1 else get_tensor_model_parallel_rank()

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                self.quant_method.__class__.__name__
                == "CompressedTensorsWNA16MoEMethod"
            )
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3", "w13"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3','w13'] but " f"got {shard_id}."
            )

        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0, "w13": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust input_scale for e4m3fnuz (AMD)
            if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                loaded_weight = loaded_weight * 2.0

            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in self.quant_method.__class__.__name__.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return
        if "ModelOpt" in self.quant_method.__class__.__name__:
            if "weight_scale_2" in weight_name or "input_scale" in weight_name:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            return

        # Case weight scales and zero_points
        if "scale" in weight_name or "zero" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust INT4 column-wise scaling number to e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 0.5

                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust FP8 per-tensor scaling number for e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 2.0

                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        is_weight = "weight" in weight_name or weight_name.endswith(
            ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
        )
        is_bias = "bias" in weight_name

        # Case model weights
        if is_weight and not is_bias:
            if checkpoint_weights_transposed:
                loaded_weight = loaded_weight.t().contiguous() # Oai model weight: [:, input channel, output channel]
            if shard_id == "w13":
                # Handle full gate_up_proj weight (w13)
                weight_param = getattr(self, "w13_weight", None)
                if weight_param is not None:
                    # Apply TP sharding to the full weight based on shard_dim
                    tp_size = get_tensor_model_parallel_world_size()
                    if tp_size > 1 and not self.use_presharded_weights:
                        # Split into gate and up parts
                        up_weight = loaded_weight[:loaded_weight.shape[0]//2, :]
                        gate_weight = loaded_weight[loaded_weight.shape[0]//2:, :]
                        assert up_weight.shape[0] == gate_weight.shape[0]
                        # Use shard_dim instead of hardcoded dim 0
                        weight_per_partition = up_weight.shape[shard_dim] // tp_size
                        start_idx = tp_rank * weight_per_partition
                        end_idx = start_idx + weight_per_partition
                        up_weight = up_weight[start_idx:end_idx, :]
                        gate_weight = gate_weight[start_idx:end_idx, :]
                        loaded_weight = torch.cat((up_weight, gate_weight), dim=0)
                    # Load into w13_weight
                    weight_param.data[expert_id].copy_(loaded_weight)
            else:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            return

        # Handle bias loading
        if is_bias:
            if shard_id == "w13":
                # Handle full gate_up_proj bias (w13)
                bias_param = getattr(self, "w13_bias", None)
                if bias_param is not None:
                    # Apply TP sharding to the full bias (bias is 1D, always shard along dim 0)
                    tp_size = get_tensor_model_parallel_world_size()
                    if tp_size > 1 and not self.use_presharded_weights:
                        # Split into gate and up parts
                        up_bias = loaded_weight[loaded_weight.shape[0]//2:]
                        gate_bias = loaded_weight[:loaded_weight.shape[0]//2]
                        assert gate_bias.shape[0] == up_bias.shape[0]
                        # For w13 bias, we shard along dim 0 (output dimension)
                        bias_per_partition = up_bias.shape[0] // tp_size
                        start_idx = tp_rank * bias_per_partition
                        end_idx = start_idx + bias_per_partition
                        up_bias = up_bias[start_idx:end_idx]
                        gate_bias = gate_bias[start_idx:end_idx]
                        loaded_weight = torch.cat((gate_bias, up_bias), dim=0)
                    # Load into w13_bias
                    bias_param.data[expert_id].copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # For w1 and w3, we need to load bias into w13_bias
                bias_param = getattr(self, "w13_bias", None)
                if bias_param is not None:
                    # Apply TP sharding to individual w1/w3 bias
                    tp_size = get_tensor_model_parallel_world_size()
                    if tp_size > 1 and not self.use_presharded_weights:
                        # w1/w3 bias needs to be sharded along output dimension
                        bias_per_partition = loaded_weight.shape[0] // tp_size
                        start_idx = tp_rank * bias_per_partition
                        end_idx = start_idx + bias_per_partition
                        loaded_weight = loaded_weight[start_idx:end_idx]
                    
                    if shard_id == "w1":
                        # Load into first half of w13_bias
                        bias_param.data[expert_id][:bias_param.data[expert_id].shape[0]//2] = loaded_weight
                    else:  # w3
                        # Load into second half of w13_bias
                        bias_param.data[expert_id][bias_param.data[expert_id].shape[0]//2:] = loaded_weight
            elif shard_id == "w2":
                # For w2, load bias into w2_bias (no TP sharding needed for w2 bias)
                bias_param = getattr(self, "w2_bias", None)
                if bias_param is not None:
                    # w2 bias is not sharded in TP (it's the output bias)
                    bias_param.data[expert_id] = loaded_weight
            return

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            custom_routing_function=self.custom_routing_function,
            correction_bias=self.correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            routed_scaling_factor=self.routed_scaling_factor,
            **(
                dict(
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    ep_rank=self.ep_rank,
                    ep_size=self.ep_size,
                )
                if self.quant_method.__class__.__name__ == "ModelOptNvFp4FusedMoEMethod"
                else {}
            ),
        )

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:

        mappings = []
        for expert_id in range(num_experts):
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]:
                # Add weight mapping
                param_name = (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                )
                mappings.append((
                    param_name,
                    f"experts.{expert_id}.{weight_name}.",
                    expert_id,
                    shard_id,
                ))
                
                # Add bias mapping
                bias_param_name = (
                    "experts.w13_bias"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_bias"
                )
                mappings.append((
                    bias_param_name,
                    f"experts.{expert_id}.{weight_name}_bias.",
                    expert_id,
                    shard_id,
                ))
        return mappings

    def _load_fp8_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        if "input_scale" in weight_name:
            if (
                param_data[expert_id] != 1
                and (param_data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param_data[expert_id]} "
                    f"vs. {loaded_weight}"
                )
            param_data[expert_id] = loaded_weight
        # Weight scales
        elif "weight_scale" in weight_name:
            # If we are in merged column case (gate_up_proj)
            if shard_id in ("w1", "w3"):
                # We have to keep the weight scales of w1 and w3 because
                # we need to re-quantize w1/w3 weights after weight loading.
                idx = 0 if shard_id == "w1" else 1
                param_data[expert_id][idx] = loaded_weight
            # If we are in the row parallel case (down_proj)
            else:
                param_data[expert_id] = loaded_weight
