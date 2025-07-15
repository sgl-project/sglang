# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright © 2025, Oracle and/or its affiliates.

import logging
import os
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase, LinearMethodBase
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.rtn_utils import (
    fix_weights,
    rtn_dequantize,
    rtn_quantize,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


# By default, use 8 bit as target precision, but it can be overridden by setting the RTN_NUM_BITS envvar
NUM_BITS = os.getenv("RTN_NUM_BITS", "8")


# By default, use group size of 128 parameters, but it can be overridden by setting the RTN_GROUP_SIZE envvar
GROUP_SIZE = os.getenv("RTN_GROUP_SIZE", "128")


class RTNConfig(QuantizationConfig):
    """Config class for RTN."""

    def __init__(
        self,
        weight_bits: int = int(NUM_BITS),
        group_size: int = int(GROUP_SIZE),
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size

        if self.weight_bits != 4 and self.weight_bits != 8:
            raise ValueError(
                "Currently, only 4-bit or 8-bit weight quantization is "
                f"supported for RTN, but got {self.weight_bits} bits."
            )

    def __repr__(self) -> str:
        return (
            f"RTNConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "rtn"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RTNConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            return RTNLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return RTNMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class RTNTensor:
    """A wrapper over Tensor that enables quantization on-the-fly by
    overloading the copy_ method.
    """

    def __init__(
        self, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> None:
        self.data = data
        self.scale = scale
        self.quant_config = quant_config

    def narrow(self, dim, start, length):
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        return RTNTensor(
            self.data.narrow(dim, start // factor, length // factor),
            self.scale.narrow(dim, start, length),
            self.quant_config,
        )

    def __getitem__(self, key):
        return RTNTensor(self.data[key], self.scale[key], self.quant_config)

    @property
    def shape(self):
        shape = self.data.shape
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        batch_present = len(shape) == 3
        if batch_present:
            return torch.Size((shape[0], shape[1] * factor, shape[2]))
        else:
            return torch.Size((shape[0] * factor, shape[1]))

    def copy_(self, loaded_weight: torch.Tensor) -> None:
        qweight, weight_scale = rtn_quantize(
            loaded_weight.cuda(),
            self.quant_config.weight_bits,
            self.quant_config.group_size,
        )

        self.data.copy_(qweight)
        self.scale.data.copy_(weight_scale)


class RTNParameter(Parameter):
    """A wrapper over Parameter that returns RTNTensor (a wrapper over Tensor)
    when its data is accessed. We need this wrapper for the data loading phase
    only, so we can intercept a weight copying function (torch.Tensor.copy_)
    and apply quantization on-the-fly.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(
        self, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> None:
        self.scale = scale
        self.quant_config = quant_config

    @property
    def data(self):
        return RTNTensor(super().data, self.scale, self.quant_config)


class RTNLinearMethod(LinearMethodBase):
    """Linear method for RTN.

    Args:
        quant_config: The RTN quantization config.
    """

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        num_groups_per_col = (
            input_size_per_partition // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )

        scale = Parameter(
            torch.empty(
                output_size_per_partition, num_groups_per_col, dtype=params_dtype
            ),
            requires_grad=False,
        )
        factor = 1 if self.quant_config.weight_bits == 8 else 2

        weight = RTNParameter(
            data=torch.empty(
                output_size_per_partition // factor,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            scale=scale,
            quant_config=self.quant_config,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        layer.register_parameter("scale", scale)
        layer.output_size_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        fix_weights(layer, "weight")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.weight
        scale = layer.scale

        weight = rtn_dequantize(qweight, scale)
        out = F.linear(x, weight)
        del weight
        if bias is not None:
            out.add_(bias)

        return out


class RTNMoEMethod:
    """MoE method for RTN."""

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

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        factor = 1 if self.quant_config.weight_bits == 8 else 2

        # Fused gate_up_proj (column parallel)
        num_groups_per_col = (
            hidden_size // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )
        w13_scale = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                num_groups_per_col,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale", w13_scale)

        w13_weight = RTNParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition // factor,
                hidden_size,
                dtype=torch.uint8,
            ),
            scale=w13_scale,
            quant_config=self.quant_config,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        num_groups_per_col = (
            intermediate_size_per_partition // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )
        w2_scale = Parameter(
            torch.zeros(
                num_experts, hidden_size, num_groups_per_col, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale", w2_scale)

        w2_weight = RTNParameter(
            data=torch.empty(
                num_experts,
                hidden_size // factor,
                intermediate_size_per_partition,
                dtype=torch.uint8,
            ),
            scale=w2_scale,
            quant_config=self.quant_config,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_bits = self.quant_config.weight_bits
        fix_weights(layer, "w13_weight", weight_bits == 4)
        fix_weights(layer, "w2_weight", weight_bits == 4)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        enable_eplb: bool = False,
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        if enable_eplb:
            raise NotImplementedError("EPLB not supported for `RTNMoEMethod` yet.")

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

        weight_bits = self.quant_config.weight_bits
        group_size = self.quant_config.group_size

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            w1_scale=layer.w13_scale,
            w2_scale=layer.w2_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            block_shape=[0, group_size],
            routed_scaling_factor=routed_scaling_factor,
        )
