# SPDX-License-Identifier: Apache-2.0
"""Serialized ConvRot W4A4 linear backed by Comfy Kitchen."""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

try:
    from comfy_kitchen import convrot_w4a4_linear
except ImportError:  # pragma: no cover - optional dependency
    convrot_w4a4_linear = None

_QUANT_GROUP_SIZE = 64


class KitchenW4A4LinearMethod(LinearMethodBase):
    """Load packed INT4 weights and execute the ConvRot W4A4 kernel."""

    def __init__(self, *, convrot_group_size: int, linear_dtype: str) -> None:
        if convrot_w4a4_linear is None:
            raise ImportError(
                "W4A4 checkpoints require a current comfy-kitchen build "
                "(`pip install -U comfy-kitchen`)."
            )
        self.convrot_group_size = convrot_group_size
        self.linear_dtype = linear_dtype

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        if input_size_per_partition % self.convrot_group_size:
            raise ValueError(
                "W4A4 needs input_size_per_partition "
                f"({input_size_per_partition}) divisible by ConvRot group size "
                f"{self.convrot_group_size}"
            )
        if input_size_per_partition % _QUANT_GROUP_SIZE:
            raise ValueError(
                "W4A4 needs input_size_per_partition "
                f"({input_size_per_partition}) divisible by quantization group size "
                f"{_QUANT_GROUP_SIZE}"
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(output_size_per_partition, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"output_dim": 0})
        set_weight_attrs(weight_scale, extra_weight_attrs)
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert convrot_w4a4_linear is not None
        return convrot_w4a4_linear(
            x.contiguous(),
            layer.weight,
            layer.weight_scale,
            bias=bias,
            convrot_groupsize=self.convrot_group_size,
            quant_group_size=_QUANT_GROUP_SIZE,
            linear_dtype=self.linear_dtype,
        )


__all__ = ["KitchenW4A4LinearMethod"]
