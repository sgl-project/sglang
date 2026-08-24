# SPDX-License-Identifier: Apache-2.0
"""Serialized grouped W4A8 linear backed by Comfy Kitchen."""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

try:
    from comfy_kitchen import w4a8_int8_linear
except ImportError:  # pragma: no cover - optional dependency
    w4a8_int8_linear = None


class KitchenW4A8LinearMethod(LinearMethodBase):
    """Load packed INT4 weights and execute the W4A8 ConvRot kernel."""

    def __init__(
        self,
        *,
        group_size: int,
        convrot_group_size: int,
        has_codebook: bool,
        has_correction: bool,
    ) -> None:
        if w4a8_int8_linear is None:
            raise ImportError(
                "W4A8 checkpoints require comfy-kitchen>=0.2.27 "
                "(`pip install -U comfy-kitchen`)."
            )
        self.group_size = group_size
        self.convrot_group_size = convrot_group_size
        self.has_codebook = has_codebook
        self.has_correction = has_correction

    @staticmethod
    def _register_weight(
        layer: torch.nn.Module,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        weight_attrs: dict,
        parallel_dims: dict[str, int] | None = None,
    ) -> None:
        weight = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        if parallel_dims is not None:
            set_weight_attrs(weight, parallel_dims)
        set_weight_attrs(weight, weight_attrs)
        layer.register_parameter(name, weight)

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
                "W4A8 needs input_size_per_partition "
                f"({input_size_per_partition}) divisible by ConvRot group size "
                f"{self.convrot_group_size}"
            )

        output_size_per_partition = sum(output_partition_sizes)
        self._register_weight(
            layer,
            "weight",
            (output_size_per_partition, input_size_per_partition // 2),
            torch.int8,
            extra_weight_attrs,
            {"input_dim": 1, "output_dim": 0},
        )
        self._register_weight(
            layer,
            "weight_s_rel",
            (output_size_per_partition, input_size_per_partition // self.group_size),
            torch.float8_e4m3fn,
            extra_weight_attrs,
            {"input_dim": 1, "output_dim": 0},
        )
        self._register_weight(
            layer,
            "weight_s_channel",
            (output_size_per_partition,),
            torch.float32,
            extra_weight_attrs,
            {"output_dim": 0},
        )
        if self.has_codebook:
            self._register_weight(
                layer,
                "weight_codebook",
                (16,),
                torch.float32,
                extra_weight_attrs,
            )
        else:
            layer.register_parameter("weight_codebook", None)
        if self.has_correction:
            self._register_weight(
                layer,
                "weight_correction",
                (
                    input_size_per_partition // self.group_size,
                    output_size_per_partition,
                ),
                torch.float32,
                extra_weight_attrs,
                {"input_dim": 0, "output_dim": 1},
            )
        else:
            layer.register_parameter("weight_correction", None)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() != 2:
            x = x.reshape(-1, original_shape[-1])
        assert w4a8_int8_linear is not None
        output = w4a8_int8_linear(
            x.contiguous(),
            layer.weight,
            layer.weight_s_rel,
            layer.weight_s_channel,
            codebook=layer.weight_codebook,
            correction=layer.weight_correction,
            bias=bias,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_group_size,
            out_dtype=x.dtype,
        )
        if len(original_shape) != 2:
            output = output.reshape(*original_shape[:-1], output.shape[-1])
        return output


__all__ = ["KitchenW4A8LinearMethod"]
