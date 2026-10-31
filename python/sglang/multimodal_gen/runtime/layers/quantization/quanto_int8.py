# SPDX-License-Identifier: Apache-2.0
"""Weight-only INT8 operations for per-channel and groupwise checkpoints."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs


class QuantoInt8LinearMethod(LinearMethodBase):
    """Keep INT8 weights packed and dequantize only the active matrix.

    Defaults preserve Quanto's per-output-channel, activation-dtype multiply.
    Groupwise checkpoints can retain FP32 scales and explicitly choose compute
    precision; their dequantization multiplies in FP32 before the final cast.
    """

    def __init__(
        self,
        group_size: int | None = None,
        *,
        scale_dtype: torch.dtype | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        if group_size is not None and group_size <= 0:
            raise ValueError("INT8 group_size must be positive")
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.compute_dtype = compute_dtype

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if self.compute_dtype is not None and params_dtype != self.compute_dtype:
            raise ValueError(f"INT8 requires {self.compute_dtype} compute")
        if (
            self.group_size is not None
            and input_size_per_partition != input_size
            and input_size_per_partition % self.group_size
        ):
            raise ValueError(
                f"INT8 TP input shards must align to {self.group_size} elements"
            )
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        attrs = {"input_dim": 1, "output_dim": 0}
        if self.group_size is not None:
            attrs["missing_param_init"] = "error"
        set_weight_attrs(weight, attrs)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                1
                if self.group_size is None
                else (input_size_per_partition + self.group_size - 1)
                // self.group_size,
                dtype=self.scale_dtype or params_dtype,
            ),
            requires_grad=False,
        )
        scale_attrs = {"output_dim": 0}
        if self.group_size is not None:
            scale_attrs.update(input_dim=1, missing_param_init="error")
        set_weight_attrs(weight_scale, scale_attrs)
        set_weight_attrs(weight_scale, extra_weight_attrs)
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.group_size is not None:
            scales = layer.weight_scale
            if not torch.isfinite(scales).all() or not (scales > 0).all():
                raise ValueError("INT8 scales must be finite and positive")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.compute_dtype is not None and x.dtype != self.compute_dtype:
            if (
                x.is_cuda
                and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == self.compute_dtype
            ):
                x = x.to(self.compute_dtype)
            else:
                dtype_name = (
                    "BF16"
                    if self.compute_dtype == torch.bfloat16
                    else str(self.compute_dtype)
                )
                raise TypeError(f"INT8 linear requires {dtype_name} input")
        if self.group_size is None:
            weight = layer.weight.to(dtype=x.dtype)
            weight.mul_(layer.weight_scale.to(dtype=x.dtype))
        else:
            out, width = layer.weight.shape
            if width % self.group_size == 0:
                weight = (
                    layer.weight.reshape(out, -1, self.group_size).float()
                    * layer.weight_scale[..., None]
                ).reshape(out, width)
            else:
                # A replicated layer may have a partial final group.
                scales = layer.weight_scale.repeat_interleave(self.group_size, dim=1)
                weight = layer.weight.float() * scales[:, :width]
            weight = weight.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def normalize_quanto_int8_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Translate flattened Quanto tensors to native linear parameter names."""

    for name, tensor in weights:
        if name.endswith((".input_scale", ".output_scale")):
            if tensor.numel() != 1 or tensor.item() != 1:
                raise ValueError(f"Quanto weight-only scale {name!r} must equal 1")
            continue
        if name.endswith(".weight._data"):
            name = name.removesuffix("._data")
        elif name.endswith(".weight._scale"):
            name = name.removesuffix("._scale") + "_scale"
        yield name, tensor


__all__ = [
    "QuantoInt8LinearMethod",
    "normalize_quanto_int8_weights",
]
