# SPDX-License-Identifier: Apache-2.0
"""Runtime operations for serialized Optimum Quanto qint8 weights."""

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
    """Keep qint8 weights packed and dequantize only the active matrix."""

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
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                1,
                dtype=params_dtype,
            ),
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
        weight = layer.weight.to(dtype=x.dtype)
        weight.mul_(layer.weight_scale.to(dtype=x.dtype))
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
