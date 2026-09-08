# SPDX-License-Identifier: Apache-2.0
"""INT8 embedding lookup shared by serialized Comfy quantization formats."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs


def is_comfy_int8_embedding(marker: dict[str, Any] | None) -> bool:
    return bool(
        marker is not None
        and marker.get("format") == "int8_tensorwise"
        and not marker.get("convrot", False)
        and (marker.get("_is_rowwise") or marker.get("_is_tensorwise_scalar"))
    )


class ComfyInt8EmbeddingMethod(QuantizeMethodBase):
    """Keep the table packed and dequantize only the requested rows."""

    def __init__(self, *, tensorwise: bool = False) -> None:
        self.tensorwise = tensorwise

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        self.output_dtype = params_dtype
        rows = sum(output_partition_sizes)
        for name, shape, dtype, dims in (
            (
                "weight",
                (rows, input_size_per_partition),
                torch.int8,
                {"input_dim": 1, "output_dim": 0},
            ),
            (
                "weight_scale",
                () if self.tensorwise else (rows, 1),
                torch.float32,
                {} if self.tensorwise else {"output_dim": 0},
            ),
        ):
            parameter = nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            set_weight_attrs(parameter, extra_weight_attrs)
            set_weight_attrs(parameter, dims)
            layer.register_parameter(name, parameter)

    def apply(self, layer: nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        raise NotImplementedError("Comfy INT8 embeddings support lookup only")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        weight = F.embedding(input_, layer.weight)
        if self.tensorwise:
            # scalar-scale exports multiply in FP32 before rounding to the activation dtype
            return (weight.float() * layer.weight_scale).to(self.output_dtype)
        scale = F.embedding(input_, layer.weight_scale).to(self.output_dtype)
        return weight.to(self.output_dtype) * scale
