# SPDX-License-Identifier: Apache-2.0
"""Diffusion Linear adapter for SRT's GGUF kernels."""

from __future__ import annotations

from typing import Any

import gguf
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.loader.gguf_weights import GGUFTensorMeta
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs
from sglang.srt.layers.quantization.gguf import (
    DEQUANT_TYPES,
    UNQUANTIZED_TYPES,
    dequantize_gguf_weight,
)


class GGUFConfig(QuantizationConfig):
    """Select a GGUF method from each checkpoint tensor's metadata."""

    def __init__(self, gguf_file: str, tensor_meta: dict[str, GGUFTensorMeta]):
        super().__init__()
        self.gguf_file = gguf_file
        self.tensor_meta = tensor_meta

    @classmethod
    def get_name(cls) -> str:
        return "gguf"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GGUFConfig:
        raise ValueError("GGUFConfig must be constructed from a GGUF checkpoint")

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None

        metadata = self.tensor_meta.get(f"{prefix}.weight")
        if metadata is None:
            raise ValueError(
                f"Linear layer {prefix!r} has no weight in the GGUF checkpoint "
                f"{self.gguf_file!r}"
            )
        weight_type = metadata.weight_type
        if weight_type in UNQUANTIZED_TYPES:
            return UnquantizedLinearMethod()
        if weight_type not in DEQUANT_TYPES:
            raise ValueError(
                f"GGUF tensor {prefix}.weight uses unsupported type {weight_type}"
            )
        return GGUFLinearMethod(metadata, prefix)


class GGUFLinearMethod(LinearMethodBase):
    """Register TP-local packed weights and reuse SRT dequantization."""

    def __init__(self, metadata: GGUFTensorMeta, prefix: str) -> None:
        self.metadata = metadata
        self.prefix = prefix
        self.weight_type = metadata.weight_type

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
        if self.metadata.logical_shape != (output_size, input_size):
            raise ValueError(
                f"GGUF tensor {self.prefix}.weight has logical shape "
                f"{self.metadata.logical_shape}, expected {(output_size, input_size)}"
            )

        block_size, type_size = gguf.GGML_QUANT_SIZES[self.weight_type]
        if input_size_per_partition % block_size:
            raise ValueError(
                f"GGUF tensor {self.prefix}.weight cannot be TP-sharded: input "
                f"partition {input_size_per_partition} is not aligned to "
                f"quantization block size {block_size}"
            )
        qweight = nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // block_size * type_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = dequantize_gguf_weight(layer.qweight, self.weight_type, x.dtype)
        return nn.functional.linear(x, weight, bias)


__all__ = ["GGUFConfig", "GGUFLinearMethod"]
