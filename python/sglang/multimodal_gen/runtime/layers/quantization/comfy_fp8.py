# SPDX-License-Identifier: Apache-2.0
"""Comfy per-layer FP8 checkpoint support."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
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
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)


class ComfyFullPrecisionFp8LinearMethod(LinearMethodBase):
    """Keep FP8 storage but honor Comfy's full-precision matmul marker."""

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
        if len(output_partition_sizes) != 1:
            raise ValueError(
                "Comfy full_precision_matrix_mult does not support fused linears"
            )
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_partition_sizes[0]
        layer.orig_dtype = params_dtype
        weight = ModelWeightParameter(
            data=torch.empty(
                output_partition_sizes[0],
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        weight_scale = PerTensorScaleParameter(
            data=torch.empty(1, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.weight = nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = nn.Parameter(layer.weight_scale.data, requires_grad=False)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Comfy's marker disables quantized GEMM for this layer. Materialize one
        # compute-dtype matrix per call so the checkpoint's FP8 residency benefit
        # is retained instead of permanently expanding every fc2 weight to BF16.
        weight = layer.weight.to(dtype=x.dtype)
        weight.mul_(layer.weight_scale[0].to(dtype=x.dtype))
        return F.linear(x, weight, bias)


class ComfyFp8Config(QuantizationConfig):
    """Dispatch each Linear according to its serialized ``comfy_quant`` marker."""

    checkpoint_uses_native_qkv_layout = True

    def __init__(self, layer_markers: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.layer_markers = layer_markers
        self.selected: list[str] = []
        self._fp8_configs = {
            activation_scheme: Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme=activation_scheme,
            )
            for activation_scheme in ("static", "dynamic")
        }

        unsupported = {
            prefix: marker.get("format")
            for prefix, marker in layer_markers.items()
            if marker.get("format") != "float8_e4m3fn"
        }
        if unsupported:
            raise ValueError(f"Unsupported Comfy FP8 layer formats: {unsupported}")

    @classmethod
    def get_name(cls) -> str:
        return "comfy_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return Fp8Config.get_supported_act_dtypes()

    @classmethod
    def get_min_capability(cls) -> int:
        return Fp8Config.get_min_capability()

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ComfyFp8Config:
        raise ValueError(
            "ComfyFp8Config must be constructed from safetensors layer markers"
        )

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        marker = self.layer_markers.get(prefix)
        if marker is None:
            return UnquantizedLinearMethod()
        self.selected.append(prefix)
        if marker.get("full_precision_matrix_mult", False):
            return ComfyFullPrecisionFp8LinearMethod()
        activation_scheme = marker.get("_activation_scheme", "static")
        return Fp8LinearMethod(self._fp8_configs[activation_scheme])


__all__ = [
    "ComfyFp8Config",
    "ComfyFullPrecisionFp8LinearMethod",
]
