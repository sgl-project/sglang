# SPDX-License-Identifier: Apache-2.0
"""Portable full-precision execution for Comfy NVFP4 checkpoints."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    _swizzled_nvfp4_scales_to_linear,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs
from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4


def _register_parameter(
    layer: nn.Module,
    name: str,
    data: torch.Tensor,
    weight_attrs: dict[str, Any],
    parallel_dims: dict[str, int] | None = None,
) -> None:
    parameter = nn.Parameter(data, requires_grad=False)
    if parallel_dims is not None:
        set_weight_attrs(parameter, parallel_dims)
    set_weight_attrs(parameter, weight_attrs)
    layer.register_parameter(name, parameter)


class ComfyRowwiseInt8EmbeddingMethod(QuantizeMethodBase):
    """Gather and dequantize only selected rows of an INT8 embedding."""

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
        del input_size, output_size
        self.output_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        _register_parameter(
            layer,
            "weight",
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            extra_weight_attrs,
            {"input_dim": 1, "output_dim": 0},
        )
        _register_parameter(
            layer,
            "weight_scale",
            torch.empty(output_size_per_partition, 1, dtype=torch.float32),
            extra_weight_attrs,
            {"output_dim": 0},
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Comfy INT8 embedding weights support lookup only")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        weight = F.embedding(input_, layer.weight).to(self.output_dtype)
        scale = F.embedding(input_, layer.weight_scale).to(self.output_dtype)
        return weight * scale


class ComfyFullPrecisionNvfp4LinearMethod(ModelOptFp4LinearMethod):
    """Keep NVFP4 storage and dequantize one active Linear for its matmul."""

    def __init__(
        self,
        quant_config: ComfyNvfp4Config,
        *,
        has_pre_quant_scale: bool,
    ) -> None:
        self.quant_config = quant_config
        self.has_pre_quant_scale = has_pre_quant_scale

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
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        # Comfy uses runtime activations directly for this weight-only path.
        layer.register_parameter("input_scale", None)
        if not self.has_pre_quant_scale:
            return
        _register_parameter(
            layer,
            "pre_quant_scale",
            torch.empty(input_size_per_partition, dtype=params_dtype),
            extra_weight_attrs,
            {"input_dim": 0},
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # The portable path consumes the serialized representation directly.
        # ModelOpt's inherited hook instead prepares a Blackwell-only kernel.
        return

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.has_pre_quant_scale:
            x = x * layer.pre_quant_scale
        weight_scale = _swizzled_nvfp4_scales_to_linear(layer.weight_scale)
        weight = dequantize_nvfp4(
            layer.weight,
            weight_scale,
            layer.weight_scale_2,
            out_dtype=x.dtype,
            high_nibble_first=True,
        )
        return F.linear(x, weight, bias)


class ComfyNvfp4Config(ModelOptFp4Config):
    """Dispatch full-precision Comfy NVFP4 linears and their INT8 embedding."""

    checkpoint_uses_comfy_quantization = True

    def __init__(self, layer_markers: dict[str, dict[str, Any]]) -> None:
        super().__init__(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[],
            checkpoint_uses_comfy_quantization=True,
        )
        self.layer_markers = layer_markers
        self.selected: list[str] = []
        for prefix, marker in layer_markers.items():
            marker_format = marker.get("format")
            if marker_format == "int8_tensorwise" and marker.get("_is_rowwise"):
                continue
            if marker_format != "nvfp4":
                raise ValueError(
                    f"Unsupported Comfy NVFP4 companion for {prefix!r}: "
                    f"{marker_format!r}"
                )
            if marker.get("full_precision_matrix_mult") is not True:
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} must request "
                    "full_precision_matrix_mult"
                )

    @classmethod
    def get_name(cls) -> str:
        return "comfy_nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ComfyNvfp4Config:
        raise ValueError(
            "comfy_nvfp4 is inferred from per-layer checkpoint metadata; "
            "it is not an online quantization method"
        )

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        marker = self.layer_markers.get(prefix)
        if isinstance(layer, VocabParallelEmbedding):
            if marker is None:
                return None
            if marker.get("format") != "int8_tensorwise" or not marker.get(
                "_is_rowwise"
            ):
                raise ValueError(
                    f"Unsupported quantized embedding marker for {prefix!r}: {marker}"
                )
            self.selected.append(prefix)
            return ComfyRowwiseInt8EmbeddingMethod()
        if not isinstance(layer, LinearBase):
            return None
        if marker is None:
            return UnquantizedLinearMethod()
        if marker.get("format") != "nvfp4":
            raise ValueError(f"Unsupported quantized linear marker for {prefix!r}")
        self.selected.append(prefix)
        return ComfyFullPrecisionNvfp4LinearMethod(
            self,
            has_pre_quant_scale=bool(marker.get("_has_pre_quant_scale")),
        )

    def quantizes_embedding(self, prefix: str) -> bool:
        marker = self.layer_markers.get(prefix)
        return bool(
            marker is not None
            and marker.get("format") == "int8_tensorwise"
            and marker.get("_is_rowwise")
        )


__all__ = [
    "ComfyFullPrecisionNvfp4LinearMethod",
    "ComfyNvfp4Config",
    "ComfyRowwiseInt8EmbeddingMethod",
]
