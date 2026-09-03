# SPDX-License-Identifier: Apache-2.0
"""Config for serialized Comfy Kitchen W4A8 ConvRot weights."""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.kitchen_w4a8 import (
    KitchenInt8EmbeddingMethod,
    KitchenW4A8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


class KitchenW4A8Config(QuantizationConfig):
    """Dispatch each linear from its serialized ``asym_w4a8_int8`` marker."""

    def __init__(self, layer_markers: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        if current_platform.is_mps():
            raise ValueError("Serialized W4A8 checkpoints are not supported on MPS")
        if current_platform.is_cuda():
            capability = current_platform.get_device_capability()
            if (
                capability is not None
                and capability.to_int() < self.get_min_capability()
            ):
                raise ValueError(
                    "Serialized W4A8 checkpoints require CUDA compute capability "
                    f">= {self.get_min_capability() / 10:.1f}; got "
                    f"{capability.to_int() / 10:.1f}"
                )
        self.layer_markers = layer_markers
        self.checkpoint_uses_native_qkv_layout = True
        self.selected: list[str] = []

        for prefix, marker in layer_markers.items():
            marker_format = marker.get("format")
            if marker_format == "int8_tensorwise" and marker.get(
                "_is_tensorwise_scalar"
            ):
                continue
            if marker_format != "asym_w4a8_int8":
                raise ValueError(
                    f"Unsupported Comfy W4A8 format for {prefix!r}: {marker_format!r}"
                )
            if marker.get("convrot") is not True:
                raise ValueError(
                    f"Serialized W4A8 layer {prefix!r} must set convrot=true"
                )

    @classmethod
    def get_name(cls) -> str:
        return "kitchen_w4a8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KitchenW4A8Config:
        raise ValueError(
            "kitchen_w4a8 is inferred from per-layer checkpoint metadata; "
            "it is not an online quantization method"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        marker = self.layer_markers.get(prefix)
        if isinstance(layer, VocabParallelEmbedding):
            if marker is None:
                return None
            if marker.get("format") != "int8_tensorwise" or not marker.get(
                "_is_tensorwise_scalar"
            ):
                raise ValueError(
                    f"Unsupported quantized embedding marker for {prefix!r}: {marker}"
                )
            self.selected.append(prefix)
            return KitchenInt8EmbeddingMethod()
        if not isinstance(layer, LinearBase):
            return None
        if marker is None:
            return UnquantizedLinearMethod()
        if marker.get("format") != "asym_w4a8_int8":
            raise ValueError(f"Unsupported quantized linear marker for {prefix!r}")

        group_size = int(marker.get("group_size", 16))
        convrot_group_size = int(marker.get("convrot_groupsize", 256))
        if not self._supports_input_size(
            layer.input_size, group_size, convrot_group_size
        ):
            raise ValueError(
                f"Serialized W4A8 layer {prefix!r} has input size "
                f"{layer.input_size}, incompatible with group_size={group_size} "
                f"and convrot_groupsize={convrot_group_size}"
            )
        self.selected.append(prefix)
        return KitchenW4A8LinearMethod(
            group_size=group_size,
            convrot_group_size=convrot_group_size,
            has_codebook=bool(marker.get("_has_codebook")),
            has_correction=bool(marker.get("_has_correction")),
        )

    @staticmethod
    def _supports_input_size(
        input_size: int, group_size: int, convrot_group_size: int
    ) -> bool:
        return (
            group_size >= 4
            and (16 % group_size == 0 or group_size % 16 == 0)
            and input_size % 16 == 0
            and input_size % group_size == 0
            and input_size % convrot_group_size == 0
        )

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        marker = self.layer_markers.get(prefix)
        if marker is None or marker.get("format") != "asym_w4a8_int8":
            return True
        return self._supports_input_size(
            input_size_per_partition,
            int(marker.get("group_size", 16)),
            int(marker.get("convrot_groupsize", 256)),
        )

    def get_scaled_act_names(self) -> list[str]:
        return []

    def quantizes_embedding(self, prefix: str) -> bool:
        marker = self.layer_markers.get(prefix)
        return bool(
            marker is not None
            and marker.get("format") == "int8_tensorwise"
            and marker.get("_is_tensorwise_scalar")
        )
