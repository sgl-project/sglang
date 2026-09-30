# SPDX-License-Identifier: Apache-2.0
"""Config for serialized Comfy Kitchen ConvRot W4A4 weights."""

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
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.kitchen_w4a4 import (
    KitchenW4A4LinearMethod,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

_QUANT_GROUP_SIZE = 64
_SUPPORTED_CONVROT_GROUP_SIZES = (16, 64, 256)
_SUPPORTED_LINEAR_DTYPES = ("int4", "int8")


class KitchenW4A4Config(QuantizationConfig):
    """Dispatch serialized W4A4 linears and their optional INT8 companions."""

    def __init__(self, layer_markers: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        if current_platform.is_mps():
            raise ValueError("Serialized W4A4 checkpoints are not supported on MPS")
        if current_platform.is_cuda():
            capability = current_platform.get_device_capability()
            if (
                capability is not None
                and capability.to_int() < self.get_min_capability()
            ):
                raise ValueError(
                    "Serialized W4A4 checkpoints require CUDA compute capability "
                    f">= {self.get_min_capability() / 10:.1f}; got "
                    f"{capability.to_int() / 10:.1f}"
                )
        self.layer_markers = layer_markers
        self.checkpoint_uses_native_qkv_layout = True
        self.selected: list[str] = []
        int8_markers = {
            prefix: marker
            for prefix, marker in layer_markers.items()
            if marker.get("format") == "int8_tensorwise"
        }
        self._int8_config = (
            KitchenInt8Config(layer_markers=int8_markers) if int8_markers else None
        )

        for prefix, marker in layer_markers.items():
            marker_format = marker.get("format")
            if marker_format == "int8_tensorwise":
                continue
            if marker_format != "convrot_w4a4":
                raise ValueError(
                    f"Unsupported Comfy W4A4 format for {prefix!r}: {marker_format!r}"
                )
            self._parse_marker(prefix, marker)

    @classmethod
    def get_name(cls) -> str:
        return "kitchen_w4a4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KitchenW4A4Config:
        raise ValueError(
            "kitchen_w4a4 is inferred from per-layer checkpoint metadata; "
            "it is not an online quantization method"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        marker = self.layer_markers.get(prefix)
        if marker is None:
            return UnquantizedLinearMethod()
        if marker.get("format") == "int8_tensorwise":
            assert self._int8_config is not None
            method = self._int8_config.get_quant_method(layer, prefix)
            self.selected.append(prefix)
            return method

        convrot_group_size, linear_dtype = self._parse_marker(prefix, marker)
        if not self._supports_input_size(layer.input_size, convrot_group_size):
            raise ValueError(
                f"Serialized W4A4 layer {prefix!r} has input size "
                f"{layer.input_size}, incompatible with quant_group_size="
                f"{_QUANT_GROUP_SIZE} and convrot_groupsize={convrot_group_size}"
            )
        self.selected.append(prefix)
        return KitchenW4A4LinearMethod(
            convrot_group_size=convrot_group_size,
            linear_dtype=linear_dtype,
        )

    @staticmethod
    def _parse_marker(prefix: str, marker: dict[str, Any]) -> tuple[int, str]:
        convrot_group_size = int(marker.get("convrot_groupsize", 256))
        if convrot_group_size not in _SUPPORTED_CONVROT_GROUP_SIZES:
            raise ValueError(
                f"Serialized W4A4 layer {prefix!r} has unsupported "
                f"convrot_groupsize={convrot_group_size}; expected one of "
                f"{_SUPPORTED_CONVROT_GROUP_SIZES}"
            )
        linear_dtype = str(marker.get("linear_dtype", "int4"))
        if linear_dtype not in _SUPPORTED_LINEAR_DTYPES:
            raise ValueError(
                f"Serialized W4A4 layer {prefix!r} has unsupported "
                f"linear_dtype={linear_dtype!r}; expected one of "
                f"{_SUPPORTED_LINEAR_DTYPES}"
            )
        return convrot_group_size, linear_dtype

    @staticmethod
    def _supports_input_size(input_size: int, convrot_group_size: int) -> bool:
        return (
            input_size % _QUANT_GROUP_SIZE == 0 and input_size % convrot_group_size == 0
        )

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        marker = self.layer_markers.get(prefix)
        if marker is None:
            return True
        if marker.get("format") == "int8_tensorwise":
            assert self._int8_config is not None
            return self._int8_config.supports_input_partition(
                prefix, input_size_per_partition
            )
        convrot_group_size, _ = self._parse_marker(prefix, marker)
        return self._supports_input_size(input_size_per_partition, convrot_group_size)

    def get_scaled_act_names(self) -> list[str]:
        return []
