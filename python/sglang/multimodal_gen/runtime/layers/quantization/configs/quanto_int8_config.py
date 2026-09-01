# SPDX-License-Identifier: Apache-2.0
"""Config and checkpoint admission for Optimum Quanto qint8 weights."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import torch
from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase as DiffusionLinearBase,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    UnquantizedLinearMethod as DiffusionUnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.quanto_int8 import (
    QuantoInt8LinearMethod,
)
from sglang.srt.layers.linear import LinearBase as SrtLinearBase
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod as SrtUnquantizedLinearMethod,
)

_FLOAT_DTYPES = {"BF16", "F16", "F32"}


class QuantoInt8Config(QuantizationConfig):
    """Dispatch linears declared qint8 in an Optimum Quanto quantization map."""

    supports_srt_linear_layers = True

    def __init__(self, layer_prefixes: set[str]) -> None:
        super().__init__()
        self.layer_prefixes = layer_prefixes
        self.selected: set[str] = set()

    @classmethod
    def get_name(cls) -> str:
        return "quanto_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QuantoInt8Config:
        raise ValueError(
            "QuantoInt8Config must be constructed from safetensors metadata"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, DiffusionLinearBase):
            unquantized_method = DiffusionUnquantizedLinearMethod
        elif isinstance(layer, SrtLinearBase):
            unquantized_method = SrtUnquantizedLinearMethod
        else:
            return None
        if prefix not in self.layer_prefixes:
            return unquantized_method()
        self.selected.add(prefix)
        return QuantoInt8LinearMethod()


def inspect_quanto_int8_checkpoint(
    file_path: str,
    param_name_mapper: Callable[[str], str] | None = None,
) -> QuantoInt8Config | None:
    """Validate a self-describing Quanto qint8 safetensors checkpoint."""

    with safe_open(file_path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("quantization_format") != "quanto":
            return None

        encoded_map = metadata.get("quantization_map_base64")
        if encoded_map is None:
            raise ValueError("Quanto checkpoint is missing quantization_map_base64")
        try:
            quantization_map = json.loads(
                base64.b64decode(encoded_map, validate=True).decode("utf-8")
            )
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid Quanto quantization_map_base64") from error
        if not isinstance(quantization_map, dict) or not quantization_map:
            raise ValueError("Quanto quantization map must be a non-empty object")
        if not all(
            isinstance(prefix, str) and isinstance(spec, dict)
            for prefix, spec in quantization_map.items()
        ):
            raise ValueError("Quanto quantization map entries must be named objects")

        checkpoint_keys = set(checkpoint.keys())
        data_suffix = ".weight._data"
        data_prefixes = {
            name.removesuffix(data_suffix)
            for name in checkpoint_keys
            if name.endswith(data_suffix)
        }
        map_prefixes = set(quantization_map)
        if data_prefixes != map_prefixes:
            missing_map = data_prefixes - map_prefixes
            missing_data = map_prefixes - data_prefixes
            raise ValueError(
                "Quanto tensor/map prefixes do not match: "
                f"missing metadata={sorted(missing_map)[:5]}, "
                f"missing tensors={sorted(missing_data)[:5]}"
            )

        mapped_prefixes: set[str] = set()
        for prefix, quantization in quantization_map.items():
            if quantization.get("weights") != "qint8":
                raise ValueError(
                    f"Unsupported Quanto weight type for {prefix!r}: "
                    f"{quantization.get('weights')!r}"
                )
            if quantization.get("activations") != "none":
                raise ValueError(
                    f"Quanto activation quantization is not supported for {prefix!r}"
                )

            names = {
                "data": f"{prefix}.weight._data",
                "scale": f"{prefix}.weight._scale",
                "input": f"{prefix}.input_scale",
                "output": f"{prefix}.output_scale",
            }
            missing = set(names.values()) - checkpoint_keys
            if missing:
                raise ValueError(
                    f"Quanto layer {prefix!r} is missing tensors: {sorted(missing)}"
                )
            if f"{prefix}.weight" in checkpoint_keys:
                raise ValueError(
                    f"Quanto layer {prefix!r} contains both packed and dense weights"
                )

            data_slice = checkpoint.get_slice(names["data"])
            scale_slice = checkpoint.get_slice(names["scale"])
            data_shape = tuple(data_slice.get_shape())
            scale_shape = tuple(scale_slice.get_shape())
            if data_slice.get_dtype() != "I8" or len(data_shape) != 2:
                raise ValueError(
                    f"Quanto layer {prefix!r} needs a 2D I8 weight, got "
                    f"{data_slice.get_dtype()} {data_shape}"
                )
            if scale_slice.get_dtype() not in _FLOAT_DTYPES or scale_shape != (
                data_shape[0],
                1,
            ):
                raise ValueError(
                    f"Quanto layer {prefix!r} has incompatible scale "
                    f"{scale_slice.get_dtype()} {scale_shape}"
                )
            for scale_name in (names["input"], names["output"]):
                scale = checkpoint.get_slice(scale_name)
                if (
                    scale.get_dtype() not in _FLOAT_DTYPES
                    or tuple(scale.get_shape()) != ()
                ):
                    raise ValueError(
                        f"Quanto auxiliary scale {scale_name!r} must be a float scalar"
                    )

            mapped_prefix = (
                param_name_mapper(prefix) if param_name_mapper is not None else prefix
            )
            if mapped_prefix in mapped_prefixes:
                raise ValueError(
                    f"Quanto layers collide after parameter mapping at {mapped_prefix!r}"
                )
            mapped_prefixes.add(mapped_prefix)

    return QuantoInt8Config(mapped_prefixes)


__all__ = ["QuantoInt8Config", "inspect_quanto_int8_checkpoint"]
