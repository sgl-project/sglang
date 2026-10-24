# SPDX-License-Identifier: Apache-2.0
"""Rowwise W8A8 FP8 quantization for diffusion linear layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod

QUANT_META_KEY = "_quantization_metadata"
FP8_FORMAT = "rowwise-float8_e4m3fn"
BF16_FORMAT = "bf16"
SUPPORTED_FORMAT_VERSIONS = frozenset({"1.1"})

_ST_FP8 = "F8_E4M3"
_ST_F32 = "F32"


class RowwiseFp8Error(ValueError):
    """Raised when a rowwise FP8 checkpoint violates its declared contract."""


class RowwiseFp8Config(QuantizationConfig):
    """Per-layer rowwise FP8 configuration read from safetensors metadata."""

    def __init__(
        self,
        *,
        format_version: str,
        fp8_layers: frozenset[str],
        input_scale_layers: frozenset[str] = frozenset(),
        num_linear_layers: int = 0,
        num_bf16_layers: int = 0,
        num_tensors: int = 0,
    ) -> None:
        super().__init__()
        self.format_version = format_version
        self.fp8_layers = fp8_layers
        self.input_scale_layers = input_scale_layers
        self.num_linear_layers = num_linear_layers
        self.num_bf16_layers = num_bf16_layers
        self.num_tensors = num_tensors
        self.is_checkpoint_fp8_serialized = True
        self.weight_qscheme = "per_channel"
        self.weight_block_size = None
        self.activation_scheme = "dynamic"

    @property
    def num_fp8_layers(self) -> int:
        return len(self.fp8_layers)

    def has_input_scale(self, prefix: str) -> bool:
        return prefix in self.input_scale_layers

    @classmethod
    def get_name(cls) -> str:
        return "rowwise_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RowwiseFp8Config:
        return cls._from_metadata(config, tensors=None)

    @classmethod
    def _from_metadata(
        cls, config: dict[str, Any], tensors: dict | None
    ) -> RowwiseFp8Config:
        version = config.get("format_version")
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise RowwiseFp8Error(
                f"unsupported quantization format_version {version!r}; "
                f"expected one of {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )

        layers = config.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise RowwiseFp8Error(f"{QUANT_META_KEY} has no 'layers' map")

        quant_configs = config.get("quant_configs", {})
        fp8_layers: set[str] = set()
        num_bf16 = 0
        for prefix, entry in layers.items():
            fmt = (entry or {}).get("format") if isinstance(entry, dict) else entry
            if fmt == BF16_FORMAT:
                num_bf16 += 1
            elif fmt == FP8_FORMAT:
                fp8_layers.add(prefix)
            else:
                raise RowwiseFp8Error(
                    f"unknown quant format {fmt!r} for layer {prefix!r}"
                )

        if not fp8_layers:
            raise RowwiseFp8Error(
                f"{QUANT_META_KEY} declares no quantized layers ({FP8_FORMAT})"
            )

        fp8_config = quant_configs.get(FP8_FORMAT)
        if not isinstance(fp8_config, dict):
            raise RowwiseFp8Error(
                f"missing quant_configs[{FP8_FORMAT!r}] layout descriptor"
            )
        recipe = fp8_config.get("scale_recipe")
        if recipe != "rowwise":
            raise RowwiseFp8Error(
                f"unsupported FP8 scale_recipe {recipe!r}; only 'rowwise' is supported"
            )

        input_scale_layers = (
            _validate_fp8_tensors(tensors, fp8_layers)
            if tensors is not None
            else set(config.get("input_scale_layers", ()))
        )
        return cls(
            format_version=version,
            fp8_layers=frozenset(fp8_layers),
            input_scale_layers=frozenset(input_scale_layers),
            num_linear_layers=len(layers),
            num_bf16_layers=num_bf16,
            num_tensors=len(tensors) if tensors is not None else 0,
        )

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None
        if prefix not in self.fp8_layers:
            return UnquantizedLinearMethod()
        return Fp8LinearMethod(self)

    def get_scaled_act_names(self) -> list[str]:
        return []


def build_rowwise_fp8_config_from_metadata(
    config: dict[str, Any], tensor_metadata: dict
) -> RowwiseFp8Config:
    """Build and validate a checkpoint-serialized rowwise FP8 config."""
    return RowwiseFp8Config._from_metadata(config, tensors=tensor_metadata)


def _validate_fp8_tensors(tensors: dict, fp8_layers: set[str]) -> set[str]:
    input_scale_layers: set[str] = set()
    for prefix in sorted(fp8_layers):
        weight_name = f"{prefix}.weight"
        weight_scale_name = f"{prefix}.weight_scale"
        input_scale_name = f"{prefix}.input_scale"
        if weight_name not in tensors:
            raise RowwiseFp8Error(
                f"FP8 layer {prefix!r} missing weight tensor {weight_name!r}"
            )
        if weight_scale_name not in tensors:
            raise RowwiseFp8Error(
                f"FP8 layer {prefix!r} missing weight_scale tensor "
                f"{weight_scale_name!r}"
            )

        weight = tensors[weight_name]
        weight_scale = tensors[weight_scale_name]
        if weight.get("dtype") != _ST_FP8:
            raise RowwiseFp8Error(
                f"{weight_name}: expected dtype {_ST_FP8}, got {weight.get('dtype')}"
            )
        weight_shape = list(weight.get("shape", []))
        if len(weight_shape) != 2:
            raise RowwiseFp8Error(
                f"{weight_name}: expected 2-D weight, got shape {weight_shape}"
            )
        out_features = weight_shape[0]
        if weight_scale.get("dtype") != _ST_F32:
            raise RowwiseFp8Error(
                f"{weight_scale_name}: expected dtype {_ST_F32}, "
                f"got {weight_scale.get('dtype')}"
            )
        weight_scale_shape = list(weight_scale.get("shape", []))
        if weight_scale_shape != [out_features]:
            raise RowwiseFp8Error(
                f"{weight_scale_name}: row-wise scale must have shape "
                f"[{out_features}], got {weight_scale_shape}"
            )
        if input_scale_name in tensors:
            input_scale = tensors[input_scale_name]
            if input_scale.get("dtype") != _ST_F32 or list(
                input_scale.get("shape", [])
            ) != [1]:
                raise RowwiseFp8Error(
                    f"{input_scale_name}: static input_scale must be float32 "
                    f"shape [1], got dtype={input_scale.get('dtype')} "
                    f"shape={input_scale.get('shape')}"
                )
            input_scale_layers.add(prefix)

    for name in tensors:
        for suffix in (".weight_scale", ".input_scale"):
            if name.endswith(suffix) and name[: -len(suffix)] not in fp8_layers:
                raise RowwiseFp8Error(
                    f"unexpected quantization tensor {name!r} for a non-FP8 layer"
                )
    return input_scale_layers
