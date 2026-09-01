# SPDX-License-Identifier: Apache-2.0
"""MXFP8 diffusion adapter backed by SRT's dense linear kernels."""

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
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP8LinearMethod,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config as SRTFp8Config
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod as SRTFp8LinearMethod


class MXFP8Config(SRTFp8Config, QuantizationConfig):
    """Route diffusion linears through SRT's MXFP8 implementation."""

    def __init__(
        self,
        *,
        is_checkpoint_fp8_serialized: bool = False,
        layer_markers: dict[str, dict[str, Any]] | None = None,
        ignored_layers: list[str] | None = None,
    ) -> None:
        if current_platform.is_mps():
            raise ValueError("MXFP8 is not supported on MPS")
        super().__init__(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme="dynamic",
            ignored_layers=ignored_layers,
            weight_block_size=[1, 32] if is_checkpoint_fp8_serialized else None,
            use_mxfp8=True,
        )
        self.layer_markers = layer_markers
        self.checkpoint_uses_native_qkv_layout = layer_markers is not None

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MXFP8Config:
        quant_method = str(cls.get_from_keys(config, ["quant_method"])).lower()
        if "mxfp8" not in quant_method:
            raise ValueError(f"Expected an MXFP8 checkpoint, got {quant_method!r}")
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        if activation_scheme != "dynamic":
            raise ValueError("MXFP8 only supports dynamic activation scaling")
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        return cls(
            is_checkpoint_fp8_serialized=True,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if self.layer_markers is not None and prefix not in self.layer_markers:
            return UnquantizedLinearMethod()
        if current_platform.is_npu():
            return NPUMXFP8LinearMethod(self)
        return SRTFp8LinearMethod(self)


__all__ = ["MXFP8Config"]
