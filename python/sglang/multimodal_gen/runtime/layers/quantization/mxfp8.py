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
from sglang.srt.layers.utils import copy_or_rebind_param


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
        return ComfyMXFP8LinearMethod(self)


class ComfyMXFP8LinearMethod(SRTFp8LinearMethod):
    """Load MXFP8 scales that the checkpoint already stores swizzled.

    ``comfy-kitchen`` serializes MXFP8 block scales in the ``SWIZZLE_32_4_4``
    byte order that FlashInfer's cutlass and cutedsl kernels consume, but
    safetensors keeps their logical ``[N, K // 32]`` shape.  Nothing in the file
    distinguishes those bytes from row-major scales, so
    :meth:`Fp8LinearMethod._process_mxfp8_linear_weight_scale` interleaves them a
    second time.  The second permutation is silent -- the checkpoint loads, the
    kernels run at full speed, and the sample decodes to noise.

    Only checkpoints that carry comfy layer markers take this path;
    :class:`MXFP8Config` resolved from a ``config.json`` has
    ``layer_markers is None`` and keeps SRT's behaviour byte for byte.
    """

    def _process_mxfp8_linear_weight_scale(self, layer: torch.nn.Module) -> None:
        backend = self.mxfp8_dense_backend
        if (
            self.use_mxfp8
            and self.quant_config.layer_markers is not None
            and backend is not None
            and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
        ):
            # Hand FlashInfer the serialized bytes flattened, not re-interleaved.
            copy_or_rebind_param(
                layer,
                "weight_scale_inv_swizzled",
                layer.weight_scale_inv.data.contiguous().view(-1),
            )
            return
        super()._process_mxfp8_linear_weight_scale(layer)


__all__ = ["ComfyMXFP8LinearMethod", "MXFP8Config"]
