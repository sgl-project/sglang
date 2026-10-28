# SPDX-License-Identifier: Apache-2.0
"""Online MXFP4 requantization of a checkpoint whose base is MXFP8-serialized and whose
routed experts are NVFP4 (MiniMax-M3.1): only the experts are requantized."""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.quark.utils import (
    Nvfp4SourceConfig,
    should_ignore_layer,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def is_mxfp8_base_nvfp4_moe_source(config: Dict[str, Any]) -> bool:
    """Whether `config` is an MXFP8-serialized base whose routed experts are NVFP4 in
    llm-compressor's `nvfp4-pack-quantized` layout, as MiniMax-M3.1 ships it.

    Unlike ModelOpt MIXED_PRECISION there is no per-layer map: every linear outside
    `ignored_layers` is MXFP8 (fp8 weight + e8m0 [1, 32] `weight_scale_inv`), and only
    the FusedMoE routed experts are NVFP4.
    """
    if str(config.get("quant_method", "")).lower() != "mxfp8":
        return False
    if str(config.get("moe_quant_algo", "")).upper() != "NVFP4":
        return False
    moe_format = str(config.get("moe_quant_format", "")).lower()
    return moe_format == "" or "nvfp4" in moe_format


def _subtree_excludes(names: List[str]) -> List[str]:
    """Exclude patterns matching each module name and every submodule of it
    (`vision_tower` -> `vision_tower.<anything>`), for producer `ignored_layers`
    lists that name whole subtrees rather than leaf projections."""
    return [f"re:^{re.escape(name)}(\\..*)?$" for name in names if name]


class QuarkMxfp8BaseConfig(QuarkConfig):
    """Only FusedMoE layers are requantized; every non-excluded linear layer loads
    as-is through `excluded_fp8_config` and the excludes themselves stay bf16."""

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return super().get_quant_method(layer, prefix)
        if should_ignore_layer(
            prefix,
            ignore=self.exclude_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            # An MXFP8-base source lists only its bf16 modules as excludes
            # (lm_head / embeddings / vision tower / routers).
            return UnquantizedLinearMethod()
        # Serialized MXFP8 (fp8 weight + e8m0 [1, 32] scale) stays as-is;
        # only the routed experts go through NVFP4 -> MXFP4.
        return Fp8LinearMethod(quant_config=self.excluded_fp8_config)

    @classmethod
    def from_source(
        cls, config: Dict[str, Any], hf_config: "PretrainedConfig"
    ) -> "QuarkMxfp8BaseConfig":
        source_excludes = _subtree_excludes(list(config.get("ignored_layers") or []))
        quant_config = QuarkConfig._create_online_mxfp4_config(
            model_type=hf_config.model_type,
            source_excludes=source_excludes,
            packed_modules_mapping=config.get("packed_modules_mapping"),
        )
        excluded_fp8_config = Fp8Config.from_config(config)
        assert excluded_fp8_config.use_mxfp8, (
            "an mxfp8 quant_method must resolve to the MXFP8 linear path"
        )
        # off Blackwell use block-FP8: Hopper has no MX matmul, gfx950's dot_scaled is slower
        excluded_fp8_config.convert_mxfp8_to_block_fp8 = not get_platform().is_blackwell
        return cls(
            quant_config=quant_config,
            hf_config=hf_config,
            is_prequantized=False,
            dequantization_config=Nvfp4SourceConfig(),
            excluded_fp8_config=excluded_fp8_config,
        )
