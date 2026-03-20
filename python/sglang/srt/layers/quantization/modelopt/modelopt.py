# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import regex as re
import torch

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_attention import RadixAttention

if TYPE_CHECKING:
    from sglang.srt.models.utils import WeightsMapper


class ModelOptQuantConfig(QuantizationConfig):
    def __init__(
        self,
        kv_cache_quant_algo: Optional[str],
        exclude_modules: Optional[List[str]],
        packed_modules_mapping: Optional[Dict[str, List[str]]],
    ):
        super().__init__()
        self.packed_modules_mapping = packed_modules_mapping
        self.exclude_modules = exclude_modules or []
        self.kv_cache_quant_algo = kv_cache_quant_algo

    def _get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        *,
        Linear: type[LinearMethodBase],
        Moe: type[FusedMoEMethodBase],
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.exclude_modules, self.packed_modules_mapping
            ) or self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            return Linear(self)
        elif self.kv_cache_quant_algo and isinstance(layer, RadixAttention):
            from sglang.srt.layers.quantization.modelopt.schemes.modelopt_fp8 import (
                ModelOptFp8KVCacheMethod,
            )

            return ModelOptFp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            # Check if MoE layer should be excluded from quantization
            # (e.g., MTP layers that have no quantization scales in checkpoint)
            if self.is_layer_excluded(prefix):
                # Falls back to default unquantized MoE
                return None
            if Moe is None:
                return None
            return Moe(self)
        return None

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        return []

    def apply_weight_name_mapper(
        self, hf_to_sglang_mapper: "WeightsMapper"
    ):  # noqa: B027
        # Map excluded module patterns from HF layout to sglang layout.
        # Ref: HF hf_quant_config.json for nvidia/Kimi-K2.5-NVFP4
        # https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/hf_quant_config.json
        if self.exclude_modules:
            mapped = hf_to_sglang_mapper.apply_list(self.exclude_modules)
            expanded: List[str] = []
            for name in mapped:
                expanded.append(name)
                if name.startswith("language_model."):
                    expanded.append(name.removeprefix("language_model."))
            # Preserve order, drop duplicates.
            self.exclude_modules = list(dict.fromkeys(expanded))

    def is_layer_excluded(self, prefix: str) -> bool:
        """Check if a layer should be excluded from quantization.

        Handles:
        - Exact matches (e.g., "lm_head" matching prefix "lm_head")
        - Glob-style wildcards (e.g., "mtp*" matching "mtp_layers")
        - Part-by-part matching (split prefix on "." and check each part)
        - language_model. prefix stripping for vision-language models
        - Fused module patterns (e.g., "q_a_proj" in "fused_qkv_a_proj_with_mqa")
        """
        if not self.exclude_modules:
            return False

        # Build prefix variants: some models wrap layers under "language_model."
        prefixes_to_check = [prefix]
        if prefix.startswith("language_model."):
            prefixes_to_check.append(prefix.removeprefix("language_model."))

        # Fused module patterns: the exclude list may reference a sub-component
        # (e.g., "q_a_proj") that is fused into a combined parameter name
        # (e.g., "fused_qkv_a_proj_with_mqa"). We check if the last segment of
        # the exclude pattern is a substring of the last segment of the prefix.
        fused_patterns = {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"}

        for pattern in self.exclude_modules:
            # Convert glob-style wildcard to regex (e.g., "mtp*" -> "mtp.*")
            regex_str = pattern.replace(".", r"\.").replace("*", r".*")

            for pfx in prefixes_to_check:
                if re.fullmatch(regex_str, pfx):
                    return True
                # Part-by-part check: handles wildcards like "mtp*" matching
                pfx_parts = pfx.split(".")
                for part in pfx_parts:
                    if re.fullmatch(regex_str, part):
                        return True

            # Check fused patterns: if the last segment of the exclude pattern
            # is a known fused component, check if it appears in the prefix's
            # last segment (handles fused_qkv_a_proj_with_mqa containing q_a_proj)
            pattern_tail = pattern.rsplit(".", maxsplit=1)[-1]
            if pattern_tail in fused_patterns:
                for pfx in prefixes_to_check:
                    if pattern_tail in pfx.rsplit(".", maxsplit=1)[-1]:
                        return True

        return False
