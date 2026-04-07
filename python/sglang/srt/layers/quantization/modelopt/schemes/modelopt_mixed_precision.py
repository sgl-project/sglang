# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.modelopt.modelopt import ModelOptQuantConfig
from sglang.srt.layers.quantization.modelopt.schemes.modelopt_fp4 import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.modelopt.schemes.modelopt_fp8 import (
    ModelOptFp8Config,
    ModelOptFp8KVCacheMethod,
    ModelOptFp8LinearMethod,
    ModelOptFp8MoEMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_attention import RadixAttention

if TYPE_CHECKING:
    from sglang.srt.models.utils import WeightsMapper


class ModelOptMixedPrecisionConfig(ModelOptQuantConfig):
    """Configuration for ModelOpt MIXED_PRECISION checkpoints."""

    def __init__(
        self,
        kv_cache_quant_algo: Optional[str],
        exclude_modules: Optional[List[str]],
        packed_modules_mapping: Optional[Dict[str, List[str]]],
        quantized_layers: Dict[str, Dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: "ModelOptFp4Config",
    ) -> None:
        super().__init__(kv_cache_quant_algo, exclude_modules, packed_modules_mapping)
        self.quantized_layers = quantized_layers
        self.fp8_config = fp8_config
        self.nvfp4_config = nvfp4_config

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant):
        if hf_quant_config is None:
            return None
        if hf_quant_config.get("quant_method", "") == "modelopt_mixed":
            return "modelopt_mixed"
        return None

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_mixed"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return ModelOptFp4Config.get_min_capability()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptMixedPrecisionConfig":
        kv_cache_quant_algo = None
        exclude_modules = None
        quantized_layers = {}

        quant_algo = config.get("quant_algo")
        if quant_algo is not None:
            kv_cache_scheme = config.get("kv_cache_scheme")
            if isinstance(kv_cache_scheme, dict):
                if (
                    kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 8
                ):
                    kv_cache_quant_algo = "FP8"
                elif (
                    kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 4
                ):
                    kv_cache_quant_algo = "NVFP4"
                else:
                    kv_cache_quant_algo = "auto"
            exclude_modules = config.get("ignore")
            quantized_layers = config.get("quantized_layers", {})
        else:
            quantization_section = cls.get_from_keys(config, ["quantization"])
            quant_algo = quantization_section.get("quant_algo")
            kv_cache_quant_algo = quantization_section.get("kv_cache_quant_algo")
            exclude_modules = quantization_section.get("exclude_modules")
            quantized_layers = quantization_section.get("quantized_layers", {})

        if quant_algo != "MIXED_PRECISION":
            raise ValueError(
                "ModelOptMixedPrecisionConfig only supports MIXED_PRECISION checkpoints."
            )
        if not quantized_layers:
            raise ValueError(
                "MIXED_PRECISION quantization requires a non-empty quantized_layers map."
            )

        group_size = None
        for layer_info in quantized_layers.values():
            if layer_info.get("quant_algo", "").upper() == "NVFP4":
                group_size = layer_info.get("group_size", 16)
                break
        if group_size is None:
            group_size = 16

        packed_modules_mapping = config.get("packed_modules_mapping")
        fp8_config = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_algo,
            exclude_modules=[],
            packed_modules_mapping=packed_modules_mapping,
        )
        nvfp4_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_algo,
            exclude_modules=[],
            packed_modules_mapping=packed_modules_mapping,
            group_size=group_size,
        )

        return cls(
            kv_cache_quant_algo=kv_cache_quant_algo,
            exclude_modules=exclude_modules,
            packed_modules_mapping=packed_modules_mapping,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
        )

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: "WeightsMapper"):
        super().apply_weight_name_mapper(hf_to_sglang_mapper)
        if self.quantized_layers:
            self.quantized_layers = hf_to_sglang_mapper.apply_dict(
                self.quantized_layers
            )

    def _resolve_quant_algo(self, prefix: str) -> Optional[str]:
        if prefix in self.quantized_layers:
            return self.quantized_layers[prefix]["quant_algo"].upper()

        proj_name = prefix.rsplit(".", 1)[-1]
        if self.packed_modules_mapping and proj_name in self.packed_modules_mapping:
            algos = set()
            base = prefix.rsplit(".", 1)[0]
            for shard_name in self.packed_modules_mapping[proj_name]:
                shard_prefix = f"{base}.{shard_name}"
                if shard_prefix in self.quantized_layers:
                    algos.add(self.quantized_layers[shard_prefix]["quant_algo"].upper())
            if len(algos) == 1:
                return algos.pop()
            if len(algos) > 1:
                raise ValueError(
                    f"Mixed quant_algo within fused layer {prefix}: {algos}. "
                    "All shards must use the same quantization."
                )

        prefix_dot = prefix + "."
        for key, info in self.quantized_layers.items():
            if key.startswith(prefix_dot):
                return info["quant_algo"].upper()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        quant_algo = self._resolve_quant_algo(prefix)

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.exclude_modules, self.packed_modules_mapping
            ) or self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptFp4LinearMethod(self.nvfp4_config)
            return UnquantizedLinearMethod()

        if self.kv_cache_quant_algo and isinstance(layer, RadixAttention):
            return ModelOptFp8KVCacheMethod(self.fp8_config)

        if isinstance(layer, FusedMoE):
            if self.is_layer_excluded(prefix):
                return None
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoEMethod(self.nvfp4_config)
            return None

        return None
