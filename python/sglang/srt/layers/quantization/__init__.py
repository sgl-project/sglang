# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import builtins
import inspect
from importlib import import_module
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch


# Define empty classes as placeholders when vllm is not available
class DummyConfig:
    def override_quantization_method(self, *args, **kwargs):
        return None


CompressedTensorsConfig = DummyConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_mps,
    is_npu,
    mxfp_supported,
)

_is_mxfp_supported = mxfp_supported()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


def _try_import_quant_configs(
    module_name: str,
    names: tuple[str, ...],
) -> Dict[str, Type[QuantizationConfig]]:
    try:
        module = import_module(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
        return {}
    return {name: getattr(module, name) for name in names if hasattr(module, name)}


_QUANT_CONFIGS: Dict[str, Type[QuantizationConfig]] = {}
for _module_name, _names in [
    ("sglang.srt.layers.quantization.auto_round", ("AutoRoundConfig",)),
    (
        "sglang.srt.layers.quantization.awq",
        ("AWQConfig", "AWQCPUConfig", "AWQMarlinConfig"),
    ),
    ("sglang.srt.layers.quantization.bitsandbytes", ("BitsAndBytesConfig",)),
    ("sglang.srt.layers.quantization.blockwise_int8", ("BlockInt8Config",)),
    (
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors",
        ("CompressedTensorsConfig",),
    ),
    ("sglang.srt.layers.quantization.fp8", ("Fp8Config",)),
    ("sglang.srt.layers.quantization.fpgemm_fp8", ("FBGEMMFp8Config",)),
    ("sglang.srt.layers.quantization.gguf", ("GGUFConfig",)),
    (
        "sglang.srt.layers.quantization.gptq",
        ("CPUGPTQConfig", "GPTQAscendConfig", "GPTQConfig", "GPTQMarlinConfig"),
    ),
    ("sglang.srt.layers.quantization.mlx", ("MlxQuantizationConfig",)),
    (
        "sglang.srt.layers.quantization.modelopt_quant",
        ("ModelOptFp4Config", "ModelOptFp8Config", "ModelOptMixedPrecisionConfig"),
    ),
    ("sglang.srt.layers.quantization.modelslim.modelslim", ("ModelSlimConfig",)),
    ("sglang.srt.layers.quantization.moe_wna16", ("MoeWNA16Config",)),
    ("sglang.srt.layers.quantization.mxfp4", ("Mxfp4Config",)),
    ("sglang.srt.layers.quantization.nvfp4_online", ("NvFp4OnlineConfig",)),
    ("sglang.srt.layers.quantization.petit", ("PetitNvFp4Config",)),
    ("sglang.srt.layers.quantization.qoq", ("QoQConfig",)),
    ("sglang.srt.layers.quantization.quark.quark", ("QuarkConfig",)),
    ("sglang.srt.layers.quantization.quark_int4fp8_moe", ("QuarkInt4Fp8Config",)),
    ("sglang.srt.layers.quantization.w4afp8", ("W4AFp8Config",)),
    ("sglang.srt.layers.quantization.w8a8_fp8", ("W8A8Fp8Config",)),
    ("sglang.srt.layers.quantization.w8a8_int8", ("W8A8Int8Config",)),
]:
    _QUANT_CONFIGS.update(_try_import_quant_configs(_module_name, _names))

CompressedTensorsConfig = _QUANT_CONFIGS.get("CompressedTensorsConfig", DummyConfig)

# Base quantization methods
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {}


def _register_quant_method(method: str, config_name: str) -> None:
    config = _QUANT_CONFIGS.get(config_name)
    if config is not None:
        BASE_QUANTIZATION_METHODS[method] = config


for _method, _config_name in [
    ("fp8", "Fp8Config"),
    ("mxfp8", "Fp8Config"),
    ("blockwise_int8", "BlockInt8Config"),
    ("modelopt", "ModelOptFp8Config"),  # Auto-detect, defaults to FP8
    ("modelopt_fp8", "ModelOptFp8Config"),
    ("modelopt_fp4", "ModelOptFp4Config"),
    ("nvfp4_online", "NvFp4OnlineConfig"),
    ("modelopt_mixed", "ModelOptMixedPrecisionConfig"),
    ("w8a8_int8", "W8A8Int8Config"),
    ("w8a8_fp8", "W8A8Fp8Config"),
    ("awq", "AWQConfig"),
    ("awq_marlin", "AWQMarlinConfig"),
    ("bitsandbytes", "BitsAndBytesConfig"),
    ("gguf", "GGUFConfig"),
    ("gptq", "GPTQConfig"),
    ("gptq_marlin", "GPTQMarlinConfig"),
    ("moe_wna16", "MoeWNA16Config"),
    ("compressed-tensors", "CompressedTensorsConfig"),
    ("qoq", "QoQConfig"),
    ("w4afp8", "W4AFp8Config"),
    ("petit_nvfp4", "PetitNvFp4Config"),
    ("fbgemm_fp8", "FBGEMMFp8Config"),
    ("quark", "QuarkConfig"),
    ("quark_mxfp4", "QuarkConfig"),
    ("auto-round", "AutoRoundConfig"),
    ("auto-round-int8", "W8A8Int8Config"),
    ("modelslim", "ModelSlimConfig"),
    ("quark_int4fp8_moe", "QuarkInt4Fp8Config"),
]:
    _register_quant_method(_method, _config_name)


if is_cpu() or is_cuda() or (_is_mxfp_supported and is_hip()):
    _register_quant_method("mxfp4", "Mxfp4Config")


if is_npu():
    _register_quant_method("gptq", "GPTQAscendConfig")


if is_mps():
    _register_quant_method("mlx_q4", "MlxQuantizationConfig")
    _register_quant_method("mlx_q8", "MlxQuantizationConfig")

# subset of above quant methods, supported on CPU
CPU_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {}


def _register_cpu_quant_method(method: str, config_name: str) -> None:
    config = _QUANT_CONFIGS.get(config_name)
    if config is not None:
        CPU_QUANTIZATION_METHODS[method] = config


for _method, _config_name in [
    ("fp8", "Fp8Config"),
    ("w8a8_int8", "W8A8Int8Config"),
    ("compressed-tensors", "CompressedTensorsConfig"),
    ("awq", "AWQCPUConfig"),
    ("gptq", "CPUGPTQConfig"),
    ("mxfp4", "Mxfp4Config"),
]:
    _register_cpu_quant_method(_method, _config_name)

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    from sglang.srt.utils import is_cpu

    if is_cpu() and cpu_has_amx_support():
        if quantization not in CPU_QUANTIZATION_METHODS:
            raise ValueError(
                f"Invalid quantization method on CPU: {quantization}. "
                f"Available methods on CPU: {list(QUANTIZATION_METHODS.keys())}"
            )
        else:
            return CPU_QUANTIZATION_METHODS[quantization]

    if current_platform.is_out_of_tree():
        config = current_platform.get_quantization_config(quantization)

        # If the platform has a quantization config, use it else use the default
        if config is not None:
            return config

    return QUANTIZATION_METHODS[quantization]


original_isinstance = builtins.isinstance
