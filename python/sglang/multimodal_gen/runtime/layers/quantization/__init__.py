# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import importlib
from typing import Literal, get_args

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)

QuantizationMethods = Literal[
    "fp8",
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "bitsandbytes",
    "modelslim",
    "mxfp8",
    "mxfp4",
    "mxfp4_npu",
]

QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

# Load only the selected quantizer. Eagerly importing every backend makes an
# unquantized diffusion model depend on unrelated SRT/LLM kernels and their
# Transformers version.
_BUILTIN_METHOD_TO_QUANT_CONFIG = {
    "modelopt": (
        "sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8",
        "ModelOptFp8Config",
    ),
    "modelopt_fp8": (
        "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant",
        "ModelOptFp8Config",
    ),
    "modelopt_fp4": (
        "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant",
        "ModelOptFp4Config",
    ),
    "bitsandbytes": (
        "sglang.multimodal_gen.runtime.layers.quantization.bitsandbytes",
        "BitsAndBytesConfig",
    ),
    "modelslim": (
        "sglang.multimodal_gen.runtime.layers.quantization.modelslim",
        "ModelSlimConfig",
    ),
    "fp8": (
        "sglang.multimodal_gen.runtime.layers.quantization.fp8",
        "Fp8Config",
    ),
    "mxfp4": (
        "sglang.multimodal_gen.runtime.layers.quantization.mxfp4",
        "Mxfp4Config",
    ),
    "mxfp8": (
        "sglang.multimodal_gen.runtime.layers.quantization.mxfp8_npu",
        "MXFP8Config",
    ),
    "mxfp4_npu": (
        "sglang.multimodal_gen.runtime.layers.quantization.mxfp4_npu",
        "NPUMXFP4Config",
    ),
}
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG: dict[str, type[QuantizationConfig]] = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.


    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            raise ValueError(
                f"The quantization method `{quantization}` is already exists."
            )
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of " "`QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]

    module_name, class_name = _BUILTIN_METHOD_TO_QUANT_CONFIG[quantization]
    return getattr(importlib.import_module(module_name), class_name)


__all__ = [
    "QuantizationMethods",
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
