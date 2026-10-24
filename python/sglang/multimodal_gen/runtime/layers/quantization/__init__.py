# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from importlib import import_module
from typing import Literal, get_args

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)

# Importing linear loads this package before LinearBase has been defined.
# Resolve implementations only when requested, after that import can finish.
_CONFIG_IMPORTS = {
    "AutoRoundConfig": ("auto_round", "AutoRoundConfig"),
    "BitsAndBytesConfig": ("bitsandbytes", "BitsAndBytesConfig"),
    "ConvRotInt8Config": ("configs.convrot_int8_config", "ConvRotInt8Config"),
    "Fp8Config": ("fp8", "Fp8Config"),
    "ModelOptFp8DiffusionConfig": ("modelopt_fp8", "ModelOptFp8Config"),
    "ModelOptFp4Config": ("modelopt_quant", "ModelOptFp4Config"),
    "ModelOptFp8Config": ("modelopt_quant", "ModelOptFp8Config"),
    "ModelSlimConfig": ("modelslim", "ModelSlimConfig"),
    "Mxfp4Config": ("mxfp4", "Mxfp4Config"),
    "NPUMXFP4Config": ("mxfp4_npu", "NPUMXFP4Config"),
    "MXFP8Config": ("mxfp8", "MXFP8Config"),
}


def __getattr__(name: str):
    # Preserve direct imports of configuration classes from this package.
    if name not in _CONFIG_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, class_name = _CONFIG_IMPORTS[name]
    config = getattr(import_module(f"{__name__}.{module_name}"), class_name)
    globals()[name] = config
    return config


QuantizationMethods = Literal[
    "auto-round",
    "fp8",
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "bitsandbytes",
    "modelslim",
    "mxfp8",
    "mxfp4",
    "mxfp4_npu",
    "convrot_int8",
]

QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

_BUILTIN_CONFIG_NAMES = {
    "auto-round": "AutoRoundConfig",
    "bitsandbytes": "BitsAndBytesConfig",
    "modelopt": "ModelOptFp8DiffusionConfig",
    "modelopt_fp8": "ModelOptFp8Config",
    "modelopt_fp4": "ModelOptFp4Config",
    "modelslim": "ModelSlimConfig",
    "fp8": "Fp8Config",
    "mxfp4": "Mxfp4Config",
    "mxfp8": "MXFP8Config",
    "mxfp4_npu": "NPUMXFP4Config",
    "convrot_int8": "ConvRotInt8Config",
}


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    name = _BUILTIN_CONFIG_NAMES[quantization]
    if name in globals():
        return globals()[name]
    return __getattr__(name)


__all__ = [
    "QUANTIZATION_METHODS",
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
]
