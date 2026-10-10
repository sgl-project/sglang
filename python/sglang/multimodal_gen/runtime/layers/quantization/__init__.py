# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from importlib import import_module
from typing import Literal, get_args

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.method_names import (
    canonical_quantization_method,
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
    # deprecated alias of convrot_int8
    "kitchen_int8",
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
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    quantization = canonical_quantization_method(quantization)
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]
    name = _BUILTIN_CONFIG_NAMES[quantization]
    if name in globals():
        return globals()[name]
    return __getattr__(name)


__all__ = [
    "QuantizationMethods",
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
