# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import Literal, get_args

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.method_names import (
    canonical_quantization_method,
)

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


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    quantization = canonical_quantization_method(quantization)
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # Quantized linear methods import LinearBase; load them after module initialization.
    from sglang.multimodal_gen.runtime.layers.quantization.auto_round import (
        AutoRoundConfig,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.bitsandbytes import (
        BitsAndBytesConfig,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
        ConvRotInt8Config,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
    from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8 import (
        ModelOptFp8Config as ModelOptFp8DiffusionConfig,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
        ModelOptFp8Config,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.modelslim import (
        ModelSlimConfig,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp4 import Mxfp4Config
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp4_npu import (
        NPUMXFP4Config,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config

    method_to_config: dict[str, type[QuantizationConfig]] = {
        "auto-round": AutoRoundConfig,
        "modelopt": ModelOptFp8DiffusionConfig,
        "modelopt_fp8": ModelOptFp8Config,
        "modelopt_fp4": ModelOptFp4Config,
        "bitsandbytes": BitsAndBytesConfig,
        "modelslim": ModelSlimConfig,
        "fp8": Fp8Config,
        "mxfp4": Mxfp4Config,
        "mxfp8": MXFP8Config,
        "mxfp4_npu": NPUMXFP4Config,
        "convrot_int8": ConvRotInt8Config,
    }

    return method_to_config[quantization]


__all__ = [
    "QuantizationMethods",
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
