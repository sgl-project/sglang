# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py

from typing import Callable, Dict, Optional, Type

import torch
from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.gptq_marlin_24 import GPTQMarlin24Config
from vllm.model_executor.layers.quantization.marlin import MarlinConfig
from vllm.model_executor.layers.quantization.qqq import QQQConfig
from vllm.model_executor.layers.quantization.tpu_int8 import Int8TpuConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "tpu_int8": Int8TpuConfig,
    "fp8": Fp8Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    "marlin": MarlinConfig,
    "gguf": GGUFConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "gptq_marlin": GPTQMarlinConfig,
    "awq_marlin": AWQMarlinConfig,
    "gptq": GPTQConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "qqq": QQQConfig,
    "experts_int8": ExpertsInt8Config,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    return QUANTIZATION_METHODS[quantization]


def fp8_get_quant_method(self, layer, prefix):
    """Enhanced get_quant_method for FP8 config."""
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    from sglang.srt.layers.linear import UnquantizedLinearMethod
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

    if isinstance(layer, LinearBase):
        if is_layer_skipped(prefix, self.ignored_layers):
            return UnquantizedLinearMethod()
        return Fp8LinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return Fp8MoEMethod(self)
    return None


def gptq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
        GPTQMarlinMoEMethod,
    )

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if isinstance(layer, LinearBase):
        return GPTQMarlinLinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return GPTQMarlinMoEMethod(self)
    return None


def awq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinLinearMethod,
        AWQMoEMethod,
    )

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if isinstance(layer, LinearBase):
        return AWQMarlinLinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return AWQMoEMethod(self)
    return None


def apply_monkey_patches():
    """Apply all monkey patches in one place."""
    setattr(Fp8Config, "get_quant_method", fp8_get_quant_method)
    setattr(GPTQMarlinConfig, "get_quant_method", gptq_get_quant_method)
    setattr(AWQMarlinConfig, "get_quant_method", awq_get_quant_method)


# Apply patches when module is imported
apply_monkey_patches()


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
