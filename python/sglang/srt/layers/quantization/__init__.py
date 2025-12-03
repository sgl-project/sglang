# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import builtins
import inspect
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch


# Define empty classes as placeholders when vllm is not available
class DummyConfig:
    def override_quantization_method(self, *args, **kwargs):
        return None


CompressedTensorsConfig = DummyConfig

from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.layers.quantization.awq import AWQConfig, AWQMarlinConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8Config
from sglang.srt.layers.quantization.gguf import GGUFConfig
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
)
from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.layers.quantization.petit import PetitNvFp4Config
from sglang.srt.layers.quantization.qoq import QoQConfig
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.utils import is_cuda, is_hip, mxfp_supported

_is_mxfp_supported = mxfp_supported()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

# Base quantization methods
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,  # Auto-detect, defaults to FP8
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "awq": AWQConfig,
    "awq_marlin": AWQMarlinConfig,
    "gguf": GGUFConfig,
    "gptq": GPTQConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "moe_wna16": MoeWNA16Config,
    "compressed-tensors": CompressedTensorsConfig,
    "qoq": QoQConfig,
    "w4afp8": W4AFp8Config,
    "petit_nvfp4": PetitNvFp4Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    "quark": QuarkConfig,
    "auto-round": AutoRoundConfig,
}


if is_cuda() or (_is_mxfp_supported and is_hip()):
    BASE_QUANTIZATION_METHODS.update(
        {
            "mxfp4": Mxfp4Config,
        }
    )

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )

    return QUANTIZATION_METHODS[quantization]


original_isinstance = builtins.isinstance
