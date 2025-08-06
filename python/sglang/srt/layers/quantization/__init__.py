# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import builtins
import inspect
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union

import torch

try:
    from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
    from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsW8A8Fp8MoEMethod,
        CompressedTensorsWNA16MoEMethod,
    )
    from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
    from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
        GPTQMarlin24Config,
    )
    from vllm.model_executor.layers.quantization.marlin import MarlinConfig
    from vllm.model_executor.layers.quantization.qqq import QQQConfig
    from vllm.model_executor.layers.quantization.tpu_int8 import Int8TpuConfig

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    # Define empty classes as placeholders when vllm is not available
    class DummyConfig:
        def override_quantization_method(self, *args, **kwargs):
            return None

    AQLMConfig = BitsAndBytesConfig = CompressedTensorsConfig = DeepSpeedFPConfig = (
        ExpertsInt8Config
    ) = FBGEMMFp8Config = GGUFConfig = GPTQMarlin24Config = MarlinConfig = QQQConfig = (
        Int8TpuConfig
    ) = DummyConfig


from sglang.srt.layers.quantization.awq import AWQConfig, AWQMarlinConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.utils import is_cuda, is_hip, mxfp_supported

is_mxfp_supported = mxfp_supported()
if is_mxfp_supported:
    from sglang.srt.layers.quantization.fp4 import MxFp4Config

from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.gptq import (
    GPTQConfig,
    GPTQLinearMethod,
    GPTQMarlinConfig,
    GPTQMarlinLinearMethod,
    GPTQMarlinMoEMethod,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
)
from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.layers.quantization.petit import PetitNvFp4Config
from sglang.srt.layers.quantization.qoq import QoQConfig
from sglang.srt.layers.quantization.utils import get_linear_quant_method
from sglang.srt.layers.quantization.w4a8_machete import W4A8MacheteConfig
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

# Base quantization methods that don't depend on vllm
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "w4a8_machete": W4A8MacheteConfig,
    "moe_wna16": MoeWNA16Config,
    "compressed-tensors": CompressedTensorsConfig,
    "qoq": QoQConfig,
    "w4afp8": W4AFp8Config,
    "petit_nvfp4": PetitNvFp4Config,
}


if is_cuda():
    BASE_QUANTIZATION_METHODS.update(
        {
            "quark": Mxfp4Config,
            "mxfp4": Mxfp4Config,
        }
    )
elif is_mxfp_supported and is_hip():
    BASE_QUANTIZATION_METHODS.update(
        {
            "quark": MxFp4Config,
            "mxfp4": MxFp4Config,
        }
    )
# VLLM-dependent quantization methods
VLLM_QUANTIZATION_METHODS = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "tpu_int8": Int8TpuConfig,
    "fbgemm_fp8": FBGEMMFp8Config,
    "marlin": MarlinConfig,
    "gguf": GGUFConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "qqq": QQQConfig,
    "experts_int8": ExpertsInt8Config,
    "gptq_marlin": GPTQMarlinConfig,
    "gptq": GPTQConfig,
}

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS, **VLLM_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    if quantization in VLLM_QUANTIZATION_METHODS and not VLLM_AVAILABLE:
        raise ValueError(
            f"{quantization} quantization requires some operators from vllm. "
            "Please install vllm by `pip install vllm==0.9.0.1`"
        )

    return QUANTIZATION_METHODS[quantization]


def gptq_get_quant_method(self, layer, prefix):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if isinstance(layer, FusedMoE):
        return GPTQMarlinMoEMethod(self)

    if isinstance(self, GPTQConfig):
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )
    elif isinstance(self, GPTQMarlinConfig):
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQMarlinLinearMethod
        )
    return None


original_isinstance = builtins.isinstance


def monkey_patch_isinstance_for_vllm_base_layer(reverse: bool = False):
    """
    Patch isinstance so that the `get_quant_method` in vllm's QuantizationConfig
    can recognize sglang layers
    """
    if not VLLM_AVAILABLE:
        return

    if reverse:
        builtins.isinstance = original_isinstance
        return

    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from sglang.srt.layers.linear import LinearBase as PatchedLinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE as PatchedFusedMoE
    from sglang.srt.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding as PatchedVocabParallelEmbedding,
    )

    def patched_isinstance(obj, classinfo):
        if classinfo is LinearBase:
            return original_isinstance(obj, PatchedLinearBase)
        if classinfo is FusedMoE:
            return original_isinstance(obj, PatchedFusedMoE)
        if classinfo is VocabParallelEmbedding:
            return original_isinstance(obj, PatchedVocabParallelEmbedding)
        return original_isinstance(obj, classinfo)

    builtins.isinstance = patched_isinstance


def monkey_patch_moe_apply(class_obj: "FusedMoEMethodBase"):
    """
    Monkey patch the apply function of vllm's FusedMoEMethodBase.
    Convert sglang arguments to vllm arguments.
    """
    original_apply = class_obj.apply
    sig = inspect.signature(original_apply)
    param_names = list(sig.parameters.keys())
    has_correction_bias = "e_score_correction_bias" in param_names

    def new_apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output: TopKOutput,
        *,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ):
        assert activation == "silu"
        assert inplace and not no_combine

        kwargs = {
            "self": self,
            "layer": layer,
            "x": x,
            "topk_output": topk_output,
        }
        return original_apply(**kwargs)

    setattr(class_obj, "apply", new_apply)


def monkey_patch_quant_configs():
    """Apply all monkey patches in one place."""
    setattr(GPTQMarlinConfig, "get_quant_method", gptq_get_quant_method)
    setattr(GPTQConfig, "get_quant_method", gptq_get_quant_method)

    monkey_patch_moe_apply(GPTQMarlinMoEMethod)
    monkey_patch_moe_apply(CompressedTensorsW8A8Fp8MoEMethod)
    monkey_patch_moe_apply(CompressedTensorsWNA16MoEMethod)


# Only apply monkey patches if vllm is available
if VLLM_AVAILABLE:
    monkey_patch_quant_configs()
