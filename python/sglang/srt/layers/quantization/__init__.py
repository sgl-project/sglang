# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from typing import Callable, Dict, List, Optional, Type

import torch
from vllm.model_executor.layers.quantization.awq_marlin import (
    AWQMarlinConfig,
    AWQMoEMethod,
)
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig

QUANTIZATION_METHODS: List[str] = {
    "aqlm",
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "blockwise_int8",
    "fbgemm_fp8",
    "marlin",
    "modelopt",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "qqq",
    "experts_int8",
    "w8a8_int8",
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}.")

    from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
    from vllm.model_executor.layers.quantization.awq import AWQConfig
    from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
    from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.layers.quantization.gptq import GPTQConfig
    from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
        GPTQMarlin24Config,
    )
    from vllm.model_executor.layers.quantization.marlin import MarlinConfig
    from vllm.model_executor.layers.quantization.qqq import QQQConfig
    from vllm.model_executor.layers.quantization.tpu_int8 import Int8TpuConfig

    from .blockwise_int8 import BlockInt8Config
    from .fp8 import Fp8Config
    from .modelopt_quant import ModelOptFp8Config
    from .w8a8_int8 import W8A8Int8Config

    method_to_config: Dict[str, Type[QuantizationConfig]] = {
        "aqlm": AQLMConfig,
        "awq": AWQConfig,
        "deepspeedfp": DeepSpeedFPConfig,
        "tpu_int8": Int8TpuConfig,
        "fp8": Fp8Config,
        "blockwise_int8": BlockInt8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "marlin": MarlinConfig,
        "modelopt": ModelOptFp8Config,
        "gguf": GGUFConfig,
        "gptq_marlin_24": GPTQMarlin24Config,
        "gptq_marlin": GPTQMarlinConfig,
        "awq_marlin": AWQMarlinConfig,
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "qqq": QQQConfig,
        "experts_int8": ExpertsInt8Config,
        "w8a8_int8": W8A8Int8Config,
    }

    return method_to_config[quantization]


def gptq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
        GPTQMarlinMoEMethod,
    )

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if isinstance(layer, LinearBase):
        return GPTQMarlinLinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return GPTQMarlinMoEMethod(self)
    return None


def awq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.quantization.awq import is_layer_skipped_awq
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinLinearMethod,
        AWQMoEMethod,
    )

    from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    if isinstance(layer, LinearBase) or (
        isinstance(layer, ParallelLMHead) and self.lm_head_quantized
    ):
        if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        return AWQMarlinLinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return AWQMoEMethod(self)
    return None


original_awq_moe_method_apply = AWQMoEMethod.apply


def awq_moe_method_apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    **kwargs,
):
    return original_awq_moe_method_apply(
        self,
        layer,
        x,
        router_logits,
        top_k,
        renormalize,
        use_grouped_topk,
        topk_group,
        num_expert_group,
        custom_routing_function,
        scoring_func,
        e_score_correction_bias,
    )


def apply_monkey_patches():
    """Apply all monkey patches in one place."""
    from vllm.model_executor.layers.quantization.awq_marlin import AWQMoEMethod

    setattr(GPTQMarlinConfig, "get_quant_method", gptq_get_quant_method)
    setattr(AWQMarlinConfig, "get_quant_method", awq_get_quant_method)
    setattr(AWQMoEMethod, "apply", awq_moe_method_apply)


# Apply patches when module is imported
apply_monkey_patches()


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
