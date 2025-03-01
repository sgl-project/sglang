# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
import re
from copy import deepcopy
from typing import Callable, Dict, Optional, Type, Union

import torch
from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_marlin import (
    AWQMarlinConfig,
    AWQMoEMethod,
)
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
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
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


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    return QUANTIZATION_METHODS[quantization]


# Match dynamic rules with module name (prefix) and override quantize
# config if module (prefix) matches a rule
def override_config(config: QuantizationConfig, prefix: str):
    weight_bits = get_dynamic_override(config, prefix, "bits", config.weight_bits)
    if isinstance(weight_bits, int):
        config.weight_bits = weight_bits
    group_size = get_dynamic_override(config, prefix, "group_size", config.group_size)
    if isinstance(group_size, int):
        config.group_size = group_size
    desc_act = get_dynamic_override(config, prefix, "desc_act", config.desc_act)
    if isinstance(desc_act, bool):
        config.desc_act = desc_act

    config.pack_factor = 32 // config.weight_bits  # packed into int32
    if config.get_name() == "gptq_marlin":
        is_sym = get_dynamic_override(config, prefix, "sym", config.is_sym)
        if isinstance(is_sym, bool):
            config.is_sym = is_sym

        if (config.weight_bits, config.is_sym) not in config.TYPE_MAP:
            raise ValueError(
                "Unsupported quantization config: "
                f"bits={config.weight_bits}, sym={config.is_sym}"
            )

        config.quant_type = config.TYPE_MAP[(config.weight_bits, config.is_sym)]
    elif config.get_name() == "gptq":
        if config.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {config.weight_bits} bits."
            )


def get_dynamic_override(
    config: QuantizationConfig,
    layer_name: str,
    key: Optional[str] = None,
    default_value: Union[int, bool, None] = None,
) -> Union[Dict, int, bool, None]:
    for pattern, pattern_dict in config.dynamic.items():
        # Negative match: matched modules are excluded from quantized init
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), layer_name):
                return False
        # Positive match: matched modules have quant properties overrides
        # base quant config
        elif re.match(pattern.removeprefix("+:"), layer_name):
            if key is None:
                return pattern_dict
            else:
                return pattern_dict.get(key, default_value)
    return default_value


def get_linear_quant_method(
    config: QuantizationConfig,
    layer: torch.nn.Module,
    prefix: str,
    linear_method_cls: type,
):

    from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )

    cloned_config = deepcopy(config)
    parallel_lm_head_quantized = (
        isinstance(layer, ParallelLMHead) and cloned_config.lm_head_quantized
    )

    if isinstance(layer, LinearBase) or parallel_lm_head_quantized:
        # False = skip module, None = no override, else = Positive match
        if (
            get_dynamic_override(  # noqa: E712
                cloned_config, layer_name=prefix  # noqa: E712
            )
            == False
        ):  # noqa: E712
            if parallel_lm_head_quantized:
                return UnquantizedEmbeddingMethod()
            return UnquantizedLinearMethod()

        if prefix:
            # Dynamic per module/layer rules may override base config
            override_config(cloned_config, prefix=prefix)

        return linear_method_cls(cloned_config)
    return None


def gptq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
        GPTQMarlinMoEMethod,
    )

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


def patch_vllm_linear_base_isinstance():
    import builtins

    from vllm.model_executor.layers.linear import LinearBase

    from sglang.srt.layers.linear import LinearBase as PatchedLinearBase

    original_isinstance = builtins.isinstance

    def patched_isinstance(obj, classinfo):
        if classinfo is LinearBase:
            return original_isinstance(obj, PatchedLinearBase)
        return original_isinstance(obj, classinfo)

    builtins.isinstance = patched_isinstance


def apply_monkey_patches():
    """Apply all monkey patches in one place."""
    from vllm.model_executor.layers.quantization.awq_marlin import AWQMoEMethod

    setattr(GPTQMarlinConfig, "get_quant_method", gptq_get_quant_method)
    setattr(GPTQConfig, "get_quant_method", gptq_get_quant_method)
    setattr(AWQMarlinConfig, "get_quant_method", awq_get_quant_method)
    setattr(AWQMoEMethod, "apply", awq_moe_method_apply)


patch_vllm_linear_base_isinstance()
# Apply patches when module is imported
apply_monkey_patches()


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
