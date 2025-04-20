# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
import builtins
import inspect
import re
from copy import deepcopy
from typing import Callable, Dict, Optional, Type, Union

import torch

try:
    from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinConfig,
        AWQMoEMethod,
    )
    from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsW8A8Fp8MoEMethod,
        CompressedTensorsWNA16MoEMethod,
    )
    from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
    from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
        GPTQMarlinMoEMethod,
    )
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

    AQLMConfig = AWQMarlinConfig = BitsAndBytesConfig = CompressedTensorsConfig = (
        DeepSpeedFPConfig
    ) = ExpertsInt8Config = FBGEMMFp8Config = GGUFConfig = GPTQMarlin24Config = (
        MarlinConfig
    ) = QQQConfig = Int8TpuConfig = DummyConfig


from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
from sglang.srt.layers.quantization.awq import AWQConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
)
from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

# Base quantization methods that don't depend on vllm
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "moe_wna16": MoeWNA16Config,
    "compressed-tensors": CompressedTensorsConfig,
}

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
            "Pleaes install vllm by `pip install vllm==0.7.2`"
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
    # Move import here to avoid circular import. This is only used in monkey patching
    # of vllm's QuantizationConfig.
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
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
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
            "router_logits": router_logits,
            "top_k": top_k,
            "renormalize": renormalize,
            "use_grouped_topk": use_grouped_topk,
            "topk_group": topk_group,
            "num_expert_group": num_expert_group,
            "custom_routing_function": custom_routing_function,
        }
        if correction_bias is not None:
            if not has_correction_bias:
                raise ValueError(
                    "Please increase the version of your vllm. Try `pip install vllm==0.7.2`"
                )
            kwargs["e_score_correction_bias"] = correction_bias
        return original_apply(**kwargs)

    setattr(class_obj, "apply", new_apply)


def monkey_patch_quant_configs():
    """Apply all monkey patches in one place."""
    setattr(GPTQMarlinConfig, "get_quant_method", gptq_get_quant_method)
    setattr(GPTQConfig, "get_quant_method", gptq_get_quant_method)

    monkey_patch_moe_apply(AWQMoEMethod)
    monkey_patch_moe_apply(GPTQMarlinMoEMethod)
    monkey_patch_moe_apply(CompressedTensorsW8A8Fp8MoEMethod)
    monkey_patch_moe_apply(CompressedTensorsWNA16MoEMethod)


# Only apply monkey patches if vllm is available
if VLLM_AVAILABLE:
    monkey_patch_quant_configs()
