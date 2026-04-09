# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py

"""Utilities for selecting and loading models."""

import concurrent.futures
import contextlib
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
import transformers
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang.srt.configs.model_config import ModelConfig, ModelImpl
from sglang.srt.layers import deep_gemm_wrapper

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _is_moe_model(model_config: ModelConfig, architectures: list[str]) -> bool:
    lowered_arches = [arch.lower() for arch in architectures]
    if any("moe" in arch or "mixtral" in arch for arch in lowered_arches):
        return True

    text_config = model_config.hf_text_config
    expert_attrs = (
        "num_local_experts",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "n_routed_experts",
    )
    for attr in expert_attrs:
        value = getattr(text_config, attr, None)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if isinstance(value, (int, float)):
            threshold = 0 if attr == "moe_intermediate_size" else 1
            if value > threshold:
                return True
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if len(value) > 0:
                return True
            continue
        if isinstance(value, str) and value == "":
            continue
        if value is not None:
            return True
    return False


def _is_sequence_classification_model(architectures: list[str]) -> bool:
    return any(
        "sequenceclassification" in lowered or "rewardmodel" in lowered
        for lowered in (arch.lower() for arch in architectures)
    )


def _get_transformers_backend_arch(
    model_config: ModelConfig, architectures: list[str]
) -> str:
    is_pooling = not model_config.is_generation
    is_multimodal = model_config.is_multimodal or (
        model_config.hf_config is not model_config.hf_text_config
    )
    is_moe = _is_moe_model(model_config, architectures)
    base_arch = "ForCausalLM"
    if is_pooling:
        base_arch = (
            "ForSequenceClassification"
            if _is_sequence_classification_model(architectures)
            else "EmbeddingModel"
        )

    arch = "Transformers"
    if is_multimodal:
        arch += "MultiModal"
    if is_moe:
        arch += "MoE"
    return arch + base_arch


def _model_impl_from_architecture(architecture: str) -> ModelImpl:
    if architecture.startswith("Transformers"):
        return ModelImpl.TRANSFORMERS
    if architecture.startswith("MindSpore"):
        return ModelImpl.MINDSPORE
    return ModelImpl.SGLANG


def resolve_transformers_arch(model_config: ModelConfig, architectures: list[str]):
    backend_arch = _get_transformers_backend_arch(model_config, architectures)

    for arch in architectures:
        if arch.startswith("Transformers"):
            continue
        auto_map: dict[str, str] = (
            getattr(model_config.hf_config, "auto_map", None) or dict()
        )
        # Make sure that config class is always initialized before model class,
        # otherwise the model class won't be able to access the config class,
        # the expected auto_map should have correct order like:
        # "auto_map": {
        #     "AutoConfig": "<your-repo-name>--<config-name>",
        #     "AutoModel": "<your-repo-name>--<config-name>",
        #     "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
        # },
        auto_modules = {}
        try:
            auto_modules = {
                name: get_class_from_dynamic_module(
                    module, model_config.model_path, revision=model_config.revision
                )
                for name, module in sorted(auto_map.items(), key=lambda x: x[0])
            }
        except Exception as e:
            logger.warning(
                "Failed to load dynamic modules from auto_map for '%s': %s. "
                "Skipping remote model compatibility checks.",
                arch,
                e,
            )
        model_module = getattr(transformers, arch, None)
        if model_module is None:
            has_auto_model = "AutoModel" in auto_modules
            if not has_auto_model and model_config.model_impl == ModelImpl.TRANSFORMERS:
                logger.warning(
                    "Cannot resolve model class for '%s' and no auto_map.AutoModel "
                    "is present. Skipping compatibility gate because "
                    "--model-impl=transformers is explicitly requested.",
                    arch,
                )
                continue
            if not has_auto_model and "AutoModel" not in auto_map:
                raise ValueError(
                    f"Cannot find model module. '{arch}' is not a registered "
                    "model in the Transformers library (only relevant if the "
                    "model is meant to be in Transformers) and 'AutoModel' is "
                    "not present in the model config's 'auto_map' (relevant "
                    "if the model is custom)."
                )
            if not has_auto_model:
                raise ValueError(
                    f"Cannot find model module. '{arch}' is not a registered "
                    "model in the Transformers library and loading the custom "
                    f"model from auto_map failed. The remote model code may be "
                    f"incompatible with the installed transformers version."
                )
            model_module = auto_modules["AutoModel"]
        if model_config.model_impl == ModelImpl.TRANSFORMERS:
            if hasattr(model_module, "is_backend_compatible") and (
                not model_module.is_backend_compatible()
            ):
                logger.warning(
                    "The Transformers implementation of %s reports it is not "
                    "backend-compatible (_supports_attention_backend=False). "
                    "Proceeding anyway because --model-impl=transformers was "
                    "explicitly requested. The model may not work correctly.",
                    arch,
                )
        if model_config.model_impl == ModelImpl.AUTO:
            if hasattr(model_module, "is_backend_compatible") and (
                not model_module.is_backend_compatible()
            ):
                raise ValueError(
                    f"{arch} has no SGlang implementation and the Transformers "
                    "implementation is not compatible with SGLang."
                )
            logger.warning(
                "%s has no SGLang implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",
                arch,
            )
    return [backend_arch]


def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    from sglang.srt.models.registry import ModelRegistry

    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = [
        "fp8",
        "compressed-tensors",
        "gptq_marlin",
        "awq_marlin",
        "quark_int4fp8_moe",
    ]

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)

    if model_config.model_impl == ModelImpl.MINDSPORE:
        architectures = ["MindSporeForCausalLM"]
    elif not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
        architectures = resolve_transformers_arch(model_config, architectures)
    model_cls, resolved_arch = ModelRegistry.resolve_model_cls(architectures)
    setattr(model_config, "_resolved_model_arch", resolved_arch)
    setattr(
        model_config,
        "_resolved_model_impl",
        _model_impl_from_architecture(resolved_arch),
    )
    return model_cls, resolved_arch


def get_resolved_model_impl(model_config: ModelConfig) -> ModelImpl:
    resolved_model_impl = getattr(model_config, "_resolved_model_impl", None)
    if resolved_model_impl is not None:
        return resolved_model_impl

    resolved_arch = getattr(model_config, "_resolved_model_arch", None)
    if resolved_arch is None:
        _, resolved_arch = get_model_architecture(model_config)

    resolved_model_impl = _model_impl_from_architecture(resolved_arch)
    setattr(model_config, "_resolved_model_arch", resolved_arch)
    setattr(model_config, "_resolved_model_impl", resolved_model_impl)
    return resolved_model_impl


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


def post_load_weights(model: nn.Module, model_config: ModelConfig):
    # Model weight loading consists of two stages:
    # 1. Initial weight loading.
    # 2. Post-processing of weights, including assigning specific member variables.
    # For `dummy_init`, only the second stage is required.
    if hasattr(model, "post_load_weights"):
        if model_config.hf_config.architectures[0] == "DeepseekV3ForCausalLMNextN":
            model.post_load_weights(is_nextn=True)
        else:
            model.post_load_weights()


def should_deepgemm_weight_requant_ue8m0(weight_block_size):
    """Should we requant fp8 weights into UE8M0 format when loading the model"""
    return (
        deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        and weight_block_size is not None
    )


def should_async_load(weight: torch.Tensor) -> bool:
    """Return True if we should load the given weight asynchronously.

    For host (CPU) tensors, using a threadpool can overlap H2D copies
    and improve throughput. For device tensors, threading often adds overhead
    (e.g., GIL contention) without benefit, so we do it synchronously.
    """
    device = getattr(weight, "device", None)
    if device is None:
        return False
    return device.type == "cpu"


def maybe_executor_submit(
    *,
    executor: concurrent.futures.ThreadPoolExecutor,
    futures: List[concurrent.futures.Future],
    use_async: bool,
    func: Callable[..., Any],
    func_args: Iterable[Any] = (),
    func_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Submit a task to the executor if async loading is enabled.

    Parameters (keyword-only):
    - executor: ThreadPoolExecutor used to submit background tasks
    - futures: a list collecting the submitted Future objects
    - use_async: whether to submit to executor or run inline
    - func: the callable to run
    - func_args: positional args for the callable (defaults to empty tuple)
    - func_kwargs: keyword args for the callable (defaults to empty dict)
    """
    if func_kwargs is None:
        func_kwargs = {}
    if use_async:
        futures.append(executor.submit(func, *func_args, **func_kwargs))
    else:
        func(*func_args, **func_kwargs)
