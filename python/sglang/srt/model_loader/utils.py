# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py

"""Utilities for selecting and loading models."""
import contextlib
import logging
from abc import ABC
from typing import Iterable, Iterator, Tuple, Type

import torch
import transformers
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang.srt.configs.model_config import ModelConfig, ModelImpl
from sglang.srt.layers.utils import get_layer_id

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def resolve_transformers_arch(model_config: ModelConfig, architectures: list[str]):
    for i, arch in enumerate(architectures):
        if arch == "TransformersForCausalLM":
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
        auto_modules = {
            name: get_class_from_dynamic_module(
                module, model_config.model_path, revision=model_config.revision
            )
            for name, module in sorted(auto_map.items(), key=lambda x: x[0])
        }
        model_module = getattr(transformers, arch, None)
        if model_module is None:
            if "AutoModel" not in auto_map:
                raise ValueError(
                    f"Cannot find model module. '{arch}' is not a registered "
                    "model in the Transformers library (only relevant if the "
                    "model is meant to be in Transformers) and 'AutoModel' is "
                    "not present in the model config's 'auto_map' (relevant "
                    "if the model is custom)."
                )
            model_module = auto_modules["AutoModel"]
        if model_config.impl == ModelImpl.TRANSFORMERS:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"The Transformers implementation of {arch} is not "
                    "compatible with vLLM."
                )
            architectures[i] = "TransformersForCausalLM"
        if model_config.impl == ModelImpl.AUTO:
            if not model_module.is_backend_compatible():
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
            architectures[i] = "TransformersForCausalLM"
    return architectures


def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    from sglang.srt.models.registry import ModelRegistry

    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = ["fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"]

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)

    if not is_native_supported or model_config.impl == ModelImpl.TRANSFORMERS:
        architectures = resolve_transformers_arch(model_config, architectures)

    return ModelRegistry.resolve_model_cls(architectures)


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


class SupportsPP(ABC):
    """Base class for PipelineParallel models with lazy weight filtering"""

    @property
    def start_layer(self) -> int:
        if hasattr(self, "model") and hasattr(self.model, "start_layer"):
            return self.model.start_layer
        return 0

    @property
    def end_layer(self) -> int:
        if hasattr(self, "model") and hasattr(self.model, "end_layer"):
            return self.model.end_layer
        if hasattr(self, "config") and hasattr(self.config, "num_hidden_layers"):
            return self.config.num_hidden_layers
        raise AttributeError(
            "No end_layer implementation found and config.num_hidden_layers not available"
        )

    @property
    def num_effective_layers(self) -> int:
        self.num_effective_layers = self.end_layer - self.start_layer

    def should_skip_layer(self, name: str) -> bool:
        """Check if a layer should be skipped based on pipeline stage"""
        layer_id = get_layer_id(name)
        return layer_id is not None and (
            layer_id < self.start_layer or layer_id >= self.end_layer
        )

    def filter_weights_by_layers(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Lazily filter weights based on pipeline stage"""
        return filter(lambda item: not self.should_skip_layer(item[0]), weights)
