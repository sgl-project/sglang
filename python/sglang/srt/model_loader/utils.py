# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py

"""Utilities for selecting and loading models."""
import contextlib
from abc import ABC
from typing import Iterable, Iterator, Tuple, Type, runtime_checkable

import torch
from torch import nn

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.utils import get_layer_id


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


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

    return ModelRegistry.resolve_model_cls(architectures)


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@runtime_checkable
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
        raise AttributeError("No end_layer implementation found")

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
