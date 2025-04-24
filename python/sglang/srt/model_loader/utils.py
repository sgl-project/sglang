# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py

"""Utilities for selecting and loading models."""
import contextlib
from typing import Tuple, Type

import torch
from torch import nn

from sglang.srt.configs.model_config import ModelConfig


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


def check_gguf_version(version: str) -> None:
    """Checks if the installed gguf version is sufficient."""
    MIN_GGUF_VERSION = "0.10.0"

    installed_version_str = getattr(version, "__version__")

    if installed_version_str < MIN_GGUF_VERSION:
        raise ImportError(
            f"Installed gguf version {installed_version_str} is insufficient. "
            f"SGLang requires gguf>={MIN_GGUF_VERSION}. "
            f"Please upgrade gguf via `pip install --upgrade gguf`."
            "You may see the following link to be useful: https://github.com/huggingface/transformers/blob/b6d65e40b256d98d9621707762b94bc8ad83b7a7/src/transformers/modeling_gguf_pytorch_utils.py#L286"
        )
