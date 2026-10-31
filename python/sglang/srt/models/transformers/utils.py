# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team


import logging
from contextlib import contextmanager

import torch
import transformers
from torch import nn
from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

logger = logging.getLogger(__name__)


def can_enable_torch_compile(config: PretrainedConfig) -> bool:
    """Check whether the model config is compatible with torch.compile."""
    text_config = getattr(config, "text_config", config)

    def is_static(parameters):
        if not isinstance(parameters, dict):
            return True
        rope_type = str(parameters.get("rope_type", parameters.get("type", "")))
        return (
            not rope_type.startswith("dynamic")
            and rope_type != "longrope"
            and all(is_static(value) for value in parameters.values())
        )

    return all(
        is_static(getattr(text_config, name, None))
        for name in ("rope_scaling", "rope_parameters")
    )


def maybe_prefix(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module):
    logger.debug("%s: %s -> %s", name, old_module, new_module)


def _getattr_first(obj, names, default=None):
    """Return the first existing attribute from *names*, else *default*."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _resolve_attention_backend_model_cls(
    config: PretrainedConfig, trust_remote_code: bool = False, revision=None
):
    model_cls = getattr(
        transformers, (getattr(config, "architectures", None) or [""])[0], None
    )
    if model_cls is not None:
        return model_cls
    if not trust_remote_code:
        return None
    auto_map = getattr(config, "auto_map", {}) or {}
    for key in ("AutoModel", "AutoModelForCausalLM"):
        if key not in auto_map:
            continue
        try:
            return get_class_from_dynamic_module(
                auto_map[key], getattr(config, "_name_or_path", ""), revision=revision
            )
        except Exception as e:
            logger.warning(
                "Failed to load dynamic module from auto_map[%s]: %s.", key, e
            )
    return None


@contextmanager
def _init_on_device_without_buffers(device: torch.device):
    """Initialize model parameters on *device* while leaving buffers on CPU."""
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
