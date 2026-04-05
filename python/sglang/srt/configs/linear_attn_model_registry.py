"""Registry for linear attention hybrid models (softmax + linear attention).

External models can register themselves without modifying SGLang core files:

    from sglang.srt.configs.linear_attn_model_registry import (
        register_linear_attn_model, LinearAttnModelSpec,
    )

    register_linear_attn_model(LinearAttnModelSpec(
        config_class=MyLinearAttnConfig,
        backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
        arch_names=["MyLinearAttnForCausalLM"],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
    ))
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LinearAttnModelSpec:
    """Specification for a hybrid (softmax + linear attention) model."""

    config_class: type
    backend_class_name: str  # fully-qualified class name, lazily imported
    arch_names: list[str] = field(default_factory=list)
    uses_mamba_radix_cache: bool = True
    support_mamba_cache: bool = True
    support_mamba_cache_extra_buffer: bool = False
    unwrap_text_config: bool = False  # call get_text_config() before isinstance check


_LINEAR_ATTN_MODEL_REGISTRY: list[LinearAttnModelSpec] = []


def register_linear_attn_model(spec: LinearAttnModelSpec) -> None:
    _LINEAR_ATTN_MODEL_REGISTRY.append(spec)
    logger.info(
        "Registered linear attn model: config=%s, backend=%s, archs=%s",
        spec.config_class.__name__,
        spec.backend_class_name.rsplit(".", 1)[-1],
        spec.arch_names,
    )


def get_linear_attn_config(hf_config: Any) -> Optional[tuple[LinearAttnModelSpec, Any]]:
    for spec in _LINEAR_ATTN_MODEL_REGISTRY:
        config = hf_config.get_text_config() if spec.unwrap_text_config else hf_config
        if isinstance(config, spec.config_class):
            return spec, config
    return None


def get_linear_attn_spec_by_arch(arch_name: str) -> Optional[LinearAttnModelSpec]:
    for spec in _LINEAR_ATTN_MODEL_REGISTRY:
        if arch_name in spec.arch_names:
            return spec
    return None


def import_backend_class(dotted_name: str) -> type:
    module_path, class_name = dotted_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
