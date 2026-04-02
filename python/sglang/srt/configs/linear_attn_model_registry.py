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

    # HuggingFace config class. Used for isinstance matching in model_runner.
    config_class: type

    # Fully-qualified class name of the linear attention backend, e.g.
    # "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend".
    # Lazily imported at backend creation time.
    backend_class_name: str

    # HuggingFace architecture names (hf_config.architectures[0]) for
    # server_args dispatch.
    arch_names: list[str] = field(default_factory=list)

    # Whether this model needs MambaRadixCache (controls is_hybrid_ssm in
    # scheduler). Set False for models that disable radix cache entirely.
    uses_mamba_radix_cache: bool = True

    # Arguments forwarded to _handle_mamba_radix_cache in server_args.
    support_mamba_cache: bool = True
    support_mamba_cache_extra_buffer: bool = False

    # If True, calls hf_config.get_text_config() before isinstance check
    # (needed for VLM wrapper configs).
    unwrap_text_config: bool = False

    # If True, asserts the model is not used with MLA backends.
    # Mirrors the inline check for GDN at attention_registry.py:198,
    # but needed here because external models can't add inline checks.
    mla_incompatible: bool = False


_LINEAR_ATTN_MODEL_REGISTRY: list[LinearAttnModelSpec] = []


def register_linear_attn_model(spec: LinearAttnModelSpec) -> None:
    """Register a linear attention hybrid model specification.

    Call this at import time (e.g. in your model package's __init__.py).
    The config_class must expose:
      - full_attention_layer_ids: list[int]
      - mamba2_cache_params (if uses_mamba_radix_cache is True)
    """
    _LINEAR_ATTN_MODEL_REGISTRY.append(spec)
    logger.info(
        "Registered linear attn model: config=%s, backend=%s, archs=%s",
        spec.config_class.__name__,
        spec.backend_class_name.rsplit(".", 1)[-1],
        spec.arch_names,
    )


def get_linear_attn_config(hf_config: Any) -> Optional[tuple[LinearAttnModelSpec, Any]]:
    """Check if hf_config matches any registered linear attention hybrid model.

    Returns (spec, resolved_config) or None. The resolved_config is the
    result of get_text_config() if unwrap_text_config is set.
    """
    for spec in _LINEAR_ATTN_MODEL_REGISTRY:
        config = hf_config.get_text_config() if spec.unwrap_text_config else hf_config
        if isinstance(config, spec.config_class):
            return spec, config
    return None


def get_linear_attn_spec_by_arch(arch_name: str) -> Optional[LinearAttnModelSpec]:
    """Look up a linear attention model spec by HuggingFace architecture name."""
    for spec in _LINEAR_ATTN_MODEL_REGISTRY:
        if arch_name in spec.arch_names:
            return spec
    return None


def import_backend_class(dotted_name: str) -> type:
    """Lazily import a backend class from its fully-qualified name."""
    module_path, class_name = dotted_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
