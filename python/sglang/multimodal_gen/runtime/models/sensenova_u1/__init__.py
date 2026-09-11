# SPDX-License-Identifier: Apache-2.0
"""Explicit SenseNova-U1 model registration for multimodal generation."""

from importlib import import_module

_LAZY_EXPORTS = {
    "NEOChatConfig": (
        "sglang.multimodal_gen.configs.transformers.configuration_neo_chat",
        "NEOChatConfig",
    ),
    "NEOLLMConfig": (
        "sglang.multimodal_gen.configs.transformers.configuration_neo_chat",
        "NEOLLMConfig",
    ),
    "NEOMoELLMConfig": (
        "sglang.multimodal_gen.configs.transformers.configuration_neo_chat",
        "NEOMoELLMConfig",
    ),
    "NEOVisionConfig": (
        "sglang.multimodal_gen.configs.transformers.configuration_neo_vit",
        "NEOVisionConfig",
    ),
    "NEOChatModel": (".modeling_neo_chat", "NEOChatModel"),
    "NEOVisionModel": (".modeling_neo_vit", "NEOVisionModel"),
    "Qwen3ForCausalLM": (".modeling_qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".modeling_qwen3_moe", "Qwen3MoeForCausalLM"),
    "set_attn_backend": (".modeling_qwen3", "set_attn_backend"),
    "get_attn_backend": (".modeling_qwen3", "get_attn_backend"),
    "effective_attn_backend": (".modeling_qwen3", "effective_attn_backend"),
    "has_flash_attn": (".modeling_qwen3", "_HAS_FLASH_ATTN"),
}


def __getattr__(name: str):
    """Preserve package-level imports without eagerly loading model code."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


_REGISTERED = False


def register() -> None:
    """Load the native model types and register them with ``transformers.Auto*``."""
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers import AutoConfig, AutoModel

    from sglang.multimodal_gen.configs.transformers.configuration_neo_chat import (
        NEOChatConfig,
    )
    from sglang.multimodal_gen.configs.transformers.configuration_neo_vit import (
        NEOVisionConfig,
    )

    from .modeling_neo_chat import NEOChatModel
    from .modeling_neo_vit import NEOVisionModel

    AutoConfig.register("neo_vision", NEOVisionConfig, exist_ok=True)
    AutoConfig.register("neo_chat", NEOChatConfig, exist_ok=True)
    AutoModel.register(NEOVisionConfig, NEOVisionModel, exist_ok=True)
    AutoModel.register(NEOChatConfig, NEOChatModel, exist_ok=True)

    _REGISTERED = True


__all__ = ["register", *_LAZY_EXPORTS]
