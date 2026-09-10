# SPDX-License-Identifier: Apache-2.0
"""Explicit SenseNova-U1 model registration for multimodal generation."""

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


__all__ = ["register"]
