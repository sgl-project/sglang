# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 native model registration for multimodal generation."""

from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify import (  # noqa: F401
    NEOChatConfig,
    NEOChatModel,
    NEOLLMConfig,
    NEOMoELLMConfig,
    NEOVisionConfig,
    NEOVisionModel,
    register,
)

register()

__all__ = [
    "NEOChatConfig",
    "NEOChatModel",
    "NEOLLMConfig",
    "NEOMoELLMConfig",
    "NEOVisionConfig",
    "NEOVisionModel",
    "register",
]
