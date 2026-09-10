# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 native model registration for multimodal generation."""

from sglang.multimodal_gen.configs.transformers.configuration_neo_chat import (
    NEOChatConfig,
    NEOLLMConfig,
    NEOMoELLMConfig,
)
from sglang.multimodal_gen.configs.transformers.configuration_neo_vit import (
    NEOVisionConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.modeling_neo_chat import (
    NEOChatModel,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.modeling_neo_vit import (
    NEOVisionModel,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.modeling_qwen3 import (
    _HAS_FLASH_ATTN as has_flash_attn,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.modeling_qwen3 import (
    Qwen3ForCausalLM,
    effective_attn_backend,
    get_attn_backend,
    set_attn_backend,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)

_REGISTERED = False


def register() -> None:
    """Register SenseNova-U1 types with ``transformers.Auto*``."""
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers import AutoConfig, AutoModel

    AutoConfig.register("neo_vision", NEOVisionConfig, exist_ok=True)
    AutoConfig.register("neo_chat", NEOChatConfig, exist_ok=True)

    AutoModel.register(NEOVisionConfig, NEOVisionModel, exist_ok=True)
    AutoModel.register(NEOChatConfig, NEOChatModel, exist_ok=True)

    _REGISTERED = True


register()

__all__ = [
    "NEOChatConfig",
    "NEOChatModel",
    "NEOLLMConfig",
    "NEOMoELLMConfig",
    "NEOVisionConfig",
    "NEOVisionModel",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "register",
    "set_attn_backend",
    "get_attn_backend",
    "effective_attn_backend",
    "has_flash_attn",
]
