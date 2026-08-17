# SPDX-License-Identifier: Apache-2.0
"""Native configuration and tokenizer wrapper for SenseNova U1."""

from __future__ import annotations

from typing import ClassVar

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    ProcessorMixin,
    Qwen3Config,
)

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)

_U1_LLM_EXTRA_FIELDS = (
    "rope_theta",
    "rope_theta_hw",
    "max_position_embeddings_hw",
    "pure_llm",
    "use_deepep",
)


def _build_llm_config(raw_config) -> Qwen3Config:
    if isinstance(raw_config, Qwen3Config):
        return raw_config
    raw_config = dict(raw_config or {})
    config = Qwen3Config(**raw_config)
    for field_name in _U1_LLM_EXTRA_FIELDS:
        if field_name in raw_config:
            setattr(config, field_name, raw_config[field_name])
    return config


class NEOChatProcessor(ProcessorMixin):
    """Tokenizer-only HF processor; SGLang owns image preprocessing."""

    attributes: ClassVar[list[str]] = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=False,
            **kwargs,
        )
        return cls(tokenizer=tokenizer)


class NEOVisionConfig(PretrainedConfig):
    model_type = "neo_vision"

    def __init__(
        self,
        hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
        num_channels: int = 3,
        patch_size: int = 16,
        downsample_ratio: float = 0.5,
        rope_theta_vision: float = 10000.0,
        max_position_embeddings_vision: int = 10000,
        min_pixels: int = 65536,
        max_pixels: int = 16777216,
        **kwargs,
    ):
        kwargs.pop("auto_map", None)
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels


@register_customized_processor(NEOChatProcessor)
class NEOChatConfig(PretrainedConfig):
    model_type = "neo_chat"
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {
        "llm_config": Qwen3Config,
        "vision_config": NEOVisionConfig,
    }

    def __init__(
        self,
        llm_config=None,
        vision_config=None,
        template: str = "neo1_0",
        patch_size: int = 16,
        downsample_ratio: float = 0.5,
        **kwargs,
    ):
        kwargs.pop("auto_map", None)
        architectures = kwargs.pop("architectures", ["NEOChatModel"])
        super().__init__(architectures=architectures, **kwargs)

        llm_config = _build_llm_config(llm_config)
        if vision_config is None:
            vision_config = NEOVisionConfig(
                patch_size=patch_size,
                downsample_ratio=downsample_ratio,
            )
        elif isinstance(vision_config, dict):
            vision_config = NEOVisionConfig(
                patch_size=vision_config.get("patch_size", patch_size),
                downsample_ratio=vision_config.get(
                    "downsample_ratio", downsample_ratio
                ),
                **{
                    key: value
                    for key, value in vision_config.items()
                    if key not in {"patch_size", "downsample_ratio"}
                },
            )

        self.llm_config = llm_config
        self.vision_config = vision_config
        self.template = template
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        # U1 supplies explicit [t, h, w] positions but applies its own split RoPE.
        self.model_is_mrope = True


def register_neo_chat_config() -> None:
    try:
        AutoConfig.register(NEOChatConfig.model_type, NEOChatConfig)
    except ValueError:
        # Importing the module more than once is harmless.
        pass


register_neo_chat_config()


__all__ = [
    "NEOChatConfig",
    "NEOChatProcessor",
    "NEOVisionConfig",
    "register_neo_chat_config",
]
