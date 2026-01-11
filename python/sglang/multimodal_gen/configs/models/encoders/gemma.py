# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


@dataclass
class GemmaArchConfig(TextEncoderArchConfig):
    """Minimal Gemma text-encoder config for tokenizer kwargs.

    Note: runtime will load the actual `text_encoder/` module from the model repo
    (e.g. Gemma3Model) via transformers; this config mainly controls tokenization.
    """

    # A sane default for prompt length.
    text_len: int = 256

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }


@dataclass
class GemmaConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=GemmaArchConfig)

    prefix: str = "gemma"

