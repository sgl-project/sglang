# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.qwen3_5 import (
    Qwen3_5EncoderArchConfig,
    Qwen3_5EncoderConfig,
)


@dataclass
class Magi2TextEncoderArchConfig(Qwen3_5EncoderArchConfig):
    """Qwen3.5-27B; ``text_config`` is read from the checkpoint at load time."""

    text_len: int = 7000

    # Conditioning is hidden_states[-(skip_layer + 1)]; the last layer shifts guidance.
    skip_layer: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs = {
            "padding": "longest",
            "truncation": True,
            "max_length": self.text_len,
            "return_tensors": "pt",
        }


@dataclass
class Magi2TextEncoderConfig(Qwen3_5EncoderConfig):
    arch_config: Magi2TextEncoderArchConfig = field(
        default_factory=Magi2TextEncoderArchConfig
    )
