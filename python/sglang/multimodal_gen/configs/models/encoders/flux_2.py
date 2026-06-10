# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 Mistral text encoder configuration and prompt formatting."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.encoders.mistral3 import (
    Mistral3EncoderArchConfig,
)

FLUX_2_SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\n"
    "attribution and actions without speculation."
)


def build_flux2_text_messages(prompts: list[str]) -> list[list[dict]]:
    cleaned_prompts = [prompt.replace("[IMG]", "") for prompt in prompts]
    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": FLUX_2_SYSTEM_MESSAGE}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_prompts
    ]


@dataclass
class Flux2MistralTextArchConfig(Mistral3EncoderArchConfig):
    """FLUX.2 text-encoder arch config.

    Inherits Mistral3 defaults (hidden_size, num_attention_heads, head_dim,
    rms_norm_eps, rope_parameters, ...) so the TP-parallel runtime has every
    field it needs even when the checkpoint config doesn't override them.
    Only the tokenizer behavior (max_length=512, padding=max_length) is
    flux2-specific.
    """

    def __post_init__(self) -> None:
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }


@dataclass
class Flux2MistralTextConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(
        default_factory=Flux2MistralTextArchConfig
    )
    prefix: str = "flux_2_mistral"
