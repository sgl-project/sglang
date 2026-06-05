# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 Mistral text encoder configuration and prompt formatting."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_layer

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
class Flux2MistralTextArchConfig(TextEncoderArchConfig):
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
    )
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_layer])

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
