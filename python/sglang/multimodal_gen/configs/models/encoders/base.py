# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class EncoderArchConfig(ArchConfig):
    architectures: list[str] = field(default_factory=lambda: [])
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA3,
            AttentionBackendEnum.TORCH_SDPA,
        }
    )
    output_hidden_states: bool = False
    use_return_dict: bool = True


@dataclass
class TextEncoderArchConfig(EncoderArchConfig):
    vocab_size: int = 0
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    pad_token_id: int = 0
    eos_token_id: int = 0
    text_len: int = 0
    hidden_state_skip_layer: int = 0
    decoder_start_token_id: int = 0
    output_past: bool = True
    scalable_attention: bool = True
    tie_word_embeddings: bool = False
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # mapping from huggingface weight names to custom names
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    _fsdp_shard_conditions: list = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        self.tokenizer_kwargs = {
            "truncation": True,
            "max_length": self.text_len,
            "return_tensors": "pt",
        }


@dataclass
class ImageEncoderArchConfig(EncoderArchConfig):
    pass


@dataclass
class BaseEncoderOutput:
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    attention_mask: torch.Tensor | None = None


@dataclass
class EncoderConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=EncoderArchConfig)

    prefix: str = ""
    quant_config: QuantizationConfig | None = None
    lora_config: Any | None = None


@dataclass
class TextEncoderConfig(EncoderConfig):
    arch_config: ArchConfig = field(default_factory=TextEncoderArchConfig)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    arch_config: ArchConfig = field(default_factory=ImageEncoderArchConfig)
