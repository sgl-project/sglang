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
class EncoderConfig(ModelConfig[EncoderArchConfig]):
    arch_config: EncoderArchConfig = field(default_factory=EncoderArchConfig)
    _internal_config_fields = (
        "_fsdp_shard_conditions",
        "_supported_attention_backends",
        "stacked_params_mapping",
    )

    prefix: str = ""
    quant_config: QuantizationConfig | None = None
    lora_config: Any | None = None
    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)
    _fsdp_shard_conditions: list = field(default_factory=list)
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.SAGE_ATTN_3,
        }
    )

    def refresh_model_config(self) -> None:
        if hasattr(self.arch_config, "stacked_params_mapping"):
            self.stacked_params_mapping = list(self.arch_config.stacked_params_mapping)
        if hasattr(self.arch_config, "_fsdp_shard_conditions"):
            self._fsdp_shard_conditions = list(self.arch_config._fsdp_shard_conditions)
        if hasattr(self.arch_config, "_supported_attention_backends"):
            self._supported_attention_backends = set(
                self.arch_config._supported_attention_backends
            )


@dataclass
class TextEncoderConfig(EncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=TextEncoderArchConfig)
    _internal_config_fields = EncoderConfig._internal_config_fields + (
        "tokenizer_kwargs",
    )

    # Use the SP Group of the transformer as the TP Group of T5.
    parallel_folding: bool = False
    # "sp" or "ulysses" or "ring"
    parallel_folding_mode: str = "sp"
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)

    def refresh_model_config(self) -> None:
        super().refresh_model_config()
        self.tokenizer_kwargs = {
            "truncation": True,
            "max_length": self.arch_config.text_len,
            "return_tensors": "pt",
        }
        if hasattr(self.arch_config, "tokenizer_kwargs"):
            self.tokenizer_kwargs = dict(self.arch_config.tokenizer_kwargs)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    arch_config: ImageEncoderArchConfig = field(default_factory=ImageEncoderArchConfig)
