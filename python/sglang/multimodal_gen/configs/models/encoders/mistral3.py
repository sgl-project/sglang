# SPDX-License-Identifier: Apache-2.0
"""Mistral3 text encoder configuration for SGLang diffusion models."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import (
    is_embed_tokens,
    is_final_norm,
    is_layer,
)


@dataclass
class Mistral3EncoderArchConfig(TextEncoderArchConfig):
    """Mistral3 text encoder architecture config for ErnieImage.

    Uses Mistral3Model (vision-language model) as text encoder,
    extracting the second-to-last hidden state layer.
    """

    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 11
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    head_dim: int = 128
    hidden_state_skip_layer: int = 2  # Use second-to-last hidden state
    text_len: int = 0

    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
    )

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_layer, is_embed_tokens, is_final_norm]
    )

    def __post_init__(self):
        # Let the parent populate tokenizer_kwargs["max_length"] = self.text_len
        super().__post_init__()


@dataclass
class Mistral3EncoderConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(
        default_factory=Mistral3EncoderArchConfig
    )
