# SPDX-License-Identifier: Apache-2.0
"""Qwen3 text encoder configuration for SGLang diffusion models."""
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


def _is_transformer_layer(n: str, m) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("embed_tokens")


def _is_final_norm(n: str, m) -> bool:
    return n.endswith("norm")


@dataclass
class Qwen3TextArchConfig(TextEncoderArchConfig):
    """Architecture config for Qwen3 text encoder.

    Qwen3 is similar to LLaMA but with QK-Norm (RMSNorm on Q and K before attention).
    """

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 151643
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    head_dim: int = 128
    text_len: int = 512
    output_hidden_states: bool = True  # Klein needs hidden states from layers 9, 18, 27

    # Stacked params for weight loading with tensor parallelism
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
    )

    # FSDP sharding conditions for CPU offload
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_transformer_layer, _is_embeddings, _is_final_norm]
    )

    def __post_init__(self) -> None:
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
            "return_tensors": "pt",
        }


@dataclass
class Qwen3TextConfig(TextEncoderConfig):
    """Top-level config for Qwen3 text encoder."""

    arch_config: TextEncoderArchConfig = field(default_factory=Qwen3TextArchConfig)
    prefix: str = "qwen3"
