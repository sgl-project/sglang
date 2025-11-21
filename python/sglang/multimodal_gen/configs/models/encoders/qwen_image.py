# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
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
class QwenImageArchConfig(TextEncoderArchConfig):
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = -1
    eos_token_id: int = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: float | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None
    hidden_state_skip_layer: int = 2
    text_len: int = 256
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),  # type: ignore
            (".gate_up_proj", ".up_proj", 1),  # type: ignore
        ]
    )
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_transformer_layer, _is_embeddings, _is_final_norm]
    )


@dataclass
class Qwen2_5VLConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=QwenImageArchConfig)
    # prefix: str = "qwen_image"
