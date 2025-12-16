# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    ImageEncoderArchConfig,
    ImageEncoderConfig,
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _is_transformer_layer(n: str, m) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("embeddings")


@dataclass
class CLIPTextArchConfig(TextEncoderArchConfig):
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    text_len: int = 77
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.TORCH_SDPA,  # Force TORCH_SDPA to support attention_mask
        }
    )
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
    )
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_transformer_layer, _is_embeddings]
    )


@dataclass
class CLIPVisionArchConfig(ImageEncoderArchConfig):
    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 32
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
    )


@dataclass
class CLIPTextConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=CLIPTextArchConfig)

    num_hidden_layers_override: int | None = None
    require_post_norm: bool | None = None
    prefix: str = "clip"


@dataclass
class CLIPVisionConfig(ImageEncoderConfig):
    arch_config: ImageEncoderArchConfig = field(default_factory=CLIPVisionArchConfig)

    num_hidden_layers_override: int | None = None
    require_post_norm: bool | None = None
    prefix: str = "clip"
