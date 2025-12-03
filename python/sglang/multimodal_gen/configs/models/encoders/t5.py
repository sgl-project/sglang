# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


def _is_transformer_layer(n: str, m) -> bool:
    return "block" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("shared")


def _is_final_layernorm(n: str, m) -> bool:
    return n.endswith("final_layer_norm")


@dataclass
class T5ArchConfig(TextEncoderArchConfig):
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int | None = None
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    dense_act_fn: str = ""
    is_gated_act: bool = False
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    classifier_dropout: float = 0.0
    text_len: int = 512
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q", "q"),
            (".qkv_proj", ".k", "k"),
            (".qkv_proj", ".v", "v"),
        ]
    )
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [
            _is_transformer_layer,
            _is_embeddings,
            _is_final_layernorm,
        ]
    )

    # Referenced from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/configuration_t5.py
    def __post_init__(self):
        super().__post_init__()
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn: str = act_info[-1]
        self.is_gated_act: bool = act_info[0] == "gated"
        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }


@dataclass
class T5Config(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=T5ArchConfig)

    prefix: str = "t5"
