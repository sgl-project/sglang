# SPDX-License-Identifier: Apache-2.0

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
class Qwen3VLArchConfig(TextEncoderArchConfig):
    """Architecture configuration for Qwen3-VL text encoder.

    Qwen3-VL-8B-Instruct is used by JoyImage model.
    Architecture is similar to Qwen2.5-VL but with Qwen3 improvements.
    """

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
    text_len: int = 2048

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
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_layer, is_embed_tokens, is_final_norm]
    )

    # JoyImage specific settings
    text_token_max_length: int = 2048
    prompt_template_encode_start_idx = {
        "image": 34,
        "video": 91,
    }

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.text_len
            + self.prompt_template_encode_start_idx["image"],
            "return_tensors": "pt",
        }


@dataclass
class Qwen3VLConfig(TextEncoderConfig):
    """Configuration for Qwen3-VL text encoder.

    Used by JoyImage model.
    """

    arch_config: TextEncoderArchConfig = field(default_factory=Qwen3VLArchConfig)
