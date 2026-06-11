# SPDX-License-Identifier: Apache-2.0
#
# Text encoder configuration for Gemma2 2B, used by SANA for text conditioning.
#
# SANA uses the hidden states from Gemma2 (not logits) as the conditioning
# signal for cross-attention in the DiT. The encoder output dimension (2304)
# is projected to the DiT's inner_dim via caption_projection.
#
# Defaults match google/gemma-2-2b-it (the model used in SANA HF checkpoints).

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
class Gemma2ArchConfig(TextEncoderArchConfig):
    vocab_size: int = 256000
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Gemma2 alternates between global and sliding-window attention
    # on odd/even layers, respectively.
    sliding_window: int = 4096

    # query_pre_attn_scalar replaces the standard 1/sqrt(head_dim) scaling.
    query_pre_attn_scalar: int = 256

    # Softcapping bounds raw attention logits via tanh(logits/cap)*cap.
    # NOTE: SDPA does not natively support softcapping; the runtime model
    # currently skips this (see Gemma2Attention.forward). Quality impact
    # is minimal for short text-encoder sequences but should be revisited
    # for longer context.
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0

    text_len: int = 300

    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "0"),
            (".gate_up_proj", ".up_proj", "1"),
        ]
    )
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_layer, is_embed_tokens, is_final_norm]
    )


@dataclass
class Gemma2Config(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=Gemma2ArchConfig)
    prefix: str = "gemma_2"
