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
class Gemma4UnifiedArchConfig(TextEncoderArchConfig):
    """Gemma-4-Unified text encoder used by LTX-2.5.

    The repo's `text_encoder/config.json` (`Gemma4UnifiedForConditionalGeneration`)
    is copied onto this config at load time; the native encoder reads its
    `text_config`. LTX-2.5 consumes all 48 hidden layers plus the embedding
    output, which is why the connector's `text_proj_in_factor` is 49 and
    `caption_channels` is 3840.
    """

    hidden_size: int = 3840
    num_hidden_layers: int = 48
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 262144
    hidden_state_skip_layer: int = 2
    text_len: int = 1024

    stacked_params_mapping: list[tuple[str, str, str | int]] = field(
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


@dataclass
class Gemma4UnifiedConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=Gemma4UnifiedArchConfig)

    prefix: str = "gemma_4_unified"
