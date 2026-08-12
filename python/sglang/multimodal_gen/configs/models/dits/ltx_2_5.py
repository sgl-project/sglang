# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.ltx_2 import (
    LTX2_PARAM_NAMES_MAPPING,
    LTX2_REVERSE_PARAM_NAMES_MAPPING,
    LTX2ArchConfig,
    LTX2Config,
    LTX2RopeType,
)

# LTX-2.5 renames the prompt adaLN modules and adds the keyframe position
# embedding; the shared LTX-2 mapping covers everything else.
LTX25_EXTRA_PARAM_NAMES_MAPPING: dict[str, str] = {
    r"^prompt_adaln\.(.*)$": r"prompt_adaln_single.\1",
    r"^audio_prompt_adaln\.(.*)$": r"audio_prompt_adaln_single.\1",
}

LTX25_EXTRA_REVERSE_PARAM_NAMES_MAPPING: dict[str, str] = {
    r"^prompt_adaln_single\.(.*)$": r"prompt_adaln.\1",
    r"^audio_prompt_adaln_single\.(.*)$": r"audio_prompt_adaln.\1",
}


@dataclass
class LTX25ArchConfig(LTX2ArchConfig):
    """LTX-2.5 DiT architecture config.

    LTX-2.5 reuses the LTX-2.3 audio-video transformer: gated attention,
    cross-attention adaLN modulation, split RoPE in double precision, and
    per-modality caption projections that live in the connector rather than the
    DiT. On top of that it drops the video feed-forward bias and carries a
    keyframe absolute-position embedding.
    """

    param_names_mapping: dict = field(
        default_factory=lambda: {
            **LTX2_PARAM_NAMES_MAPPING,
            **LTX25_EXTRA_PARAM_NAMES_MAPPING,
        }
    )
    reverse_param_names_mapping: dict = field(
        default_factory=lambda: {
            **LTX2_REVERSE_PARAM_NAMES_MAPPING,
            **LTX25_EXTRA_REVERSE_PARAM_NAMES_MAPPING,
        }
    )

    # LTX-2.3 audio-video base (`gated_attn` / `cross_attn_mod` /
    # `use_prompt_embeddings: false` in transformer/config.json).
    apply_gated_attention: bool = True
    cross_attention_adaln: bool = True
    caption_proj_before_connector: bool = True
    rope_type: LTX2RopeType = LTX2RopeType.SPLIT
    double_precision_rope: bool = True

    # LTX-2.5 specific.
    ff_bias: bool = False
    audio_ff_bias: bool = True
    use_keyframes_abs_pos_embedding: bool = True

    # Mirrored here because these also appear in transformer/config.json.
    connector_num_attention_heads: int = 32
    connector_num_layers: int = 8
    audio_connector_attention_head_dim: int = 64
    audio_connector_num_attention_heads: int = 32
    audio_connector_num_layers: int = 8


@dataclass
class LTX25Config(LTX2Config):
    arch_config: LTX25ArchConfig = field(default_factory=LTX25ArchConfig)

    prefix: str = "ltx2_5"
