# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from enum import Enum

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


class LTXModelType(Enum):
    """
    Model type enum mirroring upstream `LTXModelType`.

    Upstream reference:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model.py::LTXModelType`
    """

    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTX2RopeType(str, Enum):
    """
    Minimal RoPE type enum mirroring LTX-2 upstream `LTXRopeType`.

    Upstream reference:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/rope.py::LTXRopeType`
    """

    INTERLEAVED = "interleaved"
    SPLIT = "split"


class LTX2AttentionFunction(str, Enum):
    """
    Placeholder enum for upstream `AttentionFunction.DEFAULT`.

    Upstream reference:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/attention.py`
    """

    DEFAULT = "default"


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class LTX2ArchConfig(DiTArchConfig):
    """Architecture configuration for LTX-2 Video Transformer."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Parameter name mappings from HuggingFace checkpoint keys to SGLang module names.
            # We use upstream variable names (patchify_proj, adaln_single) but HF uses different keys.
            #
            # HF key -> SGLang key (upstream naming)
            r"^proj_in\.(.*)$": r"patchify_proj.\1",
            r"^time_embed\.(.*)$": r"adaln_single.\1",
            r"^audio_proj_in\.(.*)$": r"audio_patchify_proj.\1",
            r"^audio_time_embed\.(.*)$": r"audio_adaln_single.\1",
            # FeedForward
            r"(.*)ff\.net\.0\.proj\.(.*)$": r"\1ff.proj_in.\2",
            r"(.*)ff\.net\.2\.(.*)$": r"\1ff.proj_out.\2",
            # Attention Norms
            r"(.*)\.norm_q\.(.*)$": r"\1.q_norm.\2",
            r"(.*)\.norm_k\.(.*)$": r"\1.k_norm.\2",
            # Scale Shift Tables (Global)
            r"^av_cross_attn_video_scale_shift\.(.*)$": r"av_ca_video_scale_shift_adaln_single.\1",
            r"^av_cross_attn_audio_scale_shift\.(.*)$": r"av_ca_audio_scale_shift_adaln_single.\1",
            r"^av_cross_attn_video_a2v_gate\.(.*)$": r"av_ca_a2v_gate_adaln_single.\1",
            r"^av_cross_attn_audio_v2a_gate\.(.*)$": r"av_ca_v2a_gate_adaln_single.\1",
            # Scale Shift Tables (Block Level)
            # HF: scale_shift_table_a2v_ca_video -> SGLang: video_a2v_cross_attn_scale_shift_table
            r"(.*)scale_shift_table_a2v_ca_video": r"\1video_a2v_cross_attn_scale_shift_table",
            r"(.*)scale_shift_table_a2v_ca_audio": r"\1audio_a2v_cross_attn_scale_shift_table",
        }
    )

    reverse_param_names_mapping: dict = field(
        default_factory=lambda: {
            # Reverse mapping: SGLang module names -> HF checkpoint keys (for saving).
            r"^patchify_proj\.(.*)$": r"proj_in.\1",
            r"^adaln_single\.(.*)$": r"time_embed.\1",
            r"^audio_patchify_proj\.(.*)$": r"audio_proj_in.\1",
            r"^audio_adaln_single\.(.*)$": r"audio_time_embed.\1",
            # FeedForward
            r"(.*)ff\.proj_in\.(.*)$": r"\1ff.net.0.proj.\2",
            r"(.*)ff\.proj_out\.(.*)$": r"\1ff.net.2.\2",
            # Attention Norms
            r"(.*)\.q_norm\.(.*)$": r"\1.norm_q.\2",
            r"(.*)\.k_norm\.(.*)$": r"\1.norm_k.\2",
            # Scale Shift Tables (Global)
            r"^av_ca_video_scale_shift_adaln_single\.(.*)$": r"av_cross_attn_video_scale_shift.\1",
            r"^av_ca_audio_scale_shift_adaln_single\.(.*)$": r"av_cross_attn_audio_scale_shift.\1",
            r"^av_ca_a2v_gate_adaln_single\.(.*)$": r"av_cross_attn_video_a2v_gate.\1",
            r"^av_ca_v2a_gate_adaln_single\.(.*)$": r"av_cross_attn_audio_v2a_gate.\1",
            # Scale Shift Tables (Block Level)
            # SGLang: video_a2v_cross_attn_scale_shift_table -> HF: scale_shift_table_a2v_ca_video
            r"(.*)video_a2v_cross_attn_scale_shift_table": r"\1scale_shift_table_a2v_ca_video",
            r"(.*)audio_a2v_cross_attn_scale_shift_table": r"\1scale_shift_table_a2v_ca_audio",
        }
    )

    lora_param_names_mapping: dict = field(
        default_factory=lambda: {
            # LoRA parameter name mappings from official repo format to HF format.
            # This is applied before param_names_mapping when loading LoRA adapters.
            # Will be populated if LoRA adapters use different naming conventions.
        }
    )

    # Model type and attention configuration
    model_type: LTXModelType = LTXModelType.AudioVideo
    attention_type: LTX2AttentionFunction = LTX2AttentionFunction.DEFAULT
    rope_type: LTX2RopeType = LTX2RopeType.INTERLEAVED
    double_precision_rope: bool = False

    # Video parameters
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    norm_eps: float = 1e-6
    caption_channels: int = 3840
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: list[int] | None = None
    timestep_scale_multiplier: int = 1000
    use_middle_indices_grid: bool = True

    # Audio parameters
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048
    audio_positional_embedding_max_pos: list[int] | None = None
    av_ca_timestep_scale_multiplier: int = 1

    # SGLang-specific parameters
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512

    def __post_init__(self):
        super().__post_init__()
        # Video derived values
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels
        if self.positional_embedding_max_pos is None:
            self.positional_embedding_max_pos = [20, 2048, 2048]

        # Audio derived values
        self.audio_hidden_size = (
            self.audio_num_attention_heads * self.audio_attention_head_dim
        )
        if self.audio_positional_embedding_max_pos is None:
            self.audio_positional_embedding_max_pos = [2048]


@dataclass
class LTX2Config(DiTConfig):
    """Configuration for LTX-2 Video Transformer."""

    arch_config: LTX2ArchConfig = field(default_factory=LTX2ArchConfig)

    prefix: str = "ltx2"
