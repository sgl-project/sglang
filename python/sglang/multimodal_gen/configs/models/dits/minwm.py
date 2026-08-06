# SPDX-License-Identifier: Apache-2.0
"""Architecture config for the minWM main-branch Wan2.2-5B DMD student."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_block


@dataclass
class MinWMVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_block])
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^head\.modulation$": r"scale_shift_table",
            r"^head\.head\.(.*)$": r"proj_out.\1",
        }
    )
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})
    lora_param_names_mapping: dict = field(default_factory=lambda: {})

    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    local_attn_size: int = -1
    sink_size: int = 0
    rope_position_mode: str = "absolute"
    rope_max_frame_gap: int = 1
    prompt_first_frame_pin_enabled: bool = False
    scene_cut_rope_offset: int = 0
    scene_cut_sink_enabled: bool = False
    num_frame_first_block: int = 1
    num_frames_per_block: int = 4
    sliding_window_num_frames: int = 128
    action_type: str = "primitive_token_residual"
    action_embed_dim: int = 256
    action_hidden_dim: int = 512
    action_kernel_size: int = 3
    action_history_frames: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.action_type not in {
            "primitive_token_residual",
            "primitive_rope_token_residual",
        }:
            raise ValueError(
                "MinWM action_type must be primitive_token_residual or "
                "primitive_rope_token_residual"
            )
        if self.rope_position_mode not in {"absolute", "block_relative"}:
            raise ValueError(
                "MinWM rope_position_mode must be absolute or block_relative"
            )
        if self.rope_max_frame_gap < 1:
            raise ValueError("MinWM rope_max_frame_gap must be >= 1")
        if self.local_attn_size != -1 and self.local_attn_size <= 0:
            raise ValueError("MinWM local_attn_size must be -1 or positive")
        if self.sink_size < 0:
            raise ValueError("MinWM sink_size must be non-negative")
        if self.local_attn_size != -1 and self.sink_size >= self.local_attn_size:
            raise ValueError("MinWM sink_size must be smaller than local_attn_size")
        if (
            self.local_attn_size != -1
            and self.sliding_window_num_frames != self.local_attn_size
        ):
            raise ValueError(
                "MinWM sliding_window_num_frames must equal bounded local_attn_size"
            )
        if self.rope_position_mode == "block_relative" and self.scene_cut_rope_offset:
            raise ValueError(
                "MinWM block_relative RoPE does not support scene_cut_rope_offset"
            )
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class MinWMVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=MinWMVideoArchConfig)
    prefix: str = "Wan"
