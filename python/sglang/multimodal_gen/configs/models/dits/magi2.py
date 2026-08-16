# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig

MAGI2_NUM_MODALITY = 3

# Not a token: sinusoidal_embedding_1d(64, t) overwrites the leading 64 channels.
MAGI2_TIME_CHANNEL_DIM = 64

MAGI2_ROPE_BANDS = 16
MAGI2_ROTARY_DIM = 96


@dataclass
class Magi2PreviewArchConfig(DiTArchConfig):
    num_layers: int = 40
    hidden_size: int = 3072
    head_dim: int = 128
    num_attention_heads: int = 24
    num_channels_latents: int = 48

    num_stream: int = 4
    mhc_alpha_init: float = 0.01
    mhc_sinkhorn_iters: int = 20

    mm_layers: tuple[int, ...] = (0, 1, 38, 39)
    dense_intermediate_size: int = 8192

    moe_num_heads: int = 12
    moe_num_experts: int = 256
    moe_top_k: int = 6
    moe_expert_intermediate_size: int = 1280
    moe_shared_expert_intermediate_size: int = 1280
    moe_modality_expert_intermediate_size: int = 1280
    moe_route_scale: float = 4.9
    moe_score_func: str = "sigmoid"
    moe_route_norm: bool = True

    sink_token_num: int = 1
    rope_bands: int = MAGI2_ROPE_BANDS
    rotary_dim: int = MAGI2_ROTARY_DIM
    time_channel_dim: int = MAGI2_TIME_CHANNEL_DIM
    num_modality: int = MAGI2_NUM_MODALITY

    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120

    vae_stride: tuple[int, int, int] = (8, 16, 16)

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Released weights are EMA weights: loading raw expert_bias instead of
            # expert_bias_ema silently shifts expert selection.
            r"^block\.layers\.(\d+)\.mlp\.moe_mlp\.router\.expert_bias$": "",
            r"^block\.layers\.(\d+)\.mlp\.moe_mlp\.router\.expert_bias_ema$": (
                r"blocks.\1.mlp.moe_mlp.router.expert_bias"
            ),
            r"^block\.layers\.": r"blocks.",
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.residual_stream_dim = self.num_stream * self.hidden_size


@dataclass
class Magi2PreviewConfig(DiTConfig):
    arch_config: Magi2PreviewArchConfig = field(default_factory=Magi2PreviewArchConfig)


@dataclass
class Magi2RefinerArchConfig(DiTArchConfig):
    num_layers: int = 30
    hidden_size: int = 4096
    head_dim: int = 128
    num_attention_heads: int = 32
    num_query_groups: int = 8
    num_channels_latents: int = 48

    mm_layers: tuple[int, ...] = (0, 1, 28, 29)
    ffn_hidden_size: int = 10920

    block_t_size: int = 8
    block_size: int = 4
    block_t_radius: int = 2
    block_h_radius: int = 2
    block_w_radius: int = 2
    win_size: int = 384

    rope_bands: int = MAGI2_ROPE_BANDS
    rotary_dim: int = MAGI2_ROTARY_DIM
    num_modality: int = MAGI2_NUM_MODALITY

    # Zero: a time embedding here would overwrite 64 real embedded channels.
    time_channel_dim: int = 0

    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120

    vae_stride: tuple[int, int, int] = (4, 16, 16)

    param_names_mapping: dict = field(
        default_factory=lambda: {r"^block\.layers\.": r"blocks."}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.residual_stream_dim = self.hidden_size
        self.num_stream = 1


@dataclass
class Magi2RefinerConfig(DiTConfig):
    arch_config: Magi2RefinerArchConfig = field(default_factory=Magi2RefinerArchConfig)
