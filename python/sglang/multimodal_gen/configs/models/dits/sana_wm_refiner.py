# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_blocks_or_transformer_blocks


@dataclass
class SanaWMRefinerArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_blocks_or_transformer_blocks]
    )

    # Core dims
    in_channels: int = 128
    out_channels: int = 128
    patch_size: int = 1
    patch_size_t: int = 1
    num_layers: int = 28
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    cross_attention_dim: int = 4096
    caption_channels: int = 4096

    qk_norm: bool = True
    norm_eps: float = 1e-6
    apply_gated_attention: bool = False

    timestep_scale_multiplier: float = 1000.0
    rope_type: str = "interleaved"

    # RoPE coord generation
    sampling_rate: int = 16000
    hop_length: int = 160
    scale_factors: tuple = (8, 32, 32)
    base_num_frames: int = 20
    base_height: int = 2048
    base_width: int = 2048
    causal_offset: int = 1

    # Map Diffusers-style param keys to sglang's LTX-2 primitive naming.
    # The refiner reuses LTX2Attention / LTX2FeedForward, so it inherits the
    # same naming differences vs Diffusers:
    #   * ff.net.0.proj / ff.net.2 -> proj_in / proj_out  (LTX2FeedForward)
    #   * norm_q / norm_k          -> q_norm / k_norm     (LTX2Attention)
    # Keep this aligned with LTX2ArchConfig.param_names_mapping in ltx_2.py.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(transformer_blocks\.\d+\.ff)\.net\.0\.proj\.(.*)$": r"\1.proj_in.\2",
            r"^(transformer_blocks\.\d+\.ff)\.net\.2\.(.*)$": r"\1.proj_out.\2",
            r"(.*)\.norm_q\.(.*)$": r"\1.q_norm.\2",
            r"(.*)\.norm_k\.(.*)$": r"\1.k_norm.\2",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class SanaWMRefinerConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SanaWMRefinerArchConfig)
    prefix: str = "SanaWMRefiner"
