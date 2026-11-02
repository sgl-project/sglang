# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class NomasVAEArchConfig(VAEArchConfig):
    scaling_factor: float = 1.0
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta_t: float = 1000000.0
    rope_theta_hw: float = 10000.0
    local_attention_tile: int = 8
    temporal_window_latents: int = 10
    temporal_overlap_latents: int = 5
    num_steps: int = 1
    denoising_step_list: tuple[float, ...] = (0.999, 0.866, 0.634, 0.342)
    seed: int = 42

    def __post_init__(self):
        if not 0 <= self.temporal_overlap_latents < self.temporal_window_latents:
            raise ValueError("PixelUMM requires 0 <= overlap < window")
        steps = self.denoising_step_list[: self.num_steps]
        if not 1 <= self.num_steps <= len(self.denoising_step_list) or not all(
            0 < b <= 1 and b < a for a, b in zip((1.01, *steps[:-1]), steps)
        ):
            raise ValueError("PixelUMM requires descending DMD steps in (0,1]")
        if (self.temporal_compression_ratio, self.spatial_compression_ratio) != (4, 32):
            raise ValueError("PixelUMM requires f32t4c32 latents")
        if self.local_attention_tile < 0:
            raise ValueError("local_attention_tile cannot be negative")


@dataclass
class NomasVAEConfig(VAEConfig):
    arch_config: NomasVAEArchConfig = field(default_factory=NomasVAEArchConfig)
    load_encoder: bool = False
    use_tiling: bool = False
    use_temporal_tiling: bool = False  # Windowing is part of PixelUMM's sampling math.
    use_parallel_tiling: bool = False
    use_parallel_decode: bool = False

    def get_vae_scale_factor(self):
        return 32
