from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class LongCatImageVAEArchConfig(VAEArchConfig):
    spatial_compression_ratio: int = 8
    vae_scale_factor: int = 8
    # scaling_factor and shift_factor come from the model's config at runtime


@dataclass
class LongCatImageVAEConfig(VAEConfig):
    arch_config: LongCatImageVAEArchConfig = field(
        default_factory=LongCatImageVAEArchConfig
    )

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def get_vae_scale_factor(self):
        return self.arch_config.vae_scale_factor
