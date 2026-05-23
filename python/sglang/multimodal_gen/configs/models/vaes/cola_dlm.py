from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class ColaDLVAEArchConfig(VAEArchConfig):
    """Architecture config for Cola-DLM Text VAE (ColaTextVAEModel).

    Cola-DLM's VAE operates on token sequences (not images/video).
    It encodes token IDs to continuous latents and decodes latents back to logits.
    """

    latent_dim: int = 16
    vocab_size: int = 100278
    patch_size: int = 1
    block_size: int = 16
    scaling_factor: float = 1.0
    shifting_factor: float = 0.0
    # Token-level model — no spatial compression
    spatial_compression_ratio: int = 1
    temporal_compression_ratio: int = 1


@dataclass
class ColaDLVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=ColaDLVAEArchConfig)
    # Text VAE — no tiling needed
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
