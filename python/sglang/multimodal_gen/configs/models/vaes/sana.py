# SPDX-License-Identifier: Apache-2.0
#
# VAE configuration for SANA's DC-AE (Deep Compression AutoEncoder).
#
# DC-AE achieves a 32x spatial compression ratio (vs. 8x for standard SD VAEs),
# which means a 1024x1024 image becomes 32x32 latents with 32 channels.
# This aggressive compression is what allows SANA to run efficiently at
# high resolutions despite having a relatively small DiT.
#
# Reference: https://arxiv.org/abs/2405.17811

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class SanaVAEArchConfig(VAEArchConfig):
    spatial_compression_ratio: int = 32
    # DC-AE uses a different scaling factor than standard VAEs;
    # this value must match the pretrained checkpoint.
    scaling_factor: float = 0.41407
    latent_channels: int = 32
    in_channels: int = 3


@dataclass
class SanaVAEConfig(VAEConfig):
    arch_config: SanaVAEArchConfig = field(default_factory=SanaVAEArchConfig)

    # DC-AE does not currently support tiling in our wrapper.
    # Enable these once the diffusers AutoencoderDC adds tiling support.
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def post_init(self):
        # Called by VAELoader AFTER update_model_arch() merges the HF config.json
        # values into arch_config. Must be post_init() (not __post_init__) because
        # __post_init__ fires at dataclass creation time, before the HF config merge.
        #
        # The base VAEConfig.get_vae_scale_factor() derives from block_out_channels,
        # which DC-AE doesn't have. Set vae_scale_factor directly from the
        # spatial_compression_ratio (32x for DC-AE).
        self.arch_config.vae_scale_factor = self.arch_config.spatial_compression_ratio
