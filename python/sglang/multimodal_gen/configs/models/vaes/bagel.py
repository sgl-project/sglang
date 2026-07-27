# Copyright 2024 Black Forest Labs and contributors.
# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for BAGEL's FLUX-style autoencoder.

Source: https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/autoencoder.py
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class BagelVAEArchConfig(VAEArchConfig):
    """Architecture values stored by the official ``ae.safetensors`` file."""

    resolution: int = 256
    in_channels: int = 3
    out_channels: int = 3
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16

    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    spatial_compression_ratio: int = 8
    temporal_compression_ratio: int = 1


@dataclass
class BagelVAEConfig(VAEConfig):
    """Configurable encoder/decoder lifecycle for BAGEL generation and editing."""

    arch_config: BagelVAEArchConfig = field(default_factory=BagelVAEArchConfig)

    load_encoder: bool = False
    load_decoder: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_temporal_scaling_frames: bool = False
    use_parallel_decode: bool = False

    def get_vae_scale_factor(self) -> int:
        """Return the fixed spatial compression ratio.

        BAGEL does not expose ``block_out_channels``, which the base
        implementation uses to infer this value.
        """
        return self.arch_config.spatial_compression_ratio
