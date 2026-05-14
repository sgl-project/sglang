# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import List

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class LTXVideoVAEArchConfig(VAEArchConfig):
    # Architecture params
    in_channels: int = 3
    latent_channels: int = 128
    out_channels: int = 3
    block_out_channels: List[int] = field(
        default_factory=lambda: [256, 512, 1024, 2048]
    )
    down_block_types: List[str] = field(
        default_factory=lambda: [
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
        ]
    )
    spatio_temporal_scaling: List[bool] = field(
        default_factory=lambda: [True, True, True, True]
    )
    layers_per_block: List[int] = field(default_factory=lambda: [4, 6, 6, 2, 2])
    downsample_type: List[str] = field(
        default_factory=lambda: [
            "spatial",
            "temporal",
            "spatiotemporal",
            "spatiotemporal",
        ]
    )
    patch_size: int = 4
    patch_size_t: int = 1
    resnet_norm_eps: float = 1e-6
    encoder_causal: bool = True
    encoder_spatial_padding_mode: str = "zeros"

    decoder_block_out_channels: List[int] = field(
        default_factory=lambda: [256, 512, 1024]
    )
    decoder_spatio_temporal_scaling: List[bool] = field(
        default_factory=lambda: [True, True, True]
    )
    decoder_layers_per_block: List[int] = field(default_factory=lambda: [5, 5, 5, 5])
    decoder_causal: bool = False
    decoder_spatial_padding_mode: str = "reflect"


@dataclass
class LTXVideoVAEConfig(VAEConfig):
    arch_config: LTXVideoVAEArchConfig = field(default_factory=LTXVideoVAEArchConfig)
