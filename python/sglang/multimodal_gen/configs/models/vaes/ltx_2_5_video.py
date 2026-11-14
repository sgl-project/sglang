# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import List

from sglang.multimodal_gen.configs.models.vaes.ltx_video import (
    LTXVideoVAEArchConfig,
    LTXVideoVAEConfig,
)


@dataclass
class LTX25VideoVAEArchConfig(LTXVideoVAEArchConfig):
    """LTX-2.5 video VAE.

    The encoder is unchanged from LTX-2. The decoder gains a fourth up block and
    no longer upsamples every stage in all three dimensions -- `upsample_type`
    makes the last two stages temporal-only and spatial-only respectively.
    """

    block_out_channels: List[int] = field(
        default_factory=lambda: [256, 512, 1024, 1024]
    )
    layers_per_block: List[int] = field(default_factory=lambda: [4, 6, 4, 2, 2])

    decoder_block_out_channels: List[int] = field(
        default_factory=lambda: [256, 512, 512, 1024]
    )
    decoder_spatio_temporal_scaling: List[bool] = field(
        default_factory=lambda: [True, True, True, True]
    )
    decoder_layers_per_block: List[int] = field(default_factory=lambda: [4, 6, 4, 2, 2])
    decoder_inject_noise: List[bool] = field(
        default_factory=lambda: [False, False, False, False, False]
    )
    decoder_spatial_padding_mode: str = "zeros"

    upsample_residual: List[bool] = field(
        default_factory=lambda: [False, False, False, False]
    )
    upsample_factor: List[int] = field(default_factory=lambda: [2, 2, 1, 2])
    upsample_type: List[str] | None = field(
        default_factory=lambda: [
            "spatiotemporal",
            "spatiotemporal",
            "temporal",
            "spatial",
        ]
    )


@dataclass
class LTX25VideoVAEConfig(LTXVideoVAEConfig):
    arch_config: LTX25VideoVAEArchConfig = field(
        default_factory=LTX25VideoVAEArchConfig
    )
