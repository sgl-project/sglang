# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional, Tuple

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class LTXAudioVAEArchConfig(VAEArchConfig):
    # Architecture params
    causality_axis: str = "height"
    attn_resolutions: Optional[Tuple[int, ...]] = None
    base_channels: int = 128
    latent_channels: int = 8
    output_channels: int = 2
    ch_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    norm_type: str = "pixel"
    dropout: float = 0.0
    mid_block_add_attention: bool = False
    sample_rate: int = 16000
    mel_hop_length: int = 160
    is_causal: bool = True
    mel_bins: Optional[int] = 64
    double_z: bool = True


@dataclass
class LTXAudioVAEConfig(VAEConfig):
    arch_config: LTXAudioVAEArchConfig = field(default_factory=LTXAudioVAEArchConfig)
