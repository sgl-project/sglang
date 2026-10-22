# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class MOSSImageArchConfig(DiTArchConfig):
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0

    def __post_init__(self):
        self.hidden_size = self.dim
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.latent_channels


@dataclass
class MOSSImageDiTConfig(DiTConfig):
    arch_config: MOSSImageArchConfig = field(default_factory=MOSSImageArchConfig)
