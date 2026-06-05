# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_layer
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class Ideogram4DiTArchConfig(DiTArchConfig):
    adaln_dim: int = 512
    attention_head_dim: int = 256
    in_channels: int = 128
    intermediate_size: int = 12288
    llm_features_dim: int = 53248
    mrope_section: tuple[int, int, int] | list[int] = (24, 20, 20)
    norm_eps: float = 1e-5
    num_attention_heads: int = 18
    num_layers: int = 34
    rope_theta: int = 5_000_000
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_layer])
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels


@dataclass
class Ideogram4DiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Ideogram4DiTArchConfig)
    prefix: str = "ideogram4"
