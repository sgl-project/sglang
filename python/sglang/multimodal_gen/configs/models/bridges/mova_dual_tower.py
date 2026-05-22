# SPDX-License-Identifier: Apache-2.0
"""Configuration for MOVA dual tower bridge model."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_conditioner_block(name: str, module) -> bool:
    """Check if module is a ConditionalCrossAttentionBlock."""
    return "ConditionalCrossAttentionBlock" in type(module).__name__


@dataclass
class MOVADualTowerArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_conditioner_block]
    )

    # Model architecture parameters
    visual_layers: int = 40
    audio_layers: int = 30
    visual_hidden_dim: int = 5120
    audio_hidden_dim: int = 1536
    audio_fps: float = 50.0
    head_dim: int = 128
    interaction_strategy: str = "full"
    apply_cross_rope: bool = True
    apply_first_frame_bias_in_rope: bool = False
    trainable_condition_scale: bool = False
    pooled_adaln: bool = False
    eps: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.visual_hidden_dim
        self.num_attention_heads = self.visual_hidden_dim // self.head_dim


@dataclass
class MOVADualTowerConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=MOVADualTowerArchConfig)
