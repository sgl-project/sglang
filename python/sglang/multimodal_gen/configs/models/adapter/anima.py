# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.adapter.base import (
    AdapterArchConfig,
    AdapterConfig,
)


@dataclass
class AnimaTextConditionerArchConfig(AdapterArchConfig):
    source_dim: int = 1024
    target_dim: int = 1024
    model_dim: int = 1024
    num_layers: int = 6
    num_attention_heads: int = 16
    mlp_ratio: float = 4.0
    target_vocab_size: int = 32128
    use_self_attention: bool = True
    use_layer_norm: bool = False
    min_sequence_length: int = 512


@dataclass
class AnimaTextConditionerConfig(AdapterConfig):
    arch_config: AnimaTextConditionerArchConfig = field(
        default_factory=AnimaTextConditionerArchConfig
    )
