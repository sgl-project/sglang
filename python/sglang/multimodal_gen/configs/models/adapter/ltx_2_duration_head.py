# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.adapter.base import (
    AdapterArchConfig,
    AdapterConfig,
)


@dataclass
class LTX2DurationHeadArchConfig(AdapterArchConfig):
    """LTX-2.5 duration head.

    Field names match `duration_head/config.json` verbatim, so
    `update_model_arch` populates this directly from the checkpoint.
    """

    video_cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048
    pooler_hidden_dim: int = 256
    num_queries: int = 1
    num_pooler_heads: int = 4
    mlp_hidden_dim: int = 256


@dataclass
class LTX2DurationHeadConfig(AdapterConfig):
    arch_config: AdapterArchConfig = field(default_factory=LTX2DurationHeadArchConfig)

    prefix: str = "LTX2DurationHead"
