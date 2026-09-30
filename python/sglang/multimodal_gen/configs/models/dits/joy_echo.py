# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.ltx_2 import (
    LTX2ArchConfig,
    LTX2Config,
)


@dataclass
class JoyEchoArchConfig(LTX2ArchConfig):
    """JoyEcho DiT architecture config (LTX-2.3 AV base)."""

    caption_proj_before_connector: bool = True
    cross_attention_adaln: bool = True
    apply_gated_attention: bool = True


@dataclass
class JoyEchoConfig(LTX2Config):
    arch_config: JoyEchoArchConfig = field(default_factory=JoyEchoArchConfig)
    prefix: str = "JoyEcho"
