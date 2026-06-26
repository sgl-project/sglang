# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_layer


@dataclass
class ErnieImageArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 36
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    ffn_hidden_size: int = 12288
    text_in_dim: int = 3072
    rope_theta: int = 256
    rope_axes_dim: Tuple[int, int, int] = (32, 48, 48)
    eps: float = 1e-6
    qk_layernorm: bool = True

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"(.*)\.mlp\.gate_proj\.(.*)": (r"\1.mlp.gate_up_proj.\2", 0, 2),
            r"(.*)\.mlp\.up_proj\.(.*)": (r"\1.mlp.gate_up_proj.\2", 1, 2),
        }
    )

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_layer])

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class ErnieImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=ErnieImageArchConfig)
    prefix: str = "ernieimage"
