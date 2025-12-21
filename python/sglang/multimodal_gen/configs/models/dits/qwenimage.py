# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class QwenImageArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)
    zero_cond_t: bool = False

    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".to_added_qkv", ".add_q_proj", "q"),
            (".to_added_qkv", ".add_k_proj", "k"),
            (".to_added_qkv", ".add_v_proj", "v"),
        ]
    )

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # QKV fusion mappings
            r"(.*)\.to_q\.(weight|bias)$": (r"\1.to_qkv.\2", 0, 3),
            r"(.*)\.to_k\.(weight|bias)$": (r"\1.to_qkv.\2", 1, 3),
            r"(.*)\.to_v\.(weight|bias)$": (r"\1.to_qkv.\2", 2, 3),
            r"(.*)\.add_q_proj\.(weight|bias)$": (r"\1.to_added_qkv.\2", 0, 3),
            r"(.*)\.add_k_proj\.(weight|bias)$": (r"\1.to_added_qkv.\2", 1, 3),
            r"(.*)\.add_v_proj\.(weight|bias)$": (r"\1.to_added_qkv.\2", 2, 3),
            # LoRA mappings
            r"^(transformer_blocks\.\d+\.attn\..*\.lora_[AB])\.default$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class QwenImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=QwenImageArchConfig)

    prefix: str = "qwenimage"
