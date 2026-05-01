# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class JoyImageArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Condition embedder mappings
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^condition_embedder\.time_proj\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            # Double blocks mappings
            r"^double_blocks\.(\d+)\.attn\.(.*)$": r"double_blocks.\1.\2",
            r"^double_blocks\.(\d+)\.img_mlp\.net\.0\.proj\.(.*)$": r"double_blocks.\1.img_mlp.fc_in.\2",
            r"^double_blocks\.(\d+)\.img_mlp\.net\.2\.(.*)$": r"double_blocks.\1.img_mlp.fc_out.\2",
            r"^double_blocks\.(\d+)\.txt_mlp\.net\.0\.proj\.(.*)$": r"double_blocks.\1.txt_mlp.fc_in.\2",
            r"^double_blocks\.(\d+)\.txt_mlp\.net\.2\.(.*)$": r"double_blocks.\1.txt_mlp.fc_out.\2",
            r"^double_blocks\.(\d+)\.img_attn_qkv\.(.*)$": r"double_blocks.\1.img_attn_qkv.\2",
            r"^double_blocks\.(\d+)\.txt_attn_qkv\.(.*)$": r"double_blocks.\1.txt_attn_qkv.\2",
            r"^double_blocks\.(\d+)\.img_attn_proj\.(.*)$": r"double_blocks.\1.img_attn_proj.\2",
            r"^double_blocks\.(\d+)\.txt_attn_proj\.(.*)$": r"double_blocks.\1.txt_attn_proj.\2",
            r"^double_blocks\.(\d+)\.img_mod\.(.*)$": r"double_blocks.\1.img_mod.\2",
            r"^double_blocks\.(\d+)\.txt_mod\.(.*)$": r"double_blocks.\1.txt_mod.\2",
            r"^double_blocks\.(\d+)\.img_attn_q_norm\.(.*)$": r"double_blocks.\1.img_attn_q_norm.\2",
            r"^double_blocks\.(\d+)\.img_attn_k_norm\.(.*)$": r"double_blocks.\1.img_attn_k_norm.\2",
            r"^double_blocks\.(\d+)\.txt_attn_q_norm\.(.*)$": r"double_blocks.\1.txt_attn_q_norm.\2",
            r"^double_blocks\.(\d+)\.txt_attn_k_norm\.(.*)$": r"double_blocks.\1.txt_attn_k_norm.\2",
        }
    )

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # Model architecture parameters
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    mm_double_blocks_depth: int = 40
    freq_dim: int = 256
    text_states_dim: int = 4096
    mlp_width_ratio: float = 4.0
    rope_theta: int = 10000
    rope_dim_list: list[int] = field(default_factory=lambda: [16, 56, 56])

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class JoyImageDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=JoyImageArchConfig)
    prefix: str = "JoyImage"
