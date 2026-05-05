# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class HeliosArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Patch embeddings
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            # Condition embedder: text
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            # Condition embedder: time
            r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^condition_embedder\.time_proj\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            # Blocks: self-attention (keep attn1. prefix, drop .0. from to_out)
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.attn1.to_out.\2",
            # Blocks: cross-attention output (drop .0. from to_out)
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            # Blocks: feed-forward
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            # Blocks: cross-attn residual norm
            r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.self_attn_residual_norm.\2",
        }
    )

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    lora_param_names_mapping: dict = field(default_factory=lambda: {})

    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 226
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    # Helios-specific
    rope_dim: tuple[int, int, int] = (44, 42, 42)
    rope_theta: float = 10000.0
    guidance_cross_attn: bool = True
    zero_history_timestep: bool = True
    has_multi_term_memory_patch: bool = True
    is_amplify_history: bool = False
    history_scale_mode: str = "per_head"

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class HeliosConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=HeliosArchConfig)

    prefix: str = "Helios"
