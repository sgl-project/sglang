# SPDX-License-Identifier: Apache-2.0
"""
LongCat Video DiT configuration for native FastVideo implementation.
"""

from dataclasses import dataclass, field
from typing import Optional

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def is_longcat_blocks(n: str, m) -> bool:
    """FSDP shard condition for LongCat transformer blocks."""
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class LongCatVideoArchConfig(DiTArchConfig):
    """Architecture configuration for native LongCat Video DiT."""

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_longcat_blocks])

    # Enable torch.compile for transformer blocks (major speedup!)
    _compile_conditions: list = field(
        default_factory=lambda: [is_longcat_blocks])

    # Parameter name mapping for weight conversion
    # Maps original LongCat third_party names -> native FastVideo names
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Embedders
            r"^x_embedder\.(.*)$": r"patch_embed.\1",
            r"^t_embedder\.mlp\.0\.(.*)$": r"time_embedder.linear_1.\1",
            r"^t_embedder\.mlp\.2\.(.*)$": r"time_embedder.linear_2.\1",
            r"^y_embedder\.y_proj\.0\.(.*)$": r"caption_embedder.linear_1.\1",
            r"^y_embedder\.y_proj\.2\.(.*)$": r"caption_embedder.linear_2.\1",

            # Transformer blocks - AdaLN modulation
            r"^blocks\.(\d+)\.adaLN_modulation\.1\.(.*)$":
            r"blocks.\1.adaln_linear_1.\2",

            # Transformer blocks - Normalization
            r"^blocks\.(\d+)\.mod_norm_attn\.(.*)$": r"blocks.\1.norm_attn.\2",
            r"^blocks\.(\d+)\.mod_norm_ffn\.(.*)$": r"blocks.\1.norm_ffn.\2",
            r"^blocks\.(\d+)\.pre_crs_attn_norm\.(.*)$":
            r"blocks.\1.norm_cross.\2",

            # Self-attention: QKV fused -> separate (will need splitting in converter)
            # Original has attn.qkv.weight -> need to split into to_q, to_k, to_v
            r"^blocks\.(\d+)\.attn\.qkv\.(.*)$":
            r"blocks.\1.self_attn.qkv_fused.\2",  # Marker for splitting
            r"^blocks\.(\d+)\.attn\.proj\.(.*)$":
            r"blocks.\1.self_attn.to_out.\2",
            r"^blocks\.(\d+)\.attn\.q_norm\.(.*)$":
            r"blocks.\1.self_attn.q_norm.\2",
            r"^blocks\.(\d+)\.attn\.k_norm\.(.*)$":
            r"blocks.\1.self_attn.k_norm.\2",

            # Cross-attention
            r"^blocks\.(\d+)\.cross_attn\.q_linear\.(.*)$":
            r"blocks.\1.cross_attn.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.kv_linear\.(.*)$":
            r"blocks.\1.cross_attn.kv_fused.\2",  # Marker for splitting
            r"^blocks\.(\d+)\.cross_attn\.proj\.(.*)$":
            r"blocks.\1.cross_attn.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.q_norm\.(.*)$":
            r"blocks.\1.cross_attn.q_norm.\2",
            r"^blocks\.(\d+)\.cross_attn\.k_norm\.(.*)$":
            r"blocks.\1.cross_attn.k_norm.\2",

            # FFN (SwiGLU)
            r"^blocks\.(\d+)\.ffn\.w1\.(.*)$": r"blocks.\1.ffn.w1.\2",  # gate
            r"^blocks\.(\d+)\.ffn\.w2\.(.*)$": r"blocks.\1.ffn.w2.\2",  # down
            r"^blocks\.(\d+)\.ffn\.w3\.(.*)$": r"blocks.\1.ffn.w3.\2",  # up

            # Final layer
            r"^final_layer\.adaLN_modulation\.1\.(.*)$":
            r"final_layer.adaln_linear.\1",
            r"^final_layer\.norm_final\.(.*)$": r"final_layer.norm.\1",
            r"^final_layer\.linear\.(.*)$": r"final_layer.proj.\1",
        })

    # Reverse mapping for saving checkpoints: custom -> hf
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # LoRA parameter name mapping
    lora_param_names_mapping: dict = field(default_factory=lambda: {})

    # Model architecture parameters
    hidden_size: int = 4096
    depth: int = 48  # Number of transformer blocks
    num_attention_heads: int = 32
    attention_head_dim: int = 128  # hidden_size / num_attention_heads

    in_channels: int = 16  # Latent space channels
    out_channels: int = 16
    num_channels_latents: int = 16

    # Patch embedding
    patch_size: tuple[int, int,
                      int] = (1, 2, 2)  # [T, H, W] - no temporal compression

    # Text/caption embedding
    caption_channels: int = 4096  # UMT5 d_model

    # Timestep embedding
    adaln_tembed_dim: int = 512
    frequency_embedding_size: int = 256

    # FFN
    mlp_ratio: int = 4

    # Attention backend support
    _supported_attention_backends: tuple = field(default_factory=lambda: (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    ))

    # Text padding behavior
    text_tokens_zero_pad: bool = True

    # Block Sparse Attention (BSA)
    enable_bsa: bool = False
    bsa_params: Optional[dict] = field(
        default_factory=lambda: {
            "sparsity": 0.9375,
            "cdf_threshold": None,
            "chunk_3d_shape_q": [4, 4, 4],
            "chunk_3d_shape_k": [4, 4, 4],
        })

    # LoRA exclusions
    exclude_lora_layers: list[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        # Ensure attention_head_dim matches
        self.attention_head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class LongCatVideoConfig(DiTConfig):
    """Main configuration for LongCat Video DiT."""

    arch_config: DiTArchConfig = field(default_factory=LongCatVideoArchConfig)

    prefix: str = "longcat"
