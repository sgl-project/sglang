# SPDX-License-Identifier: Apache-2.0
#
# Architecture and model configuration for SANA-WM (World Model) DiT.
#
# SANA-WM extends the SANA T2I architecture into a 2.6B TI2V world model that:
#   - Operates on 5D video latents (B, C, T, H, W) via LTX-2 VAE (8x temporal, 32x spatial)
#   - Uses a Hybrid GDN/Softmax attention scheme:
#       * 15 "GDN" blocks: Frame-wise linear recurrent (Gated Delta Network) attention
#         with state matrix O(D^2) per head — memory-constant in frame count
#       * 5 "Softmax" blocks at positions {3, 7, 11, 15, 19}: standard full softmax
#   - Adds dual-branch 6-DoF camera conditioning:
#       * Coarse: Camera branch with UCPE (Ray-Local Unified Camera Positional Encoding)
#       * Fine: Post-attention Plücker Raymap Mixing (48-channel, frame-packed)
#   - Temporal FFN (GLUMBConvTemp) with temporal kernel_size=3
#   - wan_rope 3D positional encoding (spatial+temporal)
#
# Reference: https://arxiv.org/abs/2605.15178
# Checkpoint: Efficient-Large-Model/SANA-WM_bidirectional

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_blocks_or_transformer_blocks


@dataclass
class SanaWMArchConfig(DiTArchConfig):
    # FSDP sharding at block boundaries (same as Wan, HunyuanVideo, etc.)
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks_or_transformer_blocks])

    # Core dimensions (match config.yaml: d_model=2240, 20 heads, head_dim=112)
    patch_size: int = 1
    patch_size_t: int = 1          # temporal patch size (1 = no temporal downsampling at DiT level)
    in_channels: int = 128         # LTX-2 VAE latent channels
    out_channels: int = 128
    num_layers: int = 20

    # Self-attention heads
    num_attention_heads: int = 20
    attention_head_dim: int = 112  # head_dim (d_model / num_heads = 2240 / 20 = 112)

    # Cross-attention for text conditioning (Gemma-2)
    num_cross_attention_heads: int = 20
    cross_attention_head_dim: int = 112
    cross_attention_dim: int = 2240   # Gemma-2-2b hidden dim is 2304; projected to 2240

    # Caption (text encoder output) channels
    caption_channels: int = 2304   # Gemma-2-2b-it hidden dim

    mlp_ratio: float = 3.0         # from config.yaml mlp_ratio: 3

    # QK normalization (from config.yaml: qk_norm: true)
    qk_norm: bool = True

    norm_eps: float = 1e-6
    sample_size: int = 32          # not used for video, kept for compat
    guidance_embeds: bool = False

    # --- Hybrid GDN/Softmax attention ---
    # Blocks at these 0-based indices use full Softmax attention; all others use GDN.
    softmax_block_indices: tuple = field(default_factory=lambda: (3, 7, 11, 15, 19))

    # --- GDN attention hyper-params ---
    # Linear head dim used in GDN (from config.yaml: linear_head_dim: 112)
    gdn_linear_head_dim: int = 112
    # cam_attn_compress: compression factor for camera branch GDN keys (1 = no compression)
    cam_attn_compress: int = 1
    # Whether to use bidirectional GDN scan (checkpoint is _bidirectional)
    gdn_bidirectional: bool = True

    # --- Camera conditioning ---
    # Plücker raymap channels: VAE temporal stride=8 orig frames × 6 Plücker dims = 48
    chunk_plucker_channels: int = 48    # 8 original frames × 6D Plücker = 48 channels
    chunk_plucker_post_attn_blocks: int = 20   # all blocks receive Plücker mixing

    # --- Temporal FFN (GLUMBConvTemp) ---
    t_kernel_size: int = 3         # temporal conv kernel size in GLUMBConvTemp

    # --- Position embedding ---
    pos_embed_type: str = "wan_rope"  # 3D RoPE (spatial+temporal)

    # Weight loading: map HF "transformer." prefix → model weights
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^transformer\.(.*)$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class SanaWMConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SanaWMArchConfig)
    prefix: str = "SanaWM"
