# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_blocks_or_transformer_blocks


@dataclass
class SanaWMArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks_or_transformer_blocks])

    # --- Core dims (upstream: depth=20, hidden=2240, heads=20, linear_head_dim=112) ---
    patch_size: int = 1
    in_channels: int = 128         # LTX-2 VAE latent channels
    out_channels: int = 128
    num_layers: int = 20

    # Patch embedder uses (1, patch_size, patch_size) — temporal patch is always 1.
    patch_size_t: int = 1

    num_attention_heads: int = 20
    attention_head_dim: int = 112  # = linear_head_dim
    linear_head_dim: int = 112

    # --- Cross-attention (text conditioning) ---
    # In upstream, cross-attn uses num_heads (=20) with head_dim = hidden/num_heads = 112.
    num_cross_attention_heads: int = 20
    cross_attention_head_dim: int = 112
    cross_attention_dim: int = 2240   # query dim used inside MultiHeadCrossAttention
    cross_norm: bool = True

    # Gemma-2-2b-it hidden size (input to y_embedder.y_proj).
    caption_channels: int = 2304
    model_max_length: int = 300
    y_norm: bool = True
    y_norm_scale_factor: float = 0.01
    y_norm_eps: float = 1e-5

    mlp_ratio: float = 3.0
    qk_norm: bool = True
    norm_eps: float = 1e-6
    timestep_norm_scale_factor: float = 1.0

    # --- Hybrid GDN/Softmax attention ---
    # softmax_every_n=4 => blocks where (i+1)%4 == 0 use softmax main branch,
    # i.e. block indices {3, 7, 11, 15, 19}.
    softmax_every_n: int = 4

    # --- GDN ShortConvolution params ---
    conv_kernel_size: int = 4
    k_conv_only: bool = True
    chunk_gdn_chunk_size: int = 21
    update_rule: str = "torch_chunk"   # main branch update rule
    cam_update_rule: str = "torch_chunk"  # camera branch update rule
    # main GDN scan backend: "auto" uses the SANA-WM Triton fast path on
    # supported CUDA inference runs, otherwise falls back to the torch scan.
    gdn_backend: str = "auto"

    # --- Camera conditioning ---
    cam_attn_compress: int = 1        # cam_dim == in_dim
    init_cam_from_base: bool = True
    use_chunk_plucker_post_attn: bool = True
    use_chunk_plucker_input: bool = False
    chunk_plucker_channels: int = 48  # 8 orig frames × 6D Plücker
    chunk_plucker_post_attn_blocks: int = 20

    chunk_split_strategy: str = "first_chunk_plus_one"
    chunk_size: int = 10
    # Upstream currently forwards chunk metadata through the softmax blocks but
    # does not apply a chunk-causal mask there. Keep this disabled by default
    # for checkpoint-output parity; it can be enabled for experiments.
    use_chunked_softmax_attention: bool = False

    # --- Temporal FFN (GLUMBConvTemp) ---
    ffn_type: str = "GLUMBConvTemp"
    t_kernel_size: int = 3
    mlp_acts: tuple = field(default_factory=lambda: ("silu", "silu", None))

    # --- Position embedding ---
    pos_embed_type: str = "wan_rope"

    # --- VAE coupling (LTX-2) ---
    vae_temporal_stride: int = 8       # original-frames per latent frame
    vae_spatial_stride: int = 32       # pixels per latent token (per spatial axis)

    sample_size: int = 32              # legacy, unused
    guidance_embeds: bool = False
    class_dropout_prob: float = 0.0

    # The released checkpoint stores raw upstream parameter names (no leading
    # "transformer." prefix), so we use an identity mapping.
    param_names_mapping: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class SanaWMConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SanaWMArchConfig)
    prefix: str = "SanaWM"
