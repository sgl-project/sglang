# SPDX-License-Identifier: Apache-2.0
"""DiT config for NVIDIA OmniDreams (Cosmos-Predict2.5-2B based autoregressive
video world model; production runtime = FlashDreams).

Architecture facts mirror FlashDreams ``CosmosDiTNetworkConfig`` for the
``2b_res720p_30fps_i2v_hdmap_distilled`` checkpoint (HDMap single-view variant):
``additional_concat_ch=16`` enables HDMap conditioning, cross-view attention is
off. The flat checkpoint key names match the submodule tree one-to-one, so
``param_names_mapping`` is the identity (empty dict).
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class OmniDreamsDiTArchConfig(DiTArchConfig):
    # --- Cosmos DiT architecture (FlashDreams CosmosDiTNetworkConfig) ---
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    concat_padding_mask: bool = True
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 256
    use_crossattn_projection: bool = True
    crossattn_proj_in_channels: int = 100352
    crossattn_emb_channels: int = 1024
    timestep_scale: float = 0.001
    # HDMap variant: 16 extra latent channels routed through additional_patch_embedding.
    # Overrides the FlashDreams CosmosDiTNetworkConfig default of 0 (HDMap disabled).
    additional_concat_ch: int = 16
    # Cross-view attention is disabled for the single-view checkpoint.
    enable_cross_view_attn: bool = False
    view_condition_dim: int = 16
    n_cameras_emb: int = 7

    # Checkpoint keys equal submodule names -> identity mappings.
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        # BaseDiT-required instance attrs (also surfaced via ModelConfig.__getattr__).
        self.hidden_size = self.model_channels
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.out_channels

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads


@dataclass
class OmniDreamsDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=OmniDreamsDiTArchConfig)
    prefix: str = "OmniDreams"
