# Copied and adapted from: mossVG/mova/diffusion/models/wan_video_dit.py
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class MovaVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_blocks])

    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    dim: int = 5120
    in_dim: int = 16
    ffn_dim: int = 13824
    out_dim: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_heads: int = 40
    num_layers: int = 40
    has_image_input: bool = True
    has_image_pos_emb: bool = False
    has_ref_conv: bool = False
    add_control_adapter: bool = False
    in_dim_control_adapter: int = 24
    seperated_timestep: bool = False
    require_vae_embedding: bool = True
    require_clip_embedding: bool = True
    fuse_vae_embedding_in_latents: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.dim
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.out_dim


@dataclass
class MovaVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=MovaVideoArchConfig)
    prefix: str = "mova_video"
