from dataclasses import dataclass, field
from typing import List

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class HiDreamArchConfig(DiTArchConfig):
    """Architecture config for HiDream-O1-Image (modified Qwen3VL with diffusion heads).

    Fields are populated from the model's config.json (text_config + vision_config)
    plus diffusion-specific parameters.
    """

    # Text decoder (from config.json text_config)
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    attention_bias: bool = False
    vocab_size: int = 151936
    max_position_embeddings: int = 262144

    # Vision encoder (from config.json vision_config)
    vision_hidden_size: int = 1152
    vision_depth: int = 27
    vision_num_heads: int = 16
    vision_intermediate_size: int = 4304
    vision_patch_size: int = 16
    vision_temporal_patch_size: int = 2
    vision_spatial_merge_size: int = 2
    vision_in_channels: int = 3
    vision_out_hidden_size: int = 4096
    vision_num_position_embeddings: int = 2304
    deepstack_visual_indexes: List[int] = field(
        default_factory=lambda: [8, 16, 24]
    )

    # Diffusion-specific (from model code constants)
    dit_patch_size: int = 32
    in_channels: int = 3
    bottleneck_dim: int = 1024  # hidden_size // 4
    frequency_embedding_size: int = 256

    # Special token IDs (from config.json)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tms_token_id: int = 151673

    num_channels_latents: int = 3  # direct patch output, no VAE

    param_names_mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Ensure derived fields are consistent
        if self.bottleneck_dim == 0:
            self.bottleneck_dim = self.hidden_size // 4


@dataclass
class HiDreamDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=HiDreamArchConfig)
    prefix: str = "hidream"
