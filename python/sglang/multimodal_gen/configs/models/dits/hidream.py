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

    # Vision encoder: the SRT Qwen3VLMoeVisionModel reads config.json's
    # vision_config directly; only the merge size is consumed by the pipeline.
    vision_spatial_merge_size: int = 2

    # Diffusion-specific (from model code constants)
    dit_patch_size: int = 32
    in_channels: int = 3
    bottleneck_dim: int = 1024  # hidden_size // 4
    frequency_embedding_size: int = 256

    # Special token IDs (from config.json)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
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
