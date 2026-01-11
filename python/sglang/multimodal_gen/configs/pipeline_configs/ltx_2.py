import dataclasses
from typing import Tuple
from dataclasses import field

import torch
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config


@dataclasses.dataclass
class LTX2PipelineConfig(PipelineConfig):
    """Configuration for LTX-Video pipeline."""
    
    dit_config: LTX2Config = field(default_factory=LTX2Config)

    # Model architecture
    in_channels: int = 128
    out_channels: int = 128
    patch_size: int = 1
    patch_size_t: int = 1
    
    # Audio specific
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_sample_rate: int = 24000  # Default for LTX-2

    # Audio latent spec (matches LTX-2 audio patchifier: C * mel_bins == audio_in_channels)
    audio_latent_channels: int = 8
    audio_latent_mel_bins: int = 16
    audio_latent_downsample_factor: int = 4
    audio_hop_length: int = 160
    
    # VAE
    vae_scale_factor: int = 32  # Spatial compression
    vae_temporal_compression: int = 8
    
    # Text Encoder
    text_encoder_model_path: str = "google/gemma-3-12b-it"  # Default placeholder
    
    def __post_init__(self):
        super().__post_init__()
        self.task_type = ModelTaskType.T2V

    def get_pos_prompt_embeds(self, batch):
        # LTX-2 uses a single text encoder in our pipeline.
        return batch.prompt_embeds[0] if isinstance(batch.prompt_embeds, list) else batch.prompt_embeds

    def get_neg_prompt_embeds(self, batch):
        return (
            batch.negative_prompt_embeds[0]
            if isinstance(batch.negative_prompt_embeds, list)
            else batch.negative_prompt_embeds
        )
