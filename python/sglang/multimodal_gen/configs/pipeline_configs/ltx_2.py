import dataclasses
from dataclasses import field
from typing import Callable

import torch
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.gemma import GemmaConfig
from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config


def _gemma_postprocess_video(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    # Support both ltx-core style and transformers/diffusers style outputs.
    for key in ("video_context", "video_encoding"):
        if hasattr(outputs, key):
            return getattr(outputs, key)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    raise AttributeError(
        "Unsupported text encoder output: expected `video_context`/`video_encoding` or `last_hidden_state`."
    )


def _gemma_postprocess_audio(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    for key in ("audio_context", "audio_encoding"):
        if hasattr(outputs, key):
            return getattr(outputs, key)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    raise AttributeError(
        "Unsupported text encoder output: expected `audio_context`/`audio_encoding` or `last_hidden_state`."
    )


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
    
    # Text encoding stage (Gemma)
    # LTX-2 needs separate contexts for video/audio streams. We model this as
    # two logical encoders sharing the same underlying `text_encoder` module.
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (GemmaConfig(), GemmaConfig())
    )
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput, dict], torch.Tensor], ...
    ] = field(default_factory=lambda: (_gemma_postprocess_video, _gemma_postprocess_audio))
    
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
