# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for SANA-Video text-to-video generation."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.sana_video import SanaVideoConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def sana_video_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class SanaVideoPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2V
    should_use_guidance: bool = False
    flow_shift: float | None = 8.0
    # Linear attention deliberately accumulates its score products in FP32.
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=SanaVideoConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False
    vae_precision: str = "fp32"
    vae_decode_precision: str = "fp32"

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            {
                "padding": "max_length",
                "return_attention_mask": True,
                "add_special_tokens": True,
            }
        ]
    )
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (None,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (sana_video_postprocess_text,)
    )

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def adjust_num_frames(self, num_frames: int) -> int:
        temporal_scale = self.vae_config.arch_config.temporal_compression_ratio
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        return ((num_frames - 1) // temporal_scale) * temporal_scale + 1

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        spatial_scale = self.vae_config.arch_config.spatial_compression_ratio
        return (
            batch_size,
            self.dit_config.arch_config.num_channels_latents,
            num_frames,
            batch.height // spatial_scale,
            batch.width // spatial_scale,
        )

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        return torch.float32

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    @staticmethod
    def _unwrap_attention_mask(mask):
        if isinstance(mask, (list, tuple)):
            return mask[0] if mask else None
        return mask

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "encoder_attention_mask": self._unwrap_attention_mask(
                batch.prompt_attention_mask
            )
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "encoder_attention_mask": self._unwrap_attention_mask(
                batch.negative_attention_mask
            )
        }

    def post_denoising_loop(self, latents, batch):
        return latents

    def shard_latents_for_sp(self, batch, latents):
        return latents, False

    def gather_latents_for_sp(self, latents, batch=None):
        return latents
