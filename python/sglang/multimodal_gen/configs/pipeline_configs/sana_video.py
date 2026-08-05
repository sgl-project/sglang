# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for SANA-Video 2B 480p text-to-video."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.sana_video import SanaVideoConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def sana_video_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> torch.Tensor:
    """Use Gemma2's final hidden state as SANA-Video text conditioning."""

    return outputs.last_hidden_state


@dataclass
class SanaVideoPipelineConfig(PipelineConfig):
    """Configuration for the official 480p checkpoint with WanVAE."""

    task_type: ModelTaskType = ModelTaskType.T2V
    should_use_guidance: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=SanaVideoConfig)
    dit_precision: str = "bf16"

    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False
    vae_precision: str = "fp32"

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [{"padding": True, "return_attention_mask": True}]
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
        del device, rotary_emb, dtype
        return {
            "encoder_attention_mask": self._unwrap_attention_mask(
                batch.prompt_attention_mask
            )
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del device, rotary_emb, dtype
        return {
            "encoder_attention_mask": self._unwrap_attention_mask(
                batch.negative_attention_mask
            )
        }

    def shard_latents_for_sp(self, batch, latents):
        # SANA-Video does not yet implement sequence-parallel attention/RoPE.
        return latents, False


@dataclass
class SanaVideoOptimizedPipelineConfig(SanaVideoPipelineConfig):
    """Opt-in Sol-Engine profile.

    Packed QKV/KV projections are always enabled because they are lossless.
    This profile additionally enables BF16 linear-attention aggregation and
    prepares the model for request-scoped EasyCache. Pass
    ``enable_torch_compile=True`` to the engine to complete the acceleration
    stack.
    """

    dit_config: DiTConfig = field(
        default_factory=lambda: SanaVideoConfig(
            enable_easycache=True,
            linear_attention_aggregation_precision="bf16",
        )
    )
