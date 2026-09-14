# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import QwenImage21DitConfig
from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import QwenImage21VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


@dataclass
class QwenImage21PipelineConfig(ImagePipelineConfig):
    native_only_components: tuple[str, ...] = ("transformer", "text_encoder", "vae")
    task_type: ModelTaskType = ModelTaskType.TI2I
    should_use_guidance: bool = False
    enable_autocast: bool = False
    vae_tiling: bool = False
    vae_precision: str = "bf16"
    generator_device: str = "cpu"
    dit_config: QwenImage21DitConfig = field(default_factory=QwenImage21DitConfig)
    vae_config: QwenImage21VAEConfig = field(default_factory=QwenImage21VAEConfig)
    text_encoder_configs: tuple = field(default_factory=lambda: (Qwen3VLConfig(),))
    text_encoder_precisions: tuple[str, ...] = ("bf16",)

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def get_classifier_free_guidance_scale(self, batch, guidance_scale):
        return (
            batch.true_cfg_scale if batch.true_cfg_scale is not None else guidance_scale
        )

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        return (
            batch_size,
            1,
            self.dit_config.in_channels,
            batch.height // 16,
            batch.width // 16,
        )

    def maybe_pack_latents(self, latents, batch_size, batch):
        return latents.reshape(batch_size, self.dit_config.in_channels, -1).transpose(
            1, 2
        )

    def shard_latents_for_sp(self, batch, latents):
        # the DiT shards only the target stream; its condition prefix stays replicated
        return latents, False

    def gather_latents_for_sp(self, latents, batch=None):
        return latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return batch.extra["qwen21_positive"]

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype=None):
        return batch.extra["qwen21_negative"]

    def post_denoising_loop(self, latents, batch):
        return latents.transpose(1, 2).reshape(
            latents.shape[0], -1, 1, batch.height // 16, batch.width // 16
        )

    def get_decode_scale_and_shift(self, device, dtype, vae):
        ac = self.vae_config.arch_config
        mean = torch.tensor(ac.latents_mean, device=device, dtype=dtype).view(
            1, ac.z_dim, 1, 1, 1
        )
        std = torch.tensor(ac.latents_std, device=device, dtype=dtype).view(
            1, ac.z_dim, 1, 1, 1
        )
        return std.reciprocal(), mean

    def preprocess_condition_image(self, image, **kwargs):
        return image
