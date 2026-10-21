# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import QwenImage21DitConfig
from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import QwenImage21VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class QwenImage21PipelineConfig(ImagePipelineConfig):
    native_only_components: tuple[str, ...] = ("transformer", "text_encoder", "vae")
    task_type: ModelTaskType = ModelTaskType.TI2I
    should_use_guidance: bool = False
    enable_autocast: bool = False
    # Avoid high-resolution full-frame decode hangs on gfx1151 by default.
    # Explicit CLI, config-file, and Python API values still take precedence.
    vae_tiling: bool = field(default_factory=lambda: current_platform.is_gfx1151())
    vae_sp: bool = False
    vae_precision: str = "bf16"
    generator_device: str = "cpu"
    dit_config: QwenImage21DitConfig = field(default_factory=QwenImage21DitConfig)
    vae_config: QwenImage21VAEConfig = field(default_factory=QwenImage21VAEConfig)
    text_encoder_configs: tuple = field(default_factory=lambda: (Qwen3VLConfig(),))
    text_encoder_precisions: tuple[str, ...] = ("bf16",)
    sample_sigmas: list[float] | None = None

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        if not self.vae_tiling and current_platform.is_gfx1151():
            logger.warning(
                "VAE tiling is disabled for Qwen-Image 2.1 on gfx1151. "
                "Full-frame decoding may hang at resolutions of 896px or higher; "
                "use --vae-tiling true to enable tiling."
            )

    def supports_dynamic_batching(self):
        # the scheduler excludes reference-image requests from cross-request merging
        return True

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is None:
            sigmas = self.sample_sigmas
        return list(self._prepare_sigmas(sigmas, num_inference_steps))

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
        # decode consumes only target latents, not the condition prefix or its KV cache
        batch.extra.pop("qwen21_positive", None)
        batch.extra.pop("qwen21_negative", None)
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


def register():
    from sglang.multimodal_gen.configs.sample.qwenimage21 import (
        QwenImage21SamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=QwenImage21SamplingParams,
        pipeline_config_cls=QwenImage21PipelineConfig,
        hf_model_paths=["Qwen/Qwen-Image-2.1", "Qwen/Qwen-Image-2.1-Turbo"],
        model_detectors=[lambda hf_id: "qwen-image-2.1" in hf_id.lower()],
    )
