# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.dits.ming_image import MingImageDitConfig
from sglang.multimodal_gen.configs.models.encoders.ming_image import (
    MingImageEncoderConfig,
)
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


@dataclass
class MingImagePipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2I
    dit_config: MingImageDitConfig = field(default_factory=MingImageDitConfig)
    vae_config: QwenImageVAEConfig = field(default_factory=QwenImageVAEConfig)
    text_encoder_configs: tuple = field(
        default_factory=lambda: (MingImageEncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = ("bf16",)
    native_only_components: tuple[str, ...] = ("text_encoder", "transformer", "vae")
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False
    enable_autocast: bool = False
    generator_device: str = "cpu"

    def supports_dynamic_batching(self):
        # Reference/query sequences have different lengths across requests.
        return False

    def supports_sequential_multi_output_inference(self):
        return True

    def get_latent_dtype(self, prompt_dtype):
        return torch.float32

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        # Draw frames before channels, as in the official CPU RNG stream.
        return (
            batch_size,
            batch.extra["ming_frames"],
            16,
            batch.height // 8,
            batch.width // 8,
        )

    def maybe_pack_latents(self, latents, batch_size, batch):
        return latents.transpose(1, 2).contiguous()

    def shard_latents_for_sp(self, batch, latents):
        # Ming shards patchified tokens inside the DiT, not spatial pixels here.
        return latents, False

    def gather_latents_for_sp(self, latents, batch=None):
        return latents

    def gather_noise_pred_for_sp(self, batch, noise_pred):
        return noise_pred

    def post_denoising_loop(self, latents, batch):
        return latents

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "direct_embeddings": batch.extra["ming_direct"],
            "reference_latents": batch.extra.get("ming_reference_latents"),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "direct_embeddings": torch.zeros_like(batch.extra["ming_direct"]),
            "reference_latents": batch.extra.get("ming_reference_latents"),
        }

    def get_classifier_free_guidance_scale(self, batch, guidance_scale):
        # Official Ming uses cond + cfg * (cond - uncond).
        return guidance_scale + 1.0


@dataclass
class MingImageLayerPipelineConfig(MingImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2I


def register():
    from sglang.multimodal_gen.configs.sample.ming_image import (
        MingImageLayerSamplingParams,
        MingImageSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=MingImageLayerSamplingParams,
        pipeline_config_cls=MingImageLayerPipelineConfig,
        hf_model_paths=["inclusionAI/Ming-Image-0.1-Design-Layer"],
        model_detectors=[lambda model: "ming-image-0.1-design-layer" in model.lower()],
    )
    register_configs(
        sampling_param_cls=MingImageSamplingParams,
        pipeline_config_cls=MingImagePipelineConfig,
        hf_model_paths=["inclusionAI/Ming-Image-0.1-Design"],
        model_detectors=[
            lambda model: (
                "ming-image-0.1-design" in model.lower()
                and "design-layer" not in model.lower()
            )
        ],
    )
