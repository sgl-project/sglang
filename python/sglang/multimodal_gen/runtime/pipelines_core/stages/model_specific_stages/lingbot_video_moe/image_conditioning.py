# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.i2v import (
    COND_LATENT_KEY,
    VLM_IMAGE_KEY,
    condition_frame,
    pixel_to_vlm_image,
    preprocess_condition_image,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context,
    autocast_enabled,
    resolve_precision,
    temporary_module_dtype,
)


class LingBotVideoImageConditioningStage(ImageVAEEncodingStage):
    """Turn the condition frame into a Qwen3-VL image and a clean latent."""

    # Outputs live in batch.extra and the posterior is sampled, so no cross-request reuse.
    deduplicated_output_fields = ()

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is None:
            return batch

        pixel = preprocess_condition_image(
            condition_frame(batch.condition_image),
            height=int(batch.height),
            width=int(batch.width),
        )
        generator = batch.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        batch.extra[VLM_IMAGE_KEY] = pixel_to_vlm_image(pixel)
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        with self.use_declared_component(
            component_name=self.component_name, module=self.vae
        ) as vae:
            self.vae = vae
            batch.extra[COND_LATENT_KEY] = encode_pixels(
                vae, pixel, server_args, vae_dtype, generator
            )
        return batch


def encode_pixels(vae, pixels, server_args, vae_dtype, generator):
    normalized = (pixels.to(get_local_torch_device()).float() - 0.5) / 0.5
    cast = not autocast_enabled(vae_dtype, server_args.disable_autocast)
    with autocast_context(vae_dtype, server_args.disable_autocast):
        with temporary_module_dtype(vae, vae_dtype, enabled=cast) as encoder:
            posterior = encoder.encode(normalized.to(vae_dtype) if cast else normalized)
    if isinstance(posterior, AutoencoderKLOutput):
        posterior = posterior.latent_dist
    # Sample before drawing denoising noise, matching the reference RNG order.
    latents = posterior.sample(generator).float()
    scale, shift = server_args.pipeline_config.get_decode_scale_and_shift(
        device=latents.device, dtype=latents.dtype, vae=vae
    )
    return ImageVAEEncodingStage.scale_and_shift_encode_latents(latents, scale, shift)


def apply_cond_latent(batch: Req, latents: torch.Tensor) -> torch.Tensor:
    cond_latent = batch.extra.get(COND_LATENT_KEY)
    if cond_latent is None:
        return latents
    return torch.cat(
        [cond_latent.to(latents.dtype), latents[:, :, cond_latent.shape[2] :]], dim=2
    )
