# SPDX-License-Identifier: Apache-2.0

import PIL.Image
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
    apply_first_frame_prefix,
    pixel_to_vlm_image,
    preprocess_condition_image,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context,
    autocast_enabled,
    resolve_precision,
    temporary_module_dtype,
)

logger = init_logger(__name__)


class LingBotVideoImageConditioningStage(ImageVAEEncodingStage):
    """Turn the condition frame into a Qwen3-VL image and a clean latent."""

    deduplicated_output_fields = ()

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is None:
            return batch

        image = batch.condition_image
        if isinstance(image, list):
            image = image[0]
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(
                f"LingBot-Video I2V expects an unprocessed PIL condition image, got {type(image)}."
            )

        pixel = preprocess_condition_image(
            image, int(batch.height), int(batch.width)
        ).to(get_local_torch_device())
        generator = batch.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        batch.extra[VLM_IMAGE_KEY] = pixel_to_vlm_image(pixel)
        batch.extra[COND_LATENT_KEY] = self._encode_cond_latent(
            pixel, server_args, generator
        )
        return batch

    def _encode_cond_latent(
        self,
        pixel: torch.Tensor,
        server_args: ServerArgs,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)
        normalized = (pixel.float() - 0.5) / 0.5

        with self.use_declared_component(
            component_name=self.component_name, module=self.vae
        ) as vae:
            assert vae is not None
            self.vae = vae
            with autocast_context(vae_dtype, server_args.disable_autocast):
                if not vae_autocast_enabled:
                    normalized = normalized.to(vae_dtype)
                with temporary_module_dtype(
                    self.vae, vae_dtype, enabled=not vae_autocast_enabled
                ) as vae:
                    latent_dist = vae.encode(normalized)
                if isinstance(latent_dist, AutoencoderKLOutput):
                    latent_dist = latent_dist.latent_dist

        # The reference samples the posterior rather than taking its mode, which also
        # advances the generator before the noise draw.
        cond_latent = self.retrieve_latents(
            latent_dist, generator, sample_mode="sample"
        ).float()
        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=cond_latent.device, dtype=cond_latent.dtype, vae=self.vae
            )
        )
        return self.scale_and_shift_encode_latents(
            cond_latent, scaling_factor, shift_factor
        )


def apply_cond_latent(batch: Req, latents: torch.Tensor) -> torch.Tensor:
    cond_latent = batch.extra.get(COND_LATENT_KEY)
    if cond_latent is None:
        return latents
    return apply_first_frame_prefix(latents, cond_latent)
