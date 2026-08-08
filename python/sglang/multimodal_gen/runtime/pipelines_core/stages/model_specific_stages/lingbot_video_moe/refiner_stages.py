# SPDX-License-Identifier: Apache-2.0

import copy

import torch
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.denoising import (
    LingBotVideoDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.i2v import (
    COND_LATENT_KEY,
    TEXT_ONLY_EMBEDS_KEY,
    preprocess_condition_image,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.image_conditioning import (
    apply_cond_latent,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.refiner import (
    compute_refiner_sigmas,
    prepare_refiner_latent,
    resize_video_pixels,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context,
    autocast_enabled,
    resolve_decode_precision,
    temporary_module_dtype,
)

logger = init_logger(__name__)


class LingBotVideoRefinerUpscaleStage(DecodingStage):
    """Decode the base pass, resize to the refiner resolution, and re-encode."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        config = server_args.pipeline_config
        height, width = config.refiner_height, config.refiner_width
        # The inherited component_uses declares residency at the decode precision.
        vae_dtype = resolve_decode_precision(server_args, self.component_name)
        generator = _refiner_generator(batch, get_local_torch_device())

        with self.use_declared_component(
            component_name=self.component_name, module=self.vae
        ) as vae:
            self.vae = vae
            # The refiner resolution needs tiling whatever the pipeline-wide setting is.
            was_tiling = self.vae.use_tiling
            self.vae.use_tiling = True
            try:
                # Only the decode half has a fast path; the re-encode has none.
                with use_vae_fast_path(vae, batch.sampling_params.quality == "high"):
                    pixels = self.decode(
                        batch.latents, server_args, vae_dtype=vae_dtype
                    )
                upscaled = resize_video_pixels(pixels, height, width)
                latents = self._encode(
                    upscaled, server_args, vae_dtype=vae_dtype, generator=generator
                )
                if batch.extra.get(COND_LATENT_KEY) is not None:
                    batch.extra[COND_LATENT_KEY] = self._encode_condition_frame(
                        batch, server_args, height, width, vae_dtype, generator
                    )
            finally:
                self.vae.use_tiling = was_tiling

        latents = apply_cond_latent(batch, latents)
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        batch.latents = prepare_refiner_latent(latents, noise, config.refiner_t_thresh)
        batch.height, batch.width = height, width
        return batch

    def _encode(
        self,
        pixels: torch.Tensor,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        normalized = (pixels.to(get_local_torch_device()).float() - 0.5) / 0.5
        vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)
        with autocast_context(vae_dtype, server_args.disable_autocast):
            if not vae_autocast_enabled:
                normalized = normalized.to(vae_dtype)
            with temporary_module_dtype(
                self.vae, vae_dtype, enabled=not vae_autocast_enabled
            ) as vae:
                latent_dist = vae.encode(normalized)

        if isinstance(latent_dist, AutoencoderKLOutput):
            latent_dist = latent_dist.latent_dist
        latents = latent_dist.sample(generator).float()
        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=latents.device, dtype=latents.dtype, vae=self.vae
            )
        )
        if shift_factor is not None:
            latents = latents - shift_factor.to(latents.device)
        return latents * scaling_factor.to(latents.device)

    def _encode_condition_frame(
        self,
        batch: Req,
        server_args: ServerArgs,
        height: int,
        width: int,
        vae_dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        image = batch.condition_image
        if isinstance(image, list):
            image = image[0]
        pixel = preprocess_condition_image(image, height, width)
        latents = self._encode(
            pixel, server_args, vae_dtype=vae_dtype, generator=generator
        )
        return latents[:, :, :1]


class LingBotVideoRefinementStage(LingBotVideoDenoisingStage):
    """Second denoising pass on the refiner weights, from t_thresh down."""

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                stage_name=self._component_stage_name(stage_name),
                component_name="transformer_2",
                phase="refiner",
                memory_intensive=True,
            )
        ]

    def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs):
        config = server_args.pipeline_config
        scheduler = copy.deepcopy(batch.scheduler)
        sigmas = compute_refiner_sigmas(
            sigma_max=scheduler.sigma_max,
            sigma_min=scheduler.sigma_min,
            num_inference_steps=config.refiner_num_inference_steps,
            shift=config.refiner_flow_shift,
            t_thresh=config.refiner_t_thresh,
            tail_steps=config.refiner_sigma_tail_steps,
        )
        # The shift is already baked into the sigmas.
        scheduler.set_timesteps(
            device=get_local_torch_device(), sigmas=sigmas, shift=1.0
        )
        batch.scheduler = scheduler
        batch.timesteps = scheduler.timesteps
        batch.num_inference_steps = len(scheduler.timesteps)
        batch.guidance_scale = config.refiner_guidance_scale
        _use_text_only_conditioning(batch)
        _zero_negative_conditioning(batch)
        return super()._prepare_denoising_loop(batch, server_args)


def _use_text_only_conditioning(batch: Req) -> None:
    text_only = batch.extra.get(TEXT_ONLY_EMBEDS_KEY)
    if text_only is None:
        return
    prompt_embeds, prompt_mask = text_only
    batch.prompt_embeds = [prompt_embeds]
    batch.prompt_attention_mask = prompt_mask


def _zero_negative_conditioning(batch: Req) -> None:
    # The refiner conditions on zeros rather than the negative prompt.
    if not batch.do_classifier_free_guidance or batch.prompt_embeds is None:
        return
    batch.negative_prompt_embeds = [torch.zeros_like(batch.prompt_embeds[0])]
    batch.negative_attention_mask = batch.prompt_attention_mask.clone()


def _refiner_generator(batch: Req, device: torch.device) -> torch.Generator | None:
    # The reference reseeds for the second pass rather than continuing the base stream.
    if not batch.seeds:
        return None
    return torch.Generator(device=device).manual_seed(int(batch.seeds[0]))
