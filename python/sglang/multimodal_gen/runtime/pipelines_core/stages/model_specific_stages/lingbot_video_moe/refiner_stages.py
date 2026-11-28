# SPDX-License-Identifier: Apache-2.0

import copy

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.sample.sampling_params import quality_allows
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.denoising import (
    LingBotVideoDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.i2v import (
    COND_LATENT_KEY,
    TEXT_ONLY_EMBEDS_KEY,
    condition_frame,
    preprocess_condition_image,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.image_conditioning import (
    apply_cond_latent,
    encode_pixels,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.refiner import (
    compute_refiner_sigmas,
    prepare_refiner_latent,
    resize_video_pixels,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import (
    resolve_decode_precision,
)


class LingBotVideoRefinerUpscaleStage(DecodingStage):
    """Decode the base pass, resize to the refiner resolution, and re-encode."""

    @property
    def parallelism_type(self) -> StageParallelismType:
        # Every rank denoises the refiner latents, so none may skip the re-encode.
        return StageParallelismType.REPLICATED

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        config = server_args.pipeline_config
        height, width = config.refiner_height, config.refiner_width
        # The inherited component_uses declares residency at the default precision.
        quality = batch.sampling_params.quality
        vae_dtype = resolve_decode_precision(
            server_args, self.component_name, quality=quality
        )
        generators = _refiner_generators(batch, device=get_local_torch_device())

        with self.use_declared_component(
            component_name=self.component_name, module=self.vae
        ) as vae:
            self.vae = vae
            # The refiner resolution needs tiling whatever the pipeline-wide setting is.
            was_tiling = self.vae.use_tiling
            self.vae.use_tiling = True
            try:
                # Only the decode half has a fast path; the re-encode has none.
                with use_vae_fast_path(vae, quality_allows(quality, "lossless")):
                    pixels = self.decode(
                        batch.latents, server_args, vae_dtype=vae_dtype
                    )

                def upscale_and_encode(pixels):
                    pixels = resize_video_pixels(pixels, height, width)
                    return encode_pixels(
                        vae, pixels, server_args, vae_dtype, generators
                    )

                latents = upscale_and_encode(pixels)
                if batch.extra.get(COND_LATENT_KEY) is not None:
                    # Match the base crop before upscaling the reference with the clip.
                    frame = preprocess_condition_image(
                        condition_frame(batch.condition_image),
                        batch.height,
                        batch.width,
                    )
                    batch.extra[COND_LATENT_KEY] = upscale_and_encode(frame)[:, :, :1]
            finally:
                self.vae.use_tiling = was_tiling

        latents = apply_cond_latent(batch, latents)
        noise = randn_tensor(
            latents.shape,
            generator=generators,
            device=latents.device,
            dtype=latents.dtype,
        )
        batch.latents = prepare_refiner_latent(latents, noise, config.refiner_t_thresh)
        batch.height, batch.width = height, width
        return batch


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
        # Set once at request init from the base scale, which the refiner replaces.
        batch.do_classifier_free_guidance = config.refiner_guidance_scale > 1.0
        if (text_only := batch.extra.get(TEXT_ONLY_EMBEDS_KEY)) is not None:
            embeds, batch.prompt_attention_mask = text_only
            batch.prompt_embeds = [embeds]
        if batch.do_classifier_free_guidance:
            # The refiner conditions on zeros, not the base negative prompt.
            batch.negative_prompt_embeds = [torch.zeros_like(batch.prompt_embeds[0])]
            batch.negative_attention_mask = batch.prompt_attention_mask.clone()
        return super()._prepare_denoising_loop(batch, server_args)


def _refiner_generators(
    batch: Req, *, device: torch.device
) -> list[torch.Generator] | None:
    # The reference reseeds for the second pass rather than continuing the base stream.
    if not batch.seeds:
        return None
    return [
        torch.Generator(device=device).manual_seed(int(seed)) for seed in batch.seeds
    ]
