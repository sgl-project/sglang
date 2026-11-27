# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.sample.magi2 import (
    MAGI2_CLIP_SECONDS,
    MAGI2_REFINER_FPS,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# Both passes use this, so the handoff's 2*T-1 upsample lands exactly.
MAGI2_BASE_FRAMES = round(MAGI2_CLIP_SECONDS * MAGI2_REFINER_FPS)


def latent_grid(
    *, height: int, width: int, frames: int, vae_stride: tuple[int, int, int]
) -> tuple[int, int, int]:
    # The temporal VAE is causal, so the first frame is its own latent.
    stride_t, stride_h, stride_w = vae_stride
    return (
        (frames - 1) // stride_t + 1,
        height // stride_h,
        width // stride_w,
    )


def prompt_is_non_empty(value) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0


class Magi2InputStage(PipelineStage):
    """Its own stage because the packing stage runs once per denoise pass and both instances need these numbers."""

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, prompt_is_non_empty)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        config = server_args.pipeline_config
        params = batch.sampling_params

        # At preview size the refiner has nothing to do, even if its weights loaded.
        preview_size = (params.preview_height, params.preview_width)
        enable_refiner = (
            config.enable_refiner
            and (
                batch.height,
                batch.width,
            )
            != preview_size
        )

        batch.extra["magi2_enable_refiner"] = enable_refiner
        batch.extra["magi2_preview_grid"] = latent_grid(
            height=params.preview_height,
            width=params.preview_width,
            frames=MAGI2_BASE_FRAMES,
            vae_stride=config.dit_config.arch_config.vae_stride,
        )
        if enable_refiner:
            batch.extra["magi2_refiner_grid"] = latent_grid(
                height=batch.height,
                width=batch.width,
                frames=MAGI2_BASE_FRAMES,
                vae_stride=config.refiner_dit_config.arch_config.vae_stride,
            )
        return batch


AUDIO_LATENT_FPS = 25.0
AUDIO_LATENT_DIM = 64


def build_scheduler(
    *, shift: float, steps: int, device: torch.device
) -> FlowUniPCMultistepScheduler:
    # shift goes to set_timesteps only; passing it to both double-shifts the sigmas.
    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(steps, device=device, shift=shift)
    return scheduler


def _request_seed(batch: Req) -> int:
    # SamplingParams.seed is int | list[int].
    if batch.seeds:
        return int(batch.seeds[0])
    seed = batch.seed
    if isinstance(seed, list):
        return int(seed[0])
    return int(seed)


class Magi2LatentPreparationStage(PipelineStage):
    """Video latents are channels-last per token, not a 5-D grid: both DiTs consume one packed row per latent position."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        arch = server_args.pipeline_config.dit_config.arch_config
        frames, height, width = batch.extra["magi2_preview_grid"]

        generator = torch.Generator(device=device).manual_seed(_request_seed(batch))

        batch.raw_latent_shape = (1, arch.video_in_channels, frames, height, width)
        batch.latents = torch.randn(
            frames * height * width,
            arch.video_in_channels,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

        if batch.generate_audio:
            # From the clip length, not num_frames, which is rounded up per SP degree.
            audio_tokens = round(MAGI2_CLIP_SECONDS * AUDIO_LATENT_FPS)
            batch.raw_audio_latent_shape = (1, audio_tokens, AUDIO_LATENT_DIM)
            batch.audio_latents = torch.randn(
                audio_tokens,
                AUDIO_LATENT_DIM,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )

        batch.generator = generator
        return batch


class Magi2TimestepPreparationStage(PipelineStage):
    """A fresh scheduler, because the refiner needs a differently shifted one and UniPC carries multistep state across calls."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        config = server_args.pipeline_config
        device = get_local_torch_device()

        batch.scheduler = build_scheduler(
            shift=config.flow_shift, steps=batch.num_inference_steps, device=device
        )
        batch.timesteps = batch.scheduler.timesteps
        batch.sigmas = batch.scheduler.sigmas.tolist()

        # Own instance: UniPC keeps multistep history, so sharing mixes solver state.
        if batch.generate_audio:
            batch.extra["magi2_audio_scheduler"] = build_scheduler(
                shift=config.flow_shift,
                steps=batch.num_inference_steps,
                device=device,
            )
        return batch


# Assemble the packed sequence the DiTs consume.


class Magi2PackingStage(PipelineStage):
    """Write the row layout and 9-D coordinates for the next denoise pass."""

    def __init__(
        self,
        *,
        grid_key: str,
        conditions_on_images: bool = False,
        refiner_only: bool = False,
    ) -> None:
        super().__init__()
        self.grid_key = grid_key
        self.conditions_on_images = conditions_on_images
        self.refiner_only = refiner_only

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if self.refiner_only and not batch.extra["magi2_enable_refiner"]:
            return batch
        device = get_local_torch_device()
        grid = batch.extra[self.grid_key]
        frames, height, width = grid

        audio_tokens = (
            batch.audio_latents.shape[0] if batch.audio_latents is not None else 0
        )

        # Preview only: the refiner's sequence carries no image segment.
        ref_counts = (
            batch.extra["magi2_ref_patch_counts"] if self.conditions_on_images else []
        )
        ref_latent_hw = (
            batch.extra["magi2_ref_latent_hw"] if self.conditions_on_images else []
        )

        # One layout per branch: each text is trimmed to its own token count.
        branches = {"magi2_layout": batch.prompt_embeds[0].shape[0]}
        if batch.negative_prompt_embeds:
            branches["magi2_layout_uncond"] = batch.negative_prompt_embeds[0].shape[0]

        for key, text_tokens in branches.items():
            batch.extra[key] = packed_sequence.build_layout(
                video_latent_thw=grid,
                audio_tokens=audio_tokens,
                text_tokens=text_tokens,
                device=device,
                ref_patch_counts=ref_counts,
            )
            batch.extra[key.replace("layout", "coords")] = packed_sequence.build_coords(
                video_latent_shape=grid,
                audio_tokens=audio_tokens,
                text_tokens=text_tokens,
                device=device,
                ref_latent_hw=ref_latent_hw,
                ref_patch_counts=ref_counts,
            )
        return batch
