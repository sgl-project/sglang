# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.sample.magi2 import MAGI2_CLIP_SECONDS
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs

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
