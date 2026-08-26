# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.sample.magi2 import MAGI2_REFINER_FPS
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    renoise as magi2_renoise,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.stages.preparation import (
    build_scheduler,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Magi2StageHandoffStage(PipelineStage):
    """Audio is not refined: the refiner gets zero audio tokens and the preview's audio latent goes straight to its VAE."""

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not batch.extra["magi2_enable_refiner"]:
            return batch

        config = server_args.pipeline_config
        device = get_local_torch_device()
        arch = config.dit_config.arch_config
        frames, height, width = batch.extra["magi2_preview_grid"]
        target_frames, target_height, target_width = batch.extra["magi2_refiner_grid"]

        grid = (
            batch.latents.reshape(frames, height, width, arch.video_in_channels)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
        )
        upsampled = magi2_renoise.upsample_latent(
            grid, height=target_height, width=target_width
        )
        if upsampled.shape[2] != target_frames:
            raise ValueError(
                f"handoff produced {upsampled.shape[2]} latent frames but the "
                f"refiner grid expects {target_frames}"
            )

        noise = torch.randn(
            upsampled.shape,
            device=device,
            dtype=torch.float32,
            generator=batch.generator,
        )
        renoised = magi2_renoise.renoise(
            upsampled, noise=noise, sigma_index=config.refiner_renoise_index
        )

        batch.latents = (
            renoised.squeeze(0).permute(1, 2, 3, 0).reshape(-1, arch.video_in_channels)
        )
        batch.raw_latent_shape = (
            1,
            arch.video_in_channels,
            target_frames,
            target_height,
            target_width,
        )

        # Parked so the refiner sees no audio tokens; decoding picks it back up.
        batch.extra["magi2_preview_audio_latents"] = batch.audio_latents
        batch.audio_latents = None

        batch.scheduler = build_scheduler(
            shift=config.refiner_flow_shift,
            steps=batch.sampling_params.refiner_num_inference_steps,
            device=device,
        )
        batch.timesteps = batch.scheduler.timesteps
        batch.sigmas = batch.scheduler.sigmas.tolist()
        batch.fps = MAGI2_REFINER_FPS
        return batch
