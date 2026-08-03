# SPDX-License-Identifier: Apache-2.0

"""Terminal stage that hands MinWM latents to a remote realtime VAE."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class RealtimeLatentHandoffStage(PipelineStage):
    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        del server_args
        if not isinstance(batch.latents, torch.Tensor):
            raise ValueError("Realtime latent handoff requires tensor latents")
        if not batch.realtime_session_id or not batch.realtime_generation_id:
            raise ValueError("Realtime latent handoff requires session generation identity")

        generated_latents = batch.latents
        has_reference = isinstance(batch.image_latent, torch.Tensor)
        handoff_latents = generated_latents
        if batch.block_idx == 0 and has_reference:
            handoff_latents = torch.cat([batch.image_latent, generated_latents], dim=2)

        handoff_latents = handoff_latents.detach().to(dtype=torch.bfloat16).contiguous()
        return OutputBatch(
            realtime_latents=handoff_latents,
            realtime_handoff={
                "session_id": batch.realtime_session_id,
                "generation_id": batch.realtime_generation_id,
                "request_id": batch.request_id,
                "chunk_index": batch.block_idx,
                "event_id": batch.realtime_event_id,
                "action_version": batch.realtime_action_version,
                "prompt_version": batch.realtime_prompt_version,
                "has_reference": has_reference,
                "generated_latent_frames": int(generated_latents.shape[2]),
                "output_format": batch.realtime_output_format,
                "preview_max_width": batch.realtime_preview_max_width,
            },
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            metrics=batch.metrics,
        )
