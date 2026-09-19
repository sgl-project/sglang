# SPDX-License-Identifier: Apache-2.0
"""Loop-level rollout hooks: snapshot ``c`` at start, pack outputs at end.

Per-step ``x_t`` is recorded by ``RolloutDenoisingMixin.step_latents``.
``log π`` is recorded inside ``SchedulerRLMixin.flow_sde_sampling``
(generic loops reach it via ``scheduler.step``; H3 calls it after
mapping ``−v``).
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class DenoiseLoopObserver:
    """No-op observer. Custom loops call ``init_env`` / ``finalize`` only."""

    def init_env(
        self,
        stage: Any,
        batch: Req,
        pipeline_config: Any,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any] | None,
        guidance: torch.Tensor | None,
    ) -> None:
        return None

    def finalize(
        self,
        stage: Any,
        batch: Req,
        latents: torch.Tensor,
        num_inference_steps: int,
        final_timestep: torch.Tensor,
        server_args: ServerArgs,
    ) -> None:
        return None


class NullDenoiseLoopObserver(DenoiseLoopObserver):
    pass


class RolloutDenoiseLoopObserver(DenoiseLoopObserver):
    """Snapshot DiT cond at loop start; pack trajectory / env / log-probs at end."""

    def init_env(
        self,
        stage: Any,
        batch: Req,
        pipeline_config: Any,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any] | None,
        guidance: torch.Tensor | None,
    ) -> None:
        scheduler = getattr(batch, "scheduler", None)
        if isinstance(scheduler, SchedulerRLMixin):
            stage._maybe_prepare_rollout(batch)
        stage._maybe_init_denoising_env_collection(
            batch=batch,
            pipeline_config=pipeline_config,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            guidance=guidance,
        )

    def finalize(
        self,
        stage: Any,
        batch: Req,
        latents: torch.Tensor,
        num_inference_steps: int,
        final_timestep: torch.Tensor,
        server_args: ServerArgs,
    ) -> None:
        scheduler = getattr(batch, "scheduler", None)
        if isinstance(scheduler, SchedulerRLMixin):
            stage._postprocess_rollout_outputs(
                batch=batch,
                latents=latents,
                num_inference_steps=num_inference_steps,
                final_timestep=final_timestep,
                server_args=server_args,
            )
            return
        stage._maybe_append_dit_trajectory_step(
            batch=batch,
            latents=latents,
            timestep_value=final_timestep,
            step_index=num_inference_steps,
        )
        stage._maybe_finalize_denoising_env_collection(
            batch=batch,
            pipeline_config=server_args.pipeline_config,
        )


_NULL_OBSERVER = NullDenoiseLoopObserver()


def get_denoise_loop_observer(batch: Req) -> DenoiseLoopObserver:
    cached = getattr(batch, "_denoise_loop_observer", None)
    if cached is not None:
        return cached
    observer: DenoiseLoopObserver
    if getattr(batch, "rollout", False):
        observer = RolloutDenoiseLoopObserver()
    else:
        observer = _NULL_OBSERVER
    batch._denoise_loop_observer = observer
    return observer
