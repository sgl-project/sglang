"""Mixin for rollout-related denoising hooks.

Moved out of DenoisingStage to keep the core stage lean.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_latents_for_sp,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _kwargs_to_cpu(d: Any) -> Any:
    if isinstance(d, torch.Tensor):
        return d.detach().cpu()
    if isinstance(d, dict):
        return {k: _kwargs_to_cpu(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_kwargs_to_cpu(v) for v in d]
    if isinstance(d, tuple):
        return tuple(_kwargs_to_cpu(v) for v in d)
    return d


class RolloutDenoisingMixin:

    def _maybe_prepare_rollout(self, batch: Req):
        """Prepare denoising loop for rollout."""
        if not isinstance(self.scheduler, SchedulerRLMixin):
            if batch.rollout:
                raise ValueError(
                    f"Scheduler {type(self.scheduler)} does not support rollout"
                )
            return

        self.scheduler.release_rollout_resources(batch)
        if batch.rollout:
            self.scheduler.prepare_rollout(
                batch=batch,
                pipeline_config=self.server_args.pipeline_config,
            )

    def _maybe_collect_rollout_log_probs(self, batch: Req):
        if not isinstance(self.scheduler, SchedulerRLMixin):
            if batch.rollout:
                raise ValueError(
                    f"Scheduler {type(self.scheduler)} does not support rollout"
                )
            return

        if batch.rollout:
            if batch.rollout_trajectory_data is None:
                batch.rollout_trajectory_data = RolloutTrajectoryData()
            batch.rollout_trajectory_data.rollout_log_probs = (
                self.scheduler.collect_rollout_log_probs(batch)
            )
            if batch.rollout_debug_mode:
                batch.rollout_trajectory_data.rollout_debug_tensors = (
                    self.scheduler.collect_rollout_debug_tensors(batch)
                )
            self.scheduler.release_rollout_resources(batch)

    def _postprocess_rollout_outputs(
        self,
        batch: Req,
        latents: torch.Tensor,
        server_args: ServerArgs,
    ) -> None:
        """Finalize rollout-only outputs.

        Must be called before ``_post_denoising_loop`` so that ``latents`` (the
        last ``scheduler.step`` output) is still SP-sharded and can be gathered
        uniformly with the per-step trajectory latents.
        """
        self._maybe_collect_rollout_log_probs(batch)
        # Append the final denoised latent as the (T+1)-th entry of the
        # dit-trajectory latents list.
        state = getattr(batch, "_rollout_dit_env_state", None)
        if state is not None and batch.rollout and batch.rollout_return_dit_trajectory:
            state["step_latents"].append(latents.detach())
        self._maybe_finalize_dit_env_collection(
            batch=batch,
            pipeline_config=server_args.pipeline_config,
        )

    def _maybe_init_denoising_env_collection(
        self,
        batch,
        pipeline_config,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        guidance: torch.Tensor | None,
    ) -> None:
        collect_env = batch.rollout_return_denoising_env
        collect_traj = batch.rollout_return_dit_trajectory
        if not (collect_env or collect_traj):
            batch._rollout_dit_env_state = None
            return

        sanitize = getattr(pipeline_config, "sanitize_dit_env_kwargs", lambda x: x)
        if collect_env:
            env = RolloutDenoisingEnv(
                image_kwargs=_kwargs_to_cpu(sanitize(image_kwargs)),
                pos_cond_kwargs=_kwargs_to_cpu(sanitize(pos_cond_kwargs)),
                neg_cond_kwargs=(
                    _kwargs_to_cpu(sanitize(neg_cond_kwargs))
                    if neg_cond_kwargs
                    else None
                ),
                guidance=guidance.detach().cpu() if guidance is not None else None,
            )
            pos_src = pos_cond_kwargs
            neg_src = neg_cond_kwargs
        else:
            env = None
            pos_src = None
            neg_src = None

        batch._rollout_dit_env_state = {
            "env": env,
            "step_latents": [],
            "step_timesteps": [],
            "pos_cond_kwargs_src": pos_src,
            "neg_cond_kwargs_src": neg_src,
        }

    def _maybe_append_dit_trajectory_step(
        self,
        batch,
        latents: torch.Tensor,
        timestep_value: torch.Tensor,
    ) -> None:
        if not batch.rollout or not batch.rollout_return_dit_trajectory:
            return
        state = getattr(batch, "_rollout_dit_env_state", None)
        if state is None:
            return

        state["step_latents"].append(latents.detach())
        state["step_timesteps"].append(timestep_value.detach().cpu())

    def _maybe_finalize_dit_env_collection(self, batch, pipeline_config) -> None:
        state = getattr(batch, "_rollout_dit_env_state", None)
        if state is None:
            return

        env: RolloutDenoisingEnv | None = state["env"]
        step_latents: list[torch.Tensor] = state["step_latents"]
        step_timesteps: list[torch.Tensor] = state["step_timesteps"]

        if batch.rollout_trajectory_data is None:
            batch.rollout_trajectory_data = RolloutTrajectoryData()

        if step_latents and batch.rollout_return_dit_trajectory:
            step_latents_tensor = torch.stack(step_latents, dim=1)
            step_latents_tensor = gather_stacked_latents_for_sp(
                pipeline_config=pipeline_config,
                batch=batch,
                stacked_latents=step_latents_tensor,
            )
            batch.rollout_trajectory_data.dit_trajectory = RolloutDitTrajectory(
                latents=step_latents_tensor.cpu(),
                timesteps=torch.stack(step_timesteps, dim=0).cpu(),
            )

        if env is not None and batch.rollout_return_denoising_env:
            sanitize = getattr(pipeline_config, "sanitize_dit_env_kwargs", lambda x: x)
            gather_fn = getattr(pipeline_config, "gather_dit_env_static_for_sp", None)

            pos_src = state.get("pos_cond_kwargs_src")
            if pos_src is not None and env.pos_cond_kwargs is not None:
                gathered_pos = gather_fn(batch, pos_src) if gather_fn else pos_src
                env.pos_cond_kwargs = _kwargs_to_cpu(sanitize(gathered_pos))

            neg_src = state.get("neg_cond_kwargs_src")
            if neg_src is not None and env.neg_cond_kwargs is not None:
                gathered_neg = gather_fn(batch, neg_src) if gather_fn else neg_src
                env.neg_cond_kwargs = _kwargs_to_cpu(sanitize(gathered_neg))

            batch.rollout_trajectory_data.denoising_env = env

        batch._rollout_dit_env_state = None
