# SPDX-License-Identifier: Apache-2.0
"""Rollout hooks for denoise loops.

A loop calls ``prepare_rollout_if_enabled`` before SP sharding,
``init_rollout_env_if_enabled`` at the loop start and
``finalize_rollout_if_enabled`` at the loop end, and passes
``rollout_transition(batch)`` to ``scheduler.step``. Every hook is a no-op for
serving requests.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.flow_rollout_controller import (
    FlowRolloutController,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_latents_for_sp,
)


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


def rollout_transition(batch: Req) -> Callable[..., torch.FloatTensor] | None:
    """The ``transition`` to pass to ``scheduler.step`` for this request: the
    visual-stream rollout controller, or None to keep the scheduler's own update."""
    rollout = batch.rollout_controllers.get("visual")
    return None if rollout is None else rollout.transition


def _build_scheduler_rollout_controller(
    *, batch: Req, pipeline_config: Any
) -> FlowRolloutController:
    """The visual-stream controller of a loop that steps ``batch.scheduler``."""
    scheduler = batch.scheduler
    if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError(f"Scheduler {type(scheduler)} does not support rollout")

    controller = FlowRolloutController(
        sde_type=batch.rollout_sde_type,
        noise_level=float(batch.rollout_noise_level),
        log_prob_no_const=batch.rollout_log_prob_no_const,
        debug_mode=batch.rollout_debug_mode,
        sde_step_indices=batch.rollout_sde_step_indices,
        generator=batch.generator,
        return_dit_trajectory=batch.rollout_return_dit_trajectory,
        return_step_indices=batch.rollout_return_step_indices,
        shard_for_sp=lambda latents: pipeline_config.shard_latents_for_sp(
            batch=batch, latents=latents
        )[0],
        gather_for_sp=lambda stacked_latents: gather_stacked_latents_for_sp(
            pipeline_config=pipeline_config,
            batch=batch,
            stacked_latents=stacked_latents,
        ),
        sigmas=scheduler.sigmas,
        latents_shape=(
            tuple(batch.latents.shape) if batch.latents is not None else None
        ),
    )
    return controller


def prepare_rollout_if_enabled(
    *,
    batch: Req,
    pipeline_config: Any,
    build_controller: Callable[..., FlowRolloutController] = (
        _build_scheduler_rollout_controller
    ),
) -> None:
    """Create the rollout controller of the visual stream of this request.

    Must run before SP sharding so the controller records the full latent shape.
    ``build_controller(batch=, pipeline_config=)`` returns it; a loop without a
    scheduler passes its own.
    """
    if not batch.rollout:
        return
    controller = build_controller(batch=batch, pipeline_config=pipeline_config)
    batch.rollout_controllers = {"visual": controller}


def init_rollout_env_if_enabled(
    *,
    batch: Req,
    image_kwargs: dict[str, Any],
    pos_cond_kwargs: dict[str, Any],
    neg_cond_kwargs: dict[str, Any],
    guidance: torch.Tensor | None,
) -> None:
    """Snapshot the DiT conditioning that the trainer replays."""
    if not batch.rollout:
        return
    collect_env = batch.rollout_return_denoising_env
    collect_traj = batch.rollout_return_dit_trajectory
    if not (collect_env or collect_traj):
        batch._rollout_denoising_env_state = None
        return

    if collect_env:
        env = RolloutDenoisingEnv(
            image_kwargs=_kwargs_to_cpu(image_kwargs),
            pos_cond_kwargs=_kwargs_to_cpu(pos_cond_kwargs),
            neg_cond_kwargs=(
                _kwargs_to_cpu(neg_cond_kwargs) if neg_cond_kwargs else None
            ),
            guidance=guidance.detach().cpu() if guidance is not None else None,
        )
        pos_src = pos_cond_kwargs
        neg_src = neg_cond_kwargs
    else:
        env = None
        pos_src = None
        neg_src = None

    batch._rollout_denoising_env_state = {
        "env": env,
        "pos_cond_kwargs_src": pos_src,
        "neg_cond_kwargs_src": neg_src,
    }


def finalize_rollout_if_enabled(
    *,
    batch: Req,
    latents: torch.Tensor,
    num_inference_steps: int,
    final_timestep: torch.Tensor,
    pipeline_config: Any,
) -> None:
    """Finalize rollout-only outputs.

    Must be called before ``_post_denoising_loop`` so that ``latents`` (the
    last ``scheduler.step`` output) is still SP-sharded and can be gathered
    uniformly with the per-step trajectory latents.
    """
    if not batch.rollout:
        return
    _collect_rollout_log_probs(batch)
    # Append final denoised latent as the (T+1)-th entry (step_index=T),
    # routed through the same filter so rollout_return_step_indices can
    # include/exclude it.
    batch.rollout_controllers["visual"]._maybe_append_dit_trajectory_step(
        latents=latents,
        timestep_value=final_timestep,
        step_index=num_inference_steps,
    )
    _finalize_rollout_env(batch=batch, pipeline_config=pipeline_config)
    batch.rollout_controllers = {}


def _collect_rollout_log_probs(batch: Req) -> None:
    rollout = batch.rollout_controllers["visual"]
    if batch.rollout_trajectory_data is None:
        batch.rollout_trajectory_data = RolloutTrajectoryData()
    batch.rollout_trajectory_data.rollout_log_probs = (
        rollout.collect_rollout_log_probs()
    )
    if batch.rollout_debug_mode:
        batch.rollout_trajectory_data.rollout_debug_tensors = (
            rollout.collect_rollout_debug_tensors()
        )


def _finalize_rollout_env(*, batch: Req, pipeline_config: Any) -> None:
    state = getattr(batch, "_rollout_denoising_env_state", None)
    if state is None:
        return

    env: RolloutDenoisingEnv | None = state["env"]
    rollout = batch.rollout_controllers["visual"]
    step_latents: list[torch.Tensor] = rollout.step_latents
    step_timesteps: list[torch.Tensor] = rollout.step_timesteps

    if batch.rollout_trajectory_data is None:
        batch.rollout_trajectory_data = RolloutTrajectoryData()

    if step_latents and batch.rollout_return_dit_trajectory:
        step_latents_tensor = rollout.gather_for_sp(torch.stack(step_latents, dim=1))
        batch.rollout_trajectory_data.dit_trajectory = RolloutDitTrajectory(
            latents=step_latents_tensor.cpu(),
            timesteps=torch.stack(step_timesteps, dim=0).cpu(),
            sigmas=rollout.sigmas.detach().cpu().clone(),
            latent_step_indices=torch.tensor(
                rollout.step_latent_indices, dtype=torch.long
            ),
        )

    if env is not None and batch.rollout_return_denoising_env:
        gather_fn = getattr(pipeline_config, "gather_denoising_env_static_for_sp", None)

        pos_src = state.get("pos_cond_kwargs_src")
        if pos_src is not None and env.pos_cond_kwargs is not None:
            gathered_pos = gather_fn(batch, pos_src) if gather_fn else pos_src
            env.pos_cond_kwargs = _kwargs_to_cpu(gathered_pos)

        neg_src = state.get("neg_cond_kwargs_src")
        if neg_src is not None and env.neg_cond_kwargs is not None:
            gathered_neg = gather_fn(batch, neg_src) if gather_fn else neg_src
            env.neg_cond_kwargs = _kwargs_to_cpu(gathered_neg)

        batch.rollout_trajectory_data.denoising_env = env

    batch._rollout_denoising_env_state = None
