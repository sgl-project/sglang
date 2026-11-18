# SPDX-License-Identifier: Apache-2.0
"""Rollout hooks for denoise loops.

A loop runs inside ``rollout_scope``. Before SP sharding and before reading
``batch.timesteps`` it calls ``prepare_rollout_if_enabled`` for each of its
latent streams. It calls
``init_rollout_env_if_enabled`` at the loop start, passes
``**rollout_step_kwargs(batch, stream=...)`` to each stream's ``scheduler.step``,
calls ``finalize_rollout_if_enabled`` for each stream at the loop end and then
``publish_rollout_if_enabled`` once.
Every hook is a no-op for serving requests.
"""

from __future__ import annotations

import functools
import hashlib
from contextlib import contextmanager
from typing import Any, Callable

import torch

from sglang.multimodal_gen.configs.post_training.rl_rollout import PRIMARY_STREAM
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.flow_rollout_controller import (
    FlowRolloutController,
    truncate_rollout_sigmas,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_latents_for_sp,
)


def _drop_rollout_working_state(batch: Req) -> None:
    batch.rollout_controllers = {}
    batch._rollout_denoising_env_state = None


@contextmanager
def rollout_scope(batch: Req):
    """Reset rollout state on entry; drop the controllers and env snapshot on exit."""
    if not batch.rollout:
        yield
        return
    _drop_rollout_working_state(batch)
    batch.rollout_trajectory_data = None
    try:
        yield
    except BaseException:
        batch.rollout_trajectory_data = None
        raise
    finally:
        _drop_rollout_working_state(batch)


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


def rollout_step_kwargs(batch: Req, *, stream: str) -> dict[str, Any]:
    """The rollout-only keyword arguments of ``scheduler.step`` for one stream: its
    controller's ``transition``. Empty when the stream has no controller (a serving
    request, or a stream the request does not roll out), so that step call is
    unchanged and works with any scheduler."""
    controller = batch.rollout_controllers.get(stream)
    if controller is None:
        return {}
    return {"transition": functools.partial(controller.transition, batch=batch)}


def _requested_rollout_streams(batch: Req) -> tuple[str, ...]:
    return (PRIMARY_STREAM, *batch.rollout_extra_streams)


def pipeline_config_sp_fns(pipeline_config: Any) -> tuple[Callable, Callable]:
    """``(shard_for_sp, gather_for_sp)`` of latents the pipeline config shards for SP."""

    def shard_for_sp(batch: Req, latents: torch.Tensor) -> torch.Tensor:
        return pipeline_config.shard_latents_for_sp(batch=batch, latents=latents)[0]

    def gather_for_sp(batch: Req, stacked_latents: torch.Tensor) -> torch.Tensor:
        return gather_stacked_latents_for_sp(
            pipeline_config=pipeline_config,
            batch=batch,
            stacked_latents=stacked_latents,
        )

    return shard_for_sp, gather_for_sp


def derive_extra_stream_generators(
    generators: torch.Generator | list[torch.Generator] | None, *, stream: str
) -> torch.Generator | list[torch.Generator]:
    """RNGs from the request seed and stream name only, so every rank agrees."""
    if generators is None:
        raise ValueError(f"Rollout of stream {stream!r} requires request generators")

    def derive(generator: torch.Generator) -> torch.Generator:
        digest = hashlib.sha256(
            f"sglang/rollout/{stream}/{generator.initial_seed()}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        return torch.Generator(device=generator.device).manual_seed(seed)

    if isinstance(generators, torch.Generator):
        return derive(generators)
    return [derive(generator) for generator in generators]


def _stream_generators(
    batch: Req, stream: str
) -> torch.Generator | list[torch.Generator]:
    # The primary stream draws from the request's own RNGs, whatever else is rolled out.
    if stream == PRIMARY_STREAM:
        return batch.generator
    return derive_extra_stream_generators(batch.generator, stream=stream)


def prepare_rollout_if_enabled(
    *,
    batch: Req,
    stream: str,
    scheduler: Any,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    shard_for_sp: Callable[[Any, torch.Tensor], torch.Tensor],
    gather_for_sp: Callable[[Any, torch.Tensor], torch.Tensor],
) -> None:
    """Prepare one latent stream of the loop: its controller if the request rolls
    it out, else its scheduler's grid cut at the stop step.

    Placement: before SP sharding and before any stop is applied: ``latents`` is
    full-shape, ``scheduler`` and ``timesteps`` are uncut; ``timesteps`` holds the
    values the loop feeds the DiT for this stream. A rolled-out stream steps along
    its controller's schedule, its scheduler's grid is not cut.

    A stop also cuts ``batch.timesteps`` when the primary stream is prepared, so
    the loop must not read ``batch.timesteps`` or derive anything from it (the step
    count for warmup, cache-dit or the progress bar) before that.
    """
    if not batch.rollout:
        return
    if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError(
            f"Rollout of stream {stream!r} needs a FlowMatch Euler scheduler, "
            f"got {type(scheduler).__name__}"
        )
    for other_stream, other_controller in batch.rollout_controllers.items():
        if other_controller.uncut_num_steps != len(timesteps):
            raise ValueError(
                f"Rollout stream {stream!r} has {len(timesteps)} steps but stream "
                f"{other_stream!r} has {other_controller.uncut_num_steps}; "
                "streams step together"
            )
    stop_step_index = batch.rollout_stop_step_index
    if stream not in _requested_rollout_streams(batch):
        # Without a controller the stream keeps its scheduler's own Euler step.
        if stop_step_index is not None:
            scheduler.sigmas = truncate_rollout_sigmas(
                scheduler.sigmas, stop_step_index=stop_step_index
            )
            scheduler.timesteps = scheduler.timesteps[: stop_step_index + 1]
        return
    if stream in batch.rollout_controllers:
        raise RuntimeError(f"Rollout stream {stream!r} was already prepared")
    controller = FlowRolloutController(
        sde_type=batch.rollout_sde_type,
        noise_level=float(batch.rollout_noise_level),
        log_prob_no_const=batch.rollout_log_prob_no_const,
        debug_mode=batch.rollout_debug_mode,
        sde_step_indices=batch.rollout_sde_step_indices,
        generator=_stream_generators(batch, stream),
        return_dit_trajectory=batch.rollout_return_dit_trajectory,
        return_step_indices=batch.rollout_return_step_indices,
        shard_for_sp=shard_for_sp,
        gather_for_sp=gather_for_sp,
        sigmas=scheduler.sigmas,
        timesteps=timesteps,
        latents_shape=tuple(latents.shape),
    )
    # After the build: sigma_max comes from the uncut grid.
    if stop_step_index is not None:
        controller.stop_at(stop_step_index)
        if stream == PRIMARY_STREAM:
            # The loop iterates the primary stream's timesteps.
            batch.timesteps = controller.timesteps
    # Rebind: per-output copies of a request share the dict object.
    batch.rollout_controllers = {**batch.rollout_controllers, stream: controller}


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
    stream: str,
    latents: torch.Tensor,
    num_inference_steps: int,
    final_timestep: torch.Tensor,
) -> None:
    """Collect one stream's rollout data from its controller. A stream the request
    does not roll out has none, so the call does nothing.

    Must be called before ``_post_denoising_loop`` so that ``latents`` (the
    stream's last ``scheduler.step`` output) is still SP-sharded and can be
    gathered uniformly with the per-step trajectory latents.
    """
    if not batch.rollout:
        return
    controller = batch.rollout_controllers.get(stream)
    if controller is None:
        return
    if controller.stream_data is not None:
        raise RuntimeError(f"Rollout stream {stream!r} was already finalized")
    controller.collect_rollout_stream_data(
        batch=batch,
        final_latents=latents,
        num_inference_steps=num_inference_steps,
        final_timestep=final_timestep,
    )


def publish_rollout_if_enabled(*, batch: Req, pipeline_config: Any) -> None:
    """Publish the request's rollout result: every requested stream's data and the
    denoising env. Called once, after every stream is finalized."""
    if not batch.rollout:
        return
    requested = _requested_rollout_streams(batch)
    unprepared = set(requested) - batch.rollout_controllers.keys()
    if unprepared:
        raise RuntimeError(
            f"Requested rollout streams {sorted(unprepared)} were not prepared by "
            "the denoising stage"
        )
    unfinalized = [
        stream
        for stream in requested
        if batch.rollout_controllers[stream].stream_data is None
    ]
    if unfinalized:
        raise RuntimeError(
            f"Requested rollout streams {unfinalized} were not finalized by the "
            "denoising stage"
        )
    state = batch._rollout_denoising_env_state
    env = state["env"] if state is not None else None
    if env is not None:
        env.pos_cond_kwargs = _kwargs_to_cpu(
            pipeline_config.gather_denoising_env_static_for_sp(
                batch, state["pos_cond_kwargs_src"]
            )
        )
        if env.neg_cond_kwargs is not None:
            env.neg_cond_kwargs = _kwargs_to_cpu(
                pipeline_config.gather_denoising_env_static_for_sp(
                    batch, state["neg_cond_kwargs_src"]
                )
            )
    # One assignment: a failure before it leaves no partial trajectory.
    batch.rollout_trajectory_data = RolloutTrajectoryData(
        denoising_env=env,
        streams={
            stream: batch.rollout_controllers[stream].stream_data
            for stream in requested
        },
    )
