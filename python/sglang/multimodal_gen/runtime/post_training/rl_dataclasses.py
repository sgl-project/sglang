# SPDX-License-Identifier: Apache-2.0
"""RL-specific dataclasses used by post-training and rollout paths."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RolloutDebugTensors:
    """Container for rollout debug tensors collected during denoising."""

    rollout_variance_noises: torch.Tensor | None = None
    rollout_prev_sample_means: torch.Tensor | None = None
    rollout_noise_std_devs: torch.Tensor | None = None
    rollout_model_outputs: torch.Tensor | None = None


@dataclass
class RolloutDenoisingEnv:
    image_kwargs: dict[str, Any] | None = None
    pos_cond_kwargs: dict[str, Any] | None = None
    neg_cond_kwargs: dict[str, Any] | None = None
    guidance: torch.Tensor | None = None


@dataclass
class RolloutDitTrajectory:
    # [B, T+1, ...]: per-step noisy latents x_{t_0..t_{T-1}} followed by the
    # final denoised latent x_{t_T} (last scheduler.step output).
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None  # [T]
    # [T+1] scheduler.sigmas snapshot (post-shift, includes terminal 0).
    sigmas: torch.Tensor | None = None


@dataclass
class RolloutTrajectoryData:
    rollout_log_probs: torch.Tensor | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    denoising_env: RolloutDenoisingEnv | None = None
    dit_trajectory: RolloutDitTrajectory | None = None


def _cat_per_output(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Concatenate complete per-output tensor fields along the batch dimension."""
    if not tensors or any(tensor is None for tensor in tensors):
        return None
    return torch.cat(tensors, dim=0)


def _slice_dim(
    tensor: torch.Tensor | None, start: int, end: int
) -> torch.Tensor | None:
    """Slice a [start:end) range along the batch dimension, else pass through."""
    if tensor is None or tensor.dim() < 1 or start >= tensor.shape[0]:
        return tensor
    return tensor[start:end]


def _tensor_tree_cat(values: list[Any]) -> Any:
    """Recursively concatenate aligned conditioning trees (dict/list/tuple/Tensor)
    along the batch dimension, mirroring ``rollout_hooks._kwargs_to_cpu``'s shape."""
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {k: _tensor_tree_cat([v[k] for v in values]) for k in first}
    if isinstance(first, (list, tuple)):
        items = [_tensor_tree_cat([v[i] for v in values]) for i in range(len(first))]
        return type(first)(items)
    return first


def _tensor_tree_slice(value: Any, start: int, end: int) -> Any:
    """Recursively slice a [start:end) range out of a conditioning tree."""
    if isinstance(value, torch.Tensor):
        return _slice_dim(value, start, end)
    if isinstance(value, dict):
        return {k: _tensor_tree_slice(v, start, end) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_tensor_tree_slice(v, start, end) for v in value]
        return type(value)(items)
    return value


def _cat_envs(
    envs: list["RolloutDenoisingEnv | None"],
) -> "RolloutDenoisingEnv | None":
    """Concatenate per-request conditioning envs along the batch dimension."""
    if any(env is None for env in envs):
        return None
    return RolloutDenoisingEnv(
        image_kwargs=(
            _tensor_tree_cat([env.image_kwargs for env in envs])
            if all(env.image_kwargs is not None for env in envs)
            else None
        ),
        pos_cond_kwargs=(
            _tensor_tree_cat([env.pos_cond_kwargs for env in envs])
            if all(env.pos_cond_kwargs is not None for env in envs)
            else None
        ),
        neg_cond_kwargs=(
            _tensor_tree_cat([env.neg_cond_kwargs for env in envs])
            if all(env.neg_cond_kwargs is not None for env in envs)
            else None
        ),
        guidance=_cat_per_output([env.guidance for env in envs]),
    )


def _slice_env(
    env: "RolloutDenoisingEnv | None", start: int, end: int
) -> "RolloutDenoisingEnv | None":
    """Keep the [start:end) rows of a conditioning env."""
    if env is None:
        return None
    return RolloutDenoisingEnv(
        image_kwargs=(
            _tensor_tree_slice(env.image_kwargs, start, end)
            if env.image_kwargs is not None
            else None
        ),
        pos_cond_kwargs=(
            _tensor_tree_slice(env.pos_cond_kwargs, start, end)
            if env.pos_cond_kwargs is not None
            else None
        ),
        neg_cond_kwargs=(
            _tensor_tree_slice(env.neg_cond_kwargs, start, end)
            if env.neg_cond_kwargs is not None
            else None
        ),
        guidance=_slice_dim(env.guidance, start, end),
    )


def concat_rollout_trajectory_data(
    per_output: list[RolloutTrajectoryData | None],
) -> RolloutTrajectoryData | None:
    """Combine aligned per-output trajectories into one batched trajectory."""
    if any(data is None for data in per_output):
        return None

    first = per_output[0]
    if len(per_output) == 1:
        return first

    debug_tensors = None
    if all(data.rollout_debug_tensors is not None for data in per_output):
        debug = [data.rollout_debug_tensors for data in per_output]
        debug_tensors = RolloutDebugTensors(
            rollout_variance_noises=_cat_per_output(
                [entry.rollout_variance_noises for entry in debug]
            ),
            rollout_prev_sample_means=_cat_per_output(
                [entry.rollout_prev_sample_means for entry in debug]
            ),
            rollout_noise_std_devs=_cat_per_output(
                [entry.rollout_noise_std_devs for entry in debug]
            ),
            rollout_model_outputs=_cat_per_output(
                [entry.rollout_model_outputs for entry in debug]
            ),
        )

    dit_trajectory = None
    if all(data.dit_trajectory is not None for data in per_output):
        dit_trajectory = RolloutDitTrajectory(
            latents=_cat_per_output(
                [data.dit_trajectory.latents for data in per_output]
            ),
            timesteps=first.dit_trajectory.timesteps,
            sigmas=first.dit_trajectory.sigmas,
        )

    return RolloutTrajectoryData(
        rollout_log_probs=_cat_per_output(
            [data.rollout_log_probs for data in per_output]
        ),
        rollout_debug_tensors=debug_tensors,
        denoising_env=_cat_envs([data.denoising_env for data in per_output]),
        dit_trajectory=dit_trajectory,
    )


def slice_rollout_trajectory_range(
    rollout_trajectory_data: RolloutTrajectoryData | None,
    start: int,
    end: int,
) -> RolloutTrajectoryData | None:
    """Keep the [start:end) output rows of a batched rollout trajectory.

    Used both to select a single output (``end = start + 1``) and to split a
    dynamically-batched result back into its per-request slices.
    """
    if rollout_trajectory_data is None:
        return None

    debug_tensors = rollout_trajectory_data.rollout_debug_tensors
    if debug_tensors is not None:
        debug_tensors = RolloutDebugTensors(
            rollout_variance_noises=_slice_dim(
                debug_tensors.rollout_variance_noises, start, end
            ),
            rollout_prev_sample_means=_slice_dim(
                debug_tensors.rollout_prev_sample_means, start, end
            ),
            rollout_noise_std_devs=_slice_dim(
                debug_tensors.rollout_noise_std_devs, start, end
            ),
            rollout_model_outputs=_slice_dim(
                debug_tensors.rollout_model_outputs, start, end
            ),
        )

    dit_trajectory = rollout_trajectory_data.dit_trajectory
    if dit_trajectory is not None:
        dit_trajectory = RolloutDitTrajectory(
            latents=_slice_dim(dit_trajectory.latents, start, end),
            timesteps=dit_trajectory.timesteps,
            sigmas=dit_trajectory.sigmas,
        )

    return RolloutTrajectoryData(
        rollout_log_probs=_slice_dim(
            rollout_trajectory_data.rollout_log_probs, start, end
        ),
        rollout_debug_tensors=debug_tensors,
        denoising_env=_slice_env(rollout_trajectory_data.denoising_env, start, end),
        dit_trajectory=dit_trajectory,
    )


def select_output_rollout_trajectory(
    rollout_trajectory_data: RolloutTrajectoryData | None,
    output_index: int | None,
) -> RolloutTrajectoryData | None:
    """Select one output row from a batched rollout trajectory."""
    if rollout_trajectory_data is None or output_index is None:
        return rollout_trajectory_data
    return slice_rollout_trajectory_range(
        rollout_trajectory_data, output_index, output_index + 1
    )
