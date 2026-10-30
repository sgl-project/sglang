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


def _map_env_tensors(value: Any, fn) -> Any:
    """Apply ``fn`` to tensors nested in the denoising environment."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_env_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_env_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_env_tensors(item, fn) for item in value]
    return value


def _concat_denoising_envs(
    envs: list[RolloutDenoisingEnv | None],
) -> RolloutDenoisingEnv | None:
    if not envs or any(env is None for env in envs):
        return None

    def concat_values(values: list[Any]) -> Any:
        if all(isinstance(value, torch.Tensor) for value in values):
            tensors = values
            if all(tensor.dim() > 0 for tensor in tensors):
                return torch.cat(tensors, dim=0)
            return tensors[0]
        if all(isinstance(value, dict) for value in values):
            keys = values[0].keys()
            if not all(value.keys() == keys for value in values):
                return values[0]
            return {
                key: concat_values([value[key] for value in values]) for key in keys
            }
        if all(isinstance(value, tuple) for value in values):
            if not all(len(value) == len(values[0]) for value in values):
                return values[0]
            return tuple(
                concat_values([value[index] for value in values])
                for index in range(len(values[0]))
            )
        if all(isinstance(value, list) for value in values):
            if not all(len(value) == len(values[0]) for value in values):
                return values[0]
            return [
                concat_values([value[index] for value in values])
                for index in range(len(values[0]))
            ]
        return values[0]

    return RolloutDenoisingEnv(
        image_kwargs=concat_values([env.image_kwargs for env in envs]),
        pos_cond_kwargs=concat_values([env.pos_cond_kwargs for env in envs]),
        neg_cond_kwargs=concat_values([env.neg_cond_kwargs for env in envs]),
        guidance=concat_values([env.guidance for env in envs]),
    )


def _slice_env_tensor(tensor: torch.Tensor, output_index: int) -> torch.Tensor:
    if tensor.dim() > 0 and output_index < tensor.shape[0]:
        return tensor[output_index : output_index + 1]
    return tensor


def _select_denoising_env(
    env: RolloutDenoisingEnv | None, output_index: int
) -> RolloutDenoisingEnv | None:
    if env is None:
        return None
    return RolloutDenoisingEnv(
        image_kwargs=_map_env_tensors(
            env.image_kwargs, lambda tensor: _slice_env_tensor(tensor, output_index)
        ),
        pos_cond_kwargs=_map_env_tensors(
            env.pos_cond_kwargs, lambda tensor: _slice_env_tensor(tensor, output_index)
        ),
        neg_cond_kwargs=_map_env_tensors(
            env.neg_cond_kwargs, lambda tensor: _slice_env_tensor(tensor, output_index)
        ),
        guidance=(
            _slice_env_tensor(env.guidance, output_index)
            if env.guidance is not None
            else None
        ),
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
        denoising_env=_concat_denoising_envs(
            [data.denoising_env for data in per_output]
        ),
        dit_trajectory=dit_trajectory,
    )


def _slice_output_dim(
    tensor: torch.Tensor | None, output_index: int
) -> torch.Tensor | None:
    """Keep one output row while preserving the batch dimension."""
    if tensor is None or tensor.dim() < 1 or output_index >= tensor.shape[0]:
        return tensor
    return tensor[output_index : output_index + 1]


def select_output_rollout_trajectory(
    rollout_trajectory_data: RolloutTrajectoryData | None,
    output_index: int | None,
) -> RolloutTrajectoryData | None:
    """Select one output row from a batched rollout trajectory."""
    if rollout_trajectory_data is None or output_index is None:
        return rollout_trajectory_data

    debug_tensors = rollout_trajectory_data.rollout_debug_tensors
    if debug_tensors is not None:
        debug_tensors = RolloutDebugTensors(
            rollout_variance_noises=_slice_output_dim(
                debug_tensors.rollout_variance_noises, output_index
            ),
            rollout_prev_sample_means=_slice_output_dim(
                debug_tensors.rollout_prev_sample_means, output_index
            ),
            rollout_noise_std_devs=_slice_output_dim(
                debug_tensors.rollout_noise_std_devs, output_index
            ),
            rollout_model_outputs=_slice_output_dim(
                debug_tensors.rollout_model_outputs, output_index
            ),
        )

    dit_trajectory = rollout_trajectory_data.dit_trajectory
    if dit_trajectory is not None:
        dit_trajectory = RolloutDitTrajectory(
            latents=_slice_output_dim(dit_trajectory.latents, output_index),
            timesteps=dit_trajectory.timesteps,
            sigmas=dit_trajectory.sigmas,
        )

    return RolloutTrajectoryData(
        rollout_log_probs=_slice_output_dim(
            rollout_trajectory_data.rollout_log_probs, output_index
        ),
        rollout_debug_tensors=debug_tensors,
        denoising_env=_select_denoising_env(
            rollout_trajectory_data.denoising_env, output_index
        ),
        dit_trajectory=dit_trajectory,
    )
