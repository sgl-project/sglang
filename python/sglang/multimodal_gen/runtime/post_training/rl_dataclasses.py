# SPDX-License-Identifier: Apache-2.0
"""RL-specific dataclasses used by post-training and rollout paths."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RolloutSessionData:
    """Per-batch rollout state created by prepare_rollout(), lives on the batch object.

    Cleared by setting ``batch._rollout_session_data = None``.
    """

    pipeline_config: Any = None
    sigma_max: float = 0.0
    latents_shape: tuple | None = None
    noise_buffer: torch.Tensor | None = None

    local_log_prob_sum: list[torch.Tensor] = field(default_factory=list)
    local_log_prob_count: list[torch.Tensor] = field(default_factory=list)

    local_variance_noises: list[torch.Tensor] = field(default_factory=list)
    local_prev_sample_means: list[torch.Tensor] = field(default_factory=list)
    local_noise_std_devs: list[torch.Tensor] = field(default_factory=list)
    local_model_outputs: list[torch.Tensor] = field(default_factory=list)


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
        denoising_env=first.denoising_env,
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
        denoising_env=rollout_trajectory_data.denoising_env,
        dit_trajectory=dit_trajectory,
    )
