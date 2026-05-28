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


@dataclass
class RolloutTrajectoryData:
    rollout_log_probs: torch.Tensor | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    denoising_env: RolloutDenoisingEnv | None = None
    dit_trajectory: RolloutDitTrajectory | None = None
