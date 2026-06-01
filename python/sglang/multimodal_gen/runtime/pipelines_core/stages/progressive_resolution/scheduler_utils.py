# SPDX-License-Identifier: Apache-2.0
"""
Stage-transition math for progressive resolution growing.

Computes the sigma thresholds at which denoising should move from a coarser
to a finer latent grid, based on the Bayes-optimal frequency-activation
criterion (paper Eq. 125-129 / 142-146).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class StageTransitions:
    """Output of compute_stage_transitions."""

    # stage_sigmas[s] = sigma threshold at which stage s begins (stage 1 → 1.0)
    stage_sigmas: dict[int, float] = field(default_factory=dict)
    # transition_steps[s] = step index of first step in stage s (stage 1 → 0)
    transition_steps: dict[int, int] = field(default_factory=dict)
    num_stages: int = 1


def _P_omega(w: float, A: float, beta: float) -> float:
    """Radial power at frequency w: P(w) = A * |w|^(-beta)."""
    return A * abs(w) ** (-beta)


def _activation_time(P: float, delta: float) -> float:
    """Frequency activation time t_w = 1 / (1 + sqrt(delta / (P * (1+P-delta))))."""
    denom = P * (1.0 + P - delta)
    if denom <= 0 or delta >= 1.0 + P:
        raise ValueError(
            f"delta={delta} >= 1+P={1+P:.4f}; criterion trivially satisfied."
        )
    return 1.0 / (1.0 + math.sqrt(delta / denom))


def compute_stage_transitions(
    delta: float,
    n_levels: int,
    A: float,
    beta: float,
    H_lat: int,
    W_lat: int,
) -> dict[int, float]:
    """Compute sigma thresholds for each stage transition.

    Returns stage_sigmas: {stage: sigma_threshold}.  Stage 1 starts at 1.0.
    """
    num_stages = n_levels + 1
    sigmas: dict[int, float] = {1: 1.0}
    for s in range(2, num_stages + 1):
        H_prev = H_lat // (2 ** (num_stages - s + 1))
        W_prev = W_lat // (2 ** (num_stages - s + 1))
        w = min(H_prev, W_prev) // 2
        P_w = _P_omega(w, A, beta)
        sigmas[s] = _activation_time(P_w, delta)
    return sigmas


def find_transition_steps(
    scheduler_sigmas: torch.Tensor,
    stage_sigmas: dict[int, float],
    n_steps: int,
) -> dict[int, int]:
    """Map stage thresholds → step indices using the scheduler's sigma schedule.

    Conservative: transition fires at the first step where sigmas[i] <= threshold.
    Returns {stage: step_index} for stages >= 2.
    """
    transitions: dict[int, int] = {}
    sigmas_cpu = scheduler_sigmas.cpu()
    for s, thresh in stage_sigmas.items():
        if s == 1:
            continue
        found = n_steps
        for i in range(n_steps):
            if sigmas_cpu[i].item() <= thresh:
                found = i
                break
        transitions[s] = found
    return transitions


def reset_scheduler_at_step(scheduler: object, step_index: int) -> None:
    """Reset solver state so the scheduler continues cleanly from step_index.

    Sets _step_index = step_index and leaves _begin_index = 0 (as set by
    set_begin_index(0) in _before_denoising_loop) so that
    scheduler.step_index == _step_index + _begin_index == step_index.
    """
    if hasattr(scheduler, "model_outputs"):
        solver_order = getattr(
            getattr(scheduler, "config", None),
            "solver_order",
            len(scheduler.model_outputs),
        )
        scheduler.model_outputs = [None] * solver_order
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    if hasattr(scheduler, "this_order"):
        scheduler.this_order = 0
    if hasattr(scheduler, "timestep_list"):
        solver_order = getattr(
            getattr(scheduler, "config", None),
            "solver_order",
            len(scheduler.timestep_list),
        )
        scheduler.timestep_list = [None] * solver_order
    # _begin_index stays at 0 (set by set_begin_index(0) in _before_denoising_loop)
    scheduler._step_index = step_index
