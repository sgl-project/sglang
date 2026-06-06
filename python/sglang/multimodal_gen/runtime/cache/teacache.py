# SPDX-License-Identifier: Apache-2.0
"""
TeaCache accelerates diffusion inference by skipping redundant forward
passes when consecutive denoising steps are sufficiently similar, as measured
by the accumulated relative L1 distance of modulated inputs.

References:
- TeaCache: Accelerating Diffusion Models with Temporal Similarity
  https://arxiv.org/abs/2411.14324
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import torch


@dataclass
class TeaCacheState:
    """Tracks step progress, cached tensors, and L1 distances for a single CFG path."""

    step: int = 0
    previous_modulated_input: torch.Tensor | None = None
    previous_residual: torch.Tensor | None = None
    accumulated_rel_l1_distance: torch.Tensor | None = None


def _rescale_distance_tensor(
    coefficients: list[float], x: torch.Tensor
) -> torch.Tensor:
    """Polynomial rescaling using tensor operations (torch.compile friendly)."""
    x = (
        x.float()
    )  # upcast to float32 for numerical stability, especially with higher degree polynomials
    result = torch.zeros_like(x)
    for i, c in enumerate(coefficients):
        result = result + c * x ** (len(coefficients) - 1 - i)
    return result


def _compute_rel_l1_distance_tensor(
    current: torch.Tensor, previous: torch.Tensor
) -> torch.Tensor:
    """Compute relative L1 distance as a tensor (torch.compile friendly)."""
    current, previous = (
        current.float(),
        previous.float(),
    )  # upcast to float32 for numerical stability
    prev_mean = previous.abs().mean()
    curr_diff_mean = (current - previous).abs().mean()
    rel_distance = torch.where(
        prev_mean > 1e-9,
        curr_diff_mean / prev_mean,
        torch.where(
            current.abs().mean() < 1e-9,
            torch.zeros(1, device=current.device, dtype=current.dtype),
            torch.full((1,), float("inf"), device=current.device, dtype=current.dtype),
        ),
    )
    return rel_distance.squeeze()


class TeaCacheStrategy:
    """Implements TeaCache to skip redundant diffusion forward passes.

    TeaCacheStrategy manages two TeaCacheState objects (positive + optional
    negative CFG branch) and stores parameters needed to make the skip decision.
    """

    def __init__(
        self,
        supports_cfg: bool,
        coefficients: list[float],
        rel_l1_thresh: float,
        start_skipping: int,
        end_skipping: int,
    ) -> None:
        """Initialize cache states and all generation parameters."""
        self.supports_cfg = supports_cfg
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if supports_cfg else None
        self.coefficients = coefficients
        self.rel_l1_thresh = rel_l1_thresh
        self.start_skipping = start_skipping
        self.end_skipping = end_skipping
        if start_skipping >= end_skipping:
            logger.warning(
                f"TeaCache skip window is invalid (start_skipping={start_skipping} >= "
                f"end_skipping={end_skipping}). This can happen during warmup runs with "
                "very few steps. Skipping disabled."
            )

    def reset_states(self) -> None:
        """Reset cache states, discarding any stale tensors from a previous generation."""
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if self.supports_cfg else None

    def _get_state(self) -> TeaCacheState:
        """Select the appropriate cache state (positive/negative cfg) based on the forward context."""
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_batch = get_forward_context().forward_batch
        is_cfg_negative = (
            forward_batch.is_cfg_negative if forward_batch is not None else False
        )
        if is_cfg_negative and self.state_neg is not None:
            return self.state_neg
        return self.state

    def step(self, modulated_input: torch.Tensor) -> bool:
        """Advance state and return whether this forward pass can be skipped."""
        state = self._get_state()
        step = state.step
        state.step += 1

        # Do not skip on the first step or if we are outside the skipping window
        in_skip_window = self.start_skipping <= step < self.end_skipping
        if state.previous_modulated_input is None or not in_skip_window:
            state.accumulated_rel_l1_distance = None
            state.previous_modulated_input = modulated_input.clone()
            return False

        # Compute the relative L1 distance and update the state
        rel_l1 = _compute_rel_l1_distance_tensor(
            modulated_input, state.previous_modulated_input
        )
        rescaled = _rescale_distance_tensor(self.coefficients, rel_l1)
        state.accumulated_rel_l1_distance = (
            rescaled
            if state.accumulated_rel_l1_distance is None
            else state.accumulated_rel_l1_distance + rescaled
        )
        state.previous_modulated_input = modulated_input.clone()

        # Skip if accumulated rel l1 is small
        if state.accumulated_rel_l1_distance < self.rel_l1_thresh:
            return True

        # Otherwise reset the accumulator and do not skip
        state.accumulated_rel_l1_distance = None
        return False

    def write(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        **kwargs,
    ) -> None:
        """After the forward pass, cache the residual."""
        state = self._get_state()
        state.previous_residual = hidden_states.squeeze(0) - original_hidden_states

    def read(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Before the forward pass, read from the cache and apply it to the current hidden states."""
        return hidden_states + self._get_state().previous_residual
