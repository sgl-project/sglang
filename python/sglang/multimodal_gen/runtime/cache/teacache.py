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
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

import torch

from sglang.multimodal_gen.runtime.cache import DiffusionCache

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


def _rescale_distance_tensor(
    coefficients: list[float], x: torch.Tensor
) -> torch.Tensor:
    """Polynomial rescaling using tensor operations (torch.compile friendly)."""
    c = coefficients
    return c[0] * x**4 + c[1] * x**3 + c[2] * x**2 + c[3] * x + c[4]


def _compute_rel_l1_distance_tensor(
    current: torch.Tensor, previous: torch.Tensor
) -> torch.Tensor:
    """Compute relative L1 distance as a tensor (torch.compile friendly)."""
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


class TeaCacheState:
    """Tracks step progress, cached tensors, and L1 distances for a single CFG path. Updated every timestep."""

    def __init__(self) -> None:
        self.step: int = 0
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.accumulated_rel_l1_distance: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear all cached tensors and reset the step counter for a new generation."""
        self.step = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = None

    def update(
        self, modulated_inp: torch.Tensor | None, previous_residual: torch.Tensor | None
    ) -> None:
        """Store the current modulated input and its computed residual for possible future reuse."""
        self.previous_modulated_input = modulated_inp
        self.previous_residual = previous_residual

    def __repr__(self):
        return f"TeaCacheState(step={self.step}, accumulated_rel_l1_distance={self.accumulated_rel_l1_distance})"


class TeaCacheStrategy(DiffusionCache):
    """Implements TeaCache to skip redundant diffusion forward passes.

    TeaCacheStrategy implements teacache as a `DiffusionCache` object. It
    manages two TeaCacheState objects (positive + optional negative CFG branch)
    and stores parameters needed to make skippind decision.
    """

    def __init__(self, supports_cfg: bool) -> None:
        """Initialize cache states for positive and optional negative CFG branches."""
        # params updated every forward pass
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if supports_cfg else None
        # params updated at the start of each new generation
        # set in maybe_reset()
        self.cache_params: TeaCacheParams | None = None
        self.coefficients: list[float] = []
        self.num_steps: int = 0
        self.start_skipping: int | None = None
        self.end_skipping: int | None = None

    def _get_state(self) -> TeaCacheState:
        """Select the appropriate cache state (positive/negative cfg) based on the forward context."""
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        fb = get_forward_context().forward_batch
        is_cfg_negative = fb.is_cfg_negative if fb is not None else False
        if is_cfg_negative and self.state_neg is not None:
            return self.state_neg
        return self.state

    def maybe_reset(self, **kwargs) -> None:
        """Maybe reset the TeaCacheState by doing three things:

        1. Reset TeaCacheState if the previous generation is complete
        2. Initialize parameters if at the start of a new generation.
        3. Increment the state's timestep counter (always)

        Called on every forward pass before should_skip().
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        state = self._get_state()

        # Reset state if we completed a generation
        if state.step == self.num_steps and state.step > 0:
            state.reset()

        # Initialize values if at the start of each new generation
        if state.step == 0:

            # set the teacache parameters
            fb = get_forward_context().forward_batch
            assert (
                fb is not None
            ), "TeaCacheStrategy required the forward_batch not be None"
            self.cache_params = getattr(fb.sampling_params, "teacache_params", None)

            # set the number of inference steps
            assert (
                self.cache_params is not None
            ), "TeaCacheStrategy requires teacache_params in sampling_params"
            self.num_steps = int(fb.num_inference_steps)

            # set the teacache coefficients
            if self.cache_params.coefficients_callback:
                self.coefficients = self.cache_params.coefficients_callback(
                    self.cache_params
                )
            else:
                self.coefficients = self.cache_params.coefficients

            # set the start and end skippable steps
            if isinstance(self.cache_params.start_skipping, float):
                start_skipping = int(self.num_steps * self.cache_params.start_skipping)
            elif self.cache_params.start_skipping < 0:
                start_skipping = self.num_steps + self.cache_params.start_skipping
            else:
                start_skipping = self.cache_params.start_skipping

            if isinstance(self.cache_params.end_skipping, float):
                end_skipping = int(self.num_steps * self.cache_params.end_skipping)
            elif self.cache_params.end_skipping < 0:
                end_skipping = self.num_steps + self.cache_params.end_skipping
            else:
                end_skipping = self.cache_params.end_skipping

            if start_skipping > end_skipping:
                logger.warning(
                    f"TeaCache skip window is invalid (start_skipping={self.start_skipping} > "
                    f"end_skipping={self.end_skipping}) for num_inference_steps={self.num_steps}. "
                    "This can happen during warmup runs with very few steps. TeaCache is disabled."
                )
                self.start_skipping = self.end_skipping = None
            else:
                self.start_skipping, self.end_skipping = start_skipping, end_skipping

            # increment the number of steps always
            state.step += 1

    def should_skip(
        self, modulated_input: torch.Tensor | None = None, **kwargs
    ) -> bool:
        """Decide whether this forward pass can be skipped based on the accumulated L1 distance of the modulated input."""
        state = self._get_state()
        assert self.cache_params is not None

        # No valid skip window for this generation
        if self.start_skipping is None or self.end_skipping is None:
            return False

        # Boundary steps always compute
        if state.step < self.start_skipping or state.step >= self.end_skipping:
            return False

        # First time computing, no previous input to compare against
        if state.accumulated_rel_l1_distance is None:
            state.accumulated_rel_l1_distance = torch.zeros(
                1, device=modulated_input.device, dtype=modulated_input.dtype
            )
            return False

        # compute the accumulated relative l1 distance
        assert state.previous_modulated_input is not None
        assert modulated_input is not None
        rel_l1 = _compute_rel_l1_distance_tensor(
            modulated_input, state.previous_modulated_input
        )
        rescaled = _rescale_distance_tensor(self.coefficients, rel_l1)
        state.accumulated_rel_l1_distance += rescaled

        # If below threshold, skip the forward pass
        if state.accumulated_rel_l1_distance < self.cache_params.rel_l1_thresh:
            return True

        # If threshold exceeded, reset accumulated so next window starts fresh
        state.accumulated_rel_l1_distance = torch.zeros(
            1, device=modulated_input.device, dtype=modulated_input.dtype
        )
        return False

    def write(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        modulated_input: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """After the forward pass, cache the residual and the current modulated input."""
        assert self.cache_params is not None
        residual = hidden_states.squeeze(0) - original_hidden_states
        state = self._get_state()
        state.update(modulated_input, residual)

    def read(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Before the forward pass, read from the cache and apply it to the current hidden states."""
        return hidden_states + self._get_state().previous_residual
