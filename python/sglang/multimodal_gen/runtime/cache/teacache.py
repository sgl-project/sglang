# SPDX-License-Identifier: Apache-2.0
"""
TeaCache: Temporal similarity-based caching for diffusion models.

TeaCache accelerates diffusion inference by selectively skipping redundant
computation when consecutive diffusion steps are similar enough. This is
achieved by tracking the L1 distance between modulated inputs across timesteps.

Key concepts:
- Modulated input: The input to transformer blocks after timestep conditioning
- L1 distance: Measures how different consecutive timesteps are
- Threshold: When accumulated L1 distance exceeds threshold, force computation
- CFG support: Separate caches for positive and negative branches

References:
- TeaCache: Accelerating Diffusion Models with Temporal Similarity
  https://arxiv.org/abs/2411.14324
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class TeaCacheContext:
    """Common context extracted for TeaCache skip decision.

    This context is populated from the forward_batch and forward_context
    during each denoising step, providing all information needed to make
    cache decisions.

    Attributes:
        current_timestep: Current denoising timestep index (0-indexed).
        num_inference_steps: Total number of inference steps.
        do_cfg: Whether classifier-free guidance is enabled.
        is_cfg_negative: True if currently processing negative CFG branch.
        teacache_thresh: Threshold for accumulated L1 distance.
        coefficients: Polynomial coefficients for L1 rescaling.
        teacache_params: Full TeaCacheParams for model-specific access.
    """

    current_timestep: int
    num_inference_steps: int
    do_cfg: bool
    is_cfg_negative: bool  # For CFG branch selection
    teacache_thresh: float
    coefficients: list[float]
    teacache_params: "TeaCacheParams"  # Full params for model-specific access


class TeaCacheMixin:
    """
    Mixin class providing TeaCache optimization functionality.

    TeaCache accelerates diffusion inference by selectively skipping redundant
    computation when consecutive diffusion steps are similar enough.

    This mixin should be inherited by DiT model classes that want to support
    TeaCache optimization. It provides:
    - State management for tracking L1 distances
    - CFG-aware caching (separate caches for positive/negative branches)
    - Decision logic for when to compute vs. use cache

    Example usage in a DiT model:
        class MyDiT(TeaCacheMixin, BaseDiT):
            def __init__(self, config, **kwargs):
                super().__init__(config, **kwargs)
                self._init_teacache_state()

            def forward(self, hidden_states, timestep, ...):
                ctx = self._get_teacache_context()
                if ctx is not None:
                    # Compute modulated input (model-specific, e.g., after timestep embedding)
                    modulated_input = self._compute_modulated_input(hidden_states, timestep)
                    is_boundary = (ctx.current_timestep == 0 or
                                   ctx.current_timestep >= ctx.num_inference_steps - 1)

                    should_calc = self._compute_teacache_decision(
                        modulated_inp=modulated_input,
                        is_boundary_step=is_boundary,
                        coefficients=ctx.coefficients,
                        teacache_thresh=ctx.teacache_thresh,
                    )

                    if not should_calc:
                        # Use cached residual (must implement retrieve_cached_states)
                        return self.retrieve_cached_states(hidden_states)

                # Normal forward pass...
                output = self._transformer_forward(hidden_states, timestep, ...)

                # Cache states for next step
                if ctx is not None:
                    self.maybe_cache_states(output, hidden_states)

                return output

    Subclass implementation notes:
        - `_compute_modulated_input()`: Model-specific method to compute the input
          after timestep conditioning (used for L1 distance calculation)
        - `retrieve_cached_states()`: Must be overridden to return cached output
        - `maybe_cache_states()`: Override to store states for cache retrieval

    Attributes:
        cnt: Counter for tracking steps.
        enable_teacache: Whether TeaCache is enabled.
        previous_modulated_input: Cached modulated input for positive branch.
        previous_residual: Cached residual for positive branch.
        accumulated_rel_l1_distance: Accumulated L1 distance for positive branch.
        is_cfg_negative: Whether currently processing negative CFG branch.
        _supports_cfg_cache: Whether this model supports CFG cache separation.

    CFG-specific attributes (only when _supports_cfg_cache is True):
        previous_modulated_input_negative: Cached input for negative branch.
        previous_residual_negative: Cached residual for negative branch.
        accumulated_rel_l1_distance_negative: L1 distance for negative branch.
    """

    # Models that support CFG cache separation (wan/hunyuan/zimage)
    # Models not in this set (flux/qwen) auto-disable TeaCache when CFG is enabled
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}

    def _init_teacache_state(self) -> None:
        """Initialize TeaCache state. Call this in subclass __init__."""
        # Common TeaCache state
        self.cnt = 0
        self.enable_teacache = True
        # Flag indicating if this model supports CFG cache separation
        self._supports_cfg_cache = (
            self.config.prefix.lower() in self._CFG_SUPPORTED_PREFIXES
        )

        # Always initialize positive cache fields (used in all modes)
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.accumulated_rel_l1_distance: float = 0.0

        self.is_cfg_negative = False
        # CFG-specific fields initialized to None (created when CFG is used)
        # These are only used when _supports_cfg_cache is True AND do_cfg is True
        if self._supports_cfg_cache:
            self.previous_modulated_input_negative: torch.Tensor | None = None
            self.previous_residual_negative: torch.Tensor | None = None
            self.accumulated_rel_l1_distance_negative: float = 0.0

    def reset_teacache_state(self) -> None:
        """Reset all TeaCache state at the start of each generation task."""
        self.cnt = 0

        # Primary cache fields (always present)
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0.0
        self.is_cfg_negative = False
        self.enable_teacache = True
        # CFG negative cache fields (always reset, may be unused)
        if self._supports_cfg_cache:
            self.previous_modulated_input_negative = None
            self.previous_residual_negative = None
            self.accumulated_rel_l1_distance_negative = 0.0

    def _compute_l1_and_decide(
        self,
        modulated_inp: torch.Tensor,
        coefficients: list[float],
        teacache_thresh: float,
    ) -> tuple[float, bool]:
        """
        Compute L1 distance and decide whether to calculate or use cache.

        Args:
            modulated_inp: Current timestep's modulated input.
            coefficients: Polynomial coefficients for L1 rescaling.
            teacache_thresh: Threshold for cache decision.

        Returns:
            Tuple of (new_accumulated_distance, should_calc).
        """
        prev_modulated_inp = (
            self.previous_modulated_input_negative
            if self.is_cfg_negative
            else self.previous_modulated_input
        )

        # Defensive check: if previous input is not set, force calculation
        if prev_modulated_inp is None:
            return 0.0, True

        # Compute relative L1 distance
        diff = modulated_inp - prev_modulated_inp
        rel_l1 = (diff.abs().mean() / prev_modulated_inp.abs().mean()).cpu().item()

        # Apply polynomial rescaling
        rescale_func = np.poly1d(coefficients)

        accumulated_rel_l1_distance = (
            self.accumulated_rel_l1_distance_negative
            if self.is_cfg_negative
            else self.accumulated_rel_l1_distance
        )
        accumulated_rel_l1_distance = accumulated_rel_l1_distance + rescale_func(rel_l1)

        if accumulated_rel_l1_distance >= teacache_thresh:
            # Threshold exceeded: force compute and reset accumulator
            return 0.0, True
        # Cache hit: keep accumulated distance
        return accumulated_rel_l1_distance, False

    def _compute_teacache_decision(
        self,
        modulated_inp: torch.Tensor,
        is_boundary_step: bool,
        coefficients: list[float],
        teacache_thresh: float,
    ) -> bool:
        """
        Compute cache decision for TeaCache.

        Args:
            modulated_inp: Current timestep's modulated input.
            is_boundary_step: True for boundary timesteps that always compute.
            coefficients: Polynomial coefficients for L1 rescaling.
            teacache_thresh: Threshold for cache decision.

        Returns:
            True if forward computation is needed, False to use cache.
        """
        if not self.enable_teacache:
            return True

        if is_boundary_step:
            new_accum, should_calc = 0.0, True
        else:
            new_accum, should_calc = self._compute_l1_and_decide(
                modulated_inp=modulated_inp,
                coefficients=coefficients,
                teacache_thresh=teacache_thresh,
            )

        # Advance baseline and accumulator for the active branch
        if not self.is_cfg_negative:
            self.previous_modulated_input = modulated_inp.clone()
            self.accumulated_rel_l1_distance = new_accum
        elif self._supports_cfg_cache:
            self.previous_modulated_input_negative = modulated_inp.clone()
            self.accumulated_rel_l1_distance_negative = new_accum

        return should_calc

    def _get_teacache_context(self) -> TeaCacheContext | None:
        """
        Check TeaCache preconditions and extract common context.

        Returns:
            TeaCacheContext if TeaCache is enabled and properly configured,
            None if should skip TeaCache logic entirely.
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch

        # Early return checks
        if (
            forward_batch is None
            or not forward_batch.enable_teacache
            or forward_batch.teacache_params is None
        ):
            return None

        teacache_params = forward_batch.teacache_params

        # Extract common values
        current_timestep = forward_context.current_timestep
        num_inference_steps = forward_batch.num_inference_steps
        do_cfg = forward_batch.do_classifier_free_guidance
        is_cfg_negative = forward_batch.is_cfg_negative

        # Reset at first timestep
        if current_timestep == 0 and not self.is_cfg_negative:
            self.reset_teacache_state()

        return TeaCacheContext(
            current_timestep=current_timestep,
            num_inference_steps=num_inference_steps,
            do_cfg=do_cfg,
            is_cfg_negative=is_cfg_negative,
            teacache_thresh=teacache_params.teacache_thresh,
            coefficients=teacache_params.coefficients,
            teacache_params=teacache_params,
        )

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache states for later retrieval. Override in subclass if needed."""
        pass

    def should_skip_forward_for_cached_states(self, **kwargs: dict[str, Any]) -> bool:
        """Check if forward can be skipped using cached states."""
        return False

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retrieve cached states. Must be implemented by subclass."""
        raise NotImplementedError("retrieve_cached_states is not implemented")
