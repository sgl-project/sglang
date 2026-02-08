# SPDX-License-Identifier: Apache-2.0
"""
MagCache: Magnitude-ratio-based caching for diffusion models.

MagCache accelerates diffusion inference by skipping forward passes when
magnitude ratios of consecutive residuals are predictably similar.

Key differences from TeaCache:
- Uses magnitude ratios of residuals instead of L1 distance of inputs
- Tracks consecutive_skips counter to prevent infinite skipping
- Simpler accumulation (no polynomial rescaling)

References:
- MagCache: Fast Video Generation with Magnitude-Aware Cache
  https://openreview.net/forum?id=KZn7TDOL4J
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.multimodal_gen.configs.models import DiTConfig

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.magcache import MagCacheParams


@dataclass
class MagCacheContext:
    """
    Context for MagCache skip decision.

    This context is populated from the forward_batch and forward_context
    during each denoising step, providing all information needed to make
    cache decisions.

    Attributes:
        current_timestep: Current denoising timestep index (0-indexed).
        num_inference_steps: Total number of inference steps.
        do_cfg: Whether classifier-free guidance is enabled.
        is_cfg_negative: True if currently processing negative CFG branch.
        magcache_thresh: Threshold for accumulated error.
        max_skip_steps: Maximum consecutive skips allowed.
        retention_steps: Always compute for first N steps.
        magcache_params: Full MagCacheParams for model-specific access.
    """

    current_timestep: int
    num_inference_steps: int
    do_cfg: bool
    is_cfg_negative: bool
    magcache_thresh: float
    max_skip_steps: int
    retention_steps: int
    magcache_params: "MagCacheParams"


class MagCacheMixin:
    """
    Standalone mixin for MagCache optimization (does NOT inherit from TeaCacheMixin).

    MagCache accelerates diffusion inference by selectively skipping redundant
    computation when consecutive diffusion steps have predictable magnitude ratios.

    This mixin should be inherited by DiT model classes that want to support
    MagCache optimization. It provides:
    - State management for tracking magnitude ratios
    - CFG-aware caching (separate caches for positive/negative branches)
    - Decision logic for when to compute vs. use cache

    Note: This mixin shares some state with TeaCacheMixin (previous_residual, cnt,
    is_cfg_negative). The initialization checks if TeaCache already set these up.

    Attributes:
        enable_magcache: Whether MagCache is enabled.
        magcache_previous_residual_norm: Norm of previous residual for positive branch.
        magcache_accumulated_error: Accumulated error for positive branch.
        magcache_consecutive_skips: Consecutive skip counter for positive branch.

    CFG-specific attributes (only when _supports_cfg_cache is True):
        magcache_previous_residual_norm_negative: Norm for negative branch.
        magcache_accumulated_error_negative: Accumulated error for negative branch.
        magcache_consecutive_skips_negative: Skip counter for negative branch.
    """

    # Models that support CFG cache separation (same as TeaCache)
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}
    config: DiTConfig

    def _init_magcache_state(self) -> None:
        """
        Initialize MagCache state.

        Checks if shared state is already initialized by TeaCacheMixin.
        If not, initializes shared state, then adds MagCache-specific state.
        """
        # Initialize shared state if not present (TeaCache might have already done this)
        if not hasattr(self, "previous_residual"):
            self.previous_residual: torch.Tensor | None = None
            self.cnt = 0
            self.is_cfg_negative = False
            self._supports_cfg_cache = (
                self.config.prefix.lower() in self._CFG_SUPPORTED_PREFIXES
            )

        # Always set magcache flag
        self.enable_magcache = True

        # MagCache-specific state
        self.magcache_previous_residual_norm: float = 0.0
        self.magcache_accumulated_error: float = 0.0
        self.magcache_consecutive_skips: int = 0

        # CFG negative branch (only if supported)
        if self._supports_cfg_cache:
            if not hasattr(self, "previous_residual_negative"):
                self.previous_residual_negative: torch.Tensor | None = None

            self.magcache_previous_residual_norm_negative: float = 0.0
            self.magcache_accumulated_error_negative: float = 0.0
            self.magcache_consecutive_skips_negative: int = 0

    def reset_magcache_state(self) -> None:
        """Reset MagCache state at the start of each generation task."""
        self.magcache_previous_residual_norm = 0.0
        self.magcache_accumulated_error = 0.0
        self.magcache_consecutive_skips = 0
        self.enable_magcache = True

        if self._supports_cfg_cache:
            self.magcache_previous_residual_norm_negative = 0.0
            self.magcache_accumulated_error_negative = 0.0
            self.magcache_consecutive_skips_negative = 0

    def _compute_magcache_decision(
        self,
        residual: torch.Tensor,
        is_boundary_step: bool,
        magcache_thresh: float,
        max_skip_steps: int,
        retention_steps: int,
    ) -> bool:
        """
        Compute cache decision for MagCache.

        Args:
            residual: Current residual (output - input).
            is_boundary_step: Force computation for boundary steps.
            magcache_thresh: Threshold for accumulated error.
            max_skip_steps: Maximum consecutive skips allowed.
            retention_steps: Always compute for first N steps.

        Returns:
            True if should compute, False to use cache.
        """
        if not self.enable_magcache:
            return True

        if is_boundary_step:
            self._update_magcache_state(None, 0.0, 0)
            return True

        # Get state for current CFG branch
        prev_norm = (
            self.magcache_previous_residual_norm_negative
            if self.is_cfg_negative
            else self.magcache_previous_residual_norm
        )
        accum_error = (
            self.magcache_accumulated_error_negative
            if self.is_cfg_negative
            else self.magcache_accumulated_error
        )
        consec_skips = (
            self.magcache_consecutive_skips_negative
            if self.is_cfg_negative
            else self.magcache_consecutive_skips
        )

        if prev_norm == 0.0:
            # First step, force compute and store norm
            current_norm = residual.norm(p=2).item()
            self._update_magcache_state(current_norm, 0.0, 0)
            return True

        # Compute magnitude ratio γ = ||residual_t||₂ / ||residual_{t-1}||₂
        current_norm = residual.norm(p=2).item()
        gamma = current_norm / prev_norm

        # Accumulate error: error += |1 - γ|
        new_error = accum_error + abs(1.0 - gamma)

        # Skip conditions: all must be true to skip
        can_skip = (
            new_error <= magcache_thresh
            and consec_skips < max_skip_steps
            and self.cnt >= retention_steps
        )

        # Update state
        if can_skip:
            self._update_magcache_state(current_norm, new_error, consec_skips + 1)
            return False  # Use cache
        else:
            self._update_magcache_state(current_norm, 0.0, 0)
            return True  # Compute

    def _update_magcache_state(
        self, norm: float | None, error: float, skips: int
    ) -> None:
        """Update MagCache state for active CFG branch."""
        if norm is not None:
            if not self.is_cfg_negative:
                self.magcache_previous_residual_norm = norm
                self.magcache_accumulated_error = error
                self.magcache_consecutive_skips = skips
            elif self._supports_cfg_cache:
                self.magcache_previous_residual_norm_negative = norm
                self.magcache_accumulated_error_negative = error
                self.magcache_consecutive_skips_negative = skips

    def _get_magcache_context(self) -> MagCacheContext | None:
        """
        Check MagCache preconditions and extract common context.

        Returns:
            MagCacheContext if MagCache is enabled and properly configured,
            None if should skip MagCache logic entirely.
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch

        # Early return checks
        if forward_batch is None or not forward_batch.enable_magcache:
            return None

        magcache_params = forward_batch.magcache_params
        if magcache_params is None:
            return None

        # Reset at timestep 0
        if forward_context.current_timestep == 0 and not self.is_cfg_negative:
            self.reset_magcache_state()

        # Compute retention_steps
        if hasattr(magcache_params, "ret_steps"):
            retention_steps = magcache_params.ret_steps  # Wan-specific
        else:
            retention_steps = int(
                magcache_params.retention_ratio * forward_batch.num_inference_steps
            )

        return MagCacheContext(
            current_timestep=forward_context.current_timestep,
            num_inference_steps=forward_batch.num_inference_steps,
            do_cfg=forward_batch.do_classifier_free_guidance,
            is_cfg_negative=forward_batch.is_cfg_negative,
            magcache_thresh=magcache_params.threshold,
            max_skip_steps=magcache_params.max_skip_steps,
            retention_steps=retention_steps,
            magcache_params=magcache_params,
        )
