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
        previous_residual_norm: Norm of previous residual for positive branch.
        accumulated_error: Accumulated error for positive branch.
        consecutive_skips: Consecutive skip counter for positive branch.

    CFG-specific attributes (only when _supports_cfg_cache is True):
        previous_residual_norm_negative: Norm for negative branch.
        accumulated_error_negative: Accumulated error for negative branch.
        consecutive_skips_negative: Skip counter for negative branch.
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
        self.calibrate_magcache = True

        # MagCache-specific state
        self.previous_residual_norm: float = 0.0
        self.accumulated_error: float = 0.0
        self.consecutive_skips: int = 0

        # CFG negative branch (only if supported)
        if self._supports_cfg_cache:
            if not hasattr(self, "previous_residual_negative"):
                self.previous_residual_negative: torch.Tensor | None = None

            self.previous_residual_norm_negative: float = 0.0
            self.accumulated_error_negative: float = 0.0
            self.consecutive_skips_negative: int = 0

    def reset_magcache_state(self) -> None:
        """Reset MagCache state at the start of each generation task."""
        self.previous_residual_norm = 0.0
        self.accumulated_error = 0.0
        self.consecutive_skips = 0
        self.enable_magcache = True

        if self._supports_cfg_cache:
            self.previous_residual_norm_negative = 0.0
            self.accumulated_error_negative = 0.0
            self.consecutive_skips_negative = 0

    def _compute_magcache_decision(
        self,
        residual: torch.Tensor,
        current_timestep: int,
        mag_ratios: torch.Tensor,
        magcache_thresh: float,
        max_skip_steps: int,
        retention_steps: int,
        is_boundary_step: bool,
    ) -> bool:
        """
        Compute cache decision for MagCache.
        """

        if not self.enable_magcache:
            return True

        if is_boundary_step:
            current_norm = residual.norm(p=2).item()
            self._update_magcache_state(current_norm, 0.0, 0)
            return True

        # Select CFG branch state
        prev_norm = (
            self.previous_residual_norm_negative
            if self.is_cfg_negative
            else self.previous_residual_norm
        )
        accum_error = (
            self.accumulated_error_negative
            if self.is_cfg_negative
            else self.accumulated_error
        )
        consec_skips = (
            self.consecutive_skips_negative
            if self.is_cfg_negative
            else self.consecutive_skips
        )

        # First compute always
        if prev_norm == 0.0:
            current_norm = residual.norm(p=2).item()
            self._update_magcache_state(current_norm, 0.0, 0)
            return True


        branch_offset = 1 if self.is_cfg_negative else 0
        idx = current_timestep * 2 + branch_offset

        # must compute if out of bounds (should not happen if mag_ratios is properly calibrated/interpolated)
        if idx >= len(mag_ratios):
            return True

        gamma_hat = mag_ratios[idx].item()
        new_error = accum_error + abs(1.0 - gamma_hat)

        can_skip = (
            new_error <= magcache_thresh
            and consec_skips < max_skip_steps
            and self.cnt >= retention_steps
        )

        if can_skip:
            self._update_magcache_state(None, new_error, consec_skips + 1)
            return False
        else:
            current_norm = residual.norm(p=2).item()
            self._update_magcache_state(current_norm, 0.0, 0)
            return True

    def _calibrate_magcache(self, residual: torch.Tensor) -> None:
        """Calibration mode to collect magnitude ratios. save it json file"""
        current_norm = residual.norm(p=2).item()

        if self.previous_residual is not None:
            prev_norm = (
                self.previous_residual_norm_negative
                if self.is_cfg_negative
                else self.previous_residual_norm
            )
            ratio = current_norm / prev_norm if prev_norm > 0 else 1.0

            # Store ratio in magcache_params for later use (e.g., interpolation)
            # This is a simplified example; in practice, you might want to store
            # these ratios externally or handle them differently.
            if hasattr(self.magcache_params, "mag_ratios"):
                self.magcache_params.mag_ratios.append(ratio)

        self.previous_residual = residual.detach()

    def _update_magcache_state(
        self, norm: float | None, error: float, skips: int
    ) -> None:
        """Update MagCache state for active CFG branch."""

        if not self.is_cfg_negative:
            if norm is not None:
                self.previous_residual_norm = norm
            self.accumulated_error = error
            self.consecutive_skips = skips
        elif self._supports_cfg_cache:
            if norm is not None:
                self.previous_residual_norm_negative = norm
            self.accumulated_error_negative = error
            self.consecutive_skips_negative = skips

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

        # Get magcache_params from forward_batch, or fall back to sampling_params
        magcache_params = forward_batch.magcache_params
        if magcache_params is None and hasattr(forward_batch, 'sampling_params'):
            magcache_params = getattr(forward_batch.sampling_params, 'magcache_params', None)

        if magcache_params is None:
            ic("magcache_params is None even after fallback")
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
