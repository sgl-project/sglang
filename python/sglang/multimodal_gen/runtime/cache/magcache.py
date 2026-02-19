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

import json
import os
from dataclasses import dataclass
from datetime import datetime
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
        """Initialize MagCache state."""
        if not hasattr(self, "previous_residual"):
            self.previous_residual: torch.Tensor | None = None
            self.cnt = 0
            self.is_cfg_negative = False
            self._supports_cfg_cache = (
                self.config.prefix.lower() in self._CFG_SUPPORTED_PREFIXES
            )

        # Always set magcache flag
        self.enable_magcache = True
        self.calibrate_magcache = True # todo: fix

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
        do_cfg: bool,
        is_cfg_negative: bool,
    ) -> bool:
        """
        Compute cache decision for MagCache.
        """

        # if MagCache is not enabled, always compute
        if not self.enable_magcache:
            return True

        # must have previous residual to make decision
        if residual is None:
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

        # must compute if out of bounds (should not happen if mag_ratios is properly interpolated)
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

    def _calibrate_magcache(self, cnt:int, hidden_states:torch.Tensor, original_hidden_states:torch.Tensor, is_negative: bool = False) -> None:
        """
        Calibration mode to collect magnitude ratios and save them to a JSON file.

        This method tracks the magnitude ratios between consecutive residuals during
        diffusion inference. The ratios are collected for both positive and negative
        CFG branches (interleaved) and saved to disk when generation completes.
        """


        # only calibrate starting in the second denoising step
        if cnt == 0: return

        current_residual = hidden_states.squeeze(0) - original_hidden_states
        current_norm = current_residual.norm(p=2)

        previous_residual = self.previous_residual_negative if is_negative else self.previous_residual
        assert previous_residual is not None
        previous_norm = previous_residual.norm(p=2)

        import torch.nn.functional as F
        norm_ratio = (current_norm / previous_norm).mean().item()
        norm_std = (current_norm / previous_norm).std().item()
        cos_dis = (1-F.cosine_similarity(current_residual, previous_residual, dim=-1, eps=1e-8)).mean().item()


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

    def _get_calibration_cache_dir(self) -> str:
        """Get the directory for storing MagCache calibration files."""
        from sglang.multimodal_gen import SGLANG_DIFFUSION_CACHE_ROOT
        calibration_dir = os.path.join(SGLANG_DIFFUSION_CACHE_ROOT, "magcache_calibrations")
        os.makedirs(calibration_dir, exist_ok=True)
        return calibration_dir

    def _save_calibration_data(self, num_steps: int, do_cfg: bool) -> None:
        """
        Save collected magnitude ratios to a JSON file.

        Args:
            num_steps: Number of inference steps used during calibration.
            do_cfg: Whether classifier-free guidance was enabled.
        """
        if not self.calibration_ratios:
            return

        # Generate filename based on model and configuration
        model_name = self.config.prefix.lower()  # e.g., "wan", "flux", etc.
        cfg_suffix = "_cfg" if do_cfg else "_nocfg"
        filename = f"{model_name}_{num_steps}steps{cfg_suffix}.json"

        calibration_dir = self._get_calibration_cache_dir()
        filepath = os.path.join(calibration_dir, filename)

        # Prepare calibration data
        calibration_data = {
            "model": model_name,
            "num_steps": num_steps,
            "do_cfg": do_cfg,
            "mag_ratios": self.calibration_ratios,
            "calibration_date": datetime.now().isoformat(),
            "num_ratios": len(self.calibration_ratios),
            "recommended_threshold": 0.12 if model_name == "wan" else 0.06,
            "recommended_max_skip_steps": 4 if model_name == "wan" else 3,
            "recommended_retention_ratio": 0.2,
        }

        # Save to JSON
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"✓ MagCache calibration saved to: {filepath}")
            print(f"  - Model: {model_name}")
            print(f"  - Steps: {num_steps}")
            print(f"  - CFG: {do_cfg}")
            print(f"  - Ratios collected: {len(self.calibration_ratios)}")
        except Exception as e:
            print(f"⚠ Failed to save MagCache calibration: {e}")

    @staticmethod
    def load_calibration_data(
        model_name: str, num_steps: int, do_cfg: bool
    ) -> dict | None:
        """
        Load calibration data from disk.

        Args:
            model_name: Model identifier (e.g., "wan", "flux").
            num_steps: Number of inference steps.
            do_cfg: Whether CFG is enabled.

        Returns:
            Dictionary containing calibration data, or None if not found.
        """
        cache_root = os.path.expanduser("~/.cache/sgl_diffusion")
        calibration_dir = os.path.join(cache_root, "magcache_calibrations")

        cfg_suffix = "_cfg" if do_cfg else "_nocfg"
        filename = f"{model_name}_{num_steps}steps{cfg_suffix}.json"
        filepath = os.path.join(calibration_dir, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"⚠ Failed to load MagCache calibration from {filepath}: {e}")
            return None

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
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        server_args = get_global_server_args()
        if not server_args.enable_magcache:
            return None

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None:
            return None

        magcache_params = server_args.magcache_params

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
