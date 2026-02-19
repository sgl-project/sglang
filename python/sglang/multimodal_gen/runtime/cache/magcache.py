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
import torch.nn.functional as F

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
    # Models that support CFG cache separation (same as TeaCache)
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}
    config: DiTConfig

    def init(self, magcache_params: "MagCacheParams") -> None:

        self.num_steps = magcache_params.num_steps # todo: don't hardcode
        self.retention_ratio = magcache_params.retention_ratio
        self.magcache_thresh = magcache_params.threshold
        self.max_skip_steps = magcache_params.max_skip_steps
        self.mag_ratios = magcache_params.mag_ratios
        self.use_ret_steps = magcache_params.use_ret_steps

        # cnt = step * 2 + branch_offset, so ranges over [0, num_steps*2)
        # min_steps and max_steps must be in the same cnt space
        self.min_steps = int(self.num_steps * self.retention_ratio) * 2 if self.use_ret_steps else 2
        self.max_steps = self.num_steps * 2 if self.use_ret_steps else self.num_steps * 2 - 2

        self.previous_residual: torch.Tensor | None = None
        self.previous_residual_norm: float = 0.0
        self.norm_ratio: float = 1.0
        self.accumulated_error: float = 0.0
        self.consecutive_skips: int = 0

        # save calibrated magnitude ratios
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.calibration_path = f"cache/magcache_calibration_{timestamp}.jsonl"
        os.makedirs(os.path.dirname(self.calibration_path), exist_ok=True)

        # CFG negative branch (only if supported)
        self._supports_cfg_cache = self.config.prefix.lower() in self._CFG_SUPPORTED_PREFIXES
        if self._supports_cfg_cache:
            self.previous_residual_negative: torch.Tensor | None = None
            self.previous_residual_norm_negative: float = 0.0
            self.norm_ratio_negative: float = 1.0
            self.accumulated_error_negative: float = 0.0
            self.consecutive_skips_negative: int = 0

    def reset(self, is_cfg_negative):
        if not is_cfg_negative:
            self.norm_ratio = 1.0
            self.accumulated_error = 0.0
            self.consecutive_skips = 0
        else:
            self.norm_ratio_negative = 1.0
            self.accumulated_error_negative = 0.0
            self.consecutive_skips_negative = 0

    def should_skip_forward(self, current_timestep, do_cfg, is_cfg_negative=False):

        # cnt is the global step index that accounts for both cond and uncond passes when doing CFG
        cnt = current_timestep * 2 + (1 if is_cfg_negative else 0) if do_cfg else current_timestep
        ic(current_timestep, cnt, do_cfg, is_cfg_negative)

        # reset at timestep 0
        if current_timestep == 0:
            self.reset(is_cfg_negative)

        accumulated_error = self.accumulated_error_negative if is_cfg_negative else self.accumulated_error
        consecutive_skips = self.consecutive_skips_negative if is_cfg_negative else self.consecutive_skips
        norm_ratio = self.norm_ratio_negative if is_cfg_negative else self.norm_ratio

        # always compute first few and last few steps
        is_boundary_step = cnt < self.min_steps or cnt >= self.max_steps
        if is_boundary_step:
            ic(f"Boundary step (cnt={cnt}), computing without cache. Resetting MagCache state.")
            self.reset(is_cfg_negative)
            return False

        cur_mag_ratio = self.mag_ratios[cnt] # access pre-calibrated magnitude ratio for current step
        norm_ratio = norm_ratio * cur_mag_ratio # magnitude ratio between current step and the cached step
        consecutive_skips += 1 # skip steps plus 1
        cur_skip_err = abs(1 - norm_ratio) # skip error of current steps
        accumulated_error += cur_skip_err # accumulated error of multiple steps

        if accumulated_error < self.magcache_thresh and consecutive_skips <= self.max_skip_steps:
            # Write updated state back before returning
            if is_cfg_negative:
                self.norm_ratio_negative = norm_ratio
                self.accumulated_error_negative = accumulated_error
                self.consecutive_skips_negative = consecutive_skips
            else:
                self.norm_ratio = norm_ratio
                self.accumulated_error = accumulated_error
                self.consecutive_skips = consecutive_skips
            return True
        else:
            self.reset(is_cfg_negative)
            return False

    def calibrate(self, hidden_states, original_hidden_states, current_timestep, do_cfg, is_cfg_negative=False):
        cnt = current_timestep * 2 + (1 if is_cfg_negative else 0) if do_cfg else current_timestep

        prev_residual = self.previous_residual_negative if is_cfg_negative else self.previous_residual
        if prev_residual is None:
            # Step 0 has no previous residual to compare against — write placeholder 1.0
            with open(self.calibration_path, "a") as f:
                f.write(json.dumps({"cnt": cnt, "mag_ratio": 1.0, "mag_std": 0.0, "cos_dis": 0.0, "negative": is_cfg_negative}) + "\n")
            return None

        curr_residual = hidden_states.squeeze(0) - original_hidden_states
        mag_ratio = ((curr_residual.norm(dim=-1)/prev_residual.norm(dim=-1)).mean()).item()
        mag_std = (curr_residual.norm(dim=-1)/prev_residual.norm(dim=-1)).std().item()
        cos_dis = (1-F.cosine_similarity(curr_residual, prev_residual, dim=-1, eps=1e-8)).mean().item()

        with open(self.calibration_path, "a") as f:
            f.write(json.dumps({"cnt": cnt, "mag_ratio": mag_ratio, "mag_std": mag_std, "cos_dis": cos_dis, "negative": is_cfg_negative}) + "\n")
