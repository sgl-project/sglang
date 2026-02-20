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

from icecream import ic
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
        do_cfg: Whether classifier-free guidance is enabled.
        is_cfg_negative: True if currently processing negative CFG branch.
    """

    current_timestep: int
    cnt: int
    do_cfg: bool
    is_cfg_negative: bool
    magcache_params: "MagCacheParams | None" = None


class MagCacheMixin:
    # Models that support CFG cache separation (same as TeaCache)
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}
    config: DiTConfig

    def init(self, is_cfg_negative:bool, magcache_params:"MagCacheParams") -> None:

        self.num_steps = magcache_params.num_steps # todo: don't hardcode
        self.retention_ratio = magcache_params.retention_ratio
        self.magcache_thresh = magcache_params.threshold
        self.max_skip_steps = magcache_params.max_skip_steps
        self.mag_ratios = magcache_params.mag_ratios
        self.use_ret_steps = magcache_params.use_ret_steps

        self.min_steps = int(self.num_steps * self.retention_ratio) * 2 if self.use_ret_steps else 2
        self.max_steps = self.num_steps * 2 if self.use_ret_steps else self.num_steps * 2 - 2

        # save calibrated magnitude ratios
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.calibration_path = f"cache/magcache_calibration_{timestamp}.jsonl"
        os.makedirs(os.path.dirname(self.calibration_path), exist_ok=True)

        # track previous residual separately for positive and negative branches in CFG
        if not is_cfg_negative:
            self.previous_residual: torch.Tensor | None = None
            self.previous_residual_norm: float = 0.0
        else:
            self.previous_residual_negative: torch.Tensor | None = None
            self.previous_residual_norm_negative: float = 0.0

        # track magnitude ratio and accumulated error for skip decision
        self.reset(is_cfg_negative)

    def reset(self, is_cfg_negative):
        if not is_cfg_negative:
            self.norm_ratio = 1.0
            self.accumulated_error = 0.0
            self.consecutive_skips = 0
            self.previous_residual: torch.Tensor | None = None
            self.previous_residual_norm: float = 0.0
        else:
            self.norm_ratio_negative = 1.0
            self.accumulated_error_negative = 0.0
            self.consecutive_skips_negative = 0
            self.previous_residual_negative: torch.Tensor | None = None
            self.previous_residual_norm_negative: float = 0.0

    def should_skip_forward(self, current_timestep, cnt, do_cfg, is_cfg_negative=False):

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

    def calibrate_magcache(self, ctx, hidden_states, original_hidden_states):

        prev_residual = self.previous_residual_negative if ctx.is_cfg_negative else self.previous_residual
        if prev_residual is None:
            mag_ratio = 1.0
            mag_std = 0.0
            cos_dis = 0.0
        else:
            curr_residual = hidden_states.squeeze(0) - original_hidden_states
            mag_ratio = ((curr_residual.norm(dim=-1)/prev_residual.norm(dim=-1)).mean()).item()
            mag_std = (curr_residual.norm(dim=-1)/prev_residual.norm(dim=-1)).std().item()
            cos_dis = (1-F.cosine_similarity(curr_residual, prev_residual, dim=-1, eps=1e-8)).mean().item()

        with open(self.calibration_path, "a") as f:
            f.write(json.dumps({"cnt": ctx.cnt, "mag_ratio": mag_ratio, "mag_std": mag_std, "cos_dis": cos_dis, "negative": ctx.is_cfg_negative}) + "\n")

    def _get_magcache_context(self) -> MagCacheContext | None:
        from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None:
            return None

        # unpack parameters
        current_timestep=forward_context.current_timestep
        do_cfg=forward_batch.do_classifier_free_guidance
        is_cfg_negative=forward_batch.is_cfg_negative
        magcache_params=getattr(forward_batch.sampling_params, "magcache_params", None)
        assert magcache_params is not None, "MagCache parameters not found in sampling_params."

        # init cache at the start of each generation
        if current_timestep == 0:
            self.init(is_cfg_negative, magcache_params)

        # compute cnt index differently for cond and uncond branches in CFG
        cnt = current_timestep
        if do_cfg:
            cnt = current_timestep * 2 + (1 if is_cfg_negative else 0)

        return MagCacheContext(
            current_timestep=current_timestep,
            cnt=cnt,
            do_cfg=do_cfg,
            is_cfg_negative=is_cfg_negative,
            magcache_params=magcache_params,
        )
