# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class MagCacheParams(CacheParams):
    """
    MagCache configuration for magnitude-ratio-based caching.

    MagCache accelerates diffusion inference by skipping forward passes when
    magnitude ratios of consecutive residuals are predictably similar.

    Attributes:
        threshold: Accumulated error threshold (default 0.06 from paper).
                   Lower = higher quality but slower. Higher = faster but lower quality.
        max_skip_steps: Maximum consecutive skips allowed (default 3).
                        Prevents infinite skipping even if error is low.
        skip_start_step: Number of denoising steps at the start where skipping is disabled.
        skip_end_step: Number of denoising steps at the end where skipping is disabled (0 = active until last step).
    """

    cache_type: str = "magcache"
    threshold: float = 0.12
    max_skip_steps: int = 4
    skip_start_step: int = 10
    skip_end_step: int = 0
    mag_ratios: list[float] | None = None
