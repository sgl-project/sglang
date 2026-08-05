# SPDX-License-Identifier: Apache-2.0
"""Request-scoped configuration for EasyCache feature reuse."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class EasyCacheParams(CacheParams):
    """Controls EasyCache's online block-residual reuse policy.

    ``threshold`` is the maximum accumulated estimated relative output change
    before the transformer blocks are recomputed. A non-positive threshold
    disables reuse. ``warmup_steps`` always run densely so the online
    input/output change ratio can be established.
    """

    cache_type: str = "easycache"
    threshold: float = 0.1
    warmup_steps: int = 3
    subsample_stride: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError(
                f"EasyCache threshold must be a number, got {self.threshold!r}"
            )
        if not math.isfinite(float(self.threshold)):
            raise ValueError(
                f"EasyCache threshold must be finite, got {self.threshold!r}"
            )
        if not isinstance(self.warmup_steps, int) or isinstance(
            self.warmup_steps, bool
        ):
            raise TypeError(
                "EasyCache warmup_steps must be a non-negative integer, "
                f"got {self.warmup_steps!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                "EasyCache warmup_steps must be non-negative, "
                f"got {self.warmup_steps!r}"
            )
        if (
            not isinstance(self.subsample_stride, int)
            or isinstance(self.subsample_stride, bool)
            or self.subsample_stride <= 0
        ):
            raise ValueError(
                "EasyCache subsample_stride must be a positive integer, "
                f"got {self.subsample_stride!r}"
            )
