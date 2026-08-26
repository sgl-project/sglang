# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams

_NORM_MODES = ("mean", "peak", "none")


@dataclass
class SeaCacheParams(CacheParams):
    """
    Parameters for [SeaCache](https://arxiv.org/abs/2602.18993) spectral-evolution-aware
    caching.

    SeaCache reuses TeaCache's accumulate-and-refresh rule but measures the
    step-to-step distance after a timestep-dependent Wiener filter, so the metric
    tracks content change rather than noise. It needs no per-checkpoint fitted
    coefficients: `thresh` is the only value worth tuning.

    Attributes:
        cache_type: (`str`, defaults to `seacache`):
            A string labeling these parameters as belonging to seacache.
        thresh (`float`, defaults to `0.3`):
            Refresh threshold (paper: delta). The filtered relative-L1 distance is
            accumulated across steps and the denoiser is re-run once the sum reaches
            `thresh`. Larger values skip more steps.
        norm_mode (`str`, defaults to `mean`):
            Filter gain normalization. `mean` gives the filter unit mean gain over the
            spectrum, which is what makes distances comparable across timesteps and is
            the only mode any official call site uses. `peak` normalizes the maximum
            gain to 1 instead; `none` disables normalization and reproduces the
            paper's *w/o norm* ablation, which it reports as worse.

    CFG branches: SeaCache keeps independent state per classifier-free-guidance
    branch. Passing a negative prompt turns on two forwards per step and both branches
    accumulate separately.
    """

    cache_type: str = "seacache"
    thresh: float = 0.3
    norm_mode: str = "mean"

    def __post_init__(self) -> None:
        if (
            isinstance(self.thresh, bool)
            or not isinstance(self.thresh, (int, float))
            or not math.isfinite(self.thresh)
        ):
            raise ValueError("SeaCache thresh must be a finite number.")
        if self.thresh < 0:
            raise ValueError("SeaCache thresh must be non-negative.")
        if self.norm_mode not in _NORM_MODES:
            raise ValueError(
                f"SeaCache norm_mode must be one of {_NORM_MODES}, "
                f"got {self.norm_mode!r}."
            )
