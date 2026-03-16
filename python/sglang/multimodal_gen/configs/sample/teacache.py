# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class TeaCacheParams(CacheParams):
    """
    Parameters for [TeaCache](https://arxiv.org/abs/2411.14324).

    Attributes:
        cache_type: (`str`, defaults to `teacache`):
            A string labeling these parameters as belonging to teacache.
        teacache_thresh (`float`, defaults to `0.0`):
            Threshold for accumulated relative L1 distance. When below this threshold, the
            forward pass is skipped. Recommended values: 0.25 for ~1.5x speedup, 0.4 for ~1.8x,
            0.6 for ~2.0x.
        start_skipping (`int`, 'float', defaults to `0.2`):
            The number of timesteps after which we may skip a forward pass. These early
            steps define the global structure and are too critical to not skip.
            int: The number of timesteps after which we can skip. If negative,
                 this is an offset from the end of the schedule.
            float (0.0 - 1.0): A percentage of the total steps (e.g., 0.1
                               computes the first 10%).
        end_skipping (`int`, defaults to `-1`):
            The number of timesteps after which we are no longer able to skip
            forward passes. The last steps refine fine textures and details.
            int: The number of timesteps after which skipping ends. If negative,
                 this is an offset from the total number of steps.
            float (0.0 - 1.0): A percentage of the total steps (e.g., 0.1
                               computes the first 10%).
        coefficients (`List[float]`, defaults to `[]`):
            Polynomial coefficients for rescaling the raw relative L1 distance,
            evaluated as `c[0]*x**4 + c[1]*x**3 + c[2]*x**2 + c[3]*x + c[4]`.
        coefficients_callback (`Callable[[TeaCacheParams], List[float]]`, *optional*):
            A function that receives this `TeaCacheParams` instance and returns
            the polynomial coefficients to use. When set, it takes precedence over
            the `coefficients` field, allowing dynamic coefficient selection based
            on any property of the params (e.g., `use_ret_steps` for Wan models).
        use_ret_steps: (`bool`, `None`, defaults to `None`):
            Used exclusively for wanvideo models to select different modulated inputs.
    """
    cache_type: str = "teacache"
    teacache_thresh: float = 0.0
    start_skipping: int = 5
    end_skipping: int = -1
    coefficients: list[float] = field(default_factory=list)
    coefficients_callback: Callable[[TeaCacheParams], list[float]] | None = field(
        default=None, repr=False
    )
    use_ret_steps: bool | None = None


@dataclass
class WanTeaCacheParams(CacheParams):
    # Unfortunately, TeaCache is very different for Wan than other models
    cache_type: str = "teacache"
    teacache_thresh: float = 0.0
    use_ret_steps: bool = True
    ret_steps_coeffs: list[float] = field(default_factory=list)
    non_ret_steps_coeffs: list[float] = field(default_factory=list)

    @property
    def coefficients(self) -> list[float]:
        if self.use_ret_steps:
            return self.ret_steps_coeffs
        else:
            return self.non_ret_steps_coeffs

    @property
    def ret_steps(self) -> int:
        if self.use_ret_steps:
            return 5 * 2
        else:
            return 1 * 2

    def get_cutoff_steps(self, num_inference_steps: int) -> int:
        if self.use_ret_steps:
            return num_inference_steps * 2
        else:
            return num_inference_steps * 2 - 2
