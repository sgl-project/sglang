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
        rel_l1_thresh (`float`, defaults to `0.0`):
            Threshold for accumulated relative L1 distance. When below this threshold, the
            forward pass is skipped. Recommended values: 0.25 for ~1.5x speedup, 0.4 for ~1.8x,
            0.6 for ~2.0x.
        start_skipping (`int` or `float`, defaults to `5`):
            The number of timesteps after which we may skip a forward pass. These early
            steps define the global structure and are too critical to not skip.
            int: The number of timesteps after which we can skip. If negative,
                 this is an offset from the end of the schedule.
            float (0.0 - 1.0): A percentage of the total steps (e.g., 0.1
                               computes the first 10%).
        end_skipping (`int` or `float`, defaults to `-1`):
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
    rel_l1_thresh: float = 0.0
    start_skipping: int | float = 5
    end_skipping: int | float = -1
    coefficients: list[float] = field(default_factory=list)
    coefficients_callback: Callable[[TeaCacheParams], list[float]] | None = field(
        default=None, repr=False
    )
    use_ret_steps: bool | None = None

    def get_coefficients(self) -> list[float]:
        if self.coefficients_callback is not None:
            return self.coefficients_callback(self)
        return self.coefficients

    def get_skip_boundaries(
        self, num_inference_steps: int, do_cfg: bool
    ) -> tuple[int, int]:
        def _resolve_boundary(value: int | float) -> int:
            if isinstance(value, float):
                return int(num_inference_steps * value)
            if value < 0:
                return num_inference_steps + value
            return value

        start_skipping = _resolve_boundary(self.start_skipping)
        end_skipping = _resolve_boundary(self.end_skipping)

        if do_cfg:
            start_skipping *= 2
            end_skipping *= 2

        return start_skipping, end_skipping
