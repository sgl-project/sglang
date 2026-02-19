# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class TeaCacheParams(CacheParams):
    cache_type: str = "teacache"
    teacache_thresh: float = 0.0
    coefficients: list[float] = field(default_factory=list)


@dataclass
class WanTeaCacheParams(CacheParams):
    # Unfortunately, TeaCache is very different for Wan than other models
    cache_type: str = "teacache"
    # Default threshold and coefficients are for Wan T2V 1.3B (use_ret_steps=True).
    # For other Wan variants, override these values via --teacache-params.
    teacache_thresh: float = 0.08
    use_ret_steps: bool = True
    ret_steps_coeffs: list[float] = field(
        default_factory=lambda: [
            -5.21862437e04,
            9.23041404e03,
            -5.28275948e02,
            1.36987616e01,
            -4.99875664e-02,
        ]
    )
    non_ret_steps_coeffs: list[float] = field(
        default_factory=lambda: [
            2.39676752e03,
            -1.31110545e03,
            2.01331979e02,
            -8.29855975e00,
            1.37887774e-01,
        ]
    )

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
