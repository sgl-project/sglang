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
