# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import torch

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
        retention_ratio: Fraction of initial steps to always compute (default 0.2).
                         First 20% of steps are important, never skip them.
    """

    cache_type: str = "magcache"
    threshold: float = 0.06
    max_skip_steps: int = 3
    retention_ratio: float = 0.2
    # Note: mag_ratios calibration data would be added here in future
    # mag_ratios: dict[int, float] = field(default_factory=dict)


@dataclass
class WanMagCacheParams(CacheParams):
    """
    Wan-specific MagCache parameters.

    Wan uses special ret_steps and cutoff_steps logic similar to WanTeaCacheParams.
    """

    cache_type: str = "magcache"
    threshold: float = 0.06
    max_skip_steps: int = 3
    use_ret_steps: bool = True

    @property
    def ret_steps(self) -> int:
        """Retention steps (always compute first N steps)."""
        return 5 * 2 if self.use_ret_steps else 1 * 2

    def get_cutoff_steps(self, num_inference_steps: int) -> int:
        """Cutoff steps (always compute last few steps)."""
        return num_inference_steps * 2 if self.use_ret_steps else num_inference_steps * 2 - 2



# from https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4Wan2.1/magcache_generate.py#L912
T2V_13B_MAG_RATIOS = torch.tensor([
    1.0, 1.0,
    1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762,
    0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575,
    0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549,
    0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514,
    0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416,
    0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304,
    0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135,
    0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849,
    0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359,
    0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411,
    0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089,
    0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868,
    0.81838, 0.81939
])

def nearest_interp(data:torch.Tensor, target_len:int):
    """Simple nearest neighbor interpolation for 1D arrays."""
    indices = torch.linspace(0, len(data) - 1, target_len)
    return data[torch.round(indices).long()]

def get_interpolated_mag_ratios(sample_steps:int, raw_ratios=T2V_13B_MAG_RATIOS):
    """
    Interpolates magnitude ratios to match the number of sampling steps.
    Returns a flattened array of [cond, uncond] pairs.
    """
    # The original logic assumes ratios are stored as [cond, uncond, cond, uncond...]
    # If the current total length doesn't match steps * 2, interpolate
    if len(raw_ratios) != sample_steps * 2:
        # Separate conditional and unconditional streams
        mag_ratio_con = nearest_interp(raw_ratios[0::2], sample_steps)
        mag_ratio_ucon = nearest_interp(raw_ratios[1::2], sample_steps)

        # Zip them back together and flatten
        return torch.stack([mag_ratio_con, mag_ratio_ucon], dim=1).flatten()
    return raw_ratios
