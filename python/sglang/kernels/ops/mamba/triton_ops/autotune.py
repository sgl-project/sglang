from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping, Sequence

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import autotune_cache_kwargs

__all__ = ["autotune_cache_kwargs", "prune_oversized_tiles", "unused_index"]


# Stand-in for an absent index or flag tensor: the autotune key holds every tensor
# argument's dtype, so passing None would tune pool mode separately from the old path.
@lru_cache(maxsize=None)
def unused_index(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(1, dtype=dtype, device=device)


def _tile_footprint(config: triton.Config) -> tuple[int, int, int]:
    footprint = 1
    for block in config.kwargs.values():
        footprint *= block
    return footprint, config.num_stages, config.num_warps


def prune_oversized_tiles(
    tiled_dims: Mapping[str, Sequence[str]],
) -> Callable[..., list[triton.Config]]:
    """Build an `early_config_prune` that drops tiles wider than what they cover.

    A block larger than the dimension it tiles computes a fully masked remainder
    for the same grid, so it is dominated; `tiled_dims` maps each BLOCK_SIZE_* to
    the kernel arguments it strides over, and a block is capped by the smallest.
    """

    def early_config_prune(configs, named_args, **kwargs):
        # Triton splits the launch into positional args and kwargs; a dimension
        # can arrive through either.
        args = {**kwargs, **named_args}
        caps = {}
        for block, dims in tiled_dims.items():
            cap = triton.next_power_of_2(min(args[dim] for dim in dims))
            caps[block] = max(cap, min(c.kwargs[block] for c in configs))
        keep = [
            config
            for config in configs
            if all(config.kwargs[block] <= cap for block, cap in caps.items())
        ]
        return keep or [min(configs, key=_tile_footprint)]

    return early_config_prune
