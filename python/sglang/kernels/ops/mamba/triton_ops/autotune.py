from __future__ import annotations

from typing import Callable, Mapping, Sequence

import triton

from sglang.kernels.ops.attention.fla.utils import autotune_cache_kwargs

__all__ = [
    "autotune_cache_kwargs",
    "chain_config_prunes",
    "config_spec",
    "prune_oversized_tiles",
    "prune_wrong_configs",
]


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


def config_spec(config: triton.Config) -> tuple[tuple[tuple[str, int], ...], int, int]:
    """A config's identity for comparison: its blocks, warps and stages."""
    return (
        tuple(sorted(config.kwargs.items())),
        config.num_warps,
        config.num_stages,
    )


def prune_wrong_configs(
    wrong: Sequence[Mapping[str, object]],
) -> Callable[..., list[triton.Config]]:
    """Build an `early_config_prune` that drops configs known to be incorrect.

    These configs are not slow, they compute the wrong result: on bounded
    inputs the kernel returns non-finite values, or a result that changes
    between two identical runs. The autotuner chooses by measured time, so one
    left in the list makes the model's output depend on which config happened
    to win a benchmark.

    Each entry gives the block sizes plus `num_warps` and `num_stages`. A list
    holding only one config is kept as it is: a caller that pins a single
    config is measuring it, and dropping it would leave nothing to launch.
    """
    specs = {
        (
            tuple(sorted((k, v) for k, v in entry.items() if k.startswith("BLOCK_"))),
            entry["num_warps"],
            entry["num_stages"],
        )
        for entry in wrong
    }

    def early_config_prune(configs, named_args, **kwargs):
        if len(configs) <= 1:
            return list(configs)
        return [c for c in configs if config_spec(c) not in specs]

    return early_config_prune


def chain_config_prunes(*prunes: Callable[..., list[triton.Config]]):
    """Apply several `early_config_prune` callables in order."""

    def early_config_prune(configs, named_args, **kwargs):
        for prune in prunes:
            configs = prune(configs, named_args, **kwargs)
        return configs

    return early_config_prune
