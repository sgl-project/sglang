"""Joint sequence layout for Z-Image consumers that supply full latent grids."""

import torch

from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    build_shard_plan,
    shard_like,
)

RopeCache = tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]


def shard_block_inputs(
    hidden_states: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    rope: RopeCache,
    mask: torch.Tensor | None,
    mask_meta: dict[str, torch.Tensor | int] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], RopeCache]:
    shard = build_shard_plan(hidden_states.shape[1])
    if shard.num_pad or mask is not None or mask_meta is not None:
        raise ValueError(
            "Joint sequence SP requires uniform, evenly divisible padded sequences"
        )
    cos, sin = freqs_cis
    cache, positions, complex_freqs = rope
    if positions is not None:
        # Indices still address the full batched cache.
        positions = shard_like(positions.view(hidden_states.shape[:2]), shard).reshape(
            -1
        )
    elif cache is not None:
        cache = shard_like(cache, shard, dim=0)
    if complex_freqs is not None:
        complex_freqs = shard_like(complex_freqs, shard, dim=0)
    local = shard_like(hidden_states, shard).contiguous()
    # Canonical singleton batch strides enable fused normalization.
    local = local.view(-1, local.shape[-1]).view(local.shape)
    return (
        local,
        (shard_like(cos, shard, dim=-2), shard_like(sin, shard, dim=-2)),
        (cache, positions, complex_freqs),
    )
