"""get_rope / get_rotary_pos_embed factory functions and module-level caches."""

from collections import OrderedDict
from typing import Any

import torch

from .base import LinearScalingRotaryEmbedding, RotaryEmbedding
from .mrope import NDRotaryEmbedding, _to_tuple

_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}
_ND_ROPE_CACHE: "OrderedDict[tuple, NDRotaryEmbedding]" = OrderedDict()
_ROPE_3D_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int | float,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    max_position_embeddings = max_position
    rope_type = None
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
        if rope_type in (None, "default"):
            rope_scaling = None
        elif rope_type == "linear":
            factor = float(rope_scaling.get("factor", 1.0))
            original_max = rope_scaling.get("original_max_position_embeddings", None)
            if original_max is not None:
                max_position_embeddings = max(
                    max_position_embeddings, int(float(original_max) * factor)
                )
    key = (
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
    else:
        if rope_type == "linear":
            factor = float(rope_scaling.get("factor", 1.0))
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
                base=base,
                is_neox_style=is_neox_style,
                dtype=dtype,
                scaling_factor=factor,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling {rope_scaling}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


def get_rotary_pos_embed(
    rope_sizes,
    hidden_size,
    heads_num,
    rope_dim_list,
    rope_theta,
    theta_rescale_factor=1.0,
    interpolation_factor=1.0,
    shard_dim: int = 0,
    dtype: torch.dtype = torch.float32,
    start_frame: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rotary positional embeddings for the given sizes.

    Args:
        rope_sizes: Tuple of dimensions (t, h, w)
        hidden_size: Hidden dimension size
        heads_num: Number of attention heads
        rope_dim_list: List of dimensions for each axis, or None
        rope_theta: Base for frequency calculations
        theta_rescale_factor: Rescale factor for theta. Defaults to 1.0
        interpolation_factor: Factor to scale positions. Defaults to 1.0
        shard_dim: Which dimension to shard for sequence parallelism. Defaults to 0.

    Returns:
        Tuple of (cos, sin) tensors for rotary embeddings
    """

    target_ndim = 3
    head_dim = hidden_size // heads_num

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]

    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"

    # Get SP info - now handled within NDRotaryEmbedding
    # sp_group = get_sp_group()
    # sp_rank = sp_group.rank_in_group
    # sp_world_size = sp_group.world_size

    # Simple LRU cache keyed by parameters
    global _ND_ROPE_CACHE
    key = (
        tuple(rope_dim_list),
        float(rope_theta),
        (
            tuple(theta_rescale_factor)
            if isinstance(theta_rescale_factor, list)
            else float(theta_rescale_factor)
        ),
        (
            tuple(interpolation_factor)
            if isinstance(interpolation_factor, list)
            else float(interpolation_factor)
        ),
        dtype,
    )

    cache_hit = key in _ND_ROPE_CACHE
    if cache_hit:
        rope_emb = _ND_ROPE_CACHE.pop(key)
        _ND_ROPE_CACHE[key] = rope_emb  # move to end (most-recent)
    else:
        rope_emb = NDRotaryEmbedding(
            rope_dim_list=rope_dim_list,
            rope_theta=rope_theta,
            theta_rescale_factor=theta_rescale_factor,
            interpolation_factor=interpolation_factor,
            dtype=dtype,
        )
        _ND_ROPE_CACHE[key] = rope_emb
        if len(_ND_ROPE_CACHE) > 16:
            # pop least-recently-used
            _ND_ROPE_CACHE.pop(next(iter(_ND_ROPE_CACHE)))

    freqs_cos, freqs_sin = rope_emb.forward_from_grid(
        grid_size=_to_tuple(rope_sizes, dim=3),
        shard_dim=shard_dim,
        start_frame=start_frame,
        device=device,
    )
    return freqs_cos, freqs_sin
