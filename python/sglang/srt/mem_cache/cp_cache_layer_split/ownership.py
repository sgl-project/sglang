"""Stage-local ownership helpers for CP Cache LayerSplit."""

from __future__ import annotations

from sglang.srt.layers.cp.utils import get_layer_shard_range


def get_global_layer_shard_range(
    rank: int,
    shard_size: int,
    start_layer: int,
    layer_num: int,
) -> tuple[int, int]:
    """Global layer range owned by ``rank`` within one PP stage."""
    if shard_size <= 0 or not 0 <= rank < shard_size:
        raise ValueError(f"Invalid rank={rank} for shard_size={shard_size}")
    if start_layer < 0 or layer_num <= 0:
        raise ValueError(
            f"Invalid stage start_layer={start_layer}, layer_num={layer_num}"
        )
    local_start, local_end = get_layer_shard_range(rank, shard_size, layer_num)
    return start_layer + local_start, start_layer + local_end
