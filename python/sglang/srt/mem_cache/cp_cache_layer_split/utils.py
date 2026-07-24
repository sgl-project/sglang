"""Common configuration and ownership helpers for CP Cache LayerSplit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def should_use_cp_cache_layer_split_pool(model_runner: ModelRunner) -> bool:
    """Whether this model runner should construct a Cache LayerSplit pool."""
    return (
        not model_runner.is_draft_worker
        and model_runner.server_args.enable_cp_cache_layer_split
    )


def get_cp_cache_layer_shard_info(
    model_runner: ModelRunner,
) -> tuple[Optional[int], int]:
    """Return ``(cp_rank, cp_size)`` for a Cache LayerSplit pool.

    ``(None, 1)`` means that this model runner should use its regular pool.
    """
    if not should_use_cp_cache_layer_split_pool(model_runner):
        return None, 1

    parallel = get_parallel()
    if parallel.attn_cp_size <= 1:
        return None, 1
    return parallel.attn_cp_rank, parallel.attn_cp_size


def get_layer_shard_range(
    rank: int, shard_size: int, total_layers: int
) -> tuple[int, int]:
    """Contiguous ``[start, end)`` local-layer range owned by ``rank``.

    Layers are split as evenly as possible; the first ``total_layers %
    shard_size`` ranks own one extra layer.
    """
    base = total_layers // shard_size
    rem = total_layers % shard_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def get_global_layer_shard_range(
    rank: int,
    shard_size: int,
    start_layer: int,
    layer_num: int,
) -> tuple[int, int]:
    """Global layer range owned by ``rank`` within one pipeline stage."""
    if shard_size <= 0 or not 0 <= rank < shard_size:
        raise ValueError(f"Invalid rank={rank} for shard_size={shard_size}")
    if start_layer < 0 or layer_num <= 0:
        raise ValueError(
            f"Invalid stage start_layer={start_layer}, layer_num={layer_num}"
        )
    local_start, local_end = get_layer_shard_range(rank, shard_size, layer_num)
    return start_layer + local_start, start_layer + local_end


def get_layer_owner(local_layer_idx: int, shard_size: int, total_layers: int) -> int:
    """CP rank that owns ``local_layer_idx`` under the contiguous split."""
    for rank in range(shard_size):
        start, end = get_layer_shard_range(rank, shard_size, total_layers)
        if start <= local_layer_idx < end:
            return rank
    raise ValueError(
        f"Invalid local_layer_idx={local_layer_idx} for "
        f"shard_size={shard_size}, total_layers={total_layers}"
    )
