"""DeepSeek V4 pool layouts for CP Cache LayerSplit."""

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.mem_cache.cp_cache_layer_split.utils import (
    get_global_layer_shard_range,
)


@dataclass(frozen=True)
class CpCacheLayerSplitDeepSeekV4PoolLayout:
    """Per-rank layer-buffer counts for each V4 KV sub-pool."""

    swa_layer_num: int
    c4_layer_num: int
    c128_layer_num: int
    c4_indexer_layer_num: int
    c4_state_layer_num: int
    c128_state_layer_num: int
    c4_indexer_state_layer_num: int


def build_cp_cache_layer_split_deepseek_v4_pool_layout(
    cp_rank: int,
    cp_size: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
) -> CpCacheLayerSplitDeepSeekV4PoolLayout:
    """Buffer counts for one attention-CP rank."""
    if not 0 <= start_layer < end_layer_exclusive <= len(compression_ratios):
        raise ValueError(
            "Invalid DSV4 Cache LayerSplit stage: "
            f"start_layer={start_layer}, end_layer={end_layer_exclusive}, "
            f"compression_ratios={len(compression_ratios)}"
        )
    owned_start, owned_end = get_global_layer_shard_range(
        cp_rank,
        cp_size,
        start_layer,
        end_layer_exclusive - start_layer,
    )

    def _count(compress_ratio: int) -> int:
        return sum(
            1
            for layer_id in range(owned_start, owned_end)
            if compression_ratios[layer_id] == compress_ratio
        )

    c4_layer_num = _count(4)
    c128_layer_num = _count(128)
    return CpCacheLayerSplitDeepSeekV4PoolLayout(
        swa_layer_num=owned_end - owned_start,
        c4_layer_num=c4_layer_num,
        c128_layer_num=c128_layer_num,
        c4_indexer_layer_num=c4_layer_num,
        c4_state_layer_num=c4_layer_num,
        c128_state_layer_num=c128_layer_num,
        c4_indexer_state_layer_num=c4_layer_num,
    )


def build_cp_cache_layer_split_deepseek_v4_worst_case_pool_layout(
    cp_size: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
) -> CpCacheLayerSplitDeepSeekV4PoolLayout:
    """Max per-pool layer counts across all CP ranks."""
    layouts = [
        build_cp_cache_layer_split_deepseek_v4_pool_layout(
            cp_rank,
            cp_size,
            start_layer,
            end_layer_exclusive,
            compression_ratios,
        )
        for cp_rank in range(cp_size)
    ]
    return CpCacheLayerSplitDeepSeekV4PoolLayout(
        swa_layer_num=max(layout.swa_layer_num for layout in layouts),
        c4_layer_num=max(layout.c4_layer_num for layout in layouts),
        c128_layer_num=max(layout.c128_layer_num for layout in layouts),
        c4_indexer_layer_num=max(layout.c4_indexer_layer_num for layout in layouts),
        c4_state_layer_num=max(layout.c4_state_layer_num for layout in layouts),
        c128_state_layer_num=max(layout.c128_state_layer_num for layout in layouts),
        c4_indexer_state_layer_num=max(
            layout.c4_indexer_state_layer_num for layout in layouts
        ),
    )
