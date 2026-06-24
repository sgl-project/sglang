"""DeepSeek V4 pool layouts for CP KV LayerSplit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.cp_kv_layer_split.ownership import (
    num_owned_kv_layers,
    owned_kv_layer_range,
)


@dataclass(frozen=True)
class CpKvLayerSplitDeepSeekV4PoolLayout:
    """Per-rank layer-buffer counts for each V4 KV sub-pool."""

    swa_layer_num: int
    c4_layer_num: int
    c128_layer_num: int
    c4_indexer_layer_num: int
    c4_state_layer_num: int
    c128_state_layer_num: int
    c4_indexer_state_layer_num: int


_V4_DISABLE_SHARDING_ENVS = {
    "swa": envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_SWA_SHARDING,
    "c4": envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_C4_SHARDING,
    "c128": envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_C128_SHARDING,
    "c4_indexer": envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_C4_INDEXER_SHARDING,
}


def _shard_family(family: str) -> bool:
    return not _V4_DISABLE_SHARDING_ENVS[family].get()


def shard_cp_kv_layer_split_swa() -> bool:
    return _shard_family("swa")


def shard_cp_kv_layer_split_c4() -> bool:
    return _shard_family("c4")


def shard_cp_kv_layer_split_c128() -> bool:
    return _shard_family("c128")


def shard_cp_kv_layer_split_c4_indexer() -> bool:
    return _shard_family("c4_indexer")


def cp_kv_layer_split_sharding_flags() -> dict[str, bool]:
    return {family: _shard_family(family) for family in _V4_DISABLE_SHARDING_ENVS}


def any_cp_kv_layer_split_cache_sharded() -> bool:
    return any(cp_kv_layer_split_sharding_flags().values())


def _num_stage_compress_layers(
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
    compress_ratio: int,
) -> int:
    return sum(
        1
        for layer_id in range(start_layer, end_layer_exclusive)
        if compression_ratios[layer_id] == compress_ratio
    )


def _num_owned_compress_layers(
    cp_rank: int,
    cp_size: int,
    model_num_hidden_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
    compress_ratio: int,
) -> int:
    owned_start, owned_end = owned_kv_layer_range(
        cp_rank, cp_size, model_num_hidden_layers, start_layer, end_layer_exclusive
    )
    return _num_stage_compress_layers(
        owned_start,
        owned_end,
        compression_ratios,
        compress_ratio,
    )


def _family_layer_count(
    *,
    sharded: bool,
    cp_rank: int,
    cp_size: int,
    model_num_hidden_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
    compress_ratio: Optional[int] = None,
) -> int:
    """Count one DSV4 KV family for a pool layout."""
    if compress_ratio is None:
        if not sharded:
            return end_layer_exclusive - start_layer
        return num_owned_kv_layers(
            cp_rank,
            cp_size,
            model_num_hidden_layers,
            start_layer,
            end_layer_exclusive,
        )

    if not sharded:
        return _num_stage_compress_layers(
            start_layer,
            end_layer_exclusive,
            compression_ratios,
            compress_ratio,
        )
    return _num_owned_compress_layers(
        cp_rank,
        cp_size,
        model_num_hidden_layers,
        start_layer,
        end_layer_exclusive,
        compression_ratios,
        compress_ratio,
    )


def build_cp_kv_layer_split_deepseek_v4_pool_layout(
    cp_rank: int,
    cp_size: int,
    model_num_hidden_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
) -> CpKvLayerSplitDeepSeekV4PoolLayout:
    """Buffer counts for one attention-CP rank."""
    shard_swa = shard_cp_kv_layer_split_swa()
    shard_c4 = shard_cp_kv_layer_split_c4()
    shard_c128 = shard_cp_kv_layer_split_c128()
    shard_c4_indexer = shard_cp_kv_layer_split_c4_indexer()

    def _count(sharded: bool, compress_ratio: Optional[int] = None) -> int:
        return _family_layer_count(
            sharded=sharded,
            cp_rank=cp_rank,
            cp_size=cp_size,
            model_num_hidden_layers=model_num_hidden_layers,
            start_layer=start_layer,
            end_layer_exclusive=end_layer_exclusive,
            compression_ratios=compression_ratios,
            compress_ratio=compress_ratio,
        )

    return CpKvLayerSplitDeepSeekV4PoolLayout(
        swa_layer_num=_count(shard_swa),
        c4_layer_num=_count(shard_c4, compress_ratio=4),
        c128_layer_num=_count(shard_c128, compress_ratio=128),
        c4_indexer_layer_num=_count(shard_c4_indexer, compress_ratio=4),
        c4_state_layer_num=_count(shard_c4, compress_ratio=4),
        c128_state_layer_num=_count(shard_c128, compress_ratio=128),
        c4_indexer_state_layer_num=_count(shard_c4_indexer, compress_ratio=4),
    )


def build_cp_kv_layer_split_deepseek_v4_worst_case_pool_layout(
    cp_size: int,
    model_num_hidden_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
) -> CpKvLayerSplitDeepSeekV4PoolLayout:
    """Max per-pool layer counts across all CP ranks."""
    layouts = [
        build_cp_kv_layer_split_deepseek_v4_pool_layout(
            cp_rank,
            cp_size,
            model_num_hidden_layers,
            start_layer,
            end_layer_exclusive,
            compression_ratios,
        )
        for cp_rank in range(cp_size)
    ]
    return CpKvLayerSplitDeepSeekV4PoolLayout(
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
