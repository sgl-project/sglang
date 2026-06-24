"""CP KV LayerSplit public helpers.

Keep V4 pool/helper imports on their submodule paths; re-exporting them here
would reintroduce a circular import with ``deepseek_v4_memory_pool``.
"""

from sglang.srt.mem_cache.cp_kv_layer_split.deepseek_v4_layout import (
    CpKvLayerSplitDeepSeekV4PoolLayout,
    any_cp_kv_layer_split_cache_sharded,
    build_cp_kv_layer_split_deepseek_v4_pool_layout,
    build_cp_kv_layer_split_deepseek_v4_worst_case_pool_layout,
    cp_kv_layer_split_sharding_flags,
    shard_cp_kv_layer_split_c4,
    shard_cp_kv_layer_split_c4_indexer,
    shard_cp_kv_layer_split_c128,
    shard_cp_kv_layer_split_swa,
)
from sglang.srt.mem_cache.cp_kv_layer_split.ownership import (
    CP_KV_LAYER_SPLIT_SUPPORTED_MODEL_ARCHS,
    assert_cp_kv_layer_split_hicache_supported,
    build_owned_layer_local_index_map,
    kv_layer_owner,
    kv_layer_owner_global_rank,
    layers_per_cp_rank,
    num_owned_compress_layers,
    num_owned_kv_layers,
    num_stage_compress_layers,
    owned_kv_layer_range,
    owns_kv_layer,
    should_use_cp_kv_layer_split_pool,
    validate_cp_kv_layer_split_model_arch,
)
from sglang.srt.mem_cache.cp_kv_layer_split.pool_base import (
    CpKvLayerSplitPoolBase,
    is_cp_kv_layer_split_pool,
)


def maybe_reset_cp_kv_layer_split_active_pages(pool) -> None:
    """Drop per-forward active-page caches before a new forward."""
    if is_cp_kv_layer_split_pool(pool):
        pool.reset_batch_active_pages()


__all__ = [
    "CP_KV_LAYER_SPLIT_SUPPORTED_MODEL_ARCHS",
    "CpKvLayerSplitDeepSeekV4PoolLayout",
    "CpKvLayerSplitPoolBase",
    "any_cp_kv_layer_split_cache_sharded",
    "assert_cp_kv_layer_split_hicache_supported",
    "build_cp_kv_layer_split_deepseek_v4_pool_layout",
    "build_cp_kv_layer_split_deepseek_v4_worst_case_pool_layout",
    "build_owned_layer_local_index_map",
    "cp_kv_layer_split_sharding_flags",
    "is_cp_kv_layer_split_pool",
    "kv_layer_owner",
    "kv_layer_owner_global_rank",
    "layers_per_cp_rank",
    "maybe_reset_cp_kv_layer_split_active_pages",
    "num_owned_compress_layers",
    "num_owned_kv_layers",
    "num_stage_compress_layers",
    "owned_kv_layer_range",
    "owns_kv_layer",
    "shard_cp_kv_layer_split_c4",
    "shard_cp_kv_layer_split_c4_indexer",
    "shard_cp_kv_layer_split_c128",
    "shard_cp_kv_layer_split_swa",
    "should_use_cp_kv_layer_split_pool",
    "validate_cp_kv_layer_split_model_arch",
]
