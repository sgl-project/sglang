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
    build_owned_layer_local_index_map,
    kv_layer_owner,
    kv_layer_owner_global_rank,
    layers_per_cp_rank,
    num_owned_kv_layers,
    owned_kv_layer_range,
    owns_kv_layer,
)
from sglang.srt.mem_cache.cp_kv_layer_split.pool_base import (
    CpKvLayerSplitPoolBase,
    is_cp_kv_layer_split_pool,
)


def should_use_cp_kv_layer_split_pool(server_args=None) -> bool:
    """True when prefill CP KV LayerSplit should wire the specialized pool."""
    from sglang.srt.server_args import get_global_server_args

    args = server_args or get_global_server_args()
    return bool(
        args.enable_cp_kv_layer_split
        and args.enable_dsa_prefill_context_parallel
        and args.attn_cp_size > 1
    )


def maybe_reset_cp_kv_layer_split_active_pages(pool) -> None:
    """Drop per-forward active-page caches before a new forward."""
    if is_cp_kv_layer_split_pool(pool):
        pool.reset_batch_active_pages()


__all__ = [
    "CpKvLayerSplitDeepSeekV4PoolLayout",
    "CpKvLayerSplitPoolBase",
    "any_cp_kv_layer_split_cache_sharded",
    "build_cp_kv_layer_split_deepseek_v4_pool_layout",
    "build_cp_kv_layer_split_deepseek_v4_worst_case_pool_layout",
    "build_owned_layer_local_index_map",
    "cp_kv_layer_split_sharding_flags",
    "is_cp_kv_layer_split_pool",
    "kv_layer_owner",
    "kv_layer_owner_global_rank",
    "layers_per_cp_rank",
    "maybe_reset_cp_kv_layer_split_active_pages",
    "num_owned_kv_layers",
    "owned_kv_layer_range",
    "owns_kv_layer",
    "shard_cp_kv_layer_split_c4",
    "shard_cp_kv_layer_split_c4_indexer",
    "shard_cp_kv_layer_split_c128",
    "shard_cp_kv_layer_split_swa",
    "should_use_cp_kv_layer_split_pool",
]
