from __future__ import annotations

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def describe_tree_cache_for_oom(tree_cache: BasePrefixCache | None) -> str:
    if tree_cache is not None:
        tree_cache.pretty_print()
    return tree_cache.available_and_evictable_str()


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()


class CacheFreeSpaceProvider:
    """FreeSpaceProvider backed by a prefix cache: makes room by evicting the
    radix tree, and reports the cache's availability on OOM."""

    def __init__(self, tree_cache: BasePrefixCache | None) -> None:
        self.tree_cache = tree_cache

    def ensure_free(self, num_tokens: int) -> None:
        evict_from_tree_cache(self.tree_cache, num_tokens)

    def describe_for_oom(self) -> str:
        return describe_tree_cache_for_oom(self.tree_cache)
