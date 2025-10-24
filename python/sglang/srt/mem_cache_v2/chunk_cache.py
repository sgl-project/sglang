import torch

from sglang.srt.mem_cache_v2.cache_index import CacheIndex, MatchResult


class ChunkCache(CacheIndex):
    """Cache index without prefix sharing - treats each request independently."""

    def __init__(self, page_size: int):
        self.page_size = page_size

    def match_prefix(self, key: tuple[int, ...]):
        """No prefix matching - always returns empty match."""
        return MatchResult(
            matched_indices=torch.tensor([], dtype=torch.int32), allocation_key=None
        )

    def insert(self, key: tuple[int, ...], value: torch.Tensor, allocation_key):
        """
        No caching - return empty indices.
        This signals that nothing went into the tree, so no deduplication happens.
        """
        return allocation_key, torch.tensor([], dtype=torch.int32)

    def allocate(self, allocation_key) -> torch.Tensor:
        """No indices need loading from cache."""
        return torch.tensor([], dtype=torch.int32)

    def evict(self, num_tokens: int) -> torch.Tensor:
        """No cache to evict from."""
        return torch.tensor([], dtype=torch.int32)

    def evictable_size(self) -> int:
        """No cache to evict from."""
        return 0

    def protected_size(self) -> int:
        """No cache to protect."""
        return 0
