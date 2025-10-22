import logging
from collections import OrderedDict

import torch

# Set up logging for cache behavior
logger = logging.getLogger(__name__)


class MultiModalCache:
    """MultiModalCache is used to store vlm encoder results with LRU eviction"""

    def __init__(
        self,
        max_size: int,
    ):
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.current_size = 0

    def _allocate(self, embedding_size: int) -> bool:
        """Allocate space by evicting least recently used entries"""
        evictions = 0
        while self.current_size + embedding_size > self.max_size and self.mm_cache:
            _, old_embedding = self.mm_cache.popitem(last=False)
            evicted_size = self._get_tensor_size(old_embedding)
            self.current_size -= evicted_size
            evictions += evicted_size

        if evictions > 0:
            logger.debug(
                f"Cache eviction: evicted {evictions} bytes, remaining size: {self.current_size}/{self.max_size} bytes"
            )

        if self.current_size + embedding_size > self.max_size:
            return False
        return True

    def put(self, mm_hash: int, embedding: torch.Tensor) -> bool:
        data_size = self._get_tensor_size(embedding)
        # Lazy free cache if not enough space
        if not self._allocate(data_size):
            return False
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

    def get(self, mm_hash: int) -> torch.Tensor:
        """Get embedding and update LRU order"""
        if mm_hash in self.mm_cache:
            # Move to end (most recently used)
            self.mm_cache.move_to_end(mm_hash)
            return self.mm_cache[mm_hash]
        return None

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0

    def _get_tensor_size(self, embedding: torch.Tensor):
        return embedding.element_size() * embedding.numel()

    def __len__(self):
        return len(self.mm_cache)
