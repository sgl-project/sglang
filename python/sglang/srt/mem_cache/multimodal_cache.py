from typing import Dict

import torch


class MultiModalCache:
    """MultiModalCache is used to store vlm encoder results"""

    def __init__(
        self,
        max_size: int,
    ):
        self.max_size = max_size
        self.mm_cache: Dict[int, torch.Tensor] = {}
        self.current_size = 0

    def put(self, mm_hash: int, embedding: torch.Tensor) -> bool:
        if mm_hash in self.mm_cache:
            return True
        data_size = self._get_tensor_size(embedding)
        if self.current_size + data_size > self.max_size:
            return False
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def get(self, mm_hash: int) -> torch.Tensor:
        return self.mm_cache.get(mm_hash)

    def free(self, mm_hash: int) -> bool:
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self.current_size -= self._get_tensor_size(old_embedding)
        return True

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0

    def _get_tensor_size(self, embedding: torch.Tensor):
        return embedding.element_size() * embedding.numel()

    def __len__(self):
        return len(self.mm_cache)
