import logging
from collections import OrderedDict
from typing import Dict, Optional, List
import abc


import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ):
        ...

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """
        Get a combined hash from individual mm item hashes
        """
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract the embedding with the hash-ids of the queried items. Try combined hash first, if missed, fallback to individual hashes
        The returned tensor may not be contiguous
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """
        Set the embedding to the pre-allocated locations with a hash id
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        raise NotImplementedError()


def _get_tensor_size(embedding: torch.Tensor):
    return embedding.element_size() * embedding.numel()


class MultiModalStaticCache(MultimodalCache):
    """
    MultiModalStaticCache is a server-level cache for multimodal embedding,
    Embeddings will be computed prior, and this cache does not really pre-alloc
    """

    def __init__(
        self,
        max_size: int,
    ):
        super().__init__()
        self.max_size = max_size
        self.mm_cache: Dict[int, torch.Tensor] = {}
        self.current_size = 0

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        combined_hash = self.combine_hashes(mm_hashes)
        # MultiModalStaticCache does not fallback to individual item lookup

        print(f"getting cache with {combined_hash=}")
        return self.mm_cache.get(combined_hash)

    def set(
        self, mm_hash: int, embedding: torch.Tensor, loc: Optional[torch.Tensor] = None
    ) -> bool:
        if mm_hash in self.mm_cache:
            return True
        print(f"setting cache with {mm_hash=}")
        data_size = _get_tensor_size(embedding)
        if self.current_size + data_size > self.max_size:
            return False
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self.current_size -= _get_tensor_size(old_embedding)
        return True

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()
