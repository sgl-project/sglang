import abc
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Hashable, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ): ...

    @staticmethod
    def combine_hashes(mm_hashes: List[Hashable]) -> Optional[tuple]:
        """
        Keep the ordered identities themselves as the equality key.
        """
        if not mm_hashes:
            return None
        return tuple(mm_hashes)

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[Hashable], combined_hash: Optional[tuple] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract the embedding with the hash-ids of the queried items. Try combined hash first, if missed, fallback to individual hashes
        The returned tensor may not be contiguous
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: Hashable,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """
        Set the embedding to the pre-allocated locations with a hash id
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: Hashable) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: Hashable, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        raise NotImplementedError()


def _get_tensor_size(embedding: torch.Tensor):
    return embedding.untyped_storage().nbytes()


@dataclass(kw_only=True)
class EmbeddingResult:
    embedding: torch.Tensor


class MultiModalStaticCache(MultimodalCache):
    """
    A server-level cache for multimodal embedding.
    Embeddings are computed prior, and this cache does not really pre-alloc
    """

    def __init__(
        self,
        max_size: int,
    ):
        super().__init__()
        self.max_size = max_size
        self.mm_cache: OrderedDict[Hashable, EmbeddingResult] = OrderedDict()
        self.current_size = 0

    def get(
        self, mm_hashes: List[Hashable], combined_hash: Optional[tuple] = None
    ) -> Optional[EmbeddingResult]:
        combined_hash = self.combine_hashes(mm_hashes)
        # MultiModalStaticCache does not fallback to individual item lookup

        embedding = self.mm_cache.get(combined_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(combined_hash)
        return embedding

    def set(
        self,
        mm_hash: Hashable,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        assert isinstance(embedding, EmbeddingResult), embedding
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            return True
        tensor = embedding.embedding
        storage_size = _get_tensor_size(tensor)
        data_size = min(storage_size, tensor.element_size() * tensor.numel())
        # An uncacheable video must not evict useful entries on every chunk.
        if data_size > self.max_size:
            return False
        while self.current_size + data_size > self.max_size:
            if not self.mm_cache:
                return False
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            self.current_size -= _get_tensor_size(lru_embedding.embedding)

        if storage_size > data_size:
            # Clone admitted slices so one cached item cannot retain a whole batch.
            embedding = replace(embedding, embedding=tensor.clone())
        data_size = _get_tensor_size(embedding.embedding)
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def get_single(self, mm_hash: Hashable) -> Optional[EmbeddingResult]:
        """Get a single cached embedding by its hash (no combine_hashes)."""
        embedding = self.mm_cache.get(mm_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(mm_hash)
        return embedding

    def has(self, mm_hash: Hashable) -> bool:
        return mm_hash in self.mm_cache

    def free(
        self, mm_hash: Hashable, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self.current_size -= _get_tensor_size(old_embedding.embedding)
        return True

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()
