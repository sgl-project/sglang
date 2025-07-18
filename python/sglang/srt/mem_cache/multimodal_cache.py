import abc
from typing import Dict, Optional

import torch


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ):
        pass
        # self.hidden_size = hidden_size
        # self.max_size = max_size
        # self.page_size = page_size
        # self.dtype = dtype
        # self.device = device
        # if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        #     # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
        #     self.store_dtype = torch.uint8
        # else:
        #     self.store_dtype = dtype
        # # self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        # #     enable=enable_memory_saver
        # # )
        # self.mem_usage = 0

        # # used for chunked cpu-offloading
        # self.cpu_offloading_chunk_size = 8192

    @abc.abstractmethod
    def get_mm_embedding(self, mm_hash: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_mm_embedding(
        self, mm_hash: int, embedding: torch.Tensor, loc: Optional[torch.Tensor] = None
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, mm_hash: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()


class MultiModalStaticCache(MultimodalCache):
    """MultiModalStaticCache is used to store precomputed multimodal embeddings.
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

    def get_mm_embedding(self, mm_hash: int) -> torch.Tensor:
        return self.mm_cache.get(mm_hash)

    def set_mm_embedding(
        self, mm_hash: int, embedding: torch.Tensor, loc: Optional[torch.Tensor] = None
    ) -> bool:
        if mm_hash in self.mm_cache:
            return True
        data_size = self._get_tensor_size(embedding)
        if self.current_size + data_size > self.max_size:
            return False
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

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


class PagedMultiModalCache(MultimodalCache):
    """PagedMultiModalCache pre-allocates a buffer to store multimodal embeddings,
    and works with an external paged allocator. The allocator manages the allocation
    of token slots from this cache's buffer.
    """

    def __init__(
        self,
        size: int,
        hidden_size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.size = size  # Number of token slots
        self.hidden_size = hidden_size
        self.page_size = page_size

        self.mm_hash_to_indices: Dict[int, torch.Tensor] = {}

        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

        self.mm_buffer = torch.zeros(
            (self.size + self.page_size, self.hidden_size),
            dtype=self.store_dtype,
            device=self.device,
        )
        # self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        #     enable=enable_memory_saver
        # )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

    def get_mm_embedding(self, mm_hash: int) -> torch.Tensor:
        indices = self.mm_hash_to_indices.get(mm_hash)
        if indices is None:
            return None

        embedding = self.mm_buffer.index_select(0, indices)

        if self.store_dtype != self.dtype:
            return embedding.view(self.dtype)
        return embedding

    def set_mm_embedding(
        self, mm_hash: int, embedding: torch.Tensor, loc: Optional[torch.Tensor] = None
    ) -> bool:
        if mm_hash in self.mm_hash_to_indices:
            return True

        if embedding.dtype != self.dtype:
            embedding = embedding.to(self.dtype)

        if self.store_dtype != self.dtype:
            embedding_to_store = embedding.view(self.store_dtype)
        else:
            embedding_to_store = embedding

        self.mm_buffer.index_put_((loc,), embedding_to_store)
        self.mm_hash_to_indices[mm_hash] = loc

        return True

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_hash_to_indices

    def free(self, mm_hash: int) -> bool:
        if mm_hash in self.mm_hash_to_indices:
            del self.mm_hash_to_indices[mm_hash]
            return True
        return False

    def clear(self):
        self.mm_hash_to_indices.clear()

    def __len__(self):
        return len(self.mm_hash_to_indices)
