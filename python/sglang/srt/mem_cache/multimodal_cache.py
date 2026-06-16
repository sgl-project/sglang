import abc
import contextlib
import hashlib
import logging
import struct
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ): ...

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        # Single element: return as-is (stable across processes).
        # Multiple elements: deterministic binary hash via struct.pack.
        if not mm_hashes:
            return None
        if len(mm_hashes) == 1:
            return mm_hashes[0]
        data = struct.pack(f">{len(mm_hashes)}q", *mm_hashes)
        return int(hashlib.sha1(data).hexdigest(), 16)

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
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


@dataclass(kw_only=True)
class EmbeddingResult:
    embedding: torch.Tensor


class MultiModalStaticCache(MultimodalCache):
    """
    A server-level cache for multimodal embedding.
    Supports optional cross-process sharing via POSIX shared memory when
    SGLANG_MM_CACHE_SHM=1 and shm_resources is provided (DP encoder mode).
    """

    def __init__(
        self,
        max_size: int,
        shm_resources: Optional[dict] = None,
    ):
        super().__init__()
        self.max_size = max_size
        self._shared = envs.SGLANG_MM_CACHE_SHM.get()
        if self._shared:
            assert shm_resources is not None
            self._shm_index = shm_resources["index"]
            self._lru_order = shm_resources["lru"]
            self._current_size = shm_resources["size"]
            self._lock = shm_resources["lock"]
        else:
            self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
            self._local_size = 0
            self._lock = contextlib.nullcontext()

    @property
    def current_size(self) -> int:
        return self._current_size.value if self._shared else self._local_size

    @current_size.setter
    def current_size(self, value: int):
        if self._shared:
            self._current_size.value = value
        else:
            self._local_size = value

    @staticmethod
    def _write_to_shm(embedding: torch.Tensor) -> Tuple[SharedMemory, dict]:
        t = embedding
        if t.dtype == torch.bfloat16:
            arr = t.contiguous().cpu().view(torch.uint16).numpy()
            dtype_str = "bfloat16"
        else:
            arr = t.contiguous().cpu().numpy()
            dtype_str = str(arr.dtype)
        total = arr.nbytes
        shm = SharedMemory(create=True, size=max(total, 1))
        try:
            np.copyto(
                np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf, offset=0), arr
            )
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        return shm, {
            "shm_name": shm.name,
            "shape": arr.shape,
            "dtype_str": dtype_str,
            "total_bytes": total,
        }

    @staticmethod
    def _read_from_shm(meta: dict) -> torch.Tensor:
        shm = SharedMemory(name=meta["shm_name"], create=False)
        try:
            ds = meta["dtype_str"]
            np_dt = "uint16" if ds == "bfloat16" else ds
            t = torch.from_numpy(
                np.ndarray(meta["shape"], dtype=np_dt, buffer=shm.buf, offset=0).copy()
            )
            if ds == "bfloat16":
                t = t.view(torch.bfloat16)
        finally:
            shm.close()
        return t

    @staticmethod
    def _unlink_shm(meta: dict):
        try:
            shm = SharedMemory(name=meta["shm_name"], create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

    def _idx(self):
        return self._shm_index if self._shared else self.mm_cache

    def _store(self, k: int, embedding: EmbeddingResult) -> bool:
        if self._shared:
            try:
                shm, meta = self._write_to_shm(embedding.embedding)
                shm.close()
            except OSError as e:
                logger.error(f"MultiModalStaticCache: shm alloc failed: {e}")
                return False
            self._shm_index[k] = meta
            self._lru_order.append(k)
            self.current_size += meta["total_bytes"]
        else:
            self.mm_cache[k] = embedding
            self.current_size += _get_tensor_size(embedding.embedding)
        return True

    def _remove(self, k: int) -> bool:
        if k not in self._idx():
            return False
        if self._shared:
            meta = self._shm_index.pop(k)
            try:
                self._lru_order.remove(k)
            except ValueError:
                pass
            self.current_size -= meta["total_bytes"]
            self._unlink_shm(meta)
        else:
            emb = self.mm_cache.pop(k)
            self.current_size -= _get_tensor_size(emb.embedding)
        return True

    def _move_to_end(self, k: int):
        if self._shared:
            try:
                self._lru_order.remove(k)
            except ValueError:
                pass
            self._lru_order.append(k)
        else:
            self.mm_cache.move_to_end(k)

    def _evict_one(self) -> bool:
        if self._shared:
            return bool(self._lru_order) and self._remove(self._lru_order[0])
        if not self.mm_cache:
            return False
        _, emb = self.mm_cache.popitem(last=False)
        self.current_size -= _get_tensor_size(emb.embedding)
        return True

    def _get_by_hash(self, key: int) -> Optional[EmbeddingResult]:
        """Shared get logic for both get() and get_single()."""
        with self._lock:
            if key not in self._idx():
                return None
            self._move_to_end(key)
            if self._shared:
                meta = self._shm_index[key]
                try:
                    tensor = self._read_from_shm(meta)
                except FileNotFoundError:
                    self._remove(key)
                    return None
                return EmbeddingResult(embedding=tensor)
            return self.mm_cache[key]

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[EmbeddingResult]:
        combined_hash = self.combine_hashes(mm_hashes)
        return self._get_by_hash(combined_hash)

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        assert isinstance(embedding, EmbeddingResult), embedding
        if self._shared:
            # Pre-compute SHM outside lock to minimize lock hold time.
            try:
                shm, meta = self._write_to_shm(embedding.embedding)
                shm.close()
            except OSError as e:
                logger.error(f"MultiModalStaticCache: shm alloc failed: {e}")
                return False
            with self._lock:
                if mm_hash in self._idx():
                    self._move_to_end(mm_hash)
                    self._unlink_shm(meta)
                    return True
                data_size = meta["total_bytes"]
                while self.current_size + data_size > self.max_size:
                    if not self._evict_one():
                        self._unlink_shm(meta)
                        return False
                self._shm_index[mm_hash] = meta
                self._lru_order.append(mm_hash)
                self.current_size += data_size
            return True
        else:
            with self._lock:
                if mm_hash in self._idx():
                    self._move_to_end(mm_hash)
                    return True
                data_size = _get_tensor_size(embedding.embedding)
                while self.current_size + data_size > self.max_size:
                    if not self._evict_one():
                        return False
                self.mm_cache[mm_hash] = embedding
                self.current_size += data_size
            return True

    def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
        """Get a single cached embedding by its hash (no combine_hashes)."""
        return self._get_by_hash(mm_hash)

    def has(self, mm_hash: int) -> bool:
        with self._lock:
            return mm_hash in self._idx()

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator = None
    ) -> bool:
        with self._lock:
            return self._remove(mm_hash)

    def clear(self):
        with self._lock:
            if self._shared:
                for k in list(self._shm_index.keys()):
                    self._remove(k)
            else:
                self.mm_cache.clear()
                self.current_size = 0

    def available_size(self):
        with self._lock:
            return self.max_size - self.current_size

    def __len__(self):
        with self._lock:
            return len(self._idx())

    def get_state(self) -> Tuple[int, int]:
        with self._lock:
            return self.current_size, len(self._idx())
