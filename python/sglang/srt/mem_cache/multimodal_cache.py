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
        # Deterministic across processes (plain hash() is randomized).
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
    # Optional grid/aux carried alongside the embedding; both default to None.
    grid: Optional[torch.Tensor] = None
    aux: Optional[dict] = None


def _to_cpu_meta(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: _to_cpu_meta(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu_meta(v) for v in value)
    return value


# Aux tensors above this get their own SHM segment; smaller ones stay inline.
_AUX_SHM_TENSOR_MAX_BYTES = 64 * 1024


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
    def _write_tensor_segment(t: torch.Tensor) -> Tuple[SharedMemory, dict]:
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
    def _read_tensor_segment(seg: dict) -> torch.Tensor:
        shm = SharedMemory(name=seg["shm_name"], create=False)
        try:
            ds = seg["dtype_str"]
            np_dt = "uint16" if ds == "bfloat16" else ds
            t = torch.from_numpy(
                np.ndarray(seg["shape"], dtype=np_dt, buffer=shm.buf, offset=0).copy()
            )
            if ds == "bfloat16":
                t = t.view(torch.bfloat16)
        finally:
            shm.close()
        return t

    @staticmethod
    def _write_to_shm(embedding: torch.Tensor) -> Tuple[SharedMemory, dict]:
        return MultiModalStaticCache._write_tensor_segment(embedding)

    @staticmethod
    def _read_from_shm(meta: dict) -> torch.Tensor:
        return MultiModalStaticCache._read_tensor_segment(meta)

    @classmethod
    def _write_aux_to_shm(cls, aux: Optional[dict]) -> Tuple[dict, list, list, int]:
        # Returns (inline_aux, aux_segments, shm_objects, extra_bytes); caller closes shm_objects.
        if not aux:
            return None, [], [], 0
        inline: dict = {}
        segments: list = []
        shms: list = []
        extra = 0
        try:
            for name, v in aux.items():
                if (
                    isinstance(v, torch.Tensor)
                    and v.element_size() * v.numel() > _AUX_SHM_TENSOR_MAX_BYTES
                ):
                    shm, seg = cls._write_tensor_segment(v)
                    shms.append(shm)
                    seg["aux_key"] = name
                    segments.append(seg)
                    extra += seg["total_bytes"]
                    inline[name] = None  # placeholder; filled from segment on read
                else:
                    inline[name] = _to_cpu_meta(v)
        except BaseException:
            for s in shms:
                try:
                    s.close()
                    s.unlink()
                except FileNotFoundError:
                    pass
            raise
        return inline, segments, shms, extra

    @classmethod
    def _read_aux_from_shm(cls, meta: dict) -> Optional[dict]:
        inline = meta.get("aux")
        if inline is None:
            return None
        aux = dict(inline)
        for seg in meta.get("aux_segments", []):
            aux[seg["aux_key"]] = cls._read_tensor_segment(seg)
        return aux

    @staticmethod
    def _unlink_shm(meta: dict):
        names = [meta.get("shm_name")]
        names += [seg["shm_name"] for seg in meta.get("aux_segments", [])]
        for name in names:
            if not name:
                continue
            try:
                shm = SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    def _idx(self):
        return self._shm_index if self._shared else self.mm_cache

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
        with self._lock:
            if key not in self._idx():
                return None
            self._move_to_end(key)
            if self._shared:
                meta = self._shm_index[key]
                try:
                    tensor = self._read_from_shm(meta)
                    aux = self._read_aux_from_shm(meta)
                except FileNotFoundError:
                    # Index hit but backing SHM segment gone: evict, treat as miss.
                    logger.warning(
                        f"[mm_cache] read HIT-FAIL key=0x{key:x} "
                        f"shm={meta.get('shm_name')} reason=shm_segment_missing; "
                        f"evicting stale entry"
                    )
                    self._remove(key)
                    return None
                return EmbeddingResult(
                    embedding=tensor,
                    grid=meta.get("grid"),
                    aux=aux,
                )
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
            try:
                inline_aux, aux_segments, aux_shms, aux_bytes = self._write_aux_to_shm(
                    embedding.aux
                )
            except OSError as e:
                logger.error(f"MultiModalStaticCache: shm alloc failed (aux): {e}")
                self._unlink_shm(meta)
                return False
            for s in aux_shms:
                s.close()
            meta["grid"] = (
                _to_cpu_meta(embedding.grid) if embedding.grid is not None else None
            )
            meta["aux"] = inline_aux
            meta["aux_segments"] = aux_segments
            # total_bytes drives LRU accounting; include the aux segments.
            data_size = meta["total_bytes"] + aux_bytes
            meta["total_bytes"] = data_size
            with self._lock:
                if mm_hash in self._idx():
                    self._move_to_end(mm_hash)
                    self._unlink_shm(meta)
                    return True
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
