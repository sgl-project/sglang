import abc
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

MM_EMBEDDING_CACHE_LEASE_ID_KEY = "mm_embedding_cache_lease_id"


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ): ...

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


@dataclass(kw_only=True)
class EmbeddingResult:
    embedding: torch.Tensor


@dataclass
class _EmbeddingLease:
    entries: dict[int, EmbeddingResult]
    remaining: dict[int, int]
    expires_at: float
    admitted: bool = False


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
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size = 0
        self._leases: dict[str, _EmbeddingLease] = {}
        self._pin_counts: dict[int, int] = {}
        self._lock = threading.RLock()

    def _release_lease_locked(self, lease_id: str) -> bool:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            return False
        for mm_hash in lease.entries:
            count = self._pin_counts.get(mm_hash, 0) - 1
            if count > 0:
                self._pin_counts[mm_hash] = count
            else:
                self._pin_counts.pop(mm_hash, None)
        return True

    def _reap_expired_leases_locked(self, now: Optional[float] = None) -> int:
        now = time.monotonic() if now is None else now
        expired = [
            lease_id
            for lease_id, lease in self._leases.items()
            if not lease.admitted and lease.expires_at <= now
        ]
        for lease_id in expired:
            self._release_lease_locked(lease_id)
        return len(expired)

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[EmbeddingResult]:
        combined_hash = self.combine_hashes(mm_hashes)
        # MultiModalStaticCache does not fallback to individual item lookup
        with self._lock:
            self._reap_expired_leases_locked()
            embedding = self.mm_cache.get(combined_hash)
            if embedding is not None:
                self.mm_cache.move_to_end(combined_hash)
            return embedding

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        assert isinstance(embedding, EmbeddingResult), embedding
        with self._lock:
            self._reap_expired_leases_locked()
            if mm_hash in self.mm_cache:
                self.mm_cache.move_to_end(mm_hash)
                return True
            data_size = _get_tensor_size(embedding.embedding)
            while self.current_size + data_size > self.max_size:
                evictable_hash = next(
                    (key for key in self.mm_cache if self._pin_counts.get(key, 0) == 0),
                    None,
                )
                if evictable_hash is None:
                    return False
                evicted = self.mm_cache.pop(evictable_hash)
                self.current_size -= _get_tensor_size(evicted.embedding)

            self.mm_cache[mm_hash] = embedding
            self.current_size += data_size
            return True

    def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
        """Get a single cached embedding by its hash (no combine_hashes)."""
        with self._lock:
            self._reap_expired_leases_locked()
            embedding = self.mm_cache.get(mm_hash)
            if embedding is not None:
                self.mm_cache.move_to_end(mm_hash)
            return embedding

    def acquire_many(
        self,
        lease_id: str,
        mm_hashes: List[Optional[int]],
        ttl_s: float,
    ) -> List[bool]:
        """Atomically pin every currently available per-item embedding."""
        if ttl_s <= 0:
            raise ValueError("lease ttl must be positive")
        with self._lock:
            self._reap_expired_leases_locked()
            self._release_lease_locked(lease_id)
            entries = {
                mm_hash: self.mm_cache[mm_hash]
                for mm_hash in mm_hashes
                if mm_hash is not None and mm_hash in self.mm_cache
            }
            for mm_hash in entries:
                self._pin_counts[mm_hash] = self._pin_counts.get(mm_hash, 0) + 1
                self.mm_cache.move_to_end(mm_hash)
            if entries:
                remaining = {
                    mm_hash: sum(value == mm_hash for value in mm_hashes)
                    for mm_hash in entries
                }
                self._leases[lease_id] = _EmbeddingLease(
                    entries=entries,
                    remaining=remaining,
                    expires_at=time.monotonic() + ttl_s,
                )
            return [mm_hash is not None and mm_hash in entries for mm_hash in mm_hashes]

    def release_lease_hashes(self, lease_id: str, mm_hashes: List[int]) -> None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return
            for mm_hash in mm_hashes:
                if lease.entries.pop(mm_hash, None) is None:
                    continue
                lease.remaining.pop(mm_hash, None)
                count = self._pin_counts.get(mm_hash, 0) - 1
                if count > 0:
                    self._pin_counts[mm_hash] = count
                else:
                    self._pin_counts.pop(mm_hash, None)
            if not lease.entries:
                self._leases.pop(lease_id, None)

    def consume(self, lease_id: str, mm_hash: int) -> Optional[EmbeddingResult]:
        """Return a pinned embedding and release that item from its lease."""
        with self._lock:
            self._reap_expired_leases_locked()
            lease = self._leases.get(lease_id)
            if lease is None:
                return None
            embedding = lease.entries.pop(mm_hash, None)
            if embedding is None:
                return None
            remaining = lease.remaining[mm_hash] - 1
            if remaining > 0:
                lease.entries[mm_hash] = embedding
                lease.remaining[mm_hash] = remaining
                return embedding
            lease.remaining.pop(mm_hash, None)
            count = self._pin_counts.get(mm_hash, 0) - 1
            if count > 0:
                self._pin_counts[mm_hash] = count
            else:
                self._pin_counts.pop(mm_hash, None)
            if not lease.entries:
                self._leases.pop(lease_id, None)
            return embedding

    def get_leased(self, lease_id: str, mm_hash: int) -> Optional[EmbeddingResult]:
        """Return an admitted request's pinned embedding without releasing it.

        Chunked prefill can revisit one image in more than one scheduler step.
        The request lifecycle, rather than the first lookup, therefore owns the
        pin and releases it through ``MultimodalInputs.release_features``.
        """
        with self._lock:
            self._reap_expired_leases_locked()
            lease = self._leases.get(lease_id)
            return None if lease is None else lease.entries.get(mm_hash)

    def release_lease(self, lease_id: str) -> bool:
        with self._lock:
            return self._release_lease_locked(lease_id)

    def lease_contains(self, lease_id: str, mm_hash: int) -> bool:
        with self._lock:
            self._reap_expired_leases_locked()
            lease = self._leases.get(lease_id)
            return lease is not None and lease.remaining.get(mm_hash, 0) > 0

    def admit_lease(self, lease_id: str) -> bool:
        """Transfer a live lease to an admitted request's lifecycle.

        The five-minute TTL protects the tokenizer-to-scheduler handoff. Once
        admitted, normal request completion, cancellation, flush, or embedding
        consumption owns release, so queueing cannot invalidate the request.
        """
        with self._lock:
            self._reap_expired_leases_locked()
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            lease.admitted = True
            return True

    def lease_stats(self) -> tuple[int, int]:
        """Return active lease and pinned-entry counts for observability."""
        with self._lock:
            self._reap_expired_leases_locked()
            return len(self._leases), len(self._pin_counts)

    def has(self, mm_hash: int) -> bool:
        with self._lock:
            self._reap_expired_leases_locked()
            return mm_hash in self.mm_cache

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        with self._lock:
            self._reap_expired_leases_locked()
            if mm_hash not in self.mm_cache or self._pin_counts.get(mm_hash, 0):
                return False
            old_embedding = self.mm_cache.pop(mm_hash)
            self.current_size -= _get_tensor_size(old_embedding.embedding)
            return True

    def clear(self):
        with self._lock:
            self.mm_cache.clear()
            self._leases.clear()
            self._pin_counts.clear()
            self.current_size = 0

    def __len__(self):
        with self._lock:
            return len(self.mm_cache)

    def available_size(self):
        return self.__len__()
