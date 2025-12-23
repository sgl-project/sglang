import threading
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.srt.utils import parse_connector_type


@dataclass(frozen=True)
class SessionCacheSegment:
    """
    Represents a segment of cached tokens and their corresponding KV cache location.

    Attributes:
        token_start: The starting token ID within the session's logical token sequence.
        token_length: The number of tokens in this segment.
        kv_uri: A URI string identifying the storage backend and path for the KV cache.
        kv_start: The starting position in the physical KV cache storage.
        kv_length: The length of the KV cache data in the storage (optional).
                   If None, it's typically inferred from token_length * kv_length_per_token.
    """

    token_start: int
    token_length: int
    kv_uri: str
    kv_start: int
    kv_length: Optional[int] = None

    def __post_init__(self):
        if self.token_start < 0:
            raise ValueError(
                f"token_start must be non-negative, got {self.token_start}"
            )
        if self.token_length <= 0:
            raise ValueError(f"token_length must be positive, got {self.token_length}")
        if self.kv_start < 0:
            raise ValueError(f"kv_start must be non-negative, got {self.kv_start}")
        if self.kv_length is not None and self.kv_length <= 0:
            raise ValueError(f"kv_length must be positive if set, got {self.kv_length}")

    @classmethod
    def from_dict(cls, d: Dict) -> "SessionCacheSegment":
        """Creates an instance from a dictionary."""
        required_fields = ["token_start", "token_length", "kv_uri", "kv_start"]
        missing_fields = [f for f in required_fields if f not in d]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in SessionCacheSegment dictionary: {missing_fields}. "
                f"Required fields: {required_fields}"
            )
        return cls(
            token_start=d["token_start"],
            token_length=d["token_length"],
            kv_uri=d["kv_uri"],
            kv_start=d["kv_start"],
            kv_length=d.get("kv_length", None),
        )

    def to_dict(self) -> Dict:
        """Converts the instance to a dictionary."""
        return asdict(self)

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_length


class SessionCache:
    """
    Manages a collection of SessionCacheSegments representing a session's KV cache state.

    This class handles truncating, binding token IDs/memory indices,
    and interacting with storage backends for prefetching and backing up cache data.
    """

    def __init__(
        self,
        segments: Optional[Union[List[SessionCacheSegment], List[Dict]]] = None,
        _offset: int = 0,
        _token_ids: Optional[List[int]] = None,
        _mem_indices: Optional[torch.Tensor] = None,
    ):
        self._offset: int = _offset
        self._token_ids: Optional[List[int]] = _token_ids
        self._mem_indices: Optional[torch.Tensor] = _mem_indices
        self._backup_done: bool = False

        if segments is None:
            self._segments: Tuple[SessionCacheSegment, ...] = ()
            return

        processed: List[SessionCacheSegment] = []
        for seg in segments:
            if isinstance(seg, dict):
                processed.append(SessionCacheSegment.from_dict(seg))
            else:
                processed.append(seg)

        if not processed:
            self._segments = ()
            return

        sorted_segments = sorted(processed, key=lambda s: s.token_start)

        prev_end = sorted_segments[0].token_start
        for i, seg in enumerate(sorted_segments):
            if seg.token_start != prev_end:
                raise ValueError(
                    f"Segments are not contiguous at index {i}: "
                    f"...[{sorted_segments[i-1].token_start}, "
                    f"{sorted_segments[i-1].token_end}], "
                    f"[{seg.token_start}, {seg.token_end}]..."
                    if i > 0
                    else ""
                )
            prev_end = seg.token_end

        self._segments = tuple(sorted_segments)

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> SessionCacheSegment:
        return self._segments[index]

    def __iter__(self) -> Iterator[SessionCacheSegment]:
        return iter(self._segments)

    def __repr__(self) -> str:
        return f"SessionCache(segments={list(self._segments)}, _offset={self._offset})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SessionCache):
            return False
        return (
            self._segments == other._segments
            and self._offset == other._offset
            and self._backup_done == other._backup_done
        )

    def __hash__(self) -> int:
        return hash((self._segments, self._offset, self._backup_done))

    @property
    def token_start(self) -> int:
        """The logical start token ID of the entire cache."""
        return self._segments[0].token_start + self._offset if self._segments else 0

    @property
    def token_end(self) -> int:
        """The logical end token ID (exclusive) of the entire cache."""
        return self._segments[-1].token_end if self._segments else 0

    @property
    def token_length(self) -> int:
        """The total logical token length covered by the cache."""
        return self.token_end - self.token_start

    @property
    def token_range(self) -> Optional[Tuple[int, int]]:
        """Returns the (start, end) logical token range, or None if empty."""
        if not self._segments:
            return None
        return (self.token_start, self.token_end)

    @property
    def real_token_start(self) -> int:
        """The real start token ID of the entire cache."""
        return self._segments[0].token_start if self._segments else 0

    @property
    def real_token_end(self) -> int:
        """The real end token ID (exclusive) of the entire cache."""
        return self.token_end

    @property
    def real_token_length(self) -> int:
        """The real token length including the internal offset."""
        return self.real_token_end - self.real_token_start

    def to_dicts(self) -> List[Dict]:
        """Serializes the segments into a list of dictionaries."""
        return [seg.to_dict() for seg in self._segments]

    @classmethod
    def _from_validated_segments_and_state(
        cls,
        segments: Tuple[SessionCacheSegment, ...],
        offset: int = 0,
        token_ids: Optional[List[int]] = None,
        mem_indices: Optional[torch.Tensor] = None,
        backup_done: bool = False,  # Add backup status parameter
    ) -> "SessionCache":
        """Internal constructor for creating instances with pre-validated state."""
        instance = object.__new__(cls)
        instance._segments = segments
        instance._offset = offset
        instance._token_ids = token_ids
        instance._mem_indices = mem_indices
        instance._backup_done = backup_done  # Initialize correctly
        return instance

    def bind_token_ids(self, token_ids: List[int]) -> "SessionCache":
        """
        Returns a new SessionCache instance with token_ids bound.

        Raises:
            ValueError: If the length of token_ids doesn't match the token_length.
        """
        if len(token_ids) != self.token_length:
            raise ValueError(
                f"Token IDs length {len(token_ids)} does not match token length {self.token_length}"
            )

        return SessionCache._from_validated_segments_and_state(
            segments=self._segments,
            offset=self._offset,
            token_ids=token_ids,
            mem_indices=self._mem_indices,
            backup_done=self._backup_done,
        )

    def bind_mem_indices(self, mem_indices: torch.Tensor) -> "SessionCache":
        """
        Returns a new SessionCache instance with mem_indices bound.

        Raises:
            ValueError: If the length of mem_indices doesn't match the real_token_length.
        """
        if len(mem_indices) != self.real_token_length:
            raise ValueError(
                f"Memory indices length {len(mem_indices)} does not match real token length {self.real_token_length}"
            )

        return SessionCache._from_validated_segments_and_state(
            segments=self._segments,
            offset=self._offset,
            token_ids=self._token_ids,
            mem_indices=mem_indices,
            backup_done=self._backup_done,
        )

    @property
    def token_ids(self) -> List[int]:
        """Returns the bound token IDs.

        Raises:
            ValueError: If token_ids are not bound.
        """
        if self._token_ids is None:
            raise ValueError("token_ids are not bound")
        return self._token_ids

    @property
    def mem_indices(self) -> torch.Tensor:
        """Returns the valid bound memory indices.

        Raises:
            ValueError: If mem_indices are not bound.
        """
        if self._mem_indices is None:
            raise ValueError("mem_indices are not bound")
        return self._mem_indices[self._offset :]

    def real_mem_indices_prefix(self, offset: int = 0) -> torch.Tensor:
        """Returns the prefix of the real bound memory indices.

        Raises:
            ValueError: If mem_indices are not bound.
        """
        if self._mem_indices is None:
            raise ValueError("mem_indices are not bound")
        return self._mem_indices[: self._offset + offset]

    def _shift_real_token(self, offset: Optional[int] = None) -> "SessionCache":
        """
        Returns a new SessionCache with all segment token_start/ends shifted.

        Args:
            offset: The amount to shift. If None, shifts so that token_start becomes 0.
        """
        if not self._segments:
            return SessionCache()

        actual_offset = offset if offset is not None else -self.real_token_start

        new_segments = [
            SessionCacheSegment(
                token_start=seg.token_start + actual_offset,
                token_length=seg.token_length,
                kv_uri=seg.kv_uri,
                kv_start=seg.kv_start,
                kv_length=seg.kv_length,
            )
            for seg in self._segments
        ]

        return SessionCache._from_validated_segments_and_state(
            segments=tuple(new_segments),
            offset=self._offset,
            token_ids=self._token_ids,
            mem_indices=self._mem_indices,
            backup_done=self._backup_done,
        )

    def truncate_prefix(self, offset: int) -> "SessionCache":
        """
        Returns a new SessionCache truncated from the beginning up to 'offset'.

        Args:
            offset: The logical token offset up to which to truncate.

        Raises:
            ValueError: If offset is less than token_start.
        """
        if not self._segments:
            return SessionCache()

        if offset < self.token_start:
            raise ValueError(
                f"offset ({offset}) must be greater than or equal to the start of the cache ({self.token_start})"
            )

        if offset >= self.token_end:
            return SessionCache()

        new_segments = [seg for seg in self._segments if seg.token_end > offset]
        if not new_segments:
            return SessionCache()

        new_offset = offset - new_segments[0].token_start

        new_token_ids = None
        if self._token_ids is not None:
            new_token_ids = self._token_ids[offset - self.token_start :]

        new_mem_indices = None
        if self._mem_indices is not None:
            new_real_offset = new_segments[0].token_start - self.real_token_start
            new_mem_indices = self._mem_indices[new_real_offset:]

        return SessionCache._from_validated_segments_and_state(
            segments=tuple(new_segments),
            offset=new_offset,
            token_ids=new_token_ids,
            mem_indices=new_mem_indices,
            backup_done=self._backup_done,
        )

    def truncate_suffix(self, offset: int, kv_length_per_token: int) -> "SessionCache":
        """
        Returns a new SessionCache truncated from 'offset' to the end.

        Args:
            offset: The logical token offset from which to truncate.
            kv_length_per_token: Used to calculate the correct kv_length for partial segments.

        Raises:
            ValueError: If kv_length_per_token is not positive or if offset is invalid.
        """
        if kv_length_per_token <= 0:
            raise ValueError("kv_length_per_token must be a positive integer")

        if not self._segments:
            return SessionCache()

        if offset <= self.token_start:
            return SessionCache()

        if offset >= self.token_end:
            new_segments = [
                SessionCacheSegment(
                    token_start=seg.token_start,
                    token_length=seg.token_length,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=seg.token_length * kv_length_per_token,
                )
                for seg in self._segments
            ]
            return SessionCache._from_validated_segments_and_state(
                segments=new_segments,
                offset=self._offset,
                token_ids=self._token_ids,
                mem_indices=self._mem_indices,
                backup_done=self._backup_done,
            )

        new_segments = []
        for seg in self._segments:
            if seg.token_start >= offset:
                break
            elif seg.token_end <= offset:
                new_seg = SessionCacheSegment(
                    token_start=seg.token_start,
                    token_length=seg.token_length,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=seg.token_length * kv_length_per_token,
                )
                new_segments.append(new_seg)
            else:
                new_token_length = offset - seg.token_start
                if new_token_length > 0:
                    new_seg = SessionCacheSegment(
                        token_start=seg.token_start,
                        token_length=new_token_length,
                        kv_uri=seg.kv_uri,
                        kv_start=seg.kv_start,
                        kv_length=new_token_length * kv_length_per_token,
                    )
                    new_segments.append(new_seg)
                break

        if not new_segments:
            return SessionCache()

        new_token_ids = None
        if self._token_ids is not None:
            new_token_ids = self._token_ids[: offset - self.token_start]

        new_mem_indices = None
        if self._mem_indices is not None:
            new_mem_indices = self._mem_indices[: offset - self.real_token_start]

        return SessionCache._from_validated_segments_and_state(
            segments=tuple(new_segments),
            offset=self._offset,
            token_ids=new_token_ids,
            mem_indices=new_mem_indices,
            backup_done=self._backup_done,
        )

    def check_token_aligned(self, aligned_size: int):
        """Checks if all segment starts and lengths are aligned to `aligned_size`.

        Raises:
            ValueError: If any segment is misaligned.
        """
        for seg in self._segments:
            if seg.token_start % aligned_size != 0:
                raise ValueError(
                    f"Session cache segment token_start={seg.token_start} is not aligned to {aligned_size}"
                )
            if seg.token_length % aligned_size != 0:
                raise ValueError(
                    f"Session cache segment token_length={seg.token_length} is not aligned to {aligned_size}"
                )

    def check_kv_length(self, kv_length_per_token: int):
        """Checks if the stored kv_length matches the expected value based on token_length.

        Raises:
            ValueError: If kv_length is inconsistent.
        """
        for seg in self._segments:
            if seg.kv_length is not None:
                expected_kv_length = seg.token_length * kv_length_per_token
                if seg.kv_length != expected_kv_length:
                    raise ValueError(
                        f"Segment has inconsistent kv_length: "
                        f"token_length={seg.token_length}, "
                        f"kv_length={seg.kv_length}, "
                        f"expected kv_length={expected_kv_length} "
                        f"(based on kv_length_per_token={kv_length_per_token})"
                    )

    def prefetch(
        self, mem_pool_host: HostKVCache, storage_config: HiCacheStorageConfig
    ):
        """
        Loads KV cache data from storage into host memory pool.

        Requires mem_indices to be bound.

        Args:
            mem_pool_host: The host memory pool to load data into.
            storage_config: Configuration for accessing storage backends.

        Raises:
            ValueError: If mem_indices are not bound or checks fail.
        """
        if self._mem_indices is None:
            raise ValueError("mem_indices are not bound")

        self.check_token_aligned(mem_pool_host.page_size)
        self.check_kv_length(mem_pool_host.get_size_per_token())

        kv_cache = self._shift_real_token()
        for seg in kv_cache:
            mem_indices = self._mem_indices[
                seg.token_start : seg.token_start + seg.token_length
            ]
            page_num = len(mem_indices) // mem_pool_host.page_size

            flat_data = mem_pool_host.get_dummy_flat_data_page(page_num)

            storage, filepath = SessionCacheStorageManager.get_storage(
                seg.kv_uri, storage_config, mem_pool_host
            )

            storage.load(filepath, seg.kv_start, flat_data)

            mem_pool_host.set_from_flat_data(mem_indices, flat_data)

    def backup(
        self,
        mem_pool_device: KVCache,
        mem_pool_host: HostKVCache,
        storage_config: HiCacheStorageConfig,
    ):
        """
        Saves KV cache data from device memory pool to storage.

        Requires mem_indices to be bound.

        Args:
            mem_pool_device: The device memory pool containing the data.
            mem_pool_host: The host memory pool (used for storage backend setup).
            storage_config: Configuration for accessing storage backends.

        Raises:
            ValueError: If mem_indices are not bound.
        """
        if self._mem_indices is None:
            raise ValueError("mem_indices are not bound")

        flat_data = mem_pool_device.get_flat_data(self._mem_indices)

        kv_cache = self._shift_real_token()
        for seg in kv_cache:
            seg_data = flat_data[:, :, seg.token_start : seg.token_end, :, :]

            storage, filepath = SessionCacheStorageManager.get_storage(
                seg.kv_uri, storage_config, mem_pool_host
            )

            storage.save(filepath, seg.kv_start, seg_data)

        self._backup_done = True

    def backup_done(self) -> bool:
        """Returns whether this instance has successfully backed up its data."""
        return self._backup_done


class SessionCacheStorageManager:
    """
    Manages access to storage backends, caching them per-thread to avoid recreation overhead.

    Uses thread-local storage to hold backend instances keyed by backend type and configuration.
    """

    _local = threading.local()

    @classmethod
    def _get_thread_cache(cls) -> Dict[Tuple[str, Tuple], HiCacheStorage]:
        """Get or create the thread-local cache dict."""
        if not hasattr(cls._local, "cache"):
            cls._local.cache = {}
        return cls._local.cache

    @classmethod
    def get_storage(
        cls,
        uri: str,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: HostKVCache,
        **kwargs,
    ) -> Tuple[HiCacheStorage, str]:
        """
        Retrieves a storage backend instance for a given URI.

        Caches the instance per-thread based on backend type and parsed configuration.

        Args:
            uri: The URI specifying the backend and resource path.
            storage_config: Base configuration for storage.
            mem_pool_host: Host memory pool, passed to backend factory.
            **kwargs: Additional arguments for backend creation.

        Returns:
            A tuple of (HiCacheStorage instance, extracted filepath from URI).

        Raises:
            ValueError: If the URI is invalid or the backend cannot be found/created.
        """
        cache = cls._get_thread_cache()

        backend_name = parse_connector_type(uri)
        if not backend_name:
            raise ValueError(f"Invalid URI: missing backend name in '{uri}'")

        backend_class = StorageBackendFactory.get_backend_class(backend_name)

        extra_config, filepath = backend_class.parse_uri(uri)
        storage_config = replace(storage_config, extra_config=extra_config)
        cache_key = (
            backend_name,
            tuple(sorted(extra_config.items())) if extra_config else (),
        )
        if cache_key not in cache:
            cache[cache_key] = StorageBackendFactory.create_backend(
                backend_name=backend_name,
                storage_config=storage_config,
                mem_pool_host=mem_pool_host,
                **kwargs,
            )

        return cache[cache_key], filepath
