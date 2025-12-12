import threading
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, final

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig
from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.srt.utils import parse_connector_type


@dataclass(frozen=True)
class SessionCacheSegment:
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
        required_fields = ["token_start", "token_length", "kv_uri", "kv_start"]
        for field in required_fields:
            if field not in d:
                raise ValueError(
                    f"Missing required field '{field}' in SessionCacheSegment dictionary. "
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
        return asdict(self)

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_length


@final
class SessionCache:
    def __init__(
        self, segments: Optional[Union[List[SessionCacheSegment], List[Dict]]] = None
    ):
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
        return f"SessionCache({list(self._segments)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SessionCache):
            return False
        return self._segments == other._segments

    def __hash__(self) -> int:
        return hash(self._segments)

    @property
    def token_start(self) -> int:
        return self._segments[0].token_start if self._segments else 0

    @property
    def token_end(self) -> int:
        return self._segments[-1].token_end if self._segments else 0

    @property
    def token_range(self) -> Optional[Tuple[int, int]]:
        if not self._segments:
            return None
        return (self.token_start, self.token_end)

    @property
    def total_token_length(self) -> int:
        return self.token_end - self.token_start

    def to_dicts(self) -> List[Dict]:
        return [seg.to_dict() for seg in self._segments]

    @classmethod
    def _from_validated_segments(
        cls, segments: Tuple[SessionCacheSegment, ...]
    ) -> "SessionCache":
        instance = object.__new__(cls)
        instance._segments = segments
        return instance

    def shift_token_range(self, offset: Optional[int] = None) -> "SessionCache":
        if not self._segments:
            return SessionCache()

        if offset is None:
            offset = -self.token_start

        new_segments = []
        for seg in self._segments:
            new_seg = SessionCacheSegment(
                token_start=seg.token_start + offset,
                token_length=seg.token_length,
                kv_uri=seg.kv_uri,
                kv_start=seg.kv_start,
                kv_length=seg.kv_length,
            )
            new_segments.append(new_seg)

        return SessionCache._from_validated_segments(tuple(new_segments))

    def truncate_prefix_by_token_offset(self, token_offset: int) -> "SessionCache":
        if not self._segments:
            return SessionCache()

        if token_offset <= self.token_start:
            return self

        if token_offset >= self.token_end:
            return SessionCache()

        new_segments = []
        for seg in self._segments:
            if seg.token_end > token_offset:
                new_segments.append(seg)

        return SessionCache._from_validated_segments(tuple(new_segments))

    def truncate_suffix_by_token_offset(
        self, token_offset: int, kv_length_per_token: int
    ) -> "SessionCache":
        if kv_length_per_token <= 0:
            raise ValueError("kv_length_per_token must be a positive integer")

        if not self._segments:
            return SessionCache()

        if token_offset <= self.token_start:
            return SessionCache()

        new_segments = []
        for seg in self._segments:
            if seg.token_start >= token_offset:
                break
            elif seg.token_end <= token_offset:
                new_seg = SessionCacheSegment(
                    token_start=seg.token_start,
                    token_length=seg.token_length,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=seg.token_length * kv_length_per_token,
                )
                new_segments.append(new_seg)
            else:
                new_token_length = token_offset - seg.token_start
                new_kv_length = new_token_length * kv_length_per_token
                new_seg = SessionCacheSegment(
                    token_start=seg.token_start,
                    token_length=new_token_length,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=new_kv_length,
                )
                new_segments.append(new_seg)
                break

        return SessionCache._from_validated_segments(tuple(new_segments))

    def check_token_aligned(self, aligned_size: int):
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


class SessionCacheStorageManager:
    _local = threading.local()

    @classmethod
    def _get_thread_cache(cls) -> Dict[Tuple, HiCacheStorage]:
        """Get thread-local cache dict (keyed by (backend_name, config_key))"""
        if not hasattr(cls._local, "cache"):
            cls._local.cache = {}
        return cls._local.cache

    @classmethod
    def get_storage(
        cls,
        uri: str,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any = None,
        **kwargs,
    ) -> Tuple[HiCacheStorage, str]:
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
