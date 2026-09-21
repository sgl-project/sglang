"""Data contracts shared by transfer and L2; no serving component imports."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from typing import Any

NVCOMP_VERSION = "5.3.0.16"
FORMAT_VERSION = 2
ALIGNMENT = 256
_ids = itertools.count(1)
_id_lock = threading.Lock()


def new_page_refs(count: int) -> tuple[int, ...]:
    # Never recycle IDs, including across cache resets. The runtime is local to
    # one model/pool; IDs identify materializations, not token hashes/addresses.
    with _id_lock:
        return tuple(next(_ids) for _ in range(count))


def align_bytes(size: int) -> int:
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


class BufferDrainError(RuntimeError):
    """A submitted access may still be running; borrowed memory cannot be freed."""


class CompressionCapacityError(RuntimeError):
    """A bounded allocation/queue cannot admit more work."""


class KVVerificationError(RuntimeError):
    """Restored KV must not be published or used for attention."""


@dataclass(frozen=True)
class EncodedPage:
    data: Any
    encoding: str
    raw_bytes: int
    raw_sha256: bytes | None = None

    @property
    def nbytes(self) -> int:
        return self.data.numel()


class Lease:
    """Idempotent consumer release; the producer owns completion/drain rules."""

    def __init__(self, future, release, *, source="unknown"):
        self.future = future
        self.source = source
        self._release = release
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            release, self._release = self._release, None
        if release is not None:
            release()
            self.future = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
