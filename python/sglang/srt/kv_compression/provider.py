"""Read-only representation access, independent of cache and transport policy."""

from dataclasses import dataclass
from typing import Protocol

from .types import FORMAT_VERSION, Lease


@dataclass(frozen=True)
class RepresentationSpec:
    layout: str
    mode: str
    force: bool = False
    verify: bool = False
    version: int = FORMAT_VERSION


class EncodedKVProvider(Protocol):
    def acquire(self, page_ref: int, spec: RepresentationSpec) -> Lease | None: ...


class HostEncodedKVProvider:
    def __init__(self, pool, spec: RepresentationSpec):
        self._pool, self.spec = pool, spec

    def acquire(self, page_ref: int, spec: RepresentationSpec) -> Lease | None:
        if (spec.layout, spec.version, spec.mode) != (
            self.spec.layout,
            self.spec.version,
            self.spec.mode,
        ):
            return None
        lease = self._pool.acquire_ref(page_ref, spec.mode)
        if lease is None:
            return None
        page = lease.future.result()  # Host acquire returns an already-ready value.
        if (
            page.encoding not in ((spec.mode,) if spec.force else ("raw", spec.mode))
        ) or (spec.verify and page.raw_sha256 is None):
            lease.close()
            return None
        return lease
