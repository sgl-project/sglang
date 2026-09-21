"""Dependency-free contracts for the experimental Mooncake compression path."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass

from sglang.srt.kv_compression.types import NVCOMP_VERSION
from sglang.srt.kv_compression.types import BufferDrainError as BufferDrainError

PROTOCOL_VERSION = 2
MODES = ("off", "passthrough", "lz4")


CHUNK_READY = b"COMPRESSED_CHUNK_READY_V2"


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(
            f"SGLANG_PD_KV_COMPRESSION must be one of {MODES}, got {mode!r}"
        )
    return mode


def capability(mode: str, force=None) -> str:
    validate_mode(mode)
    if mode == "off":
        return "off"
    if force is None:
        force = os.environ.get("SGLANG_PD_KV_COMPRESSION_FORCE", "0").lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
    suffix = "/forced-test" if force else ""
    return f"pd-kv-v{PROTOCOL_VERSION}/{mode}/nvcomp-{NVCOMP_VERSION}/native-lz4-65536{suffix}"


def check_peer(local: str, remote: str) -> None:
    if local != remote:
        raise ValueError(
            f"P/D KV compression mismatch: local={local!r}, peer={remote!r}"
        )


@dataclass(frozen=True)
class ChunkDescriptor:
    nonce: str
    encoding: str
    raw_bytes: int
    wire_bytes: int
    sha256: str = ""
    version: int = PROTOCOL_VERSION
    # [offset, actual_bytes, encoding] in logical page order. Per-request
    # destinations stay outside cached objects.
    pages: tuple = ()

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode("ascii")

    @classmethod
    def from_bytes(cls, data: bytes) -> ChunkDescriptor:
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("Compression descriptor exceeds its byte budget")
        value = json.loads(data)
        if not isinstance(value, dict) or set(value) != {
            "nonce",
            "encoding",
            "raw_bytes",
            "wire_bytes",
            "sha256",
            "version",
            "pages",
        }:
            raise ValueError("Invalid compression descriptor fields")
        if not isinstance(value["pages"], list):
            raise ValueError("Invalid page manifest")
        value["pages"] = tuple(tuple(p) for p in value["pages"])
        result = cls(**value)
        if type(result.version) is not int or result.version != PROTOCOL_VERSION:
            raise ValueError("Unsupported compression protocol version")
        if result.encoding not in ("raw", "lz4", "pages"):
            raise ValueError("Unknown KV representation")
        if not isinstance(result.nonce, str) or not 1 <= len(result.nonce) <= 64:
            raise ValueError("Invalid request nonce")
        if any(
            type(n) is not int or n <= 0 for n in (result.raw_bytes, result.wire_bytes)
        ):
            raise ValueError("Invalid payload lengths")
        if not isinstance(result.sha256, str) or (
            result.sha256
            and (
                len(result.sha256) != 64
                or any(c not in "0123456789abcdef" for c in result.sha256)
            )
        ):
            raise ValueError("Invalid verification digest")
        if result.encoding == "raw" and result.raw_bytes != result.wire_bytes:
            raise ValueError("Raw payload length mismatch")
        if result.encoding == "pages":
            if not result.pages:
                raise ValueError("Empty page manifest")
            end = 0
            for page in result.pages:
                if len(page) != 3:
                    raise ValueError("Invalid page entry")
                offset, size, encoding = page
                if (
                    type(offset) is not int
                    or type(size) is not int
                    or size <= 0
                    or offset != (end + 255) // 256 * 256
                    or encoding not in ("raw", "lz4")
                ):
                    raise ValueError("Invalid page range/encoding")
                end = offset + size
            if end != result.wire_bytes:
                raise ValueError("Page manifest does not cover the payload")
        elif result.pages:
            raise ValueError("Unexpected page manifest")
        return result

    def validate(
        self, *, nonce: str, raw_bytes: int, capacity: int, mode: str, page_bytes=None
    ) -> None:
        if self.nonce != nonce:
            raise ValueError("Stale compression descriptor")
        if self.raw_bytes != raw_bytes or self.wire_bytes > capacity:
            raise ValueError("Compression payload does not fit the assigned KV range")
        if self.encoding == "pages":
            if page_bytes is None or len(self.pages) * page_bytes != raw_bytes:
                raise ValueError("Page count/layout mismatch")
            for _, length, encoding in self.pages:
                if encoding == "raw" and length != page_bytes:
                    raise ValueError("Raw page size mismatch")
                if mode == "passthrough" and encoding != "raw":
                    raise ValueError("Compressed page in passthrough mode")
        elif mode == "passthrough" and self.encoding != "raw":
            raise ValueError("Compressed data received in passthrough mode")


class RoomTasks:
    """Keep source pages alive from enqueue through all GPU/RDMA accesses."""

    def __init__(self):
        self._counts = {}
        self._cv = threading.Condition()

    def add(self, room: int) -> None:
        with self._cv:
            self._counts[room] = self._counts.get(room, 0) + 1

    def finish(self, room: int) -> None:
        with self._cv:
            count = self._counts[room] - 1
            if count:
                self._counts[room] = count
            else:
                del self._counts[room]
                self._cv.notify_all()

    def pending(self, room: int) -> bool:
        with self._cv:
            return self._counts.get(room, 0) > 0

    def drain(self, room: int) -> None:
        # Cancellation is exceptional. Never turn a timeout into permission to
        # reuse pages while a transfer worker can still read them.
        with self._cv:
            self._cv.wait_for(lambda: room not in self._counts)
