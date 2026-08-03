# SPDX-License-Identifier: Apache-2.0

"""Wire primitives shared by the realtime gateway and remote VAE worker."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import msgspec.msgpack

PROTOCOL_VERSION = 1
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class ProtocolViolation(ValueError):
    """Raised when a peer sends an invalid or stale realtime VAE message."""


class AcceptDisposition(str, Enum):
    ACCEPT = "accept"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class LatentChunkHeader:
    session_id: str
    generation_id: str
    request_id: str
    chunk_index: int
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    checksum: str
    event_id: int | None = None
    action_version: int = 0
    prompt_version: int = 0
    deadline_epoch_ms: int = 0
    has_reference: bool = False

    def validate(self) -> None:
        if not self.session_id or not self.generation_id or not self.request_id:
            raise ProtocolViolation("session, generation, and request IDs are required")
        if self.chunk_index < 0:
            raise ProtocolViolation("chunk index must be non-negative")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ProtocolViolation(f"unsupported latent dtype: {self.dtype}")
        if len(self.shape) != 5 or any(int(dim) <= 0 for dim in self.shape):
            raise ProtocolViolation("latent shape must contain five positive dimensions")
        if self.byte_length <= 0:
            raise ProtocolViolation("latent byte length must be positive")
        if not self.checksum:
            raise ProtocolViolation("latent checksum is required")


class ChunkSequenceTracker:
    """Accept exactly-once, monotonically ordered chunks for one generation."""

    def __init__(self, session_id: str, generation_id: str) -> None:
        self.session_id = session_id
        self.generation_id = generation_id
        self.next_chunk_index = 0

    def accept(self, header: LatentChunkHeader) -> AcceptDisposition:
        header.validate()
        if header.session_id != self.session_id:
            raise ProtocolViolation("wrong session")
        if header.generation_id != self.generation_id:
            raise ProtocolViolation("stale generation")
        if header.chunk_index == self.next_chunk_index:
            self.next_chunk_index += 1
            return AcceptDisposition.ACCEPT
        if header.chunk_index < self.next_chunk_index:
            return AcceptDisposition.DUPLICATE
        raise ProtocolViolation("out-of-order chunk")


def checksum_payload(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def encode_message(
    message_type: str,
    *,
    header: LatentChunkHeader | dict[str, Any] | None = None,
    payload: bytes | None = None,
    **fields: Any,
) -> bytes:
    message: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "type": message_type,
        **fields,
    }
    if header is not None:
        message["header"] = asdict(header) if isinstance(header, LatentChunkHeader) else header
    if payload is not None:
        message["payload"] = payload
    return msgspec.msgpack.encode(message)


def decode_message(
    wire: bytes,
    *,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> dict[str, Any]:
    if len(wire) > max_message_bytes:
        raise ProtocolViolation(
            f"message exceeds {max_message_bytes} byte protocol limit"
        )
    try:
        message = msgspec.msgpack.decode(wire)
    except msgspec.DecodeError as exc:
        raise ProtocolViolation("invalid MessagePack message") from exc
    if not isinstance(message, dict):
        raise ProtocolViolation("protocol message must be a map")
    if message.get("version") != PROTOCOL_VERSION:
        raise ProtocolViolation("unsupported protocol version")
    if not isinstance(message.get("type"), str):
        raise ProtocolViolation("protocol message type is required")
    return message


def latent_header_from_message(message: dict[str, Any]) -> LatentChunkHeader:
    raw = message.get("header")
    if not isinstance(raw, dict):
        raise ProtocolViolation("latent header is required")
    try:
        header = LatentChunkHeader(
            **{
                **raw,
                "shape": tuple(raw.get("shape", ())),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation("invalid latent header") from exc
    header.validate()
    return header


def validate_payload(header: LatentChunkHeader, payload: bytes) -> None:
    if len(payload) != header.byte_length:
        raise ProtocolViolation("latent payload length mismatch")
    if checksum_payload(payload) != header.checksum:
        raise ProtocolViolation("latent payload checksum mismatch")
