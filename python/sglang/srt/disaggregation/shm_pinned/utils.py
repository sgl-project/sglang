"""
Utility types for the shm_pinned backend.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from enum import IntEnum

SHM_MAGIC = b"SHP1"
SHM_VERSION = 1

DEFAULT_SLOT_COUNT = 32
DEFAULT_CHUNK_TOKENS = 512


class SlotState(IntEnum):
    FREE = 0
    WRITING = 1
    READY = 2
    READING = 3


@dataclass
class SlotMeta:
    """
    64-byte metadata entry for one ring-buffer slot.
    """

    state: SlotState = SlotState.FREE
    room: int = 0
    index_start: int = 0
    index_len: int = 0
    is_last: int = 0
    layer_start: int = 0
    layer_count: int = 0xFFFFFFFF
    valid_bytes: int = 0
    seqno: int = 0
    owner_pid: int = 0

    _STRUCT_FORMAT = "<IQQIIIIQQI8s"
    _STRUCT_SIZE = 64

    def pack(self) -> bytes:
        return struct.pack(
            self._STRUCT_FORMAT,
            int(self.state),
            int(self.room),
            int(self.index_start),
            int(self.index_len),
            int(self.is_last),
            int(self.layer_start),
            int(self.layer_count),
            int(self.valid_bytes),
            int(self.seqno),
            int(self.owner_pid),
            b"\x00" * 8,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "SlotMeta":
        (
            state,
            room,
            index_start,
            index_len,
            is_last,
            layer_start,
            layer_count,
            valid_bytes,
            seqno,
            owner_pid,
            _reserved,
        ) = struct.unpack(cls._STRUCT_FORMAT, data)

        return cls(
            state=SlotState(state),
            room=room,
            index_start=index_start,
            index_len=index_len,
            is_last=is_last,
            layer_start=layer_start,
            layer_count=layer_count,
            valid_bytes=valid_bytes,
            seqno=seqno,
            owner_pid=owner_pid,
        )

    @classmethod
    def size(cls) -> int:
        return cls._STRUCT_SIZE


@dataclass
class ShmHeader:
    magic: bytes = SHM_MAGIC
    version: int = SHM_VERSION
    slot_count: int = DEFAULT_SLOT_COUNT
    slot_bytes: int = 0
    write_idx: int = 0
    read_idx: int = 0

    _STRUCT_FORMAT = "<4sIIQII36s"
    _STRUCT_SIZE = 64

    def pack(self) -> bytes:
        return struct.pack(
            self._STRUCT_FORMAT,
            self.magic,
            self.version,
            int(self.slot_count),
            int(self.slot_bytes),
            int(self.write_idx),
            int(self.read_idx),
            b"\x00" * 36,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ShmHeader":
        magic, version, slot_count, slot_bytes, write_idx, read_idx, _reserved = (
            struct.unpack(cls._STRUCT_FORMAT, data)
        )
        return cls(
            magic=magic,
            version=version,
            slot_count=slot_count,
            slot_bytes=slot_bytes,
            write_idx=write_idx,
            read_idx=read_idx,
        )

    @classmethod
    def size(cls) -> int:
        return cls._STRUCT_SIZE

    def validate(self) -> bool:
        return self.magic == SHM_MAGIC and self.version == SHM_VERSION


@dataclass
class ShmPinnedInfo:
    data_shm_name: str = ""
    meta_shm_name: str = ""
    sem_free_name: str = ""
    sem_ready_name: str = ""
    sem_slot_name: str = ""
    slot_count: int = DEFAULT_SLOT_COUNT
    slot_bytes: int = 0
    session_id: str = ""
    kv_item_lens: list[int] = field(default_factory=list)
    decode_host: str = ""
    decode_port: int = 0

    def to_dict(self) -> dict:
        return {
            "data_shm_name": self.data_shm_name,
            "meta_shm_name": self.meta_shm_name,
            "sem_free_name": self.sem_free_name,
            "sem_ready_name": self.sem_ready_name,
            "sem_slot_name": self.sem_slot_name,
            "slot_count": self.slot_count,
            "slot_bytes": self.slot_bytes,
            "session_id": self.session_id,
            "kv_item_lens": self.kv_item_lens,
            "decode_host": self.decode_host,
            "decode_port": self.decode_port,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShmPinnedInfo":
        return cls(
            data_shm_name=data["data_shm_name"],
            meta_shm_name=data["meta_shm_name"],
            sem_free_name=data["sem_free_name"],
            sem_ready_name=data["sem_ready_name"],
            sem_slot_name=data.get("sem_slot_name", ""),
            slot_count=int(data["slot_count"]),
            slot_bytes=int(data["slot_bytes"]),
            session_id=data.get("session_id", ""),
            kv_item_lens=list(data.get("kv_item_lens", [])),
            decode_host=data.get("decode_host", ""),
            decode_port=int(data.get("decode_port", 0)),
        )


def generate_shm_names(session_id: str) -> tuple[str, str, str, str, str]:
    pid = os.getpid()
    prefix = f"sglang_shm_{pid}_{session_id}"
    return (
        f"/{prefix}_data",
        f"/{prefix}_meta",
        f"/{prefix}_free",
        f"/{prefix}_ready",
        f"/{prefix}_slot",
    )


def calculate_slot_bytes(
    chunk_pages: int,
    kv_item_lens: list[int],
    extra_slot_bytes: int = 0,
) -> int:
    if not kv_item_lens:
        raise ValueError("kv_item_lens is required for shm_pinned sizing")
    return int(chunk_pages) * int(sum(kv_item_lens)) + int(extra_slot_bytes)


def calculate_meta_shm_size(slot_count: int) -> int:
    return ShmHeader.size() + int(slot_count) * SlotMeta.size()


def get_slot_meta_offset(slot_idx: int) -> int:
    return ShmHeader.size() + int(slot_idx) * SlotMeta.size()
