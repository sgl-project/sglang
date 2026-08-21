#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

MAGIC = b"SGLANG-EXPERTPACK-v1\0\0\0\0"
VERSION = 1
ROLE_NAMES = ("gate", "up", "down")
ROLE_IDS = {name: index for index, name in enumerate(ROLE_NAMES)}
FLAG_IDENTITY_PAYLOAD = 1 << 0
FLAG_TRIPLET_OBJECTS = 1 << 1

# magic, version, header bytes, entry bytes, flags, index count, data start,
# alignment, layer count, expert count, top-k, role count, and three digests.
HEADER_STRUCT = struct.Struct("<24sIIIIQQQIIII32s32s32s")

# layer, expert, role, rank, GGML dtype id, dtype, tensor name, six ranges,
# source tensor/slice and pack hashes, logical role shape, quant/transform,
# quant block size, and generation.
ENTRY_STRUCT = struct.Struct("<HHBBH16s80sQQQQQQ32s32s32s4Q16s16sQQ")


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256(value: str, field: str) -> bytes:
    if len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal SHA-256 digest") from exc
    if len(result) != 32:
        raise ValueError(f"{field} must decode to 32 bytes")
    return result


def encode_fixed(value: str, size: int, field: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) >= size:
        raise ValueError(f"{field} is too long for its {size}-byte field: {value!r}")
    return encoded + bytes(size - len(encoded))


def decode_fixed(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8")


@dataclass(frozen=True)
class PackHeader:
    flags: int
    index_count: int
    data_start: int
    alignment: int
    num_layers: int
    num_experts: int
    top_k: int
    role_count: int
    model_identity_sha256: str
    source_blob_sha256: str
    config_sha256: str

    @property
    def header_bytes(self) -> int:
        return HEADER_STRUCT.size

    @property
    def entry_bytes(self) -> int:
        return ENTRY_STRUCT.size

    def pack(self) -> bytes:
        if self.role_count != len(ROLE_NAMES):
            raise ValueError(f"role_count must be {len(ROLE_NAMES)}")
        expected_entries = self.num_layers * self.num_experts * self.role_count
        if self.index_count != expected_entries:
            raise ValueError(
                f"index_count {self.index_count} does not match {expected_entries}"
            )
        minimum_data_start = self.header_bytes + self.index_count * self.entry_bytes
        if self.data_start < minimum_data_start or self.data_start % self.alignment:
            raise ValueError("data_start is too small or is not aligned")
        return HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            self.header_bytes,
            self.entry_bytes,
            self.flags,
            self.index_count,
            self.data_start,
            self.alignment,
            self.num_layers,
            self.num_experts,
            self.top_k,
            self.role_count,
            parse_sha256(self.model_identity_sha256, "model_identity_sha256"),
            parse_sha256(self.source_blob_sha256, "source_blob_sha256"),
            parse_sha256(self.config_sha256, "config_sha256"),
        )

    @classmethod
    def unpack(cls, raw: bytes) -> PackHeader:
        if len(raw) != HEADER_STRUCT.size:
            raise ValueError("expert-pack header is truncated")
        (
            magic,
            version,
            header_bytes,
            entry_bytes,
            flags,
            index_count,
            data_start,
            alignment,
            num_layers,
            num_experts,
            top_k,
            role_count,
            model_identity_digest,
            source_digest,
            config_digest,
        ) = HEADER_STRUCT.unpack(raw)
        if magic != MAGIC:
            raise ValueError("expert-pack magic does not match")
        if version != VERSION:
            raise ValueError(f"unsupported expert-pack version {version}")
        if header_bytes != HEADER_STRUCT.size or entry_bytes != ENTRY_STRUCT.size:
            raise ValueError(
                "expert-pack struct sizes do not match this implementation"
            )
        result = cls(
            flags=flags,
            index_count=index_count,
            data_start=data_start,
            alignment=alignment,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            role_count=role_count,
            model_identity_sha256=model_identity_digest.hex(),
            source_blob_sha256=source_digest.hex(),
            config_sha256=config_digest.hex(),
        )
        result.pack()
        return result


@dataclass(frozen=True)
class IndexEntry:
    layer: int
    expert: int
    role: str
    dtype_id: int
    dtype: str
    tensor_name: str
    source_tensor_offset: int
    source_tensor_nbytes: int
    source_slice_offset: int
    source_slice_nbytes: int
    pack_offset: int
    pack_nbytes: int
    source_tensor_sha256: str
    source_slice_sha256: str
    checksum: str
    shape: tuple[int, ...]
    quant_scheme: str
    transform_id: str
    block_size: int
    generation: int

    @property
    def key(self) -> tuple[int, int, int]:
        return self.layer, self.expert, ROLE_IDS[self.role]

    def pack(self) -> bytes:
        if self.role not in ROLE_IDS:
            raise ValueError(f"unknown expert role {self.role!r}")
        if not 0 <= self.layer <= 0xFFFF or not 0 <= self.expert <= 0xFFFF:
            raise ValueError("layer/expert does not fit the pack index")
        if not 0 <= self.dtype_id <= 0xFFFF:
            raise ValueError("dtype_id does not fit the pack index")
        if not 1 <= len(self.shape) <= 4 or any(value <= 0 for value in self.shape):
            raise ValueError(f"invalid role shape {self.shape}")
        dims = self.shape + (0,) * (4 - len(self.shape))
        return ENTRY_STRUCT.pack(
            self.layer,
            self.expert,
            ROLE_IDS[self.role],
            len(self.shape),
            self.dtype_id,
            encode_fixed(self.dtype, 16, "dtype"),
            encode_fixed(self.tensor_name, 80, "tensor_name"),
            self.source_tensor_offset,
            self.source_tensor_nbytes,
            self.source_slice_offset,
            self.source_slice_nbytes,
            self.pack_offset,
            self.pack_nbytes,
            parse_sha256(self.source_tensor_sha256, "source_tensor_sha256"),
            parse_sha256(self.source_slice_sha256, "source_slice_sha256"),
            parse_sha256(self.checksum, "checksum"),
            *dims,
            encode_fixed(self.quant_scheme, 16, "quant_scheme"),
            encode_fixed(self.transform_id, 16, "transform_id"),
            self.block_size,
            self.generation,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> IndexEntry:
        if len(raw) != ENTRY_STRUCT.size:
            raise ValueError("expert-pack index entry is truncated")
        values = ENTRY_STRUCT.unpack(raw)
        role_id = values[2]
        rank = values[3]
        if role_id >= len(ROLE_NAMES) or not 1 <= rank <= 4:
            raise ValueError("expert-pack index contains an invalid role or rank")
        shape = tuple(values[16 : 16 + rank])
        return cls(
            layer=values[0],
            expert=values[1],
            role=ROLE_NAMES[role_id],
            dtype_id=values[4],
            dtype=decode_fixed(values[5]),
            tensor_name=decode_fixed(values[6]),
            source_tensor_offset=values[7],
            source_tensor_nbytes=values[8],
            source_slice_offset=values[9],
            source_slice_nbytes=values[10],
            pack_offset=values[11],
            pack_nbytes=values[12],
            source_tensor_sha256=values[13].hex(),
            source_slice_sha256=values[14].hex(),
            checksum=values[15].hex(),
            shape=shape,
            quant_scheme=decode_fixed(values[20]),
            transform_id=decode_fixed(values[21]),
            block_size=values[22],
            generation=values[23],
        )

    def to_dict(self) -> dict[str, object]:
        value = dict(self.__dict__)
        value["shape"] = list(self.shape)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> IndexEntry:
        fields = dict(value)
        fields["shape"] = tuple(int(item) for item in fields["shape"])
        return cls(**fields)


def read_header(stream: BinaryIO) -> PackHeader:
    stream.seek(0)
    return PackHeader.unpack(stream.read(HEADER_STRUCT.size))


def read_index(stream: BinaryIO, header: PackHeader) -> list[IndexEntry]:
    stream.seek(header.header_bytes)
    entries = []
    for _ in range(header.index_count):
        entries.append(IndexEntry.unpack(stream.read(header.entry_bytes)))
    return entries


def write_index(
    stream: BinaryIO, header: PackHeader, entries: Iterable[IndexEntry]
) -> str:
    ordered = sorted(entries, key=lambda entry: entry.key)
    if len(ordered) != header.index_count:
        raise ValueError(
            f"expected {header.index_count} index entries, got {len(ordered)}"
        )
    digest = hashlib.sha256()
    stream.seek(header.header_bytes)
    for entry in ordered:
        raw = entry.pack()
        stream.write(raw)
        digest.update(raw)
    return digest.hexdigest()


def inspect_pack(path: Path, limit: int = 12) -> dict[str, object]:
    with path.open("rb", buffering=0) as stream:
        header = read_header(stream)
        entries = read_index(stream, header)
    summary = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "header": header.__dict__,
        "role_counts": {
            role: sum(entry.role == role for entry in entries) for role in ROLE_NAMES
        },
        "payload_bytes": sum(entry.pack_nbytes for entry in entries),
        "entries": [entry.to_dict() for entry in entries[:limit]],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary
