# SPDX-License-Identifier: Apache-2.0
"""Host-side tensor and meta buffers for diffusion disaggregation."""

from __future__ import annotations

import json
import logging
import struct
import uuid
from dataclasses import dataclass, field
from multiprocessing import shared_memory

import numpy as np
import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.allocator import (
    BuddyAllocator,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    str_to_dtype,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.pinned_memory import (
    PinnedHostMemoryRegistration,
    register_pinned_host_memory,
)

logger = logging.getLogger(__name__)

_META_MAGIC = b"DMTA"
_META_VERSION = 1
_META_HEADER = struct.Struct("<4sIIIII40s")
_META_DESC = struct.Struct("<BBHIIII44s")
_META_HEADER_SIZE = _META_HEADER.size
_META_DESC_SIZE = _META_DESC.size
_ALIGN_64 = 64


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


@dataclass
class SlotHandle:
    request_id: str
    offset: int
    size: int
    tensor_views: dict[str, torch.Tensor | list[torch.Tensor]] = field(
        default_factory=dict
    )
    slot_id: int | None = None


@dataclass
class BufferDescriptor:
    pool_ptr: int
    pool_size: int
    shm_name: str | None = None


class _SharedMemoryRegion:
    def __init__(
        self,
        role_name: str,
        tag: str,
        size: int,
        *,
        pin_memory: bool = False,
        pin_memory_strict: bool = False,
    ):
        self._name = f"sgl_diff_{role_name}_{tag}_{uuid.uuid4().hex}"
        self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=size)
        self._np_array = np.ndarray((size,), dtype=np.uint8, buffer=self._shm.buf)
        self._tensor = torch.from_numpy(self._np_array)
        self._pin_registration: PinnedHostMemoryRegistration | None = None
        if pin_memory:
            try:
                self._pin_registration = register_pinned_host_memory(
                    int(self._np_array.ctypes.data),
                    size,
                    enabled=True,
                    strict=pin_memory_strict,
                )
            except Exception:
                self.cleanup()
                raise

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_pinned(self) -> bool:
        return bool(self._pin_registration and self._pin_registration.registered)

    @property
    def pin_memory_status(self) -> str:
        if self._pin_registration is None:
            return "disabled"
        return self._pin_registration.status

    @property
    def pin_memory_error(self) -> str | None:
        if self._pin_registration is None:
            return None
        return self._pin_registration.error

    def cleanup(self) -> None:
        if self._pin_registration is not None:
            self._pin_registration.unregister()
            self._pin_registration = None
        try:
            self._shm.close()
        except FileNotFoundError:
            pass
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


class TransferTensorBuffer:
    """Memory pool for staging tensor payloads between diffusion roles.

    Diffusion disaggregation keeps these buffers on the host so that both
    inter-machine RDMA and same-host shared-memory fast paths operate on the
    same memory model.
    """

    def __init__(
        self,
        pool_size: int,
        min_block_size: int = 1 << 20,
        role_name: str = "unknown",
        device: str = "cpu",
        shared_memory_backing: bool | None = None,
        pin_memory: bool = False,
        pin_memory_strict: bool = False,
    ):
        self._role_name = role_name
        self._device = device
        self._allocator = BuddyAllocator(pool_size, min_block_size)
        actual_size = self._allocator.pool_size
        self._shared_region: _SharedMemoryRegion | None = None

        if device == "cpu":
            if shared_memory_backing is None:
                shared_memory_backing = True
            if shared_memory_backing:
                self._shared_region = _SharedMemoryRegion(
                    role_name,
                    "tensor",
                    actual_size,
                    pin_memory=pin_memory,
                    pin_memory_strict=pin_memory_strict,
                )
                self._pool = self._shared_region.tensor
            else:
                self._pool = torch.empty(actual_size, dtype=torch.uint8)
        else:
            self._pool = torch.empty(actual_size, dtype=torch.uint8, device=device)
        self._pool_ptr = self._pool.data_ptr()

        pool_location = (
            f"host-shm{'+pinned' if self.pinned_shared_memory else ''}({self.shared_memory_name})"
            if self._shared_region is not None
            else ("host" if device == "cpu" else f"GPU ({device})")
        )
        logger.info(
            "TransferTensorBuffer[%s]: allocated %d MiB %s memory (min_block=%d KiB)",
            role_name,
            actual_size >> 20,
            pool_location,
            min_block_size >> 10,
        )

    @property
    def pool_size(self) -> int:
        return self._allocator.pool_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def pool_data_ptr(self) -> int:
        return self._pool_ptr

    @property
    def shared_memory_name(self) -> str | None:
        return self._shared_region.name if self._shared_region is not None else None

    @property
    def uses_shared_memory(self) -> bool:
        return self._shared_region is not None

    @property
    def pinned_shared_memory(self) -> bool:
        return bool(self._shared_region and self._shared_region.is_pinned)

    @property
    def pin_memory_status(self) -> str:
        if self._shared_region is None:
            return "not_shared_memory"
        return self._shared_region.pin_memory_status

    @property
    def pin_memory_error(self) -> str | None:
        if self._shared_region is None:
            return None
        return self._shared_region.pin_memory_error

    def descriptor(self) -> BufferDescriptor:
        return BufferDescriptor(
            pool_ptr=self.pool_data_ptr,
            pool_size=self.pool_size,
            shm_name=self.shared_memory_name,
        )

    def cleanup(self) -> None:
        if self._shared_region is not None:
            self._shared_region.cleanup()
            self._shared_region = None

    def allocate(self, size: int, request_id: str) -> SlotHandle | None:
        offset = self._allocator.allocate(size, request_id=request_id)
        if offset is None:
            logger.warning(
                "TransferTensorBuffer[%s]: allocation failed for %s (%d bytes). Pool stats: %s",
                self._role_name,
                request_id,
                size,
                self._allocator.get_stats(),
            )
            return None

        block = self._allocator.get_block_info(offset)
        return SlotHandle(
            request_id=request_id,
            offset=offset,
            size=block.size if block else size,
        )

    def free(self, handle: SlotHandle) -> bool:
        return self._allocator.free(handle.offset)

    def write_tensor(
        self,
        handle: SlotHandle,
        name: str,
        tensor: torch.Tensor,
        byte_offset: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> int:
        src_tensor = tensor.contiguous()
        nbytes = src_tensor.numel() * src_tensor.element_size()

        if byte_offset + nbytes > handle.size:
            raise ValueError(
                f"Write exceeds slot: offset={byte_offset}, nbytes={nbytes}, slot_size={handle.size}"
            )

        dst = self._pool[
            handle.offset + byte_offset : handle.offset + byte_offset + nbytes
        ]
        src_bytes = src_tensor.view(torch.uint8).reshape(-1)

        if stream is not None and src_tensor.is_cuda:
            with torch.cuda.stream(stream):
                dst.copy_(src_bytes, non_blocking=True)
        else:
            dst.copy_(src_bytes, non_blocking=src_tensor.is_cuda)

        return nbytes

    def read_tensor(
        self,
        handle: SlotHandle,
        shape: list[int],
        dtype: torch.dtype,
        byte_offset: int = 0,
        device: torch.device | str = "cpu",
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        nbytes = torch.tensor([], dtype=dtype).element_size()
        for dim in shape:
            nbytes *= dim

        raw = self._pool[
            handle.offset + byte_offset : handle.offset + byte_offset + nbytes
        ]
        src = raw.view(dtype).reshape(shape)

        if str(device) == str(self._pool.device):
            if stream is not None and str(device).startswith("cuda"):
                with torch.cuda.stream(stream):
                    return src.clone()
            return src.clone()

        if stream is not None and str(device).startswith("cuda"):
            with torch.cuda.stream(stream):
                return src.to(device, non_blocking=True)
        return src.to(device, non_blocking=str(device).startswith("cuda"))

    def write_tensors_from_gpu(
        self,
        handle: SlotHandle,
        tensors: dict[str, torch.Tensor | list[torch.Tensor] | None],
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, list[dict]]:
        manifest: dict[str, list[dict]] = {}
        byte_offset = 0
        if stream is not None and torch.cuda.is_available():
            stream.wait_stream(torch.cuda.current_stream())

        for name, value in tensors.items():
            if value is None:
                continue

            entries = []
            if isinstance(value, torch.Tensor):
                nbytes = self.write_tensor(handle, name, value, byte_offset, stream)
                entries.append(
                    {
                        "offset": byte_offset,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype).replace("torch.", ""),
                    }
                )
                byte_offset = _align_up(byte_offset + nbytes, 512)
            elif isinstance(value, list):
                for i, tensor in enumerate(value):
                    if tensor is None:
                        continue
                    nbytes = self.write_tensor(
                        handle, f"{name}[{i}]", tensor, byte_offset, stream
                    )
                    entries.append(
                        {
                            "offset": byte_offset,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                            "list_index": i,
                        }
                    )
                    byte_offset = _align_up(byte_offset + nbytes, 512)

            if entries:
                manifest[name] = entries

        return manifest

    def read_tensors_from_manifest(
        self,
        handle: SlotHandle | None,
        manifest: dict[str, list[dict]],
        device: torch.device | str = "cpu",
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if handle is None:
            return {}

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for name, entries in manifest.items():
            if not entries:
                continue
            has_list_index = any("list_index" in entry for entry in entries)
            if has_list_index:
                max_idx = max(entry.get("list_index", 0) for entry in entries) + 1
                tensors = [None] * max_idx
                for entry in entries:
                    tensors[entry["list_index"]] = self.read_tensor(
                        handle,
                        entry["shape"],
                        str_to_dtype(entry["dtype"]),
                        entry["offset"],
                        device,
                        stream,
                    )
                result[name] = tensors
            else:
                entry = entries[0]
                result[name] = self.read_tensor(
                    handle,
                    entry["shape"],
                    str_to_dtype(entry["dtype"]),
                    entry["offset"],
                    device,
                    stream,
                )
        return result

    def free_slots_count(self, typical_request_size: int) -> int:
        return self._allocator.count_free_slots(typical_request_size)

    def get_stats(self) -> dict:
        alloc_stats = self._allocator.get_stats()
        alloc_stats["role"] = self._role_name
        alloc_stats["uses_shared_memory"] = self.uses_shared_memory
        alloc_stats["shared_memory_name"] = self.shared_memory_name
        alloc_stats["pinned_shared_memory"] = self.pinned_shared_memory
        alloc_stats["pin_memory_status"] = self.pin_memory_status
        alloc_stats["pin_memory_error"] = self.pin_memory_error
        return alloc_stats


def _encode_meta_value(value) -> tuple[int, bytes, int]:
    if value is None:
        return 0, b"", _ALIGN_64
    if isinstance(value, bool):
        return 1, struct.pack("<?", value), _ALIGN_64
    if isinstance(value, int) and not isinstance(value, bool):
        return 2, struct.pack("<q", value), _ALIGN_64
    if isinstance(value, float):
        return 3, struct.pack("<d", value), _ALIGN_64
    if isinstance(value, str):
        return 4, value.encode("utf-8"), _ALIGN_64
    return 5, json.dumps(value, separators=(",", ":")).encode("utf-8"), _ALIGN_64


def _decode_meta_value(kind: int, payload: bytes):
    if kind == 0:
        return None
    if kind == 1:
        return struct.unpack("<?", payload[:1])[0]
    if kind == 2:
        return struct.unpack("<q", payload[:8])[0]
    if kind == 3:
        return struct.unpack("<d", payload[:8])[0]
    if kind == 4:
        return payload.decode("utf-8")
    return json.loads(payload.decode("utf-8"))


def pack_transfer_metadata(manifest: dict, scalar_fields: dict) -> bytes:
    payload = bytearray()
    descriptors = []

    def append_aligned(data: bytes, alignment: int) -> tuple[int, int]:
        offset = _align_up(len(payload), alignment)
        if offset > len(payload):
            payload.extend(b"\0" * (offset - len(payload)))
        payload.extend(data)
        return offset, len(data)

    for key, value in scalar_fields.items():
        key_bytes = key.encode("utf-8")
        key_offset, key_length = append_aligned(key_bytes, 8)
        value_kind, value_bytes, value_alignment = _encode_meta_value(value)
        value_offset, value_length = append_aligned(value_bytes, value_alignment)
        descriptors.append(
            (1, value_kind, key_offset, key_length, value_offset, value_length)
        )

    for key, entries in manifest.items():
        key_bytes = key.encode("utf-8")
        key_offset, key_length = append_aligned(key_bytes, 8)
        value_bytes = json.dumps(entries, separators=(",", ":")).encode("utf-8")
        value_offset, value_length = append_aligned(value_bytes, _ALIGN_64)
        descriptors.append((2, 5, key_offset, key_length, value_offset, value_length))

    descriptor_bytes = len(descriptors) * _META_DESC_SIZE
    payload_offset = _META_HEADER_SIZE + descriptor_bytes
    result = bytearray(payload_offset + len(payload))
    result[:_META_HEADER_SIZE] = _META_HEADER.pack(
        _META_MAGIC,
        _META_VERSION,
        len(descriptors),
        descriptor_bytes,
        payload_offset,
        len(payload),
        b"",
    )

    for idx, desc in enumerate(descriptors):
        start = _META_HEADER_SIZE + idx * _META_DESC_SIZE
        result[start : start + _META_DESC_SIZE] = _META_DESC.pack(
            desc[0],
            desc[1],
            0,
            desc[2],
            desc[3],
            desc[4],
            desc[5],
            b"",
        )

    result[payload_offset:] = payload
    return bytes(result)


def unpack_transfer_metadata(blob: bytes) -> tuple[dict, dict]:
    header = _META_HEADER.unpack(blob[:_META_HEADER_SIZE])
    magic, version, num_records, descriptor_bytes, payload_offset, payload_bytes, _ = (
        header
    )
    if magic != _META_MAGIC:
        raise ValueError("Invalid transfer metadata magic")
    if version != _META_VERSION:
        raise ValueError(f"Unsupported transfer metadata version: {version}")

    payload = blob[payload_offset : payload_offset + payload_bytes]
    manifest: dict = {}
    scalar_fields: dict = {}
    for idx in range(num_records):
        start = _META_HEADER_SIZE + idx * _META_DESC_SIZE
        (
            section_kind,
            value_kind,
            _,
            key_offset,
            key_length,
            value_offset,
            value_length,
            _,
        ) = _META_DESC.unpack(blob[start : start + _META_DESC_SIZE])
        key = payload[key_offset : key_offset + key_length].decode("utf-8")
        value_blob = bytes(payload[value_offset : value_offset + value_length])
        value = _decode_meta_value(value_kind, value_blob)
        if section_kind == 1:
            scalar_fields[key] = value
        elif section_kind == 2:
            manifest[key] = value

    return manifest, scalar_fields


def estimate_transfer_meta_bytes(manifest: dict, scalar_fields: dict) -> int:
    return len(pack_transfer_metadata(manifest, scalar_fields))


class TransferMetaBuffer:
    """Fixed-slot host buffer for request metadata payloads."""

    def __init__(
        self,
        slot_count: int,
        slot_size: int,
        role_name: str = "unknown",
        *,
        shared_memory_backing: bool = True,
    ):
        self._role_name = role_name
        self._slot_count = max(1, int(slot_count))
        self._slot_size = _align_up(max(_META_HEADER_SIZE, int(slot_size)), _ALIGN_64)
        total_size = self._slot_count * self._slot_size
        self._shared_region: _SharedMemoryRegion | None = None
        if shared_memory_backing:
            self._shared_region = _SharedMemoryRegion(role_name, "meta", total_size)
            self._pool = self._shared_region.tensor
        else:
            self._pool = torch.empty(total_size, dtype=torch.uint8)
        self._pool_ptr = self._pool.data_ptr()
        self._free_slots = list(range(self._slot_count))
        logger.info(
            "TransferMetaBuffer[%s]: allocated %d slots x %d bytes (%s)",
            role_name,
            self._slot_count,
            self._slot_size,
            (
                f"host-shm({self.shared_memory_name})"
                if self._shared_region is not None
                else "host"
            ),
        )

    @property
    def pool_size(self) -> int:
        return self._slot_count * self._slot_size

    @property
    def pool_data_ptr(self) -> int:
        return self._pool_ptr

    @property
    def slot_size(self) -> int:
        return self._slot_size

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def shared_memory_name(self) -> str | None:
        return self._shared_region.name if self._shared_region is not None else None

    def descriptor(self) -> BufferDescriptor:
        return BufferDescriptor(
            pool_ptr=self.pool_data_ptr,
            pool_size=self.pool_size,
            shm_name=self.shared_memory_name,
        )

    def cleanup(self) -> None:
        if self._shared_region is not None:
            self._shared_region.cleanup()
            self._shared_region = None

    def allocate(self, request_id: str) -> SlotHandle | None:
        if not self._free_slots:
            logger.warning(
                "TransferMetaBuffer[%s]: allocation failed for %s, no free slots",
                self._role_name,
                request_id,
            )
            return None
        slot_id = self._free_slots.pop(0)
        return SlotHandle(
            request_id=request_id,
            offset=slot_id * self._slot_size,
            size=self._slot_size,
            slot_id=slot_id,
        )

    def free(self, handle: SlotHandle) -> None:
        if handle.slot_id is None:
            return
        if handle.slot_id not in self._free_slots:
            self._free_slots.append(handle.slot_id)
            self._free_slots.sort()

    def write_metadata(
        self,
        handle: SlotHandle,
        manifest: dict,
        scalar_fields: dict,
    ) -> int:
        blob = pack_transfer_metadata(manifest, scalar_fields)
        if len(blob) > handle.size:
            raise ValueError(
                f"Metadata exceeds slot capacity: size={len(blob)}, slot_size={handle.size}"
            )
        self._pool[handle.offset : handle.offset + handle.size].zero_()
        target = self._pool[handle.offset : handle.offset + len(blob)]
        target.copy_(torch.tensor(list(blob), dtype=torch.uint8))
        return len(blob)

    def read_metadata(self, handle: SlotHandle) -> tuple[dict, dict]:
        raw = self._pool[handle.offset : handle.offset + handle.size]
        blob = bytes(raw.tolist())
        return unpack_transfer_metadata(blob)

    def get_slot_addr(self, handle: SlotHandle) -> int:
        return self._pool_ptr + handle.offset
