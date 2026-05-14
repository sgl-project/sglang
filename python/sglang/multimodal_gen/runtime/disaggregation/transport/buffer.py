# SPDX-License-Identifier: Apache-2.0
"""TransferTensorBuffer: memory staging area for disaggregated tensor transfer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.allocator import (
    BuddyAllocator,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    str_to_dtype,
)

logger = logging.getLogger(__name__)


@dataclass
class SlotHandle:
    request_id: str
    offset: int  # byte offset in the pool
    size: int  # allocated size in bytes
    tensor_views: dict[str, torch.Tensor | list[torch.Tensor]] = field(
        default_factory=dict
    )


class TransferTensorBuffer:
    """Memory pool for staging tensor payloads between roles.

    Wraps a contiguous block of memory (CPU pinned or GPU) with a BuddyAllocator.
    """

    def __init__(
        self,
        pool_size: int,
        min_block_size: int = 1 << 20,
        role_name: str = "unknown",
        device: str = "cpu",
    ):
        self._role_name = role_name
        self._device = device
        self._allocator = BuddyAllocator(pool_size, min_block_size)
        actual_size = self._allocator.pool_size

        if device == "cpu":
            self._pool = torch.empty(actual_size, dtype=torch.uint8, pin_memory=True)
        else:
            self._pool = torch.empty(actual_size, dtype=torch.uint8, device=device)
        self._pool_ptr = self._pool.data_ptr()

        pool_location = "pinned CPU" if device == "cpu" else f"GPU ({device})"
        logger.info(
            "TransferTensorBuffer[%s]: allocated %d MiB %s memory "
            "(min_block=%d KiB)",
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

    def allocate(self, size: int, request_id: str) -> SlotHandle | None:
        """Allocate a slot. Returns None if pool is full."""
        offset = self._allocator.allocate(size, request_id=request_id)
        if offset is None:
            logger.warning(
                "TransferTensorBuffer[%s]: allocation failed for %s (%d bytes). "
                "Pool stats: %s",
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
        """Copy a tensor into the pool slot. Returns bytes written."""
        src_tensor = tensor.contiguous()
        nbytes = src_tensor.numel() * src_tensor.element_size()

        if byte_offset + nbytes > handle.size:
            raise ValueError(
                f"Write exceeds slot: offset={byte_offset}, nbytes={nbytes}, "
                f"slot_size={handle.size}"
            )

        dst = self._pool[
            handle.offset + byte_offset : handle.offset + byte_offset + nbytes
        ]
        src_bytes = src_tensor.view(torch.uint8).reshape(-1)

        if stream is not None:
            with torch.cuda.stream(stream):
                dst.copy_(src_bytes, non_blocking=True)
        else:
            dst.copy_(src_bytes, non_blocking=True)

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
        """Read a tensor from the pool slot. Returns a clone on target device."""
        nbytes = 1
        for s in shape:
            nbytes *= s
        nbytes *= torch.tensor([], dtype=dtype).element_size()

        raw = self._pool[
            handle.offset + byte_offset : handle.offset + byte_offset + nbytes
        ]
        src = raw.view(dtype).reshape(shape)

        pool_dev = str(self._pool.device)
        target_dev = str(device)

        same_device = pool_dev == target_dev

        if same_device:
            # Clone to decouple tensor lifetime from pool slot
            if stream is not None:
                with torch.cuda.stream(stream):
                    return src.clone()
            return src.clone()

        if stream is not None:
            with torch.cuda.stream(stream):
                return src.to(device, non_blocking=True)
        return src.to(device, non_blocking=True)

    def write_tensors_from_gpu(
        self,
        handle: SlotHandle,
        tensors: dict[str, torch.Tensor | list[torch.Tensor] | None],
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, list[dict]]:
        """Batch-write GPU tensors into a slot. Returns a manifest for later reads."""
        manifest: dict[str, list[dict]] = {}
        byte_offset = 0

        # Ensure copy stream sees all prior compute kernels
        if stream is not None:
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
                byte_offset += nbytes
                byte_offset = (byte_offset + 511) & ~511  # align to 512B

            elif isinstance(value, list):
                for i, t in enumerate(value):
                    if t is None:
                        continue
                    nbytes = self.write_tensor(
                        handle, f"{name}[{i}]", t, byte_offset, stream
                    )
                    entries.append(
                        {
                            "offset": byte_offset,
                            "shape": list(t.shape),
                            "dtype": str(t.dtype).replace("torch.", ""),
                            "list_index": i,
                        }
                    )
                    byte_offset += nbytes
                    byte_offset = (byte_offset + 511) & ~511

            if entries:
                manifest[name] = entries

        return manifest

    def read_tensors_from_manifest(
        self,
        handle: SlotHandle,
        manifest: dict[str, list[dict]],
        device: torch.device | str = "cpu",
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Batch-read tensors from a slot using a manifest."""
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {}

        for name, entries in manifest.items():
            if not entries:
                continue
            has_list_index = any("list_index" in e for e in entries)

            if has_list_index:
                max_idx = max(e.get("list_index", 0) for e in entries) + 1
                tensors = [None] * max_idx
                for entry in entries:
                    t = self.read_tensor(
                        handle,
                        entry["shape"],
                        str_to_dtype(entry["dtype"]),
                        entry["offset"],
                        device,
                        stream,
                    )
                    tensors[entry["list_index"]] = t
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
        """Estimate how many requests of typical size can still be buffered."""
        return self._allocator.count_free_slots(typical_request_size)

    def get_stats(self) -> dict:
        alloc_stats = self._allocator.get_stats()
        alloc_stats["role"] = self._role_name
        return alloc_stats
