"""MNNVL FABRIC transport for tokenizer-to-encoder multimodal features.

The producer owns one bounded VMM allocation. Consumers import its 64-byte
FABRIC handle once per device, then copy each leased slice directly into local
HBM. Ready and acknowledgement words live in the same allocation and use CUDA
stream memory operations, so neither payloads nor lifecycle messages detour
through host memory.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_CONTROL_WORD_BYTES = 4
_DEFAULT_MAX_INFLIGHT_SLICES = 4096
_DATA_ALIGNMENT = 256


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
    return (value // alignment) * alignment


def _driver_modules():
    from cuda.bindings import driver as cuda

    from sglang.srt.distributed.device_communicators.vmm_utils import check_drv

    return cuda, check_drv


@dataclass
class _FabricPoolMapping:
    va: int
    size: int
    device_id: int
    byte_tensor: torch.Tensor

    @classmethod
    def open(
        cls, fabric_handle: bytes, allocation_size: int, device_id: int
    ) -> _FabricPoolMapping:
        from sglang.srt.distributed.device_communicators.vmm_utils import (
            import_and_map_alloc,
        )
        from sglang.srt.layers.moe.dwdp.vmm import tensor_from_ptr

        with torch.cuda.device(device_id):
            va = import_and_map_alloc(
                fabric_handle,
                None,
                allocation_size,
                device_id,
                use_fabric=True,
                peer_rank=-1,
            )
            byte_tensor = tensor_from_ptr(
                va, (allocation_size,), torch.uint8, device_id
            )
        return cls(
            va=va,
            size=allocation_size,
            device_id=device_id,
            byte_tensor=byte_tensor,
        )

    def close(self) -> None:
        from sglang.srt.layers.moe.dwdp.vmm import free_va, unmap_va

        self.byte_tensor = None
        try:
            unmap_va(self.va, self.size)
        finally:
            free_va(self.va, self.size)
            self.va = 0


_mapping_cache: dict[tuple[bytes, int, int], _FabricPoolMapping] = {}
_mapping_cache_lock = threading.Lock()


def _get_or_open_mapping(
    fabric_handle: bytes, allocation_size: int, device_id: int
) -> _FabricPoolMapping:
    key = (fabric_handle, allocation_size, device_id)
    mapping = _mapping_cache.get(key)
    if mapping is None:
        with _mapping_cache_lock:
            mapping = _mapping_cache.get(key)
            if mapping is None:
                mapping = _FabricPoolMapping.open(
                    fabric_handle, allocation_size, device_id
                )
                _mapping_cache[key] = mapping
    return mapping


def _close_mapping_cache() -> None:
    with _mapping_cache_lock:
        mappings = list(_mapping_cache.values())
        _mapping_cache.clear()
    for mapping in mappings:
        try:
            mapping.close()
        except Exception:
            # CUDA may already be unloading during interpreter shutdown.
            pass


atexit.register(_close_mapping_cache)


def _stream_wait_value32(device_id: int, address: int, value: int) -> None:
    cuda, check_drv = _driver_modules()
    stream = torch.cuda.current_stream(device_id)
    check_drv(
        cuda.cuStreamWaitValue32(stream.cuda_stream, address, value, 0),
        "cuStreamWaitValue32(mm FABRIC)",
    )


def _stream_write_value32(device_id: int, address: int, value: int) -> None:
    cuda, check_drv = _driver_modules()
    stream = torch.cuda.current_stream(device_id)
    check_drv(
        cuda.cuStreamWriteValue32(stream.cuda_stream, address, value, 0),
        "cuStreamWriteValue32(mm FABRIC)",
    )


def _resolve_consumer_rank(
    total_consumer_count: int, consumer_rank: Optional[int] = None
) -> int:
    if total_consumer_count == 1:
        return 0
    if consumer_rank is None:
        try:
            from sglang.srt.runtime_context import get_parallel

            # The producer creates one acknowledgement slot per global TP rank.
            # attn_tp_rank may be local to an attention/DCP subgroup and can
            # therefore alias another consumer's slot.
            rank = int(get_parallel().tp_rank)
        except Exception as exc:
            raise RuntimeError(
                "Cannot resolve the multimodal FABRIC consumer rank before parallel "
                "state initialization"
            ) from exc
    else:
        rank = int(consumer_rank)
    if not 0 <= rank < total_consumer_count:
        raise RuntimeError(
            f"FABRIC consumer rank {rank} is outside [0, {total_consumer_count})"
        )
    return rank


class FabricTensorTransportProxy:
    """Pickle-friendly reference to one slice of a remote FABRIC pool."""

    def __init__(
        self,
        *,
        fabric_handle: bytes,
        allocation_size: int,
        data_byte_offset: int,
        nbytes: int,
        shape: torch.Size,
        dtype: torch.dtype,
        ready_byte_offset: int,
        ack_byte_offset: int,
        generation: int,
        total_consumer_count: int,
    ) -> None:
        self.fabric_handle = fabric_handle
        self.allocation_size = allocation_size
        self.data_byte_offset = data_byte_offset
        self.nbytes = nbytes
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ready_byte_offset = ready_byte_offset
        self.ack_byte_offset = ack_byte_offset
        self.generation = generation
        self.total_consumer_count = total_consumer_count
        self.reconstruct_tensor: Optional[torch.Tensor] = None
        self._consumer_acknowledged = False

    def _mapping(self, device_id: int) -> _FabricPoolMapping:
        return _get_or_open_mapping(self.fabric_handle, self.allocation_size, device_id)

    def _wait_until_ready(self, mapping: _FabricPoolMapping, device_id: int) -> None:
        _stream_wait_value32(
            device_id, mapping.va + self.ready_byte_offset, self.generation
        )

    def _acknowledge_on_stream(
        self,
        mapping: _FabricPoolMapping,
        device_id: int,
        consumer_count: int,
        consumer_rank: Optional[int] = None,
    ) -> None:
        if self._consumer_acknowledged:
            return
        if consumer_count == self.total_consumer_count:
            ranks = range(self.total_consumer_count)
        elif consumer_count == 1:
            ranks = (
                _resolve_consumer_rank(
                    self.total_consumer_count, consumer_rank=consumer_rank
                ),
            )
        else:
            raise ValueError(
                "FABRIC acknowledgements support one consumer or the complete "
                f"consumer group, got {consumer_count}/{self.total_consumer_count}"
            )
        for rank in ranks:
            _stream_write_value32(
                device_id,
                mapping.va + self.ack_byte_offset + rank * _CONTROL_WORD_BYTES,
                self.generation,
            )
        self._consumer_acknowledged = True

    def acknowledge_consumption(
        self, consumer_count: int = 1, consumer_rank: Optional[int] = None
    ) -> None:
        """Release a slice after an embedding-cache hit skips the data copy."""
        if self._consumer_acknowledged:
            return
        device_id = torch.cuda.current_device()
        with torch.cuda.device(device_id):
            mapping = self._mapping(device_id)
            self._wait_until_ready(mapping, device_id)
            self._acknowledge_on_stream(
                mapping, device_id, consumer_count, consumer_rank
            )

    def reconstruct_on_target_device(
        self,
        rebuild_device_idx: int,
        consumer_count: int = 1,
        consumer_rank: Optional[int] = None,
    ) -> torch.Tensor:
        rebuild_device = torch.device(f"cuda:{rebuild_device_idx}")
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor

        with torch.cuda.device(rebuild_device_idx):
            mapping = self._mapping(rebuild_device_idx)
            self._wait_until_ready(mapping, rebuild_device_idx)
            source = mapping.byte_tensor[
                self.data_byte_offset : self.data_byte_offset + self.nbytes
            ]
            result = torch.empty(
                self.shape, dtype=self.dtype, device=rebuild_device
            ).contiguous()
            result.view(torch.uint8).reshape(-1).copy_(source, non_blocking=True)
            self._acknowledge_on_stream(
                mapping, rebuild_device_idx, consumer_count, consumer_rank
            )

        self.reconstruct_tensor = result
        return result


@dataclass(frozen=True)
class _PoolChunk:
    start: int
    end: int
    slot: int
    generation: int

    @property
    def size(self) -> int:
        return self.end - self.start


class FabricMmFeatureMemoryPool:
    """Bounded producer allocation shared across an MNNVL fabric domain."""

    def __init__(
        self,
        memory_size: int,
        recycle_interval: float,
        base_gpu_id: int,
        consumer_count: int,
        max_inflight_slices: int = _DEFAULT_MAX_INFLIGHT_SLICES,
    ) -> None:
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if consumer_count <= 0:
            raise ValueError("consumer_count must be positive")

        from sglang.srt.layers.moe.dwdp.vmm import (
            create_fabric_handle,
            get_allocation_granularity,
            map_handle,
            reserve_va,
            set_access,
            shareable_handle_types,
            tensor_from_ptr,
        )

        cuda, check_drv = _driver_modules()
        fabric_type = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        if not (shareable_handle_types(base_gpu_id) & int(fabric_type)):
            raise RuntimeError(
                "CUDA FABRIC handles are unavailable; verify GB200/GB300 MNNVL "
                "and the IMEX channel allocation"
            )

        self.device_id = base_gpu_id
        self.consumer_count = consumer_count
        self.control_words_per_slot = 1 + consumer_count
        self.max_inflight_slices = max_inflight_slices
        self.granularity = get_allocation_granularity(base_gpu_id)
        self.allocation_size = _align_down(memory_size, self.granularity)
        self._allocation_handle = 0
        self._va = 0
        self._mapped = False
        self._byte_tensor = None
        self._control_words = None
        self._closed = False
        raw_control_bytes = (
            max_inflight_slices * self.control_words_per_slot * _CONTROL_WORD_BYTES
        )
        self.data_start = _align_up(raw_control_bytes, _DATA_ALIGNMENT)
        if self.allocation_size <= self.data_start:
            raise ValueError(
                "FABRIC pool is too small after control metadata: "
                f"allocation={self.allocation_size}, control={self.data_start}"
            )

        try:
            with torch.cuda.device(base_gpu_id):
                self._allocation_handle = create_fabric_handle(
                    self.allocation_size, base_gpu_id
                )
                self._va = reserve_va(self.allocation_size, self.granularity)
                map_handle(self._va, self.allocation_size, self._allocation_handle)
                self._mapped = True
                set_access(self._va, self.allocation_size, base_gpu_id)
                self._byte_tensor = tensor_from_ptr(
                    self._va, (self.allocation_size,), torch.uint8, base_gpu_id
                )
                control_word_count = max_inflight_slices * self.control_words_per_slot
                self._control_words = (
                    self._byte_tensor[: control_word_count * _CONTROL_WORD_BYTES]
                    .view(torch.int32)
                    .view(max_inflight_slices, self.control_words_per_slot)
                )
                self._control_words.zero_()
                torch.cuda.synchronize(base_gpu_id)

            exported = check_drv(
                cuda.cuMemExportToShareableHandle(
                    self._allocation_handle, fabric_type, 0
                ),
                "cuMemExportToShareableHandle(mm FABRIC)",
            )
            self.fabric_handle = bytes(exported.data)
            if len(self.fabric_handle) != 64:
                raise RuntimeError(
                    f"Unexpected CUDA FABRIC handle size: {len(self.fabric_handle)}"
                )
        except Exception:
            self._byte_tensor = None
            self._control_words = None
            self._release_vmm_resources()
            self._closed = True
            raise

        self._available_ranges = [(self.data_start, self.allocation_size)]
        self._available_slots = list(reversed(range(max_inflight_slices)))
        self._slot_generations = [0] * max_inflight_slices
        self._occupied: dict[int, _PoolChunk] = {}
        self._lock = threading.Lock()
        self._pool_full_warned = False
        self._recycle_interval = recycle_interval
        self._stop_recycler = False
        self._recycle_thread = threading.Thread(
            target=self._recycle_loop,
            name="FabricMmFeaturePoolRecycler",
            daemon=True,
        )
        self._recycle_thread.start()

    @property
    def usable_size(self) -> int:
        return self.allocation_size - self.data_start

    def _allocate_locked(self, nbytes: int) -> Optional[_PoolChunk]:
        candidates = [
            (end - start, index, start, end)
            for index, (start, end) in enumerate(self._available_ranges)
            if end - start >= nbytes
        ]
        if not candidates or not self._available_slots:
            return None
        _, index, start, end = min(candidates)
        self._available_ranges.pop(index)
        if start + nbytes < end:
            self._available_ranges.append((start + nbytes, end))
        slot = self._available_slots.pop()
        generation = self._slot_generations[slot] + 1
        if generation > 0x7FFFFFFF:
            # int32 control words intentionally avoid the unsigned/signed
            # comparison ambiguity in PyTorch's recycler readback.
            raise RuntimeError("FABRIC pool slot generation exhausted")
        self._slot_generations[slot] = generation
        chunk = _PoolChunk(
            start=start,
            end=start + nbytes,
            slot=slot,
            generation=generation,
        )
        self._occupied[slot] = chunk
        return chunk

    def _release_chunk_locked(self, chunk: _PoolChunk) -> None:
        self._occupied.pop(chunk.slot, None)
        self._available_slots.append(chunk.slot)
        self._available_ranges.append((chunk.start, chunk.end))

    def _merge_ranges_locked(self) -> None:
        merged = []
        for start, end in sorted(self._available_ranges):
            if merged and merged[-1][1] == start:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        self._available_ranges = merged

    def _recycle_ready_chunks_locked(self) -> None:
        if not self._occupied:
            return
        chunks = list(self._occupied.values())
        slot_indices = torch.tensor(
            [chunk.slot for chunk in chunks],
            dtype=torch.long,
            device=f"cuda:{self.device_id}",
        )
        # Every word must belong to the active slot generation. Generation
        # tokens avoid an ABA race when a recycled row still contains the
        # previous lease's acknowledgements on another CUDA stream.
        expected = torch.tensor(
            [chunk.generation for chunk in chunks],
            dtype=torch.int32,
            device=f"cuda:{self.device_id}",
        ).unsqueeze(1)
        completed = (
            (self._control_words.index_select(0, slot_indices) == expected)
            .all(dim=1)
            .cpu()
            .tolist()
        )
        for chunk, is_complete in zip(chunks, completed):
            if is_complete:
                self._release_chunk_locked(chunk)
        self._merge_ranges_locked()

    def _recycle_loop(self) -> None:
        torch.cuda.set_device(self.device_id)
        while not self._stop_recycler:
            try:
                with self._lock, torch.cuda.device(self.device_id):
                    self._recycle_ready_chunks_locked()
            except Exception:
                logger.warning("FABRIC multimodal pool recycle failed", exc_info=True)
            time.sleep(self._recycle_interval)

    def wrap_tensor(self, tensor: torch.Tensor) -> Optional[FabricTensorTransportProxy]:
        if not tensor.is_cuda:
            raise ValueError("FABRIC transport requires a CUDA tensor")
        source = tensor.contiguous()
        nbytes = source.numel() * source.element_size()
        with self._lock:
            chunk = self._allocate_locked(nbytes)
        if chunk is None:
            self._warn_pool_full_once(nbytes)
            return None

        try:
            with torch.cuda.device(self.device_id):
                destination = self._byte_tensor[chunk.start : chunk.end]
                destination.copy_(
                    source.view(torch.uint8).reshape(-1), non_blocking=True
                )
                ready_byte_offset = (
                    chunk.slot * self.control_words_per_slot * _CONTROL_WORD_BYTES
                )
                _stream_write_value32(
                    self.device_id,
                    self._va + ready_byte_offset,
                    chunk.generation,
                )
        except Exception:
            with self._lock:
                self._release_chunk_locked(chunk)
                self._merge_ranges_locked()
            raise

        return FabricTensorTransportProxy(
            fabric_handle=self.fabric_handle,
            allocation_size=self.allocation_size,
            data_byte_offset=chunk.start,
            nbytes=nbytes,
            shape=source.shape,
            dtype=source.dtype,
            ready_byte_offset=ready_byte_offset,
            ack_byte_offset=ready_byte_offset + _CONTROL_WORD_BYTES,
            generation=chunk.generation,
            total_consumer_count=self.consumer_count,
        )

    def _warn_pool_full_once(self, nbytes: int) -> None:
        if self._pool_full_warned:
            return
        self._pool_full_warned = True
        logger.warning(
            "FABRIC multimodal feature pool has no free slice for %.2f MiB "
            "(usable pool %.2f MiB, %d/%d leases active); falling back to CPU "
            "for this tensor. Consider increasing SGLANG_MM_FEATURE_CACHE_MB.",
            nbytes / (1024 * 1024),
            self.usable_size / (1024 * 1024),
            len(self._occupied),
            self.max_inflight_slices,
        )

    def _release_vmm_resources(self) -> None:
        from sglang.srt.layers.moe.dwdp.vmm import (
            free_va,
            release_handle,
            unmap_va,
        )

        with torch.cuda.device(self.device_id):
            try:
                if self._mapped:
                    unmap_va(self._va, self.allocation_size)
                    self._mapped = False
            finally:
                try:
                    if self._va:
                        free_va(self._va, self.allocation_size)
                        self._va = 0
                finally:
                    if self._allocation_handle:
                        release_handle(self._allocation_handle)
                        self._allocation_handle = 0

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_recycler = True
        if (
            getattr(self, "_recycle_thread", None) is not None
            and self._recycle_thread.is_alive()
        ):
            self._recycle_thread.join(timeout=1.0)

        self._control_words = None
        self._byte_tensor = None
        self._release_vmm_resources()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
