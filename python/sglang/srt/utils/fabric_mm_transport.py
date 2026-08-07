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
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.utils.mm_gpu_memory_pool import (
    DEFAULT_MAX_INFLIGHT_SLICES,
    StreamOrderedMmFeaturePool,
    StreamOrderedPoolConsumerMixin,
)

logger = logging.getLogger(__name__)


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


class FabricTensorTransportProxy(StreamOrderedPoolConsumerMixin):
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
        self._init_stream_ordered_consumer(
            ready_byte_offset=ready_byte_offset,
            ack_byte_offset=ack_byte_offset,
            generation=generation,
            total_consumer_count=total_consumer_count,
            transport_name="FABRIC",
        )
        self.reconstruct_tensor: Optional[torch.Tensor] = None

    def _mapping(self, device_id: int) -> _FabricPoolMapping:
        return _get_or_open_mapping(self.fabric_handle, self.allocation_size, device_id)

    def acknowledge_consumption(
        self, consumer_count: int = 1, consumer_rank: Optional[int] = None
    ) -> None:
        """Release a slice after an embedding-cache hit skips the data copy."""
        if self._consumer_acknowledged:
            return
        device_id = torch.cuda.current_device()
        with torch.cuda.device(device_id):
            mapping = self._mapping(device_id)
            self._wait_until_ready(mapping.va, device_id)
            self._acknowledge_on_stream(
                mapping.va, device_id, consumer_count, consumer_rank
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
            self._wait_until_ready(mapping.va, rebuild_device_idx)
            source = mapping.byte_tensor[
                self.data_byte_offset : self.data_byte_offset + self.nbytes
            ]
            result = torch.empty(
                self.shape, dtype=self.dtype, device=rebuild_device
            ).contiguous()
            result.view(torch.uint8).reshape(-1).copy_(source, non_blocking=True)
            self._acknowledge_on_stream(
                mapping.va, rebuild_device_idx, consumer_count, consumer_rank
            )

        self.reconstruct_tensor = result
        return result


class FabricMmFeatureMemoryPool:
    """Bounded producer allocation shared across an MNNVL fabric domain."""

    def __init__(
        self,
        memory_size: int,
        recycle_interval: float,
        base_gpu_id: int,
        consumer_count: int,
        max_inflight_slices: int = DEFAULT_MAX_INFLIGHT_SLICES,
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
        self.max_inflight_slices = max_inflight_slices
        self.granularity = get_allocation_granularity(base_gpu_id)
        self.allocation_size = _align_down(memory_size, self.granularity)
        self._allocation_handle = 0
        self._va = 0
        self._mapped = False
        self._byte_tensor = None
        self._pool = None
        self._closed = False

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
            self._pool = StreamOrderedMmFeaturePool(
                memory_size=self.allocation_size,
                byte_tensor=self._byte_tensor,
                base_address=self._va,
                device_id=base_gpu_id,
                consumer_count=consumer_count,
                recycle_interval=recycle_interval,
                transport_name="FABRIC",
                max_inflight_slices=max_inflight_slices,
            )
        except Exception:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
            self._byte_tensor = None
            self._release_vmm_resources()
            self._closed = True
            raise

        self._pool_full_warned = False

    @property
    def usable_size(self) -> int:
        return self._pool.usable_size

    @property
    def active_lease_count(self) -> int:
        return self._pool.active_lease_count

    def wrap_tensor(self, tensor: torch.Tensor) -> Optional[FabricTensorTransportProxy]:
        nbytes = tensor.numel() * tensor.element_size()
        lease, _ = self._pool.copy_tensor(tensor)
        if lease is None:
            self._warn_pool_full_once(nbytes)
            return None

        return FabricTensorTransportProxy(
            fabric_handle=self.fabric_handle,
            allocation_size=self.allocation_size,
            data_byte_offset=lease.start,
            nbytes=nbytes,
            shape=tensor.shape,
            dtype=tensor.dtype,
            ready_byte_offset=lease.ready_byte_offset,
            ack_byte_offset=lease.ack_byte_offset,
            generation=lease.generation,
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
            self._pool.active_lease_count,
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
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self._byte_tensor = None
        self._release_vmm_resources()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
