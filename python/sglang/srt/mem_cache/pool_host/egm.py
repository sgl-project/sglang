"""CUDA HOST_NUMA VMM allocation for an EGM-backed host KV pool."""

from __future__ import annotations

import logging
import math
import socket
from dataclasses import dataclass
from functools import lru_cache

import torch

from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator

logger = logging.getLogger(__name__)


def _check(result, what):
    from cuda.bindings import driver as drv

    if result[0] != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {result[0]}")
    return result[1] if len(result) > 1 else None


@lru_cache(maxsize=None)
def _allocation_properties(device: int):
    """Return the device-local host NUMA allocation policy and granularity."""
    from cuda.bindings import driver as drv

    _check(drv.cuInit(0), "cuInit")
    dev = _check(drv.cuDeviceGet(device), "cuDeviceGet")
    numa = int(
        _check(
            drv.cuDeviceGetAttribute(
                drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev
            ),
            "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID",
        )
    )
    prop = drv.CUmemAllocationProp()
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = numa
    prop.requestedHandleTypes = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    granularity = int(
        _check(
            drv.cuMemGetAllocationGranularity(
                prop,
                drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
    )
    return prop, numa, granularity


class EgmAllocation:
    """Own one host NUMA allocation and its local VMM mapping."""

    def __init__(self, nbytes: int, device: int):
        from cuda.bindings import driver as drv

        if nbytes <= 0:
            raise ValueError("allocation must be nonempty")
        self.address = None
        self.handle = None
        self.mapped = False
        self.size = 0
        self.device = device
        prop, numa, granularity = _allocation_properties(device)
        self.size = ((nbytes + granularity - 1) // granularity) * granularity
        try:
            self.handle = _check(drv.cuMemCreate(self.size, prop, 0), "cuMemCreate")
            self.address = int(
                _check(
                    drv.cuMemAddressReserve(self.size, granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            _check(drv.cuMemMap(self.address, self.size, 0, self.handle, 0), "cuMemMap")
            self.mapped = True
            descs = []
            for kind, ident in (
                (drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, int(numa)),
                (drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device),
            ):
                desc = drv.CUmemAccessDesc()
                desc.location.type, desc.location.id = kind, ident
                desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                descs.append(desc)
            _check(
                drv.cuMemSetAccess(self.address, self.size, descs, len(descs)),
                "cuMemSetAccess",
            )
            logger.info(
                "Allocated EGM host pool: device=%d host_numa=%d bytes=%d",
                device,
                numa,
                self.size,
            )
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"EGM allocation cleanup failed: {cleanup_error}")
            raise

    def close(self):
        from cuda.bindings import driver as drv

        if self.mapped:
            _check(drv.cuMemUnmap(self.address, self.size), "cuMemUnmap")
            self.mapped = False
        if self.address is not None:
            _check(drv.cuMemAddressFree(self.address, self.size), "cuMemAddressFree")
            self.address = None
        if self.handle is not None:
            _check(drv.cuMemRelease(self.handle), "cuMemRelease")
            self.handle = None

    def export(self) -> bytes:
        from cuda.bindings import driver as drv

        if self.handle is None:
            raise RuntimeError("allocation is closed")
        exported = _check(
            drv.cuMemExportToShareableHandle(
                self.handle, drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC, 0
            ),
            "cuMemExportToShareableHandle",
        )
        return bytes(exported.data)


@dataclass
class _RemoteMapping:
    address: int
    size: int
    owned: EgmAllocation
    mapped: bool = True
    reserved: bool = True

    def unmap(self):
        from cuda.bindings import driver as drv

        if self.mapped:
            _check(drv.cuMemUnmap(self.address, self.size), "cuMemUnmap(remote EGM)")
            self.mapped = False
        if self.reserved:
            _check(
                drv.cuMemAddressFree(self.address, self.size),
                "cuMemAddressFree(remote EGM)",
            )
            self.reserved = False


class EgmRemoteTensorAllocator(HostTensorAllocator):
    """Back each rank's L2 with an allocation owned by its TP peer.

    The two ranks exchange one allocation per pool in construction order. Each
    rank alone manages the page indices of its imported pool. The owner keeps
    its physical allocation alive until both ranks have stopped and unmapped.
    """

    @classmethod
    def free_hugetlb_bytes(cls):
        return 0

    @classmethod
    def supports_hugetlb(cls):
        return False

    @classmethod
    def round_mapping_lengths(cls, lengths, device):
        current = torch.cuda.current_device()
        if device is not None:
            target = torch.device(device)
            if (
                target.type == "cuda"
                and target.index is not None
                and target.index != current
            ):
                raise ValueError(
                    f"EGM host pool device {target} differs from current CUDA device {current}"
                )
        _, _, granularity = _allocation_properties(current)
        return [
            ((length + granularity - 1) // granularity) * granularity
            for length in lengths
        ]

    def __init__(self):
        super().__init__()
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        if group.world_size != 2:
            raise ValueError("remote EGM HiCache currently requires TP=2")
        self.group = group.cpu_group
        self.rank = group.rank_in_group
        self.device_id = torch.cuda.current_device()
        self._regions = []
        self._closed = False
        self._preflight_complete = False
        locality = None
        error = None
        try:
            _, numa, _ = _allocation_properties(self.device_id)
            locality = (socket.gethostname(), numa)
        except Exception as exc:
            error = repr(exc)
        localities = self._exchange((locality, error))
        if any(item[1] is not None for item in localities):
            raise RuntimeError(f"remote EGM capability check failed: {localities}")
        if localities[0][0] == localities[1][0]:
            raise ValueError("remote EGM requires distinct hosts or host NUMA nodes")

    def _exchange(self, value):
        import torch.distributed as dist

        values = [None, None]
        dist.all_gather_object(values, value, group=self.group)
        return values

    def preflight(self, lengths, device, *, supported=True):
        """Make allocation budget and device failures visible to both ranks."""
        from sglang.srt.mem_cache.pool_host.base import host_memory_preflight

        result = None
        error = None
        try:
            if not supported:
                raise ValueError(
                    "remote EGM currently supports full-attention MHA host pools"
                )
            result = host_memory_preflight(lengths, self, device)
            if result.allocation_bytes > result.available_bytes:
                raise ValueError(
                    f"need {result.allocation_bytes} host bytes, have {result.available_bytes}"
                )
        except Exception as exc:
            error = repr(exc)
        errors = self._exchange(error)
        if any(item is not None for item in errors):
            raise RuntimeError(f"remote EGM preflight failed: {errors}")
        self._preflight_complete = True
        return result

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        from cuda.bindings import driver as drv

        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        if self._closed:
            raise RuntimeError("remote EGM allocator is closed")
        # The first supported pool is full-attention MHA, layer_first:
        # (K/V, layer, token, head, dimension).
        metadata = (tuple(dims), str(dtype), device, self._preflight_complete)
        metadata_all = self._exchange(metadata)
        if (
            metadata_all[0] != metadata_all[1]
            or device != "cpu"
            or len(dims) != 5
            or dims[0] != 2
            or not self._preflight_complete
        ):
            raise ValueError(
                f"remote EGM requires matching layer_first MHA pools: {metadata_all}"
            )
        nbytes = math.prod(dims) * dtype.itemsize
        owned = None
        error = None
        try:
            owned = EgmAllocation(nbytes, self.device_id)
            descriptor = (owned.export(), owned.size)
        except Exception as exc:
            error = repr(exc)
            descriptor = None
        descriptors = self._exchange((descriptor, error))
        if any(item[1] is not None for item in descriptors):
            if owned is not None:
                owned.close()
            raise RuntimeError(
                f"remote EGM allocation failed: {[x[1] for x in descriptors]}"
            )

        handle = None
        va = None
        mapped = False
        error = None
        tensor = None
        peer_size = 0
        try:
            peer_handle, peer_size = descriptors[1 - self.rank][0]
            if not isinstance(peer_handle, bytes) or len(peer_handle) != 64:
                raise ValueError("invalid remote EGM FABRIC handle")
            if not isinstance(peer_size, int) or peer_size < nbytes:
                raise ValueError("remote EGM allocation is smaller than the pool")
            handle = _check(
                drv.cuMemImportFromShareableHandle(
                    peer_handle, drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
                ),
                "cuMemImportFromShareableHandle",
            )
            va = int(
                _check(
                    drv.cuMemAddressReserve(peer_size, 0, 0, 0), "cuMemAddressReserve"
                )
            )
            _check(drv.cuMemMap(va, peer_size, 0, handle, 0), "cuMemMap")
            mapped = True
            access = drv.CUmemAccessDesc()
            access.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = self.device_id
            access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _check(drv.cuMemSetAccess(va, peer_size, [access], 1), "cuMemSetAccess")
            _check(drv.cuMemRelease(handle), "cuMemRelease(import)")
            handle = None
            tensor = tensor_from_pointer(
                va, nbytes, shape=dims, dtype=dtype, device_id=self.device_id
            )
        except Exception as exc:
            error = repr(exc)
        errors = self._exchange(error)
        if any(item is not None for item in errors):
            cleanup_error = None
            try:
                if mapped:
                    _check(drv.cuMemUnmap(va, peer_size), "cuMemUnmap(rollback)")
                if va is not None:
                    _check(
                        drv.cuMemAddressFree(va, peer_size),
                        "cuMemAddressFree(rollback)",
                    )
                if handle is not None:
                    _check(drv.cuMemRelease(handle), "cuMemRelease(rollback)")
            except Exception as exc:
                cleanup_error = repr(exc)
            cleanup_errors = self._exchange(cleanup_error)
            if not any(item is not None for item in cleanup_errors):
                owned.close()
            raise RuntimeError(
                f"remote EGM import failed: {errors}; rollback: {cleanup_errors}"
            )
        self._regions.append(_RemoteMapping(va, peer_size, owned))
        logger.info("Mapped remote EGM host pool from TP rank %d", 1 - self.rank)
        return tensor

    def close(self):
        if self._closed:
            return
        error = None
        try:
            torch.cuda.synchronize(self.device_id)
        except Exception as exc:
            error = repr(exc)
        errors = self._exchange(error)
        if any(item is not None for item in errors):
            raise RuntimeError(f"remote EGM drain failed: {errors}")
        error = None
        try:
            for region in self._regions:
                region.unmap()
        except Exception as exc:
            error = repr(exc)
        errors = self._exchange(error)
        if any(item is not None for item in errors):
            raise RuntimeError(f"remote EGM unmap failed: {errors}")
        error = None
        try:
            for region in self._regions:
                region.owned.close()
        except Exception as exc:
            error = repr(exc)
        errors = self._exchange(error)
        if any(item is not None for item in errors):
            raise RuntimeError(f"remote EGM owner release failed: {errors}")
        self._regions.clear()
        self._closed = True
        logger.info("Closed remote EGM host pool on TP rank %d", self.rank)
