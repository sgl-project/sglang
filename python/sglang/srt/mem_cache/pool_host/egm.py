"""CUDA HOST_NUMA VMM allocation for an EGM-backed host KV pool."""

from __future__ import annotations

import ctypes
import logging
import math
import weakref
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
    """Own one host-accessible FABRIC allocation and its local VMM mapping.

    The caller must ensure all local and imported accesses have stopped before
    close. Keeping a mapping alive does not protect individual KV pages from
    being reused by their allocator.
    """

    def __init__(self, nbytes: int, device: int):
        from cuda.bindings import driver as drv

        if nbytes <= 0:
            raise ValueError("allocation must be nonempty")
        self.nbytes = nbytes
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

    def export(self) -> bytes:
        from cuda.bindings import driver as drv

        if self.handle is None:
            raise RuntimeError("allocation is closed")
        handle = _check(
            drv.cuMemExportToShareableHandle(
                self.handle, drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC, 0
            ),
            "cuMemExportToShareableHandle",
        )
        return bytes(handle.data)

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


class EgmHostTensorAllocator(HostTensorAllocator):
    """Allocate CPU tensors backed by exportable host NUMA memory."""

    def __init__(self):
        super().__init__()
        self.allocations = {}

    @classmethod
    def free_hugetlb_bytes(cls):
        return 0

    @classmethod
    def supports_hugetlb(cls):
        return False

    @classmethod
    def round_mapping_lengths(cls, lengths):
        _, _, granularity = _allocation_properties(torch.cuda.current_device())
        return [
            ((length + granularity - 1) // granularity) * granularity
            for length in lengths
        ]

    def export_buffer(self, tensor: torch.Tensor) -> tuple[bytes, int, int]:
        """Return FABRIC handle, allocation size and buffer offset in bytes.

        The host pool must retain the allocation and pin the relevant pages
        until all readers have finished and unmapped their imports.
        """
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("EGM export requires a contiguous host tensor")
        address = tensor.data_ptr()
        length = tensor.numel() * tensor.element_size()
        for base, alloc in self.allocations.items():
            if base <= address and address + length <= base + alloc.nbytes:
                return alloc.export(), alloc.size, address - base
        raise ValueError("tensor does not belong to this EGM allocator")

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        if device != "cpu":
            raise ValueError("EGM host tensors require device=cpu")
        self.dtype = dtype
        self.dims = dims
        nbytes = math.prod(dims) * dtype.itemsize
        alloc = EgmAllocation(nbytes, torch.cuda.current_device())
        try:
            array = (ctypes.c_uint8 * nbytes).from_address(alloc.address)
            tensor = torch.frombuffer(array, dtype=dtype).reshape(dims)
        except BaseException:
            alloc.close()
            raise
        self.allocations[alloc.address] = alloc

        def release():
            # Transfers using a raw pointer may outlive the last Python view.
            # Keep the mapping if synchronization fails.
            torch.cuda.synchronize(alloc.device)
            address = alloc.address
            alloc.close()
            self.allocations.pop(address, None)

        weakref.finalize(array, release).atexit = False
        return tensor
