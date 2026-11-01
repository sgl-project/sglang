# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature-owned allocations admitted for Blackwell hardware decompression.

Each rank owns its persistent host input arena; allocations are never exported
or imported.
Device buffers use cuMemAlloc directly, without changing PyTorch's allocator.
"""

import ctypes
import logging
import math
import weakref
from functools import cache

logger = logging.getLogger(__name__)
_device_owners = weakref.WeakValueDictionary()


@cache
def _driver():
    from cuda.bindings import driver

    _check(driver.cuInit(0), "cuInit")
    return driver


def _check(result, operation):
    code = result[0]
    if code:
        raise RuntimeError(f"GPU-delta allocation {operation} failed: CUDA {code}")
    return result[1] if len(result) > 1 else None


@cache
def _pointer_attribute_query():
    # cuda-bindings 13.3.1 exports the new enum but its typed result dispatcher
    # rejects it before calling CUDA. Keep the image's bindings and query the
    # same driver API directly in that specific case. CUdeviceptr is uint64;
    # IS_HW_DECOMPRESS_CAPABLE writes an unsigned-int boolean.
    library = ctypes.CDLL("libcuda.so.1")
    query = library.cuPointerGetAttribute
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    query.restype = ctypes.c_int
    return query


def require_de_capable(pointer):
    driver = _driver()
    attribute = driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE
    try:
        result = driver.cuPointerGetAttribute(attribute, pointer)
    except TypeError as error:
        if (
            str(error)
            != "Unsupported attribute: CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE"
        ):
            raise
        capable = ctypes.c_uint(0)
        status = _pointer_attribute_query()(
            ctypes.byref(capable), int(attribute), int(pointer)
        )
        result = status, capable.value
    capable = _check(result, "IS_HW_DECOMPRESS_CAPABLE")
    if not capable:
        raise RuntimeError("GPU-delta pointer is not hardware-decompression capable")


class _DeviceAllocation:
    """Foreign tensor storage owns this object, including through escaped views.

    Never retain the wrapping tensor here or free while a tensor alias survives.
    Finalization drains recorded streams even on partial-construction failures;
    the normal receiver path already joins them before dropping its leases.
    """

    def __init__(self, size, device, stream):
        self.address = 0
        self.release_attempted = False
        self.device = device
        self.driver = driver = _driver()
        self.streams = {stream.cuda_stream: stream}
        self.context = _check(driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
        if not int(self.context):
            raise RuntimeError("GPU-delta device allocation requires a current context")
        if int(_check(driver.cuCtxGetDevice(), "cuCtxGetDevice")) != device.index:
            raise RuntimeError("GPU-delta device allocation context/device mismatch")
        self.size = size
        self.address = int(_check(driver.cuMemAlloc(size), "cuMemAlloc"))
        try:
            require_de_capable(self.address)
        except BaseException:
            self._release()
            raise
        _device_owners[self.address] = self

    @property
    def __cuda_array_interface__(self):
        # Newly allocated bytes have no producer work. PyTorch's CAI importer
        # retains this owner in the storage deleter; stream safety is explicit.
        return {
            "shape": (self.size,),
            "typestr": "|u1",
            "data": (self.address, False),
            "version": 3,
        }

    def _release(self):
        if not self.address or self.release_attempted:
            return
        # Cleanup failure is retained as a leak rather than an uncertain retry.
        self.release_attempted = True
        driver = self.driver
        _check(driver.cuCtxPushCurrent(self.context), "cuCtxPushCurrent")
        try:
            for stream in self.streams.values():
                stream.synchronize()
            _check(driver.cuMemFree(self.address), "cuMemFree")
            self.address = 0
        finally:
            _check(driver.cuCtxPopCurrent(), "cuCtxPopCurrent")

    def __del__(self):
        try:
            self._release()
        except Exception:
            # A failed fence/free must never trigger replay of a committed XOR.
            # Do not force-free a buffer whose stream completion is uncertain.
            logger.exception("GPU-delta device allocation cleanup failed")


def device_empty(shape, dtype, device):
    """Allocate only a DE-facing tensor outside the global Torch allocator."""
    import torch

    device = torch.device(device)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if device.type != "cuda" or device.index is None:
        raise ValueError("GPU-delta device allocation requires an explicit CUDA device")
    if any(not isinstance(size, int) or size < 0 for size in shape):
        raise ValueError("GPU-delta device allocation requires nonnegative dimensions")
    if dtype not in (torch.uint8, torch.int32, torch.int64):
        raise ValueError("Unsupported GPU-delta device allocation dtype")
    size = math.prod(shape) * dtype.itemsize
    if not size:
        # cuMemAlloc(0) is invalid. Empty tensors have no DE-addressable storage.
        return torch.empty(shape, dtype=dtype, device=device)
    with torch.cuda.device(device):
        # Establish the existing PyTorch primary context; create no new context.
        stream = torch.cuda.current_stream(device)
        owner = _DeviceAllocation(size, device, stream)
        value = torch.as_tensor(owner, dtype=torch.uint8, device=device)
        if value.data_ptr() != owner.address or value.device != device:
            raise RuntimeError("GPU-delta CUDA array import must preserve storage")
        return value.view(dtype).reshape(shape)


def record_device_stream(tensor, stream):
    """Record foreign-storage readers; Tensor.record_stream only covers Torch pools."""
    owner = _device_owners.get(tensor.untyped_storage().data_ptr())
    if owner is not None:
        if stream.device != owner.device:
            raise ValueError("GPU-delta allocation consumer stream/device mismatch")
        owner.streams[stream.cuda_stream] = stream


class HostAllocation:
    """One rank's original mapping; close after CPU tasks and GPU readers drain."""

    def __init__(self, capacity, device):
        self.driver = driver = _driver()
        self.handle = self.address = 0
        self.capacity = capacity
        self.mapped = False
        self.view = None
        try:
            supported = _check(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                    device,
                ),
                "HOST_NUMA_VMM_SUPPORTED",
            )
            if not supported:
                raise RuntimeError("GPU-delta requires CUDA HOST_NUMA VMM support")
            numa = _check(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
                ),
                "HOST_NUMA_ID",
            )
            properties = driver.CUmemAllocationProp()
            properties.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            properties.requestedHandleTypes = (
                driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
            )
            properties.location.type = (
                driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            )
            properties.location.id = max(0, numa)
            properties.allocFlags.usage = driver.CU_MEM_CREATE_USAGE_HW_DECOMPRESS
            granularity = _check(
                driver.cuMemGetAllocationGranularity(
                    properties,
                    driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                ),
                "cuMemGetAllocationGranularity",
            )
            self.capacity = (capacity + granularity - 1) // granularity * granularity
            self.handle = _check(
                driver.cuMemCreate(self.capacity, properties, 0), "cuMemCreate"
            )
            properties = _check(
                driver.cuMemGetAllocationPropertiesFromHandle(self.handle),
                "cuMemGetAllocationPropertiesFromHandle",
            )
            # Drivers may omit the requested usage flags from this query. The
            # mapped-pointer capability check below admits actual HW-DE support.
            if (
                properties.location.type
                != driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            ):
                raise RuntimeError("GPU-delta host allocation is not HOST_NUMA")
            self.address = _check(
                driver.cuMemAddressReserve(self.capacity, 0, 0, 0),
                "cuMemAddressReserve",
            )
            _check(
                driver.cuMemMap(self.address, self.capacity, 0, self.handle, 0),
                "cuMemMap",
            )
            self.mapped = True
            host_access = driver.CUmemAccessDesc()
            host_access.location.type = (
                driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            )
            host_access.location.id = 0  # Ignored for host access.
            host_access.flags = (
                driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            )
            device_access = driver.CUmemAccessDesc()
            device_access.location.type = (
                driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            )
            device_access.location.id = device
            device_access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
            _check(
                driver.cuMemSetAccess(
                    self.address, self.capacity, [host_access, device_access], 2
                ),
                "cuMemSetAccess",
            )
            require_de_capable(self.address)
            self.view = memoryview(
                (ctypes.c_ubyte * self.capacity).from_address(int(self.address))
            ).cast("B")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.view is not None:
            self.view.release()
            self.view = None
        if self.mapped:
            _check(self.driver.cuMemUnmap(self.address, self.capacity), "cuMemUnmap")
            self.mapped = False
        if int(self.address):
            _check(
                self.driver.cuMemAddressFree(self.address, self.capacity),
                "cuMemAddressFree",
            )
            self.address = 0
        if int(self.handle):
            _check(self.driver.cuMemRelease(self.handle), "cuMemRelease")
            self.handle = 0
