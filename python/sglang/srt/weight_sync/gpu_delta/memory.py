# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Original HOST_NUMA allocations admitted for Blackwell hardware decompression.

Each rank owns its persistent host input arena; allocations are never exported
or imported.
This module allocates no device memory and submits no CUDA stream work.
"""

import ctypes
from functools import cache


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocationProperties(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocationFlags),
    ]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


@cache
def _driver():
    library = ctypes.CDLL("libcuda.so.1")
    pointer, size, address = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetAttribute": [pointer, ctypes.c_int, ctypes.c_int],
        "cuMemGetAllocationGranularity": [pointer, pointer, ctypes.c_int],
        "cuMemCreate": [pointer, size, pointer, ctypes.c_uint64],
        "cuMemGetAllocationPropertiesFromHandle": [pointer, address],
        "cuMemAddressReserve": [pointer, size, size, address, address],
        "cuMemMap": [address, size, size, address, address],
        "cuMemSetAccess": [address, size, pointer, size],
        "cuPointerGetAttribute": [pointer, ctypes.c_int, address],
        "cuMemUnmap": [address, size],
        "cuMemAddressFree": [address, size],
        "cuMemRelease": [address],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
    _check(library.cuInit(0), "cuInit")
    return library


def _check(code, operation):
    if code:
        raise RuntimeError(f"GPU-delta host allocation {operation} failed: CUDA {code}")


def require_de_capable(pointer):
    capable = ctypes.c_uint()
    _check(
        _driver().cuPointerGetAttribute(ctypes.byref(capable), 21, pointer),
        "IS_HW_DECOMPRESS_CAPABLE",
    )
    if not capable.value:
        raise RuntimeError("GPU-delta pointer is not hardware-decompression capable")


class HostAllocation:
    """One rank's original mapping; close after CPU tasks and GPU readers drain."""

    def __init__(self, capacity, device):
        self.driver = _driver()
        self.handle, self.address = ctypes.c_uint64(), ctypes.c_uint64()
        self.capacity = capacity
        self.mapped = False
        self.view = None
        try:
            supported = ctypes.c_int()
            _check(
                self.driver.cuDeviceGetAttribute(ctypes.byref(supported), 141, device),
                "HOST_NUMA_VMM_SUPPORTED",
            )
            if not supported.value:
                raise RuntimeError("GPU-delta requires CUDA HOST_NUMA VMM support")
            numa = ctypes.c_int()
            _check(
                self.driver.cuDeviceGetAttribute(ctypes.byref(numa), 134, device),
                "HOST_NUMA_ID",
            )
            properties = _AllocationProperties()
            properties.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
            properties.requestedHandleTypes = 0  # Not shareable.
            properties.location = _Location(3, max(0, numa.value))  # HOST_NUMA
            properties.allocFlags.usage = 2  # HW_DECOMPRESS
            granularity = ctypes.c_size_t()
            _check(
                self.driver.cuMemGetAllocationGranularity(
                    ctypes.byref(granularity), ctypes.byref(properties), 0
                ),
                "cuMemGetAllocationGranularity",
            )
            self.capacity = (
                (capacity + granularity.value - 1)
                // granularity.value
                * granularity.value
            )
            _check(
                self.driver.cuMemCreate(
                    ctypes.byref(self.handle),
                    self.capacity,
                    ctypes.byref(properties),
                    0,
                ),
                "cuMemCreate",
            )
            properties = _AllocationProperties()
            _check(
                self.driver.cuMemGetAllocationPropertiesFromHandle(
                    ctypes.byref(properties), self.handle.value
                ),
                "cuMemGetAllocationPropertiesFromHandle",
            )
            # Drivers may omit the requested usage flags from this query. The
            # mapped-pointer capability check below admits actual HW-DE support.
            if properties.location.type != 3:
                raise RuntimeError("GPU-delta host allocation is not HOST_NUMA")
            _check(
                self.driver.cuMemAddressReserve(
                    ctypes.byref(self.address), self.capacity, 0, 0, 0
                ),
                "cuMemAddressReserve",
            )
            _check(
                self.driver.cuMemMap(
                    self.address.value, self.capacity, 0, self.handle.value, 0
                ),
                "cuMemMap",
            )
            self.mapped = True
            access = (_Access * 2)(
                _Access(_Location(3, 0), 3),  # CPU read/write; NUMA id ignored.
                _Access(_Location(1, device), 1),  # This GPU reads compressed bytes.
            )
            _check(
                self.driver.cuMemSetAccess(
                    self.address.value, self.capacity, access, 2
                ),
                "cuMemSetAccess",
            )
            require_de_capable(self.address.value)
            self.view = memoryview(
                (ctypes.c_ubyte * self.capacity).from_address(self.address.value)
            ).cast("B")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.view is not None:
            self.view.release()
            self.view = None
        if self.mapped:
            _check(
                self.driver.cuMemUnmap(self.address.value, self.capacity), "cuMemUnmap"
            )
            self.mapped = False
        if self.address.value:
            _check(
                self.driver.cuMemAddressFree(self.address.value, self.capacity),
                "cuMemAddressFree",
            )
            self.address.value = 0
        if self.handle.value:
            _check(self.driver.cuMemRelease(self.handle.value), "cuMemRelease")
            self.handle.value = 0
