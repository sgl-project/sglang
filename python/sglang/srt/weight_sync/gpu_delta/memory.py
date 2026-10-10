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


@cache
def _driver():
    from cuda.bindings import driver

    _check(driver.cuInit(0), "cuInit")
    return driver


def _check(result, operation):
    code = result[0]
    if code:
        raise RuntimeError(f"GPU-delta host allocation {operation} failed: CUDA {code}")
    return result[1] if len(result) > 1 else None


def require_de_capable(pointer):
    driver = _driver()
    capable = _check(
        driver.cuPointerGetAttribute(
            driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE,
            pointer,
        ),
        "IS_HW_DECOMPRESS_CAPABLE",
    )
    if not capable:
        raise RuntimeError("GPU-delta pointer is not hardware-decompression capable")


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
