"""Typed CUDA host-allocation admission and cleanup mocks."""

import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.weight_sync.gpu_delta import memory as memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class Driver:
    CUdevice_attribute = SimpleNamespace(
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=141,
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID=134,
    )
    CUpointer_attribute = SimpleNamespace(
        CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE=21
    )
    CUmemAllocationType = SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1)
    CUmemAllocationHandleType = SimpleNamespace(CU_MEM_HANDLE_TYPE_NONE=0)
    CUmemLocationType = SimpleNamespace(
        CU_MEM_LOCATION_TYPE_HOST_NUMA=3, CU_MEM_LOCATION_TYPE_DEVICE=1
    )
    CUmemAllocationGranularity_flags = SimpleNamespace(
        CU_MEM_ALLOC_GRANULARITY_MINIMUM=0
    )
    CUmemAccess_flags = SimpleNamespace(
        CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3, CU_MEM_ACCESS_FLAGS_PROT_READ=1
    )
    CU_MEM_CREATE_USAGE_HW_DECOMPRESS = 2

    def __init__(self, failure=None):
        self.calls = []
        self.backing = ctypes.create_string_buffer(8192)
        self.failure = failure

    @staticmethod
    def CUmemAllocationProp():
        return SimpleNamespace(location=SimpleNamespace(), allocFlags=SimpleNamespace())

    @staticmethod
    def CUmemAccessDesc():
        return SimpleNamespace(location=SimpleNamespace())

    def cuDeviceGetAttribute(self, attribute, device):
        assert attribute in (141, 134) and device == 2
        return 0, 1 if attribute == 141 else 7

    def cuMemGetAllocationGranularity(self, properties, mode):
        assert mode == 0
        assert (
            properties.type,
            properties.requestedHandleTypes,
            properties.location.type,
        ) == (
            1,
            0,
            3,
        )
        assert properties.location.id == 7 and properties.allocFlags.usage == 2
        return 0, 4096

    def cuMemCreate(self, size, properties, flags):
        assert size == 8192 and flags == 0
        self.calls.append("create")
        return 0, 11

    def cuMemGetAllocationPropertiesFromHandle(self, handle):
        assert handle == 11
        properties = self.CUmemAllocationProp()
        properties.location.type = 3
        # CUDA may not echo HW_DECOMPRESS; the mapped pointer still must admit it.
        properties.allocFlags.usage = 0
        return 0, properties

    def cuMemAddressReserve(self, size, alignment, hint, flags):
        assert (size, alignment, hint, flags) == (8192, 0, 0, 0)
        self.calls.append("reserve")
        return 0, ctypes.addressof(self.backing)

    def cuMemMap(self, address, size, offset, handle, flags):
        self.calls.append("map")
        return (1 if self.failure == "map" else 0,)

    def cuMemSetAccess(self, address, size, descriptors, count):
        assert count == 2
        assert [(x.location.type, x.location.id, x.flags) for x in descriptors] == [
            (3, 0, 3),
            (1, 2, 1),
        ]
        self.calls.append("access")
        return (0,)

    def cuPointerGetAttribute(self, attribute, address):
        assert attribute == 21
        self.calls.append("capability")
        return 0, self.failure != "capability"

    def cuMemUnmap(self, address, size):
        self.calls.append("unmap")
        return (0,)

    def cuMemAddressFree(self, address, size):
        self.calls.append("free_address")
        return (0,)

    def cuMemRelease(self, handle):
        self.calls.append("release")
        return (0,)


@pytest.mark.parametrize("failure", [None, "map", "capability"])
def test_original_host_vmm_admission_and_cleanup(failure):
    driver = Driver(failure)
    with patch.object(memory, "_driver", return_value=driver):
        if failure:
            error = (
                "cuMemMap" if failure == "map" else "not hardware-decompression capable"
            )
            with pytest.raises(RuntimeError, match=error):
                memory.HostAllocation(4097, 2)
        else:
            allocation = memory.HostAllocation(4097, 2)
            assert allocation.capacity == 8192
            allocation.view[:4] = b"test"
            assert bytes(driver.backing[:4]) == b"test"
            allocation.close()
            allocation.close()
    expected = ["create", "reserve", "map"]
    if failure != "map":
        expected += ["access", "capability", "unmap"]
    assert driver.calls == expected + ["free_address", "release"]


def test_cuda_driver_initialization_is_cached_and_checks_status(monkeypatch):
    calls = []
    driver = SimpleNamespace(cuInit=lambda flags: (calls.append(flags) or 0,))
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(driver=driver))
    memory._driver.cache_clear()
    try:
        assert memory._driver() is driver
        assert memory._driver() is driver
        assert calls == [0]
        memory._driver.cache_clear()
        driver.cuInit = lambda flags: (1,)
        with pytest.raises(RuntimeError, match="cuInit failed"):
            memory._driver()
        assert memory._driver.cache_info().currsize == 0
    finally:
        memory._driver.cache_clear()
