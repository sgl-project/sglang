"""Original CUDA host-allocation ABI, admission and cleanup mocks."""

import ctypes
from unittest.mock import patch

import pytest

from sglang.srt.weight_sync.gpu_delta import memory as memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class Driver:
    def __init__(self, failure=None):
        self.calls = []
        self.backing = ctypes.create_string_buffer(8192)
        self.failure = failure

    def cuDeviceGetAttribute(self, result, attribute, device):
        result._obj.value = 1 if attribute == 141 else 7
        return 0

    def cuMemGetAllocationGranularity(self, result, properties, mode):
        value = properties._obj
        assert (value.type, value.requestedHandleTypes, value.location.type) == (
            1,
            0,
            3,
        )
        assert value.location.id == 7 and value.allocFlags.usage == 2
        result._obj.value = 4096
        return 0

    def cuMemCreate(self, handle, size, properties, flags):
        assert size == 8192 and flags == 0
        handle._obj.value = 11
        self.calls.append("create")
        return 0

    def cuMemGetAllocationPropertiesFromHandle(self, properties, handle):
        properties._obj.location.type = 3
        # CUDA may not echo HW_DECOMPRESS; the mapped pointer still must admit it.
        properties._obj.allocFlags.usage = 0
        return 0

    def cuMemAddressReserve(self, address, size, alignment, hint, flags):
        address._obj.value = ctypes.addressof(self.backing)
        self.calls.append("reserve")
        return 0

    def cuMemMap(self, address, size, offset, handle, flags):
        self.calls.append("map")
        return 1 if self.failure == "map" else 0

    def cuMemSetAccess(self, address, size, descriptors, count):
        assert count == 2
        assert [(x.location.type, x.location.id, x.flags) for x in descriptors] == [
            (3, 0, 3),
            (1, 2, 1),
        ]
        self.calls.append("access")
        return 0

    def cuPointerGetAttribute(self, capable, attribute, address):
        assert attribute == 21
        capable._obj.value = self.failure != "capability"
        self.calls.append("capability")
        return 0

    def cuMemUnmap(self, address, size):
        self.calls.append("unmap")
        return 0

    def cuMemAddressFree(self, address, size):
        self.calls.append("free_address")
        return 0

    def cuMemRelease(self, handle):
        self.calls.append("release")
        return 0


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


def test_cuda13_allocation_struct_layout():
    # Exact CUDA13.0.96 public header ABI; no driver or GPU is loaded.
    assert ctypes.sizeof(memory._Location) == 8
    assert ctypes.sizeof(memory._AllocationProperties) == 32
    assert memory._AllocationProperties.allocFlags.offset == 24
    assert ctypes.sizeof(memory._Access) == 12
