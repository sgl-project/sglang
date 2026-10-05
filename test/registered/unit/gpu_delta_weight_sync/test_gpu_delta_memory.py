"""CUDA host-allocation ABI/lifetime mocks and real Unix FD transfer."""

import ctypes
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from sglang.srt.weight_sync import gpu_delta_memory as memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class Driver:
    def __init__(self, capable=True):
        self.calls = []
        self.backing = ctypes.create_string_buffer(8192)
        self.capable = capable
        self.fd = None

    def cuDeviceGetAttribute(self, result, attribute, device):
        result._obj.value = 1 if attribute == 141 else 7
        return 0

    def cuMemGetAllocationGranularity(self, result, properties, mode):
        value = properties._obj
        assert (value.type, value.requestedHandleTypes, value.location.type) == (
            1,
            1,
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

    def cuMemExportToShareableHandle(self, result, handle, kind, flags):
        assert (handle, kind, flags) == (11, 1, 0)
        self.fd = os.open(os.devnull, os.O_RDONLY)
        result._obj.value = self.fd
        return 0

    def cuMemImportFromShareableHandle(self, handle, fd, kind):
        assert isinstance(fd, ctypes.c_void_p) and kind == 1
        os.fstat(fd.value)  # Import takes the received FD value, not int*.
        handle._obj.value = 12
        self.calls.append("import")
        return 0

    def cuMemGetAllocationPropertiesFromHandle(self, properties, handle):
        properties._obj.location.type = 3
        properties._obj.allocFlags.usage = 2
        return 0

    def cuMemAddressReserve(self, address, size, alignment, hint, flags):
        address._obj.value = ctypes.addressof(self.backing)
        self.calls.append("reserve")
        return 0

    def cuMemMap(self, address, size, offset, handle, flags):
        self.calls.append("map")
        return 0

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
        capable._obj.value = self.capable
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


@pytest.mark.parametrize("capable", [True, False])
def test_host_vmm_properties_fd_import_and_cleanup(capable):
    driver = Driver(capable)
    with tempfile.TemporaryDirectory() as directory:
        # Filesystem sockets allow the CPU oracle to run on macOS too. Native
        # Linux uses an abstract address and validates the connecting peer UID.
        with (
            patch.object(memory, "_driver", return_value=driver),
            patch.object(
                memory,
                "_socket_address",
                side_effect=lambda name: str(Path(directory) / name[-32:]),
            ),
            patch.object(memory, "_same_user", return_value=True),
        ):
            if not capable:
                with pytest.raises(
                    RuntimeError, match="not hardware-decompression capable"
                ):
                    memory.SharedHostAllocation(4097, 2)
            else:
                owner = memory.SharedHostAllocation(4097, 2)
                assert owner.capacity == 8192
                owner.view[:4] = b"test"
                imported = memory.SharedHostAllocation(
                    owner.capacity, 2, owner.shareable
                )
                assert bytes(imported.view[:4]) == b"test"
                assert driver.calls.count("create") == 1
                assert driver.calls.count("import") == 1
                imported.close()
                assert bytes(owner.view[:4]) == b"test"
                owner.close()
                owner.close()
    assert driver.calls[-3:] == ["unmap", "free_address", "release"]
    with pytest.raises(OSError):
        os.fstat(driver.fd)


def test_cuda13_allocation_struct_layout():
    # Exact CUDA13.0.96 public header ABI; no driver or GPU is loaded.
    assert ctypes.sizeof(memory._Location) == 8
    assert ctypes.sizeof(memory._AllocationProperties) == 32
    assert memory._AllocationProperties.allocFlags.offset == 24
    assert ctypes.sizeof(memory._Access) == 12
