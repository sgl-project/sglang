# SPDX-License-Identifier: Apache-2.0
"""Host VMM descriptor, resource ownership, and failure cleanup contracts."""

import ctypes
import os
import socket
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.weight_cache.host_vmm import HostVmmAllocation
from sglang.srt.weight_cache.model_state import host_table_backend
from sglang.srt.weight_cache.transport import TorchIpcTransportBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeDriver:
    CUmemAllocationType = NS(CU_MEM_ALLOCATION_TYPE_PINNED=1)
    CUmemAllocationHandleType = NS(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR=1)
    CUmemLocationType = NS(
        CU_MEM_LOCATION_TYPE_HOST_NUMA=3, CU_MEM_LOCATION_TYPE_DEVICE=1
    )
    CUmemAllocationGranularity_flags = NS(CU_MEM_ALLOC_GRANULARITY_MINIMUM=0)
    CUmemAccess_flags = NS(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3)
    CUdevice_attribute = NS(
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED=2,
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID=3,
    )

    def __init__(self, failure=None, supported=True, numa_id=0):
        self.failure, self.supported, self.numa_id = failure, supported, numa_id
        self.calls = []
        self.buffer = (ctypes.c_uint8 * 32)()

    def CUmemAllocationProp(self):
        return NS(type=1, requestedHandleTypes=1, location=NS(type=3, id=self.numa_id))

    def CUmemAccessDesc(self):
        return NS(location=NS(type=None, id=None), flags=None)

    def cuDeviceGetAttribute(self, attribute, device):
        return (0, self.numa_id if attribute == 3 else int(self.supported))

    def cuMemGetAllocationGranularity(self, prop, flags):
        return (0, 16)

    def cuMemCreate(self, size, prop, flags):
        self.calls.append(("create", size))
        return (0, 101)

    def cuMemExportToShareableHandle(self, handle, kind, flags):
        return (0, os.memfd_create("vmm-unit", 0))

    def cuMemImportFromShareableHandle(self, fd, kind):
        self.calls.append(("import", fd))
        return (0, 101)

    def cuMemGetAllocationPropertiesFromHandle(self, handle):
        return (0, self.CUmemAllocationProp())

    def cuMemAddressReserve(self, size, *args):
        self.calls.append(("reserve", size))
        return (0, ctypes.addressof(self.buffer))

    def cuMemMap(self, *args):
        self.calls.append(("map",))
        return (1,) if self.failure == "map" else (0,)

    def cuMemSetAccess(self, *args):
        self.calls.append(("access",))
        return (1,) if self.failure == "access" else (0,)

    def cuMemUnmap(self, *args):
        self.calls.append(("unmap",))
        return (0,)

    def cuMemAddressFree(self, *args):
        self.calls.append(("free_address",))
        return (0,)

    def cuMemRelease(self, *args):
        self.calls.append(("release",))
        return (0,)


class TestHostVmm(CustomTestCase):
    def test_backend_config(self):
        self.assertEqual(host_table_backend(None), "memfd")
        self.assertEqual(
            host_table_backend('{"dsv41_host_table_backend":"cuda_vmm"}'), "cuda_vmm"
        )
        with self.assertRaises(ValueError):
            host_table_backend({"dsv41_host_table_backend": "invalid"})

    def test_create_rounds_storage_and_close_is_idempotent(self):
        driver = FakeDriver()
        with patch(
            "sglang.srt.weight_cache.host_vmm._cuda_driver", return_value=(driver, 0)
        ):
            allocation = HostVmmAllocation.create(17)
        self.assertEqual(allocation.allocation_nbytes, 32)
        allocation.bytes.fill_(7)
        self.assertEqual(driver.buffer[16], 7)
        fd = allocation.fd
        self.assertFalse(os.get_inheritable(fd))
        allocation.close()
        calls = driver.calls[:]
        allocation.close()
        self.assertEqual(driver.calls, calls)
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_gpu_view_import_skips_cpu_access_permissions(self):
        driver = FakeDriver()
        fd = os.memfd_create("vmm-unit", 0)
        import torch

        tensor = torch.frombuffer(driver.buffer, dtype=torch.uint8)[:17]
        with (
            patch(
                "sglang.srt.weight_cache.host_vmm._cuda_driver",
                return_value=(driver, 0),
            ),
            patch(
                "sglang.srt.weight_cache.host_vmm.torch.as_tensor", return_value=tensor
            ) as view,
            patch.object(
                driver, "cuMemSetAccess", wraps=driver.cuMemSetAccess
            ) as access,
        ):
            allocation = HostVmmAllocation.from_fd(fd, 17, 32, 0, gpu_view=True)
        descriptors = access.call_args.args[2]
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0].location.type, 1)
        self.assertEqual(view.call_args.kwargs["device"], "cuda:0")
        self.assertEqual(allocation.bytes.data_ptr(), allocation.device_ptr)
        allocation.close()

    def test_import_failure_unwinds_only_acquired_resources(self):
        for failure, unmap in (("map", False), ("access", True)):
            with self.subTest(failure=failure):
                driver = FakeDriver(failure=failure)
                fd = os.memfd_create("vmm-unit", 0)
                with patch(
                    "sglang.srt.weight_cache.host_vmm._cuda_driver",
                    return_value=(driver, 0),
                ):
                    with self.assertRaises(RuntimeError):
                        HostVmmAllocation.from_fd(fd, 17, 32, 0)
                self.assertEqual(("unmap",) in driver.calls, unmap)
                self.assertIn(("free_address",), driver.calls)
                self.assertIn(("release",), driver.calls)
                with self.assertRaises(OSError):
                    os.fstat(fd)

    def test_wrong_numa_and_padding_rejected_before_mapping(self):
        for actual_numa, size in ((1, 32), (0, 48)):
            driver = FakeDriver(numa_id=actual_numa)
            fd = os.memfd_create("vmm-unit", 0)
            with patch(
                "sglang.srt.weight_cache.host_vmm._cuda_driver",
                return_value=(driver, 0),
            ):
                with self.assertRaises(ValueError):
                    HostVmmAllocation.from_fd(fd, 17, size, 0)
            self.assertNotIn(("map",), driver.calls)
            self.assertIn(("release",), driver.calls)
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_unsupported_device_does_not_allocate_or_import(self):
        driver = FakeDriver(supported=False)
        fd = os.memfd_create("vmm-unit", 0)
        with patch(
            "sglang.srt.weight_cache.host_vmm._cuda_driver", return_value=(driver, 0)
        ):
            with self.assertRaises(RuntimeError):
                HostVmmAllocation.from_fd(fd, 17, 32, 0)
        self.assertEqual(driver.calls, [])
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_invalid_descriptor_closes_received_fd(self):
        fd = os.memfd_create("vmm-unit", 0)
        with self.assertRaises(ValueError):
            HostVmmAllocation.from_fd(fd, -1, 32, 0)
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_transport_uses_cuda_handle_metadata_not_memfd_size(self):
        import torch

        from sglang.srt.layers.engram import _HostTable
        from sglang.srt.weight_cache.protocol import recv_msg

        fd = os.memfd_create("not-a-regular-host-table", 0)
        desc = {
            "index": 0,
            "module": "layer",
            "layout": "per_rank",
            "nbytes": 17,
            "backend": "cuda_vmm",
            "allocation_nbytes": 32,
            "numa_id": 0,
        }
        entry = {
            "host_table": desc,
            "host_offset": 0,
            "shape": [17],
            "dtype": "uint8",
            "is_param": False,
        }
        sender = TorchIpcTransportBackend()
        sender.configure_host_table_exports({"layer.weight": entry}, [fd])
        receiver = TorchIpcTransportBackend()
        a, b = socket.socketpair()
        try:
            sender.send_fetch_state_response(
                a, config={}, entries={"layer.weight": entry}, pid=os.getpid()
            )
            receiver.recv_fetch_state_response(b, recv_msg(b))
            table = NS(bytes=torch.arange(17, dtype=torch.uint8))

            def adopt(layout, size, received_fd, padded, numa):
                os.close(received_fd)
                return table

            with patch.object(_HostTable, "from_vmm_fd", side_effect=adopt) as imported:
                result = receiver.import_tensor(entry)
            self.assertEqual(result.tolist(), list(range(17)))
            self.assertEqual(imported.call_count, 1)
        finally:
            a.close()
            b.close()
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
