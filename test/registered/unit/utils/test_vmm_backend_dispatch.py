"""get_vmm_backend dispatches on the platform's get_vmm_backend_class, with no driver."""

import os
import socket
import unittest
from unittest import mock

from sglang.srt.platforms.cpu import CpuSRTPlatform
from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.utils import vmm_backend
from sglang.srt.utils.vmm_common import recv_fd, send_fd, visible_physical_device_ids
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingBackend:
    def __init__(self, device_id: int) -> None:
        self.device_id = device_id


class TestVmmBackendDispatch(CustomTestCase):
    def setUp(self) -> None:
        vmm_backend.get_vmm_backend.cache_clear()
        self.addCleanup(vmm_backend.get_vmm_backend.cache_clear)

    def test_platform_class_is_built_once_per_device(self) -> None:
        platform = mock.Mock(get_vmm_backend_class=lambda: _RecordingBackend)
        with mock.patch.object(vmm_backend, "current_platform", platform):
            first = vmm_backend.get_vmm_backend(3)
            self.assertIsInstance(first, _RecordingBackend)
            self.assertEqual(first.device_id, 3)
            self.assertIs(vmm_backend.get_vmm_backend(3), first)

    def test_platform_without_backend_raises(self) -> None:
        with mock.patch.object(vmm_backend, "current_platform", CpuSRTPlatform()):
            with self.assertRaisesRegex(RuntimeError, "has no VMM backend"):
                vmm_backend.get_vmm_backend(0)

    def test_cuda_platform_names_the_cuda_backend(self) -> None:
        self.assertIs(
            CudaSRTPlatform().get_vmm_backend_class(), vmm_backend.CudaVmmBackend
        )


class TestPhysicalDeviceIds(CustomTestCase):
    def test_ordinals_map_through_the_visibility_variable(self) -> None:
        """With one visible device per process every rank's ordinal is 0, so the peer check must compare physical ids."""
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "6"}):
            self.assertEqual(
                visible_physical_device_ids("CUDA_VISIBLE_DEVICES", 1), ["6"]
            )
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
            self.assertEqual(
                visible_physical_device_ids("CUDA_VISIBLE_DEVICES", 2), ["2", "3"]
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                visible_physical_device_ids("CUDA_VISIBLE_DEVICES", 2), ["0", "1"]
            )


class TestFdTransfer(CustomTestCase):
    def test_fd_round_trip_keeps_rank_and_index(self) -> None:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        with left, right, open(__file__, "rb") as source:
            send_fd(left, source.fileno(), src_rank=2, base_idx=5)
            src_rank, base_idx, fd = recv_fd(right)
            try:
                self.assertEqual((src_rank, base_idx), (2, 5))
                self.assertNotEqual(fd, source.fileno())
                self.assertEqual(os.fstat(fd).st_ino, os.fstat(source.fileno()).st_ino)
            finally:
                os.close(fd)
            left.close()
            self.assertIsNone(recv_fd(right))


if __name__ == "__main__":
    unittest.main()
