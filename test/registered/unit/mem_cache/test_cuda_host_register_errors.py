"""CUDA errors from host-memory registration reach the caller with their text.

torch.cuda.cudart() returns cudaError values, and its cudaGetErrorString only
accepts those, so formatting the message from an int raised a TypeError that
replaced the CUDA error. These tests trigger real CUDA errors through torch's
binding (the fake runtime in test_hicache_host_register.py takes ints).
"""

import ctypes
import unittest

import torch

from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    find_loaded_library,
)
from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_register,
    _cuda_host_unregister,
    _cuda_host_unregister_ranges,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _clear_cuda_error() -> None:
    # The forced failures set the runtime's last error; reset it so later
    # kernel-launch checks in this process do not report it.
    ctypes.CDLL(find_loaded_library("libcudart")).cudaGetLastError()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.cuda is not None,
    "needs the CUDA runtime",
)
class TestCudaHostRegisterErrors(CustomTestCase):
    def setUp(self):
        torch.cuda.init()

    def test_registration_error_names_the_cuda_error(self):
        buffer = torch.empty(1 << 20, dtype=torch.uint8)
        _cuda_host_register(buffer)
        self.addCleanup(_cuda_host_unregister, buffer)
        # Registering the same range twice is a real CUDA error.
        with self.assertRaisesRegex(
            RuntimeError, r"cudaHostRegister failed \(rc=\d+, [^)]*already"
        ):
            _cuda_host_register(buffer)
        _clear_cuda_error()

    def test_unregister_error_is_logged_with_the_cuda_error(self):
        buffer = torch.empty(4096, dtype=torch.uint8)
        ranges = [(buffer.data_ptr(), 4096)]
        with self.assertLogs(
            "sglang.srt.mem_cache.pool_host.common", level="WARNING"
        ) as logs:
            failed = _cuda_host_unregister_ranges(
                torch.cuda.cudart(), ranges, operation="test"
            )
        _clear_cuda_error()
        self.assertEqual(failed, ranges)
        self.assertRegex(
            logs.output[0], r"cudaHostUnregister failed during test \(rc=\d+, \w"
        )


if __name__ == "__main__":
    unittest.main()
