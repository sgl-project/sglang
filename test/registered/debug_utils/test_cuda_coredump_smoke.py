"""Smoke test: intentionally trigger a CUDA illegal memory access
to verify the coredump collection pipeline works end-to-end.

DELETE THIS FILE after verification.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-a-test-1")


class _CudaArrayView:
    """Expose an arbitrary (data_ptr, size) pair as __cuda_array_interface__."""

    def __init__(self, data_ptr: int, size_in_bytes: int):
        self._ptr = data_ptr
        self._size = size_in_bytes

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self._size,),
            "typestr": "|u1",
            "data": (self._ptr, False),
            "version": 3,
        }


class TestCudaCoredumpSmoke(unittest.TestCase):
    def test_trigger_illegal_memory_access(self):
        x = torch.zeros(10, device="cuda")
        # Create a view 1000x larger than the actual allocation, then fill it.
        # This guarantees an illegal memory access on the GPU.
        bogus = torch.as_tensor(_CudaArrayView(x.data_ptr(), x.numel() * 1000))
        bogus.fill_(1)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
