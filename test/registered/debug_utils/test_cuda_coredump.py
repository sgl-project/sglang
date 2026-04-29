"""Intentionally trigger a CUDA illegal memory access
to verify the coredump collection pipeline works end-to-end.

Manual use:  python3 test/registered/debug_utils/test_cuda_coredump.py
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10,
    suite="stage-a-test-1-gpu-small",
    disabled="Manual only: triggers intentional CUDA crash for coredump verification",
)


class TestCudaCoredump(unittest.TestCase):
    def test_trigger_illegal_memory_access(self):
        x = torch.zeros(10, device="cuda")
        y = torch.arange(10, device="cuda")
        x[y * y] = 1
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
