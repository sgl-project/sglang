"""SMOKE TEST — DELETE BEFORE MERGING PR.

Deliberately triggers a device-side assertion on the GPU so the CUDA
coredump pipeline (artifact upload + sglang-ci-stats#2 tracker issue
comment) can be exercised end-to-end on a real CI run.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")


class TestCudaCoredumpSmoke(CustomTestCase):
    def test_force_warp_assert(self):
        torch._assert_async(torch.tensor(False, device="cuda"))
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
