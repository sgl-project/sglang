import unittest

import torch
from triton.runtime.errors import OutOfResources

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _InjectSharedMemoryFailure:
    def __init__(self, kernel):
        self.kernel = kernel
        self.num_stages = []

    def __getitem__(self, grid):
        real_launch = self.kernel[grid]

        def launch(*args, **kwargs):
            num_stages = kwargs["num_stages"]
            self.num_stages.append(num_stages)
            if num_stages == 3:
                raise OutOfResources(106496, 101376, "shared memory")
            return real_launch(*args, **kwargs)

        return launch


class TestPersistentKernelFallbackCuda(CustomTestCase):
    def setUp(self):
        batch_invariant_ops._PERSISTENT_KERNEL_FALLBACK_CACHE.clear()
        torch.manual_seed(29149)
        self.a = torch.randn((6, 512), device="cuda", dtype=torch.float16)
        self.b = torch.randn((512, 512), device="cuda", dtype=torch.float16)

    def tearDown(self):
        batch_invariant_ops._PERSISTENT_KERNEL_FALLBACK_CACHE.clear()

    def test_matmul_fallback_remains_batch_invariant(self):
        """Float16 matmul stays batch invariant when its launch exhausts smem."""
        original = batch_invariant_ops.matmul_kernel_persistent
        proxy = _InjectSharedMemoryFailure(original)
        batch_invariant_ops.matmul_kernel_persistent = proxy
        try:
            with self.assertLogs(batch_invariant_ops.__name__, level="WARNING"):
                full = batch_invariant_ops._matmul_persistent_triton(self.a, self.b)
            with self.assertNoLogs(batch_invariant_ops.__name__, level="WARNING"):
                single = batch_invariant_ops._matmul_persistent_triton(
                    self.a[:1].contiguous(), self.b
                )
        finally:
            batch_invariant_ops.matmul_kernel_persistent = original

        self.assertEqual(proxy.num_stages, [3, 2, 2])
        self.assertTrue(torch.equal(full[:1], single))

    def test_bmm_fallback_remains_batch_invariant(self):
        """Float16 BMM stays batch invariant when its launch exhausts smem."""
        original = batch_invariant_ops.bmm_kernel_persistent
        proxy = _InjectSharedMemoryFailure(original)
        batch_invariant_ops.bmm_kernel_persistent = proxy
        try:
            with self.assertLogs(batch_invariant_ops.__name__, level="WARNING"):
                full = batch_invariant_ops.bmm_batch_invariant(
                    self.a.unsqueeze(0), self.b.unsqueeze(0)
                )
            with self.assertNoLogs(batch_invariant_ops.__name__, level="WARNING"):
                single = batch_invariant_ops.bmm_batch_invariant(
                    self.a[:1].contiguous().unsqueeze(0), self.b.unsqueeze(0)
                )
        finally:
            batch_invariant_ops.bmm_kernel_persistent = original

        self.assertEqual(proxy.num_stages, [3, 2, 2])
        self.assertTrue(torch.equal(full[:, :1], single))


if __name__ == "__main__":
    unittest.main()
