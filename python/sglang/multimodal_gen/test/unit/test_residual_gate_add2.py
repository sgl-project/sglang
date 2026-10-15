"""The two-branch gated residual must equal addcmul(residual, gate, a + b) exactly."""

import unittest

import torch

from sglang.kernels.ops.diffusion import (
    can_use_residual_gate_add2_cuda,
    residual_gate_add2_cuda,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestResidualGateAdd2Cuda(unittest.TestCase):
    def test_kernel_is_bit_identical_to_addcmul(self):
        torch.manual_seed(0)
        # 1366 rows is the fewest at hidden 3072 that clear the kernel's numel
        # floor; hidden 250 is not a multiple of the vector width.
        for dtype in (torch.bfloat16, torch.float16):
            for shape in ((1366, 3072), (17_000, 250)):
                residual, a, b, full_gate = (
                    torch.randn(shape, device="cuda", dtype=dtype) for _ in range(4)
                )
                for gate in (full_gate, full_gate[:1]):
                    # Raises instead of falling back when the predicate rejects.
                    got = residual_gate_add2_cuda(residual, a, b, gate)
                    want = torch.addcmul(residual, gate, a + b)
                    self.assertTrue(torch.equal(got, want), (dtype, shape, gate.shape))
                expanded_gate = full_gate[:1].expand_as(residual)
                self.assertFalse(
                    can_use_residual_gate_add2_cuda(residual, a, b, expanded_gate)
                )


if __name__ == "__main__":
    unittest.main()
