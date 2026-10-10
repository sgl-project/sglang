# SPDX-License-Identifier: Apache-2.0
"""A mounted fast path falls back on inputs it cannot serve.

The fused RMSNorm+modulate launcher raises on anything outside its contract,
which is right for a caller that has already committed. The LTX-2 site has
not committed: it is request-gated, so an input it cannot serve has to go
down the reference chain instead of failing the request.

It did fail one. The site guarded on dtype and width only, while the launcher
also requires `scale` and `shift` to be contiguous `[B, 1, D]` rows, and
LTX-2.5's are not -- so every denoising step raised
"scale and shift must be contiguous [B, 1, D] tensors" as soon as the fusion
was mounted. Nothing caught it earlier because the fusion only mounted above
the default level.
"""

import sys
import unittest

import pytest
import torch

from sglang.kernels.ops.diffusion import can_use_fused_rmsnorm_modulation


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestFusedRmsNormModulationGuard(unittest.TestCase):
    """The guard answers exactly the question the launcher would raise on."""

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 16, 2048, device="cuda", dtype=torch.bfloat16)
        self.rows = torch.randn(2, 1, 2048, device="cuda", dtype=torch.bfloat16)

    def test_admits_the_layout_the_kernel_serves(self):
        self.assertTrue(
            can_use_fused_rmsnorm_modulation(self.x, self.rows, self.rows.clone())
        )

    def test_rejects_modulation_rows_that_are_not_contiguous(self):
        # LTX-2.5's shape: the rows arrive as a stride into a wider tensor.
        wide = torch.randn(2, 1, 4096, device="cuda", dtype=torch.bfloat16)
        strided = wide[:, :, ::2]
        self.assertFalse(strided.is_contiguous())
        self.assertFalse(can_use_fused_rmsnorm_modulation(self.x, strided, self.rows))

    def test_rejects_a_per_token_modulation(self):
        per_token = torch.randn(2, 16, 2048, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(can_use_fused_rmsnorm_modulation(self.x, per_token, self.rows))

    def test_rejects_mismatched_dtype_and_width(self):
        self.assertFalse(
            can_use_fused_rmsnorm_modulation(self.x, self.rows.float(), self.rows)
        )
        narrow = torch.randn(2, 16, 1024, device="cuda", dtype=torch.bfloat16)
        rows = torch.randn(2, 1, 1024, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(can_use_fused_rmsnorm_modulation(narrow, rows, rows))

    def test_whatever_the_guard_rejects_the_launcher_would_have_raised(self):
        # The guard is only worth having if it is exactly the launcher's
        # contract: a rejected input must be one the launcher refuses, and an
        # admitted one must go through.
        from sglang.kernels.ops.diffusion import fused_rmsnorm_scale_shift_bitexact

        weight = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
        wide = torch.randn(2, 1, 4096, device="cuda", dtype=torch.bfloat16)
        for name, scale, shift in (
            ("non-contiguous rows", wide[:, :, ::2], self.rows),
            ("per-token modulation", self.x, self.rows),
            ("fp32 rows", self.rows.float(), self.rows),
        ):
            with self.subTest(name):
                self.assertFalse(can_use_fused_rmsnorm_modulation(self.x, scale, shift))
                with self.assertRaises(RuntimeError):
                    fused_rmsnorm_scale_shift_bitexact(
                        self.x, weight, scale, shift, 1e-6
                    )

        self.assertTrue(can_use_fused_rmsnorm_modulation(self.x, self.rows, self.rows))
        fused_rmsnorm_scale_shift_bitexact(self.x, weight, self.rows, self.rows, 1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
