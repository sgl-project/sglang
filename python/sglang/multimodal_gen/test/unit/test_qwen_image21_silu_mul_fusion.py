# SPDX-License-Identifier: Apache-2.0
"""Device guard and bit-exactness coverage for the Qwen-Image 2.1 SiLU-mul fusion.

The fused activation is plain Triton, not a CUDA-only JIT/PTX kernel, so it is
attempted on every CUDA-like device instead of NVIDIA hardware only. These tests
pin the two facts that guard depends on: the guard itself tracks the device
family, and the fused kernel still rounds bit-exactly against the eager
``silu(a) * b`` chain it replaces.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import fused_silu_mul_bitexact
from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as model_module
from sglang.multimodal_gen.runtime.platforms import current_platform


class TestSiluMulFusionDeviceGuard(unittest.TestCase):
    def test_guard_tracks_the_cuda_like_device_family(self):
        expected = current_platform.is_cuda() or current_platform.is_hip()
        self.assertEqual(model_module._silu_mul_device, expected)
        if torch.cuda.is_available():
            self.assertTrue(
                model_module._silu_mul_device,
                "the fused SiLU-mul path is unreachable on a CUDA-like device",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-like device")
    def test_fused_silu_mul_is_bit_exact(self):
        # The MLP widths Qwen-Image 2.1 actually feeds this kernel.
        for shape in ((1, 2048, 3072), (1, 1024, 4096), (37, 1, 7)):
            with self.subTest(shape=shape):
                torch.manual_seed(0)
                gate = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
                value = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
                self.assertTrue(
                    torch.equal(
                        fused_silu_mul_bitexact(gate, value), F.silu(gate) * value
                    )
                )


if __name__ == "__main__":
    unittest.main()
