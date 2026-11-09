"""per_token_quant_int8 on XPU (the dynamic activation quantizer of W8A8 int8)."""

import unittest

import torch

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an XPU device"
)
class TestPerTokenQuantInt8XPU(CustomTestCase):
    def test_compiles_and_matches_reference(self):
        """Regression: the kernel used tl.extra.cuda.libdevice.round (`__nv_roundf`) on
        every non-HIP backend, and loading it on XPU failed with
        ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED."""
        torch.manual_seed(0)
        x = torch.randn(17, 1280, device=DEVICE, dtype=torch.bfloat16) * 3
        q, scale = per_token_quant_int8(x)
        absmax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        torch.testing.assert_close(scale, absmax / 127, rtol=1e-6, atol=0)
        v = x.float() * (127 / absmax)
        ref = torch.sign(v) * torch.floor(v.abs() + 0.5)  # round half away from zero
        self.assertLessEqual((q.float() - ref).abs().max().item(), 1.0)
        self.assertGreaterEqual((q.float() == ref).float().mean().item(), 0.999)


if __name__ == "__main__":
    unittest.main()
