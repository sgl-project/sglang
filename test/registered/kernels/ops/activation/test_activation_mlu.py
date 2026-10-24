"""MLU SiLU gating against the native reference."""

import unittest

import torch
import torch_mlu  # noqa: F401

from sglang.srt.layers.activation import SiluAndMul
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=30, suite="pr-test-1-mlu")


class TestActivationMLU(CustomTestCase):
    def test_silu_and_mul(self):
        act = SiluAndMul()
        for dtype in (torch.bfloat16, torch.float16):
            for shape in ((7, 128), (2, 7, 128)):
                with self.subTest(dtype=dtype, shape=shape):
                    x = torch.randn(shape, device="mlu", dtype=dtype)
                    torch.testing.assert_close(
                        act(x), act.forward_native(x), rtol=0.02, atol=0.02
                    )
                    self.assertEqual(act._forward_method, act.forward_mlu)


if __name__ == "__main__":
    unittest.main()
