import unittest

import torch
import torch.nn.functional as F
from sgl_kernel.common_ops import silu_and_mul_cpu as silu_and_mul

from sglang.test.test_utils import CustomTestCase


class TestActivation(CustomTestCase):
    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def _run_single_test(self, shape, dtype, device):
        x = torch.randn(shape, dtype=dtype).to(device=device)

        out = silu_and_mul(x)
        ref_out = self._forward_native(x)

        torch.testing.assert_close(out, ref_out)

    def test_activation(self):
        self._run_single_test([128, 22016], torch.bfloat16, "cpu")
        self._run_single_test([129, 22016], torch.float16, "cpu")


if __name__ == "__main__":
    unittest.main()
