import itertools
import unittest

import sgl_kernel
import torch
import torch.nn.functional as F
from utils import SiluAndMul, precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestActivation(CustomTestCase):
    M = [128, 129, 257]
    N = [22016, 22018]
    dtype = [torch.float16, torch.bfloat16]

    def _activation_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)

        out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
        ref_out = SiluAndMul(x)

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_activation(self):
        for params in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=params[0], n=params[1], dtype=params[2]):
                self._activation_test(*params)


if __name__ == "__main__":
    unittest.main()
