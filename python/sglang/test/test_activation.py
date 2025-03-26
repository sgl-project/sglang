import itertools
import unittest

import torch

from sglang.srt.layers.activation import GeluAndMul
from sglang.test.test_utils import CustomTestCase


class TestGeluAndMul(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 2048]
    D = [512, 4096, 5120, 13824]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_gelu_and_mul_test(self, num_tokens, d, dtype, seed):
        torch.manual_seed(seed)

        layer = GeluAndMul().to(dtype=dtype)
        x = torch.randn(num_tokens, 2 * d, dtype=dtype)

        with torch.inference_mode():
            ref_out = layer.forward_native(x)
            out = layer.forward_cuda(x)

        if dtype == torch.bfloat16:
            atol = rtol = 1e-2
        else:
            atol = rtol = 1e-3

        self.assertTrue(torch.allclose(out, ref_out, atol=atol, rtol=rtol))

    def test_gelu_and_mul(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.D,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                d=params[1],
                dtype=params[2],
                seed=params[3],
            ):
                self._run_gelu_and_mul_test(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
