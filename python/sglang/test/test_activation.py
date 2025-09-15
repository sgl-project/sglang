import itertools
import unittest

import torch

from sglang.srt.layers.activation import GeluAndMul, QuickGELU
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

_is_hip = is_hip()


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


class TestQuickGELU(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 2048]  # batch = sequence length
    DIMS = [512, 4096, 5120, 13824]  # all multiples of 16 bytes
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_gelu_quick_test(self, n_tok: int, dim: int, dtype: torch.dtype, seed: int):
        torch.manual_seed(seed)

        layer = QuickGELU().to(dtype=dtype)

        x = torch.randn(n_tok, dim, dtype=dtype, device="cuda")

        with torch.inference_mode():
            ref = layer.forward_native(x)  # x * sigmoid(1.702 * x), fp32 math
            if _is_hip:
                out = layer.forward_hip(x)  # 128-bit vectorised kernel from sgl-kernel
            else:
                out = layer.forward_cuda(x)

        tol = 1e-2 if dtype is torch.bfloat16 else 1e-3
        self.assertTrue(
            torch.allclose(out, ref, atol=tol, rtol=tol),
            msg=f"Mismatch @ B={n_tok}, D={dim}, dtype={dtype}",
        )
        print(f"Match @ B={n_tok}, D={dim}, dtype={dtype}")

    def test_quick_gelu(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.DIMS, self.DTYPES, self.SEEDS
        ):
            with self.subTest(
                num_tokens=params[0],
                dim=params[1],
                dtype=params[2],
                seed=params[3],
            ):
                self._run_gelu_quick_test(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
