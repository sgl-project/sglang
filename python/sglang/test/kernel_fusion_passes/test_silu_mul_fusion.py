import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../srt"))

import torch
import torch.nn as nn
from fx_pass_manager import apply_sglang_fx_optimization


class TestModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.dim = dim

    def forward(self, x):
        x1 = x[..., : self.dim]
        x2 = x[..., self.dim :]
        activated = self.silu(x1)
        result = activated * x2
        return result


class TestSiLUMulFusion(unittest.TestCase):
    DTYPES = [torch.float32, torch.half]
    BATCH_SIZES = [1, 4, 16]
    DIMS = [64, 128, 256]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fusion_test(self, batch_size, dim, dtype, seed):
        torch.manual_seed(seed)

        model = TestModel(dim).to(dtype=dtype)
        x = torch.randn(batch_size, 2 * dim, dtype=dtype)

        with torch.inference_mode():
            original_output = model(x)
            optimized_model = apply_sglang_fx_optimization(model)
            optimized_output = optimized_model(x)

        self.assertTrue(
            torch.allclose(original_output, optimized_output, atol=1e-3, rtol=1e-3)
        )

    def test_silu_mul_fusion(self):
        for batch_size in self.BATCH_SIZES:
            for dim in self.DIMS:
                for dtype in self.DTYPES:
                    for seed in self.SEEDS:
                        with self.subTest(
                            batch_size=batch_size,
                            dim=dim,
                            dtype=dtype,
                            seed=seed,
                        ):
                            self._run_fusion_test(batch_size, dim, dtype, seed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
