"""Validate H20-3e decode tables, including nearest-M interpolation.

Run on an H20-3e with the proposed configuration files installed:
    python -m unittest test.manual.quant.test_h20_block_fp8_configs -v
"""

import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    get_w8a8_block_fp8_configs,
    w8a8_block_fp8_matmul_triton,
)

SHAPES = [
    (1792, 5120),
    (25600, 6144),
    (4096, 1280),
    (5120, 1024),
    (5120, 288),
    (576, 5120),
]
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}


class TestH20BlockFP8Configs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_name() != "NVIDIA H20-3e"
        ):
            raise unittest.SkipTest(
                "These configurations are specific to NVIDIA H20-3e"
            )
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_untuned_batch_sizes_keep_default(self):
        for n, k in SHAPES:
            configs = get_w8a8_block_fp8_configs(n, k, 32, 32)
            self.assertIsNotNone(configs, (n, k))
            for m in [0, *range(65, 8193), 32768, 131072, 1048576]:
                selected = configs[min(configs, key=lambda key: abs(key - m))]
                self.assertEqual(selected, DEFAULT_CONFIG, (m, n, k))

    def test_all_decode_batches_against_fp32(self):
        # Independent seeds and both scale strides exercise masked M tails,
        # K=288's partial Split-K partition, and both supported output dtypes.
        for seed in (1234, 5678):
            torch.manual_seed(seed)
            for n, k in SHAPES:
                b = (torch.randn(n, k, device="cuda") * 0.2).to(torch.float8_e4m3fn)
                bs = torch.rand(n // 32, k // 32, device="cuda") + 0.1
                b_dequant = b.float() * bs.repeat_interleave(32, 0).repeat_interleave(
                    32, 1
                )
                for m in range(1, 66):
                    a = (torch.randn(m, k, device="cuda") * 0.2).to(torch.float8_e4m3fn)
                    scales = torch.rand(m, k // 32, device="cuda") + 0.1
                    reference = (
                        a.float() * scales.repeat_interleave(32, 1)
                    ) @ b_dequant.T
                    for column_major in (False, True):
                        a_scales = scales.T.contiguous().T if column_major else scales
                        for dtype in (torch.bfloat16, torch.float16):
                            with self.subTest(
                                seed=seed,
                                m=m,
                                n=n,
                                k=k,
                                column_major=column_major,
                                dtype=dtype,
                            ):
                                actual = w8a8_block_fp8_matmul_triton(
                                    a, b, a_scales, bs, [32, 32], dtype
                                )
                                expected = reference.to(dtype).float()
                                self.assertTrue(torch.isfinite(actual).all().item())
                                relative_l2 = torch.linalg.vector_norm(
                                    actual.float() - expected
                                ) / torch.linalg.vector_norm(expected)
                                self.assertLess(relative_l2.item(), 0.01)


if __name__ == "__main__":
    unittest.main()
