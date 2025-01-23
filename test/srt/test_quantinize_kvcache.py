import itertools
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import quantize_cache_kv


class TestQuantizeKVCache(unittest.TestCase):
    DTYPES = [torch.float32, torch.half, torch.bfloat16]
    BATCH_SIZES = [1, 16, 32, 128]
    HEAD_NUMS = [1, 8, 16, 32]
    HEAD_DIMS = [32, 64, 128, 256]
    SEEDS = [0, 42, 123]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _quantize_tensor(self, tensor):
        """Quantize a tensor to uint8 with scale and zero point."""
        bs, head_num, head_dim = tensor.shape
        quantized = torch.empty(
            (bs, head_num, head_dim), dtype=torch.uint8, device=tensor.device
        )
        scale_zeros = torch.empty(
            (bs, head_num, 2), dtype=torch.float32, device=tensor.device
        )

        for i in range(bs):
            for h in range(head_num):
                min_val = tensor[i, h].min().to(torch.float32)
                max_val = tensor[i, h].max().to(torch.float32)
                scale = (max_val - min_val) / 255.0
                zero_point = -min_val / scale
                quantized[i, h] = (
                    (tensor[i, h] / scale + zero_point + 0.5)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                scale_zeros[i, h, 0] = scale
                scale_zeros[i, h, 1] = zero_point

        return quantized, scale_zeros

    def _native_quantize_cache_kv(self, k, v, dest_idx):
        """Native PyTorch implementation of quantize_cache_kv for reference."""
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        k_quantized, k_scale_zeros = self._quantize_tensor(k)
        v_quantized, v_scale_zeros = self._quantize_tensor(v)

        return k_quantized, k_scale_zeros, v_quantized, v_scale_zeros

    def _test_quantize_cache_kv(self, batch_size, head_num, head_dim, dtype, seed):
        # Set print options to display all elements
        torch.set_printoptions(profile="full", linewidth=120)

        print(
            f"\nTesting with params: batch_size={batch_size}, head_num={head_num}, "
            f"head_dim={head_dim}, dtype={dtype}, seed={seed}"
        )

        torch.manual_seed(seed)

        # Create input data
        k = torch.randn(batch_size, head_num, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch_size, head_num, head_dim, dtype=dtype, device="cuda")
        dest_idx = torch.arange(batch_size, device="cuda")

        # Create output tensors
        k_quantized = torch.empty(
            (batch_size, head_num, head_dim), dtype=torch.uint8, device="cuda"
        )
        v_quantized = torch.empty(
            (batch_size, head_num, head_dim), dtype=torch.uint8, device="cuda"
        )
        k_scale_zeros = torch.empty(
            (batch_size, head_num, 2), dtype=torch.float32, device="cuda"
        )
        v_scale_zeros = torch.empty(
            (batch_size, head_num, 2), dtype=torch.float32, device="cuda"
        )

        # Run Triton implementation
        with torch.inference_mode():
            quantize_cache_kv(
                k,
                v,
                dest_idx,
                k_quantized,
                k_scale_zeros,
                v_quantized,
                v_scale_zeros,
            )

        # Run reference implementation
        ref_k_quantized, ref_k_scale_zeros, ref_v_quantized, ref_v_scale_zeros = (
            self._native_quantize_cache_kv(k, v, dest_idx)
        )

        # Assert results
        self.assertTrue(torch.allclose(k_quantized, ref_k_quantized, atol=1))
        self.assertTrue(torch.allclose(v_quantized, ref_v_quantized, atol=1))
        self.assertTrue(torch.allclose(k_scale_zeros, ref_k_scale_zeros, rtol=1e-2))
        self.assertTrue(torch.allclose(v_scale_zeros, ref_v_scale_zeros, rtol=1e-2))

    def test_quantize_cache_kv(self):
        total_tests = (
            len(self.BATCH_SIZES)
            * len(self.HEAD_NUMS)
            * len(self.HEAD_DIMS)
            * len(self.DTYPES)
            * len(self.SEEDS)
        )

        current_test = 0
        for params in itertools.product(
            self.BATCH_SIZES, self.HEAD_NUMS, self.HEAD_DIMS, self.DTYPES, self.SEEDS
        ):
            current_test += 1
            with self.subTest(
                batch_size=params[0],
                head_num=params[1],
                head_dim=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._test_quantize_cache_kv(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
