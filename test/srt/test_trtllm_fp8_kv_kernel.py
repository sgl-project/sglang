"""
Unit tests for TRTLLM FP8 KV cache fusion kernel.
"""

import unittest

import torch

from sglang.srt.layers.attention.trtllm_fp8_kv_kernel import fused_fp8_set_kv_buffer
from sglang.test.test_utils import CustomTestCase


class TestTRTLLMFP8KVKernel(CustomTestCase):
    """Test fused FP8 KV cache write kernel correctness."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        if torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("FP8 requires compute capability >= 9.0")

    def _test_kernel_correctness(
        self, num_tokens, num_kv_heads, head_dim, page_size, use_scale, input_ndim, cache_ndim
    ):
        """Compare Triton kernel output against naive implementation."""
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Create input tensors
        if input_ndim == 3:
            k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
            v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
        else:
            k = torch.randn(num_tokens, num_kv_heads * head_dim, device=device, dtype=dtype)
            v = torch.randn(num_tokens, num_kv_heads * head_dim, device=device, dtype=dtype)

        # Create cache tensors (use FP8 to match real runtime behavior)
        num_pages = 128
        total_slots = num_pages * page_size
        cache_dtype = torch.float8_e4m3fn
        if cache_ndim == 3:
            k_cache_triton = torch.zeros(
                total_slots, num_kv_heads, head_dim, device=device, dtype=cache_dtype
            )
            v_cache_triton = torch.zeros(
                total_slots, num_kv_heads, head_dim, device=device, dtype=cache_dtype
            )
            k_cache_naive = torch.zeros(
                total_slots, num_kv_heads, head_dim, device=device, dtype=cache_dtype
            )
            v_cache_naive = torch.zeros(
                total_slots, num_kv_heads, head_dim, device=device, dtype=cache_dtype
            )
        else:
            k_cache_triton = torch.zeros(
                num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=cache_dtype
            )
            v_cache_triton = torch.zeros(
                num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=cache_dtype
            )
            k_cache_naive = torch.zeros(
                num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=cache_dtype
            )
            v_cache_naive = torch.zeros(
                num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=cache_dtype
            )

        # Create cache locations (ensure unique indices to avoid race conditions)
        cache_loc = torch.randperm(total_slots, device=device, dtype=torch.int32)[:num_tokens]

        # Optional scales
        k_scale = 0.5 if use_scale else None
        v_scale = 0.75 if use_scale else None

        # Run Triton kernel
        fused_fp8_set_kv_buffer(
            k.clone(),
            v.clone(),
            k_cache_triton,
            v_cache_triton,
            cache_loc,
            k_scale,
            v_scale,
            page_size,
            use_triton=True,
        )

        # Run naive fallback
        fused_fp8_set_kv_buffer(
            k.clone(),
            v.clone(),
            k_cache_naive,
            v_cache_naive,
            cache_loc,
            k_scale,
            v_scale,
            page_size,
            use_triton=False,
        )

        # Compare results (bit-exact match expected)
        self.assertTrue(
            torch.equal(k_cache_triton, k_cache_naive),
            "K cache mismatch between Triton and naive",
        )
        self.assertTrue(
            torch.equal(v_cache_triton, v_cache_naive),
            "V cache mismatch between Triton and naive",
        )

    def test_basic_3d_input_3d_cache(self):
        """Test basic case: 3D input, 3D cache, no scale."""
        self._test_kernel_correctness(
            num_tokens=16,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            use_scale=False,
            input_ndim=3,
            cache_ndim=3,
        )

    def test_basic_3d_input_4d_cache(self):
        """Test basic case: 3D input, 4D cache, no scale."""
        self._test_kernel_correctness(
            num_tokens=16,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            use_scale=False,
            input_ndim=3,
            cache_ndim=4,
        )

    def test_with_scale_3d_cache(self):
        """Test with scale: 3D input, 3D cache."""
        self._test_kernel_correctness(
            num_tokens=16,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            use_scale=True,
            input_ndim=3,
            cache_ndim=3,
        )

    def test_with_scale_4d_cache(self):
        """Test with scale: 3D input, 4D cache."""
        self._test_kernel_correctness(
            num_tokens=16,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            use_scale=True,
            input_ndim=3,
            cache_ndim=4,
        )

    def test_2d_input_3d_cache(self):
        """Test 2D input (flattened): 2D input, 3D cache."""
        self._test_kernel_correctness(
            num_tokens=16,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            use_scale=False,
            input_ndim=2,
            cache_ndim=3,
        )

    def test_2d_input_4d_cache(self):
        """Test 2D input (flattened): 2D input, 4D cache."""
        self._test_kernel_correctness(
            num_tokens=16,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            use_scale=False,
            input_ndim=2,
            cache_ndim=4,
        )

    def test_single_token(self):
        """Test edge case: single token."""
        self._test_kernel_correctness(
            num_tokens=1,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            use_scale=True,
            input_ndim=3,
            cache_ndim=3,
        )

    def test_large_batch(self):
        """Test larger batch size."""
        self._test_kernel_correctness(
            num_tokens=128,
            num_kv_heads=16,
            head_dim=64,
            page_size=16,
            use_scale=True,
            input_ndim=3,
            cache_ndim=4,
        )

    def test_different_head_dims(self):
        """Test different head dimensions."""
        for head_dim in [64, 128]:
            self._test_kernel_correctness(
                num_tokens=16,
                num_kv_heads=8,
                head_dim=head_dim,
                page_size=16,
                use_scale=False,
                input_ndim=3,
                cache_ndim=3,
            )

    def test_empty_input(self):
        """Test edge case: empty input (0 tokens)."""
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_kv_heads = 8
        head_dim = 128
        page_size = 16
        num_tokens = 0

        # Empty inputs
        k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

        # Cache (use FP8 to match real runtime behavior)
        total_slots = 128
        k_cache = torch.zeros(
            total_slots, num_kv_heads, head_dim, device=device, dtype=torch.float8_e4m3fn
        )
        v_cache = torch.zeros(
            total_slots, num_kv_heads, head_dim, device=device, dtype=torch.float8_e4m3fn
        )

        # Empty cache locations
        cache_loc = torch.empty(num_tokens, device=device, dtype=torch.int32)

        # Should not crash
        fused_fp8_set_kv_buffer(
            k,
            v,
            k_cache,
            v_cache,
            cache_loc,
            k_scale=None,
            v_scale=None,
            page_size=page_size,
        )


if __name__ == "__main__":
    unittest.main()
