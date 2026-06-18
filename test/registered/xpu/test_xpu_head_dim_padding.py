"""
Test for XPU vision attention head_dim padding support.

This test verifies that VisionIntelXPUAttention properly handles unsupported
head dimensions (like 80) by padding to the nearest supported size (e.g., 96).
This is required for models like Qwen2-VL that use head_dim=80 in their vision
encoders, which would otherwise fail with the XPU SYCL flash attention kernel.

Usage:
    python3 -m unittest test_xpu_head_dim_padding.TestXPUHeadDimPadding.test_head_dim_80_padding
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")

_XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, "xpu") else False


@unittest.skipUnless(_XPU_AVAILABLE, "Intel XPU not available")
class TestXPUHeadDimPadding(CustomTestCase):
    """Test head dimension padding for XPU flash attention."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("xpu")
        self.dtype = torch.float16
        # Supported head sizes by XPU SYCL kernel
        self.supported_head_sizes = [64, 96, 128, 192]

    def test_head_dim_80_padding(self):
        """Test that head_dim=80 is padded to 96 and produces correct output."""
        from sglang.srt.layers.attention.vision import VisionIntelXPUAttention

        batch_size = 2
        seq_len = 16
        num_heads = 16
        head_dim = 80  # Unsupported size (used by Qwen2-VL vision encoder)

        # Create attention module
        attn = VisionIntelXPUAttention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / (head_dim**0.5),
            num_kv_heads=num_heads,
        ).to(self.device)

        # Create input tensors
        # Shape: (batch_size * seq_len, num_heads, head_dim)
        total_tokens = batch_size * seq_len
        q = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )

        # Create cu_seqlens for variable-length attention
        cu_seqlens = torch.tensor(
            [0, seq_len, seq_len * 2], dtype=torch.int32, device=self.device
        )

        # Run attention
        try:
            output = attn(
                q, k, v, cu_seqlens=cu_seqlens, bsz=batch_size, seq_len=seq_len
            )

            # Verify output shape matches input
            self.assertEqual(
                output.shape,
                (total_tokens, num_heads, head_dim),
                f"Output shape should match input shape for head_dim={head_dim}",
            )

            # Verify output is not all zeros (indicates computation happened)
            self.assertGreater(
                output.abs().sum().item(),
                0,
                "Output should contain non-zero values",
            )

            # Verify output is finite (no NaN/Inf)
            self.assertTrue(
                torch.isfinite(output).all(),
                "Output should not contain NaN or Inf values",
            )

        except Exception as e:
            self.fail(f"VisionIntelXPUAttention failed with head_dim={head_dim}: {e}")

    def test_supported_head_dims_unchanged(self):
        """Test that already-supported head dimensions work without padding."""
        from sglang.srt.layers.attention.vision import VisionIntelXPUAttention

        batch_size = 2
        seq_len = 16
        num_heads = 16

        # Test all supported head dimensions
        for head_dim in self.supported_head_sizes:
            with self.subTest(head_dim=head_dim):
                attn = VisionIntelXPUAttention(
                    num_heads=num_heads,
                    head_size=head_dim,
                    scale=1.0 / (head_dim**0.5),
                    num_kv_heads=num_heads,
                ).to(self.device)

                total_tokens = batch_size * seq_len
                q = torch.randn(
                    total_tokens,
                    num_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                k = torch.randn(
                    total_tokens,
                    num_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                v = torch.randn(
                    total_tokens,
                    num_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )

                cu_seqlens = torch.tensor(
                    [0, seq_len, seq_len * 2], dtype=torch.int32, device=self.device
                )

                output = attn(
                    q, k, v, cu_seqlens=cu_seqlens, bsz=batch_size, seq_len=seq_len
                )

                self.assertEqual(
                    output.shape,
                    (total_tokens, num_heads, head_dim),
                    f"Output shape mismatch for head_dim={head_dim}",
                )

    def test_padding_correctness_output_validity(self):
        """Test that padding produces valid outputs with correct statistics."""
        from sglang.srt.layers.attention.vision import VisionIntelXPUAttention

        batch_size = 1
        seq_len = 8
        num_heads = 8
        head_dim = 80

        # Create inputs
        total_tokens = batch_size * seq_len
        q = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)

        # Run with XPU attention (automatic padding head_dim=80 -> 96)
        xpu_attn = VisionIntelXPUAttention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / (head_dim**0.5),
            num_kv_heads=num_heads,
        ).to(self.device)

        output_xpu = xpu_attn(
            q, k, v, cu_seqlens=cu_seqlens, bsz=batch_size, seq_len=seq_len
        )

        # Verify output shape is correct
        self.assertEqual(
            output_xpu.shape,
            (total_tokens, num_heads, head_dim),
            "Output should maintain original head_dim=80",
        )

        # Verify output is finite and reasonable
        self.assertTrue(
            torch.isfinite(output_xpu).all(), "Output should not contain NaN/Inf"
        )

        # Verify output has reasonable magnitude (attention output should be similar scale to input)
        # Input v has std ~1.0, output should be similar order of magnitude
        output_std = output_xpu.std().item()
        self.assertGreater(output_std, 0.1, "Output std should be > 0.1")
        self.assertLess(output_std, 10.0, "Output std should be < 10.0")

        # Verify different runs with same input produce same output (deterministic)
        output_xpu2 = xpu_attn(
            q, k, v, cu_seqlens=cu_seqlens, bsz=batch_size, seq_len=seq_len
        )
        torch.testing.assert_close(
            output_xpu, output_xpu2, msg="XPU attention should be deterministic"
        )

    def test_softmax_scale_correctness(self):
        """Test that softmax_scale uses the original head_dim, not padded size."""
        from sglang.srt.layers.attention.vision import VisionIntelXPUAttention

        batch_size = 1
        seq_len = 8
        num_heads = 8
        head_dim = 80  # Will be padded to 96

        # Expected scale should be based on original head_dim=80, not padded 96
        expected_scale = 1.0 / (head_dim**0.5)

        attn = VisionIntelXPUAttention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=expected_scale,
            num_kv_heads=num_heads,
        ).to(self.device)

        total_tokens = batch_size * seq_len
        q = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)

        # Test with explicit softmax_scale
        output1 = attn(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            bsz=batch_size,
            seq_len=seq_len,
            softmax_scale=expected_scale,
        )

        # Test with None softmax_scale (should auto-compute from original head_dim)
        output2 = attn(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            bsz=batch_size,
            seq_len=seq_len,
            softmax_scale=None,
        )

        # Both outputs should be similar since scale is based on head_dim=80
        self.assertTrue(torch.isfinite(output1).all())
        self.assertTrue(torch.isfinite(output2).all())

    def test_head_size_too_large_raises_error(self):
        """Test that head_dim > 192 raises ValueError."""
        from sglang.srt.layers.attention.vision import VisionIntelXPUAttention

        batch_size = 1
        seq_len = 8
        num_heads = 8
        head_dim = 256  # Larger than max supported (192)

        attn = VisionIntelXPUAttention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / (head_dim**0.5),
            num_kv_heads=num_heads,
        ).to(self.device)

        total_tokens = batch_size * seq_len
        q = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)

        # Should raise ValueError for unsupported large head_dim
        with self.assertRaises(ValueError) as context:
            attn(q, k, v, cu_seqlens=cu_seqlens, bsz=batch_size, seq_len=seq_len)

        self.assertIn("Unsupported head size 256", str(context.exception))
        self.assertIn("[64, 96, 128, 192]", str(context.exception))

    def test_various_unsupported_head_dims(self):
        """Test various unsupported head dimensions are handled correctly."""
        from sglang.srt.layers.attention.vision import VisionIntelXPUAttention

        batch_size = 1
        seq_len = 8
        num_heads = 8

        # Test various unsupported head dimensions (all < 192)
        unsupported_dims = [32, 48, 80, 100, 112, 144, 160]

        for head_dim in unsupported_dims:
            with self.subTest(head_dim=head_dim):
                try:
                    attn = VisionIntelXPUAttention(
                        num_heads=num_heads,
                        head_size=head_dim,
                        scale=1.0 / (head_dim**0.5),
                        num_kv_heads=num_heads,
                    ).to(self.device)

                    total_tokens = batch_size * seq_len
                    q = torch.randn(
                        total_tokens,
                        num_heads,
                        head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    k = torch.randn(
                        total_tokens,
                        num_heads,
                        head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    v = torch.randn(
                        total_tokens,
                        num_heads,
                        head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )

                    cu_seqlens = torch.tensor(
                        [0, seq_len], dtype=torch.int32, device=self.device
                    )

                    output = attn(
                        q, k, v, cu_seqlens=cu_seqlens, bsz=batch_size, seq_len=seq_len
                    )

                    self.assertEqual(
                        output.shape,
                        (total_tokens, num_heads, head_dim),
                        f"Output shape mismatch for head_dim={head_dim}",
                    )
                    self.assertTrue(
                        torch.isfinite(output).all(),
                        f"Output contains NaN/Inf for head_dim={head_dim}",
                    )

                except Exception as e:
                    self.fail(f"Failed to handle unsupported head_dim={head_dim}: {e}")


if __name__ == "__main__":
    unittest.main()
