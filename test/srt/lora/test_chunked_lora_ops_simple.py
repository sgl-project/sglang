import unittest

import torch
import triton

from sglang.srt.lora.triton_ops.chunked_lora_expand import chunked_lora_expand_forward
from sglang.srt.lora.triton_ops.chunked_lora_shrink import chunked_lora_shrink_forward
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.test_utils import CustomTestCase


class TestChunkedLoRAOpsSimple(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Set random seed for reproducibility
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.float16

    def create_simple_batch_info(self, bs=4, seq_len=32, rank=64):
        """Create a simple batch info for testing."""
        seg_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cpu")
        total_tokens = seg_lens.sum().item()

        # Create seg_indptr
        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(seg_lens.to(self.device), dim=0)

        # All sequences use adapter 0
        weight_indices = torch.zeros((bs,), dtype=torch.int32, device=self.device)

        # Single adapter with specified rank
        lora_ranks = torch.tensor([rank], dtype=torch.int64, device=self.device)

        # Default scaling
        scalings = torch.ones((1,), dtype=torch.float, device=self.device)

        # Identity permutation
        permutation = torch.arange(total_tokens, dtype=torch.int32, device=self.device)

        return LoRABatchInfo(
            bs=bs,
            num_segments=bs,
            seg_lens=seg_lens.to(self.device),
            seg_indptr=seg_indptr,
            max_len=seq_len,
            use_cuda_graph=False,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=permutation,
        )

    def test_chunked_lora_expand_shape_and_properties(self):
        """Test chunked_lora_expand_forward output shape and basic properties."""
        bs = 2
        seq_len = 16
        rank = 32
        q_dim = 1024
        kv_dim = 128
        total_output_dim = q_dim + 2 * kv_dim
        
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        x = torch.randn((total_tokens, 3 * rank), device=self.device, dtype=self.dtype)
        lora_weight_b = torch.randn((1, total_output_dim, rank), device=self.device, dtype=self.dtype)
        slice_offsets = torch.tensor([0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim], dtype=torch.int32, device=self.device)
        max_qkv_out_dim = max(q_dim, kv_dim)

        output = chunked_lora_expand_forward(
            x=x,
            lora_weight_b=lora_weight_b,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max_qkv_out_dim
        )

        # Check output shape
        self.assertEqual(output.shape, (total_tokens, total_output_dim))
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(output.device, x.device)

        # Check that output is finite (no NaN or Inf)
        self.assertTrue(torch.isfinite(output).all())

    def test_chunked_lora_shrink_shape_and_properties(self):
        """Test chunked_lora_shrink_forward output shape and basic properties."""
        bs = 2
        seq_len = 16
        input_dim = 2048
        rank = 32
        num_slices = 3
        
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        x = torch.randn((total_tokens, input_dim), device=self.device, dtype=self.dtype)
        weights = torch.randn((1, num_slices * rank, input_dim), device=self.device, dtype=self.dtype)

        output = chunked_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        # Check output shape
        self.assertEqual(output.shape, (total_tokens, num_slices * rank))
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(output.device, x.device)

        # Check that output is finite (no NaN or Inf)
        self.assertTrue(torch.isfinite(output).all())

    def test_chunked_lora_expand_zero_input(self):
        """Test chunked_lora_expand_forward with zero input."""
        bs = 2
        seq_len = 16
        rank = 32
        q_dim = 512
        kv_dim = 64
        total_output_dim = q_dim + 2 * kv_dim
        
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        # Zero input
        x = torch.zeros((total_tokens, 3 * rank), device=self.device, dtype=self.dtype)
        lora_weight_b = torch.randn((1, total_output_dim, rank), device=self.device, dtype=self.dtype)
        slice_offsets = torch.tensor([0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim], dtype=torch.int32, device=self.device)

        output = chunked_lora_expand_forward(
            x=x,
            lora_weight_b=lora_weight_b,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max(q_dim, kv_dim)
        )

        # Output should be zero (since input is zero)
        expected_zero = torch.zeros((total_tokens, total_output_dim), device=self.device, dtype=self.dtype)
        torch.testing.assert_close(output, expected_zero, atol=1e-6, rtol=1e-6)

    def test_chunked_lora_shrink_zero_input(self):
        """Test chunked_lora_shrink_forward with zero input."""
        bs = 2
        seq_len = 16
        input_dim = 1024
        rank = 32
        num_slices = 2
        
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        # Zero input
        x = torch.zeros((total_tokens, input_dim), device=self.device, dtype=self.dtype)
        weights = torch.randn((1, num_slices * rank, input_dim), device=self.device, dtype=self.dtype)

        output = chunked_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        # Output should be zero (since input is zero)
        expected_zero = torch.zeros((total_tokens, num_slices * rank), device=self.device, dtype=self.dtype)
        torch.testing.assert_close(output, expected_zero, atol=1e-6, rtol=1e-6)

    def test_chunked_lora_expand_different_slice_sizes(self):
        """Test chunked_lora_expand_forward with different Q/K/V dimensions."""
        bs = 1
        seq_len = 8
        rank = 16
        
        # Test case 1: Q larger than K/V
        q_dim = 512
        kv_dim = 64
        total_output_dim = q_dim + 2 * kv_dim
        
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        x = torch.randn((total_tokens, 3 * rank), device=self.device, dtype=self.dtype)
        lora_weight_b = torch.randn((1, total_output_dim, rank), device=self.device, dtype=self.dtype)
        slice_offsets = torch.tensor([0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim], dtype=torch.int32, device=self.device)

        output = chunked_lora_expand_forward(
            x=x,
            lora_weight_b=lora_weight_b,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max(q_dim, kv_dim)
        )

        self.assertEqual(output.shape, (total_tokens, total_output_dim))
        self.assertTrue(torch.isfinite(output).all())

    def test_chunked_lora_expand_with_base_output_accumulation(self):
        """Test that chunked_lora_expand_forward correctly accumulates with base output."""
        bs = 1
        seq_len = 8
        rank = 16
        q_dim = 256
        kv_dim = 32
        total_output_dim = q_dim + 2 * kv_dim
        
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        x = torch.randn((total_tokens, 3 * rank), device=self.device, dtype=self.dtype)
        lora_weight_b = torch.randn((1, total_output_dim, rank), device=self.device, dtype=self.dtype)
        slice_offsets = torch.tensor([0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim], dtype=torch.int32, device=self.device)

        # Test with base output
        base_output = torch.randn((total_tokens, total_output_dim), device=self.device, dtype=self.dtype)
        original_base = base_output.clone()

        output = chunked_lora_expand_forward(
            x=x,
            lora_weight_b=lora_weight_b,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max(q_dim, kv_dim),
            base_output=base_output
        )

        # Should return the same tensor (modified in place)
        self.assertTrue(torch.equal(output, base_output))
        
        # Should not be the same as the original base
        self.assertFalse(torch.equal(output, original_base))

    def test_chunked_lora_shrink_scaling(self):
        """Test that chunked_lora_shrink doesn't apply scaling (that's for expand)."""
        bs = 1
        seq_len = 8
        input_dim = 512
        rank = 16
        num_slices = 2
        
        # Use a different scaling value
        seg_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cpu")
        total_tokens = seg_lens.sum().item()

        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(seg_lens.to(self.device), dim=0)

        weight_indices = torch.zeros((bs,), dtype=torch.int32, device=self.device)
        lora_ranks = torch.tensor([rank], dtype=torch.int64, device=self.device)
        
        # Different scaling - should not affect shrink output
        scalings = torch.tensor([2.0], dtype=torch.float, device=self.device)
        permutation = torch.arange(total_tokens, dtype=torch.int32, device=self.device)

        batch_info = LoRABatchInfo(
            bs=bs,
            num_segments=bs,
            seg_lens=seg_lens.to(self.device),
            seg_indptr=seg_indptr,
            max_len=seq_len,
            use_cuda_graph=False,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=permutation,
        )

        x = torch.randn((total_tokens, input_dim), device=self.device, dtype=self.dtype)
        weights = torch.randn((1, num_slices * rank, input_dim), device=self.device, dtype=self.dtype)

        # Get output with scaling = 2.0
        output_scaled = chunked_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        # Get output with scaling = 1.0
        batch_info.scalings[0] = 1.0
        output_normal = chunked_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        # Shrink should not be affected by scaling (scaling is applied in expand)
        torch.testing.assert_close(output_scaled, output_normal, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()