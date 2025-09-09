import unittest

import torch
import triton

from sglang.srt.lora.triton_ops.chunked_lora_expand import chunked_lora_expand_forward
from sglang.srt.lora.triton_ops.chunked_lora_shrink import chunked_lora_shrink_forward
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.test_utils import CustomTestCase


class TestChunkedLoRAOps(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Set random seed for reproducibility
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.float16

    def create_simple_batch_info(self, bs=4, seq_len=32, rank=64, segment_size=16):
        """Create a simple batch info for testing."""
        seq_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cpu")
        total_tokens = seq_lens.sum().item()

        # Calculate segments: each sequence gets split into chunks of segment_size
        # All sequences use the same adapter (weight index 0)
        num_segs_per_seq = (seq_len + segment_size - 1) // segment_size
        total_segments = bs * num_segs_per_seq

        # Create segment lengths and indptr
        seg_lens_list = []
        for _ in range(bs):
            # Split each sequence into segments
            remaining = seq_len
            for _ in range(num_segs_per_seq):
                seg_len = min(segment_size, remaining)
                seg_lens_list.append(seg_len)
                remaining -= seg_len

        seg_lens = torch.tensor(seg_lens_list, dtype=torch.int32, device=self.device)
        seg_indptr = torch.zeros((total_segments + 1,), dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

        # All segments use adapter 0
        weight_indices = torch.zeros((total_segments,), dtype=torch.int32, device=self.device)

        # Single adapter with specified rank
        lora_ranks = torch.tensor([rank], dtype=torch.int64, device=self.device)

        # Default scaling
        scalings = torch.ones((1,), dtype=torch.float, device=self.device)

        # Identity permutation (no reordering in this simple case)
        permutation = torch.arange(total_tokens, dtype=torch.int32, device=self.device)

        return LoRABatchInfo(
            bs=bs,
            num_segments=total_segments,
            seg_lens=seg_lens,
            seg_indptr=seg_indptr,
            max_len=seq_len,
            use_cuda_graph=False,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=permutation,
        )

    def create_qkv_slice_offsets(self, q_dim=2048, kv_dim=256):
        """Create slice offsets for Q/K/V projections."""
        return torch.tensor([0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim], dtype=torch.int32, device=self.device)

    def test_chunked_lora_expand_basic(self):
        """Test basic functionality of chunked_lora_expand_forward."""
        bs = 4
        seq_len = 32
        rank = 64
        q_dim = 2048
        kv_dim = 256
        total_output_dim = q_dim + 2 * kv_dim
        
        # Create batch info
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        # Create input tensor (result of LoRA A projection)
        # Shape: (total_tokens, 3 * rank) for Q/K/V
        x = torch.randn((total_tokens, 3 * rank), device=self.device, dtype=self.dtype)

        # Create LoRA B weights
        # Shape: (num_lora, total_output_dim, rank)
        lora_weight_b = torch.randn((1, total_output_dim, rank), device=self.device, dtype=self.dtype)

        # Create slice offsets
        slice_offsets = self.create_qkv_slice_offsets(q_dim, kv_dim)
        max_qkv_out_dim = max(q_dim, kv_dim)

        # Call the function
        output = chunked_lora_expand_forward(
            x=x,
            lora_weight_b=lora_weight_b,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max_qkv_out_dim
        )

        # Check output shape
        self.assertEqual(output.shape, (total_tokens, total_output_dim))

        # Compute reference using PyTorch
        # Q projection: x[:, :rank] @ lora_weight_b[0, :q_dim, :].T
        # K projection: x[:, rank:2*rank] @ lora_weight_b[0, q_dim:q_dim+kv_dim, :].T
        # V projection: x[:, 2*rank:3*rank] @ lora_weight_b[0, q_dim+kv_dim:, :].T
        expected = torch.zeros((total_tokens, total_output_dim), device=self.device, dtype=self.dtype)
        
        # Q projection
        expected[:, :q_dim] = torch.mm(x[:, :rank], lora_weight_b[0, :q_dim, :].T)
        # K projection
        expected[:, q_dim:q_dim+kv_dim] = torch.mm(x[:, rank:2*rank], lora_weight_b[0, q_dim:q_dim+kv_dim, :].T)
        # V projection
        expected[:, q_dim+kv_dim:] = torch.mm(x[:, 2*rank:3*rank], lora_weight_b[0, q_dim+kv_dim:, :].T)

        # Apply scaling
        expected *= batch_info.scalings[0]

        # Check results
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)

    def test_chunked_lora_expand_with_base_output(self):
        """Test chunked_lora_expand_forward with base output accumulation."""
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
        slice_offsets = self.create_qkv_slice_offsets(q_dim, kv_dim)
        max_qkv_out_dim = max(q_dim, kv_dim)

        # Create base output
        base_output = torch.randn((total_tokens, total_output_dim), device=self.device, dtype=self.dtype)
        original_base = base_output.clone()

        # Call function with base output
        output = chunked_lora_expand_forward(
            x=x,
            lora_weight_b=lora_weight_b,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max_qkv_out_dim,
            base_output=base_output
        )

        # Should return the same tensor as base_output (modified in-place)
        self.assertTrue(torch.equal(output, base_output))
        
        # Output should be base + lora computation
        self.assertFalse(torch.equal(output, original_base))

    def test_chunked_lora_expand_zero_rank(self):
        """Test chunked_lora_expand_forward with zero rank (no-op case)."""
        bs = 2
        seq_len = 16
        rank = 0  # Zero rank
        q_dim = 1024
        kv_dim = 128
        total_output_dim = q_dim + 2 * kv_dim
        
        # Create batch info with zero rank
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        batch_info.lora_ranks[0] = 0  # Explicitly set to zero
        
        total_tokens = bs * seq_len

        # Input should have shape (total_tokens, 0) but we'll create minimal tensor
        x = torch.empty((total_tokens, 0), device=self.device, dtype=self.dtype)
        lora_weight_b = torch.empty((1, total_output_dim, 0), device=self.device, dtype=self.dtype)
        slice_offsets = self.create_qkv_slice_offsets(q_dim, kv_dim)
        max_qkv_out_dim = max(q_dim, kv_dim)

        # Call function
        output = chunked_lora_expand_forward(
            x=x,
            lora_weight_b=lora_weight_b,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max_qkv_out_dim
        )

        # Should return all zeros
        expected = torch.zeros((total_tokens, total_output_dim), device=self.device, dtype=self.dtype)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)

    def test_chunked_lora_shrink_basic(self):
        """Test basic functionality of chunked_lora_shrink_forward."""
        bs = 4
        seq_len = 32
        input_dim = 4096
        rank = 64
        num_slices = 3  # For Q/K/V
        
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        # Create input tensor
        x = torch.randn((total_tokens, input_dim), device=self.device, dtype=self.dtype)

        # Create LoRA A weights 
        # Shape: (num_lora, num_slices * rank, input_dim)
        weights = torch.randn((1, num_slices * rank, input_dim), device=self.device, dtype=self.dtype)

        # Call function
        output = chunked_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        # Check output shape
        self.assertEqual(output.shape, (total_tokens, num_slices * rank))

        # Compute reference using PyTorch
        expected = torch.mm(x, weights[0].T)

        # Check results
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)

    def test_chunked_lora_shrink_zero_rank(self):
        """Test chunked_lora_shrink_forward with zero rank (no-op case)."""
        bs = 2
        seq_len = 16
        input_dim = 2048
        rank = 0  # Zero rank
        num_slices = 2
        
        # Create batch info with zero rank
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        batch_info.lora_ranks[0] = 0  # Explicitly set to zero
        
        total_tokens = bs * seq_len

        x = torch.randn((total_tokens, input_dim), device=self.device, dtype=self.dtype)
        # Weights should have shape (1, 0, input_dim) 
        weights = torch.empty((1, 0, input_dim), device=self.device, dtype=self.dtype)

        # Call function
        output = chunked_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        # Should return tensor with shape (total_tokens, 0)
        self.assertEqual(output.shape, (total_tokens, 0))

    def test_chunked_lora_shrink_different_num_slices(self):
        """Test chunked_lora_shrink_forward with different num_slices values."""
        bs = 2
        seq_len = 16
        input_dim = 1024
        rank = 32
        
        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len
        x = torch.randn((total_tokens, input_dim), device=self.device, dtype=self.dtype)

        # Test with num_slices = 2 (for gate_up_proj)
        num_slices = 2
        weights = torch.randn((1, num_slices * rank, input_dim), device=self.device, dtype=self.dtype)
        
        output = chunked_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        self.assertEqual(output.shape, (total_tokens, num_slices * rank))
        
        # Verify correctness
        expected = torch.mm(x, weights[0].T)
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)

    def test_chunked_lora_shrink_multiple_adapters(self):
        """Test chunked_lora_shrink_forward with multiple LoRA adapters."""
        bs = 4
        seq_len = 16
        input_dim = 2048
        rank1, rank2 = 32, 64
        num_slices = 3
        
        # Create batch info with different adapters for different sequences
        seg_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cpu")
        total_tokens = seg_lens.sum().item()

        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(seg_lens.to(self.device), dim=0)

        # First two sequences use adapter 0, last two use adapter 1
        weight_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=self.device)
        
        # Two adapters with different ranks
        lora_ranks = torch.tensor([rank1, rank2], dtype=torch.int64, device=self.device)
        scalings = torch.ones((2,), dtype=torch.float, device=self.device)
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
        
        # Weights for both adapters (use max rank for consistent shape)
        max_rank = max(rank1, rank2)
        weights = torch.randn((2, num_slices * max_rank, input_dim), device=self.device, dtype=self.dtype)

        # Call function
        output = chunked_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        self.assertEqual(output.shape, (total_tokens, num_slices * max_rank))

        # Verify first sequence uses adapter 0
        seq1_start = seg_indptr[0].item()
        seq1_end = seg_indptr[1].item()
        expected_seq1 = torch.mm(x[seq1_start:seq1_end], weights[0, :num_slices*rank1].T)
        torch.testing.assert_close(
            output[seq1_start:seq1_end, :num_slices*rank1], 
            expected_seq1, 
            atol=1e-2, rtol=1e-2
        )

    def test_integration_expand_shrink_pipeline(self):
        """Test the integration of shrink -> expand pipeline (typical LoRA flow)."""
        bs = 2
        seq_len = 16
        input_dim = 2048
        rank = 32
        q_dim = 1024
        kv_dim = 128
        output_dim = q_dim + 2 * kv_dim
        num_slices = 3

        batch_info = self.create_simple_batch_info(bs, seq_len, rank)
        total_tokens = bs * seq_len

        # Step 1: Create input and LoRA A weights for shrink operation
        x_input = torch.randn((total_tokens, input_dim), device=self.device, dtype=self.dtype)
        lora_a_weights = torch.randn((1, num_slices * rank, input_dim), device=self.device, dtype=self.dtype)

        # Step 2: Apply LoRA A (shrink)
        intermediate = chunked_lora_shrink_forward(
            x=x_input,
            weights=lora_a_weights,
            batch_info=batch_info,
            num_slices=num_slices
        )

        self.assertEqual(intermediate.shape, (total_tokens, num_slices * rank))

        # Step 3: Apply LoRA B (expand)
        lora_b_weights = torch.randn((1, output_dim, rank), device=self.device, dtype=self.dtype)
        slice_offsets = self.create_qkv_slice_offsets(q_dim, kv_dim)
        max_qkv_out_dim = max(q_dim, kv_dim)

        final_output = chunked_lora_expand_forward(
            x=intermediate,
            lora_weight_b=lora_b_weights,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max_qkv_out_dim
        )

        self.assertEqual(final_output.shape, (total_tokens, output_dim))

        # Verify end-to-end computation
        # Expected: x_input @ lora_a_weights[0].T @ lora_b_weights[0] (with slicing)
        intermediate_ref = torch.mm(x_input, lora_a_weights[0].T)
        
        expected_final = torch.zeros((total_tokens, output_dim), device=self.device, dtype=self.dtype)
        # Q projection
        expected_final[:, :q_dim] = torch.mm(intermediate_ref[:, :rank], lora_b_weights[0, :q_dim, :].T)
        # K projection  
        expected_final[:, q_dim:q_dim+kv_dim] = torch.mm(intermediate_ref[:, rank:2*rank], lora_b_weights[0, q_dim:q_dim+kv_dim, :].T)
        # V projection
        expected_final[:, q_dim+kv_dim:] = torch.mm(intermediate_ref[:, 2*rank:3*rank], lora_b_weights[0, q_dim+kv_dim:, :].T)
        
        # Apply scaling
        expected_final *= batch_info.scalings[0]

        torch.testing.assert_close(final_output, expected_final, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()