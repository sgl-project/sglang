import unittest

import torch
import triton

from sglang.srt.lora.triton_ops.sgemm_lora_a_chunked import (
    _sgemm_lora_a_kernel_chunked,
    sgemm_lora_a_fwd_chunked,
)
from sglang.srt.lora.triton_ops.sgemm_lora_a import (
    _sgemm_lora_a_kernel as _sgemm_lora_a_kernel_non_chunked,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.test_utils import CustomTestCase


class TestTritonSgemmLoraA(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Set random seed for reproducibility
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.float16

    @staticmethod
    def reorder_and_prepare_chunks(weight_indices, seg_lens, chunk_size: int):
        # Create a weight index for each row by repeating weight_indices according to seg_lens
        row_weight_indices = torch.repeat_interleave(weight_indices, seg_lens)

        # Sort rows by weight index (stable sort keeps relative order within each weight)
        index_map = torch.argsort(row_weight_indices, stable=True)

        # Get reordered weights to find group boundaries
        weights_reordered = row_weight_indices[index_map]

        # Get unique weights and their counts
        unique_weights, counts = torch.unique_consecutive(
            weights_reordered, return_counts=True
        )

        # Build chunk arrays
        chunk_to_weight = []
        cu_chunk_lens = [0]

        cumulative_pos = 0
        for weight_idx, group_len in zip(unique_weights, counts):
            group_len = group_len.item()
            num_chunks = (group_len + chunk_size - 1) // chunk_size

            chunk_to_weight.extend([weight_idx.item()] * num_chunks)

            # Add boundaries for each chunk
            for i in range(1, num_chunks):
                cu_chunk_lens.append(cumulative_pos + i * chunk_size)
            cu_chunk_lens.append(cumulative_pos + group_len)

            cumulative_pos += group_len

        chunk_to_weight = torch.tensor(
            chunk_to_weight, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        cu_chunk_lens = torch.tensor(
            cu_chunk_lens, dtype=torch.int32, pin_memory=True, device="cpu"
        )

        return index_map, chunk_to_weight, cu_chunk_lens

    def create_batch_info(self, bs):
        """
        Create batch info similar to how it is handled in production.
        This creates a LoRABatchInfo following the production patterns.
        """
        # Create basic batch info structure - use fixed sequence lengths for simplicity
        seg_lens = torch.randint(8, 32, (bs,), dtype=torch.int32, device="cpu")
        total_tokens = seg_lens.sum().item()

        # Create proper seg_indptr for non-chunked kernel
        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(seg_lens.to(self.device), dim=0)

        # Weight indices - all sequences use the same adapter (index 0)
        weight_indices = torch.zeros((bs,), dtype=torch.int32, device="cpu")

        # LoRA ranks - single adapter with rank 64
        lora_ranks = torch.zeros((1,), dtype=torch.int64, device=self.device)
        lora_ranks[0] = 64  # rank of the single adapter

        # Scalings - single adapter with default scaling
        scalings = torch.ones((1,), dtype=torch.float, device=self.device)

        # Use the production reorder_and_prepare_chunks function to create proper chunking
        index_map, chunk_to_weight, cu_chunk_lens = (
            TestTritonSgemmLoraA.reorder_and_prepare_chunks(
                weight_indices, seg_lens, chunk_size=16
            )
        )

        # Move tensors to device
        index_map = index_map.to(self.device)
        chunk_to_weight = chunk_to_weight.to(self.device)
        cu_chunk_lens = cu_chunk_lens.to(self.device)

        # Number of chunks
        num_chunks = len(chunk_to_weight)

        # Get max sequence length
        max_len = seg_lens.max().item()

        return LoRABatchInfo(
            bs=bs,
            num_segments=num_chunks,
            seg_lens=seg_lens.to(self.device),
            seg_indptr=cu_chunk_lens,
            max_len=max_len,
            use_cuda_graph=False,
            weight_indices=chunk_to_weight,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=index_map,
        )

    def test_chunked_kernel_basic(self):
        """Test the sgemm_lora_a_fwd_chunked function with simple inputs."""
        # Simple test case: single sequence, single LoRA adapter
        bs = 32
        input_dim = 4096
        rank = 64
        stack_num = 3

        # Create batch info for a single sequence
        batch_info = self.create_batch_info(bs)

        # Create test tensors (not reordered, the chunked function handles reordering internally)
        total_len = len(batch_info.permutation)
        x = torch.randn((total_len, input_dim), device=self.device, dtype=self.dtype)
        weights = torch.randn(
            (8, rank * stack_num, input_dim), device=self.device, dtype=self.dtype
        )

        # Call the high-level chunked function 
        output = sgemm_lora_a_fwd_chunked(
            x, weights, batch_info, stack_num=stack_num
        )

        # Compute reference result using PyTorch
        expected = torch.mm(x, weights[0].T)

        # Check that kernel output matches reference
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)



if __name__ == "__main__":
    unittest.main()
