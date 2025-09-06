import unittest

import torch
import random

from sglang.srt.lora.triton_ops.sgemm_lora_a_chunked import sgemm_lora_a_fwd_chunked 
from sglang.srt.lora.triton_ops.sgemm_lora_a import sgemm_lora_a_fwd 
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.test_utils import CustomTestCase


class TestTritonSgemmLoraA(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Set random seed for reproducibility
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.float16

    def _create_batch_info(
        self, bs, seg_lens, weight_indices, lora_ranks, scalings=None
    ):
        """Create LoRABatchInfo from test parameters."""
        seg_lens_tensor = torch.tensor(seg_lens, device=self.device, dtype=torch.int32)
        weight_indices_tensor = torch.tensor(
            weight_indices, device=self.device, dtype=torch.int32
        )
        lora_ranks_tensor = torch.tensor(
            lora_ranks, device=self.device, dtype=torch.int32
        )

        # Create cumulative indices
        seg_indptr = torch.cumsum(
            torch.cat([torch.zeros(1, device=self.device, dtype=torch.int32), seg_lens_tensor]),
            dim=0,
        )

        if scalings is None:
            scalings = torch.ones(len(lora_ranks), device=self.device, dtype=self.dtype)
        else:
            scalings = torch.tensor(scalings, device=self.device, dtype=self.dtype)

        # Use the new reorder_and_prepare_chunks function to generate the required fields
        BLOCK_S = 16  # Match the block size used in the kernel
        index_map, chunk_to_weight, cu_chunk_lens = LoRAManager.reorder_and_prepare_chunks(
            weight_indices_tensor, seg_lens_tensor, BLOCK_S, self.device
        )
        num_chunks = len(chunk_to_weight)

        return LoRABatchInfo(
            bs=bs,
            seg_lens=seg_lens_tensor,
            seg_indptr=seg_indptr,
            max_len=max(seg_lens),
            weight_indices=weight_indices_tensor,
            lora_ranks=lora_ranks_tensor,
            scalings=scalings,
            index_map=index_map,
            chunk_to_weight=chunk_to_weight,
            cu_chunk_lens=cu_chunk_lens,
            num_chunks=num_chunks,
        )

    def _reference_sgemm_lora_a(
        self, x, weights, batch_info, stack_num=1
    ):
        """Reference implementation using standard PyTorch operations."""
        S, K = x.shape
        num_lora, N, _ = weights.shape
        
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
        
        for i in range(batch_info.bs):
            start = batch_info.seg_indptr[i].item()
            length = batch_info.seg_lens[i].item()
            w_idx = batch_info.weight_indices[i].item()
            rank = batch_info.lora_ranks[w_idx].item()
            
            if length == 0 or rank == 0:
                continue
                
            # Extract input segment
            x_seg = x[start : start + length]  # (length, K)
            
            # Get the actual output dimension for this LoRA adapter
            actual_N = min(N, rank * stack_num)
            
            # Compute matrix multiplication
            w_seg = weights[w_idx, :actual_N, :]  # (actual_N, K)
            result = torch.mm(x_seg, w_seg.T)  # (length, actual_N)
            
            # Store result
            output[start : start + length, :actual_N] = result
            
        return output

    def test_basic_functionality(self):
        """Test basic sgemm_lora_a functionality with simple inputs."""
        # Test parameters
        bs = 31
        seg_lens = [random.randint(1, 32) for _ in range(bs)]  # Random lengths between 1 and 32
        input_dim = 64
        num_lora = 1
        stack_num = 3

        # Create batch info with mixed ranks
        lora_ranks = [64] #[4, 16, 32, 64, 4, 16, 32, 64]  # Mixture of different ranks
        weight_indices = [0 for i in range(bs)]  # Cycle through LoRA adapters
        batch_info = self._create_batch_info(bs, seg_lens, weight_indices, lora_ranks)

        # Create input tensor
        total_len = sum(seg_lens)
        x = torch.randn(total_len, input_dim, device=self.device, dtype=self.dtype)
        
        # Create weights tensor - use max rank for tensor size
        max_rank = max(lora_ranks)
        weights = torch.randn(
            num_lora, max_rank * stack_num, input_dim, device=self.device, dtype=self.dtype
        )

        # Run triton implementation
        triton_output = sgemm_lora_a_fwd(x, weights, batch_info, stack_num)

        # Run reference implementation
        ref_output = self._reference_sgemm_lora_a(x, weights, batch_info, stack_num)

        # Check shapes
        self.assertEqual(triton_output.shape, (total_len, max_rank * stack_num))
        self.assertEqual(ref_output.shape, triton_output.shape)

        # Check numerical accuracy
        torch.testing.assert_close(
            triton_output, ref_output, atol=1e-2, rtol=1e-2
        )

    def test_mixed_batch(self):
        """Test basic sgemm_lora_a functionality with simple inputs."""
        # Test parameters
        bs = 31
        seg_lens = [random.randint(1, 128) for _ in range(bs)]  # Random lengths between 1 and 128
        input_dim = 512
        num_lora = 1
        stack_num = 3

        # Create batch info with mixed ranks
        lora_ranks =  [4, 16, 32, 64, 4, 16, 32, 64]  # Mixture of different ranks
        weight_indices = [i % num_lora for i in range(bs)]  # Cycle through LoRA adapters
        batch_info = self._create_batch_info(bs, seg_lens, weight_indices, lora_ranks)

        # Create input tensor
        total_len = sum(seg_lens)
        x = torch.randn(total_len, input_dim, device=self.device, dtype=self.dtype)
        
        # Create weights tensor - use max rank for tensor size
        max_rank = max(lora_ranks)
        weights = torch.randn(
            num_lora, max_rank * stack_num, input_dim, device=self.device, dtype=self.dtype
        )

        # Run triton implementation
        triton_output = sgemm_lora_a_fwd(x, weights, batch_info, stack_num)

        # Run reference implementation
        ref_output = self._reference_sgemm_lora_a(x, weights, batch_info, stack_num)

        # Check shapes
        self.assertEqual(triton_output.shape, (total_len, max_rank * stack_num))
        self.assertEqual(ref_output.shape, triton_output.shape)

        # Check numerical accuracy
        torch.testing.assert_close(
            triton_output, ref_output, atol=1e-2, rtol=1e-2
        )

if __name__ == "__main__":
    unittest.main()