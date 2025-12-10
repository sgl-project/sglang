"""Test ChunkedSgmvLoRABackend.run_lora_a_embedding() method."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.triton_ops import embedding_lora_a_fwd
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase


class TestChunkedLoRAEmbedding(CustomTestCase):
    """Test embedding LoRA for ChunkedSgmvLoRABackend (requires CUDA)."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_run_lora_a_embedding(self):
        """Test that run_lora_a_embedding uses embedding_batch_info correctly."""
        device = torch.device("cuda")
        vocab_size = 1024
        rank = 16
        num_loras = 2
        batch_size = 2
        seq_len = 4

        # Create mock server_args
        server_args = MagicMock()
        server_args.max_lora_chunk_size = 128

        # Create backend
        backend = ChunkedSgmvLoRABackend(
            max_loras_per_batch=num_loras,
            device=device,
            server_args=server_args,
        )

        # Create mock forward_batch
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=torch.randint(
                0, vocab_size, (batch_size, 3), dtype=torch.int32, device=device
            ),
            req_pool_indices=None,
            seq_lens=None,
            out_cache_loc=None,
            seq_lens_sum=seq_len,
            extend_seq_lens=torch.tensor([2, 2], dtype=torch.int32, device=device),
            extend_seq_lens_cpu=[2, 2],
        )

        # Prepare batch (this sets up embedding_batch_info)
        backend.prepare_lora_batch(
            forward_batch=forward_batch,
            weight_indices=[0, 1],
            lora_ranks=[rank, rank],
            scalings=[1.0, 0.5],
            use_cuda_graph=False,
        )

        # Verify embedding_batch_info was created
        self.assertIsNotNone(backend.embedding_batch_info)
        self.assertEqual(backend.embedding_batch_info.num_segments, batch_size)
        self.assertIsNone(backend.embedding_batch_info.permutation)  # Original order

        # Create input and weights
        input_ids = torch.randint(
            0, vocab_size, (seq_len,), dtype=torch.int32, device=device
        )
        weights = torch.randn(
            num_loras, rank, vocab_size, dtype=torch.float16, device=device
        )

        # Run the method
        output = backend.run_lora_a_embedding(
            input_ids=input_ids,
            weights=weights,
            vocab_size=vocab_size,
        )

        # Verify output shape
        self.assertEqual(output.shape, (seq_len, rank))

        # Verify it matches direct call to kernel with embedding_batch_info
        expected = embedding_lora_a_fwd(
            input_ids=input_ids,
            weights=weights,
            batch_info=backend.embedding_batch_info,
            vocab_size=vocab_size,
        )
        self.assertTrue(torch.allclose(output, expected))


if __name__ == "__main__":
    unittest.main()
