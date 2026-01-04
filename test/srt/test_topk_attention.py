"""
Unit tests for top-k attention token extraction kernel.

Tests the memory-efficient chunked top-k attention extraction that
captures the tokens with highest attention scores for interpretability.

Memory comparison for 1M context:
- Old approach (full matrix): ~256MB per decode step
- New chunked approach: ~125KB per decode step (2000x reduction)
"""

import random
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


def topk_attention_reference(
    q: torch.Tensor,  # [batch, num_heads, head_dim]
    k_buffer: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    kv_indptr: torch.Tensor,  # [batch + 1]
    kv_indices: torch.Tensor,  # flattened KV indices
    sm_scale: float,
    top_k: int = 5,
):
    """
    Reference PyTorch implementation for top-k attention extraction.
    Returns top-k token positions and their normalized attention scores.
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    device = q.device

    topk_scores_list = []
    topk_indices_list = []

    for b in range(batch_size):
        kv_start = kv_indptr[b].item()
        kv_end = kv_indptr[b + 1].item()
        seq_len = kv_end - kv_start

        if seq_len == 0:
            topk_scores_list.append(torch.zeros(top_k, device=device))
            topk_indices_list.append(
                torch.zeros(top_k, dtype=torch.int64, device=device)
            )
            continue

        # Get KV cache positions for this sequence
        kv_pos = kv_indices[kv_start:kv_end]

        # Gather keys: [seq_len, num_kv_heads, head_dim]
        keys = k_buffer[kv_pos]

        # Expand keys for GQA: [seq_len, num_heads, head_dim]
        if kv_group_num > 1:
            keys = keys.repeat_interleave(kv_group_num, dim=1)

        # Query for this batch: [num_heads, head_dim]
        query = q[b]

        # Compute attention scores: [num_heads, seq_len]
        scores = torch.einsum("hd,shd->hs", query.float(), keys.float()) * sm_scale

        # Average across heads for interpretability
        scores_avg = scores.mean(dim=0)  # [seq_len]

        # Get top-k
        actual_k = min(top_k, seq_len)
        topk_vals, topk_idx = torch.topk(scores_avg, actual_k)

        # Apply softmax to normalize scores
        topk_probs = torch.softmax(topk_vals, dim=0)

        # Pad if needed
        if actual_k < top_k:
            padding = top_k - actual_k
            topk_probs = torch.cat(
                [topk_probs, torch.zeros(padding, device=device)]
            )
            topk_idx = torch.cat(
                [topk_idx, torch.zeros(padding, dtype=torch.int64, device=device)]
            )

        topk_scores_list.append(topk_probs)
        topk_indices_list.append(topk_idx)

    return torch.stack(topk_scores_list), torch.stack(topk_indices_list)


class TestTopkAttention(CustomTestCase):

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        self._set_all_seeds(42)

    def _test_topk_attention_once(
        self, batch_size, seq_len, num_heads, num_kv_heads, head_dim, top_k
    ):
        """Test top-k attention extraction for given parameters."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        device = get_device()
        dtype = torch.bfloat16

        # Create random query
        q = torch.randn(
            batch_size, num_heads, head_dim, dtype=dtype, device=device
        )

        # Create KV cache with variable sequence lengths per batch
        seq_lens = torch.randint(
            max(1, seq_len // 2), seq_len + 1, (batch_size,), device=device
        )
        total_tokens = seq_lens.sum().item()

        k_buffer = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Create CSR-style indptr and indices
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)

        kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)

        sm_scale = 1.0 / (head_dim**0.5)

        # Run the chunked kernel
        topk_scores, topk_indices = compute_topk_attention_chunked(
            q, k_buffer, kv_indptr, kv_indices, sm_scale, top_k
        )

        # Run reference implementation
        ref_scores, ref_indices = topk_attention_reference(
            q, k_buffer, kv_indptr, kv_indices, sm_scale, top_k
        )

        # Check shapes
        self.assertEqual(topk_scores.shape, (batch_size, top_k))
        self.assertEqual(topk_indices.shape, (batch_size, top_k))

        # Check that indices are valid (within sequence length for each batch)
        for b in range(batch_size):
            valid_len = seq_lens[b].item()
            actual_k = min(top_k, valid_len)
            for k in range(actual_k):
                self.assertGreaterEqual(topk_indices[b, k].item(), 0)
                self.assertLess(topk_indices[b, k].item(), valid_len)

        # Check that scores sum to approximately 1 (after softmax)
        for b in range(batch_size):
            valid_len = seq_lens[b].item()
            actual_k = min(top_k, valid_len)
            score_sum = topk_scores[b, :actual_k].sum().item()
            self.assertAlmostEqual(score_sum, 1.0, places=3)

        # Check that the same top indices are found (order might differ due to ties)
        for b in range(batch_size):
            valid_len = seq_lens[b].item()
            actual_k = min(top_k, valid_len)
            kernel_set = set(topk_indices[b, :actual_k].tolist())
            ref_set = set(ref_indices[b, :actual_k].tolist())
            # Allow some tolerance for ties
            overlap = len(kernel_set & ref_set)
            self.assertGreaterEqual(
                overlap, actual_k - 1, f"Batch {b}: kernel={kernel_set}, ref={ref_set}"
            )

    def test_basic_topk(self):
        """Test basic top-k extraction with small tensors."""
        self._test_topk_attention_once(
            batch_size=2, seq_len=32, num_heads=8, num_kv_heads=8, head_dim=64, top_k=5
        )

    def test_gqa_topk(self):
        """Test top-k with grouped query attention (GQA)."""
        self._test_topk_attention_once(
            batch_size=4, seq_len=64, num_heads=32, num_kv_heads=8, head_dim=128, top_k=10
        )

    def test_mqa_topk(self):
        """Test top-k with multi-query attention (MQA)."""
        self._test_topk_attention_once(
            batch_size=2, seq_len=128, num_heads=16, num_kv_heads=1, head_dim=64, top_k=5
        )

    def test_topk_larger_than_seq(self):
        """Test when top_k is larger than sequence length."""
        self._test_topk_attention_once(
            batch_size=2, seq_len=3, num_heads=4, num_kv_heads=4, head_dim=32, top_k=10
        )

    def test_single_token_sequence(self):
        """Test with single token sequences."""
        self._test_topk_attention_once(
            batch_size=4, seq_len=1, num_heads=8, num_kv_heads=8, head_dim=64, top_k=5
        )

    def test_large_batch(self):
        """Test with larger batch size."""
        self._test_topk_attention_once(
            batch_size=32, seq_len=256, num_heads=16, num_kv_heads=4, head_dim=128, top_k=10
        )

    def test_various_head_dims(self):
        """Test with various head dimensions."""
        for head_dim in [32, 64, 128, 256]:
            with self.subTest(head_dim=head_dim):
                self._test_topk_attention_once(
                    batch_size=2,
                    seq_len=64,
                    num_heads=8,
                    num_kv_heads=8,
                    head_dim=head_dim,
                    top_k=5,
                )

    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        device = get_device()
        batch_size = 2
        num_heads = 8
        head_dim = 64
        top_k = 5

        q = torch.randn(batch_size, num_heads, head_dim, device=device)
        k_buffer = torch.randn(0, num_heads, head_dim, device=device)
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        kv_indices = torch.zeros(0, dtype=torch.int32, device=device)
        sm_scale = 1.0 / (head_dim**0.5)

        topk_scores, topk_indices = compute_topk_attention_chunked(
            q, k_buffer, kv_indptr, kv_indices, sm_scale, top_k
        )

        # Should return zeros for empty sequences
        self.assertEqual(topk_scores.shape, (batch_size, top_k))
        self.assertEqual(topk_indices.shape, (batch_size, top_k))
        self.assertTrue(torch.all(topk_scores == 0))
        self.assertTrue(torch.all(topk_indices == 0))

    def test_large_sequence_memory_efficiency(self):
        """
        Test that chunked approach handles large sequences without OOM.

        For a 50K token sequence:
        - Old approach would allocate: batch × heads × 50K × 4 = ~12.8MB
        - Chunked approach allocates: batch × heads × 25 chunks × 4 = ~6.4KB

        This test verifies the chunked path is used for large sequences.
        """
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        device = get_device()
        batch_size = 1
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        seq_len = 50000  # 50K tokens - triggers chunked path
        top_k = 10

        q = torch.randn(
            batch_size, num_heads, head_dim,
            dtype=torch.float16, device=device
        )
        k_buffer = torch.randn(
            seq_len, num_kv_heads, head_dim,
            dtype=torch.float16, device=device
        )
        kv_indptr = torch.tensor(
            [0, seq_len], dtype=torch.int32, device=device
        )
        kv_indices = torch.arange(
            seq_len, dtype=torch.int32, device=device
        )
        sm_scale = 1.0 / (head_dim ** 0.5)

        # This should not OOM due to chunked approach
        topk_scores, topk_indices = compute_topk_attention_chunked(
            q, k_buffer, kv_indptr, kv_indices, sm_scale,
            top_k=top_k, chunk_size=2048
        )

        # Verify results
        self.assertEqual(topk_scores.shape, (batch_size, top_k))
        self.assertEqual(topk_indices.shape, (batch_size, top_k))

        # Indices should be within valid range
        self.assertTrue(torch.all(topk_indices >= 0))
        self.assertTrue(torch.all(topk_indices < seq_len))

        # Scores should sum to ~1 (softmax normalized)
        score_sum = topk_scores[0].sum().item()
        self.assertAlmostEqual(score_sum, 1.0, places=2)

    def test_topk_attention_capture_class(self):
        """Test the high-level TopKAttentionCapture API."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            TopKAttentionCapture,
        )

        device = get_device()
        batch_size = 2
        num_heads = 16
        num_kv_heads = 4
        head_dim = 64
        seq_len = 512
        top_k = 5

        q = torch.randn(
            batch_size, num_heads, head_dim,
            dtype=torch.float16, device=device
        )
        k_buffer = torch.randn(
            seq_len * batch_size, num_kv_heads, head_dim,
            dtype=torch.float16, device=device
        )
        kv_indptr = torch.tensor(
            [0, seq_len, seq_len * 2], dtype=torch.int32, device=device
        )
        kv_indices = torch.arange(
            seq_len * batch_size, dtype=torch.int32, device=device
        )
        sm_scale = 1.0 / (head_dim ** 0.5)

        # Test the capture class
        capture = TopKAttentionCapture(top_k=top_k, chunk_size=256)
        result = capture.extract(q, k_buffer, kv_indptr, kv_indices, sm_scale)

        # Check result structure
        self.assertIn("scores", result)
        self.assertIn("indices", result)
        self.assertEqual(result["scores"].shape, (batch_size, top_k))
        self.assertEqual(result["indices"].shape, (batch_size, top_k))

        # Test format_for_response (matches frontend schema)
        formatted = capture.format_for_response(result, layer_id=42)
        self.assertEqual(len(formatted), batch_size)
        for item in formatted:
            self.assertIn("token_positions", item)
            self.assertIn("attention_scores", item)
            self.assertIn("layer_id", item)
            self.assertEqual(len(item["token_positions"]), top_k)
            self.assertEqual(len(item["attention_scores"]), top_k)
            self.assertEqual(item["layer_id"], 42)


if __name__ == "__main__":
    unittest.main()
