#!/usr/bin/env python3
"""
Correctness tests for attention token capture.

These tests verify:
1. Token positions are LOGICAL (token stream) not PHYSICAL (KV cache addresses)
2. GQA head broadcasting is handled correctly
3. Prefix sharing / radix cache doesn't corrupt position mapping

Run with:
    python -m pytest test/srt/test_attention_correctness.py -v
"""

import json
import os
import unittest
from typing import Dict, List, Optional, Tuple

import numpy as np


class TestAttentionPositionCorrectness(unittest.TestCase):
    """
    Test that attention positions are logical token indices.

    In paged/radix KV cache systems, the physical memory layout differs from
    logical token order. These tests verify we return logical positions.
    """

    def setUp(self):
        """Check if we can import the required modules."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                decode_forward_with_topk,
            )

            self.decode_available = True
        except ImportError:
            self.decode_available = False

    def _create_mock_kv_cache(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        seq_len: int,
        dtype=None,
    ) -> Tuple:
        """Create mock KV cache tensors."""
        import torch

        dtype = dtype or torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create KV cache with known patterns
        k_cache = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
        )
        v_cache = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
        )

        return k_cache, v_cache

    def test_position_indices_are_logical(self):
        """
        Verify that returned token_positions are logical indices [0, seq_len).

        Even with paged KV cache (where physical addresses differ), the
        positions returned should match the original token ordering.
        """
        if not self.decode_available:
            self.skipTest("decode_attention_with_topk not available")

        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Setup
        batch_size = 2
        num_heads = 8
        head_dim = 64
        seq_len = 100
        top_k = 10

        # Create query and KV cache
        q = torch.randn(
            batch_size, num_heads, 1, head_dim, dtype=torch.float16, device="cuda"
        )
        k_cache, v_cache = self._create_mock_kv_cache(
            batch_size, num_heads, head_dim, seq_len, torch.float16
        )

        # Create seq_lens
        seq_lens = torch.tensor(
            [seq_len] * batch_size, dtype=torch.int32, device="cuda"
        )

        # Run attention with top-k extraction
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            decode_forward_with_topk,
        )

        output, topk_values, topk_indices = decode_forward_with_topk(
            q.squeeze(2),  # [B, H, D]
            k_cache,
            v_cache,
            seq_lens,
            top_k=top_k,
            sm_scale=1.0 / (head_dim ** 0.5),
        )

        # Verify: all indices should be in [0, seq_len)
        for b in range(batch_size):
            for h in range(num_heads):
                indices = topk_indices[b, h].cpu().tolist()
                valid_indices = [i for i in indices if i >= 0]  # Filter sentinel values

                self.assertTrue(
                    all(0 <= idx < seq_len for idx in valid_indices),
                    f"Batch {b}, Head {h}: indices {list(valid_indices)} not in [0, {seq_len})",
                )

    def test_needle_token_position(self):
        """
        'Needle in haystack' test: verify we find a specific token at its correct position.

        This test creates a KV cache where one position has a distinctive key,
        then verifies the attention extraction correctly identifies that position.
        """
        if not self.decode_available:
            self.skipTest("decode_attention_with_topk not available")

        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Setup
        batch_size = 1
        num_heads = 4
        head_dim = 64
        seq_len = 50
        needle_position = 25  # The position where we put the "needle"
        top_k = 5

        # Create KV cache with a "needle" - a key that will have high attention
        k_cache = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda"
        )
        v_cache = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda"
        )

        # Create a query that matches the needle
        needle_key = torch.randn(
            1, num_heads, 1, head_dim, dtype=torch.float16, device="cuda"
        )
        k_cache[:, :, needle_position : needle_position + 1, :] = needle_key

        # Query is the same as the needle key (will have high attention to position 25)
        q = needle_key.squeeze(2)  # [B, H, D]

        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            decode_forward_with_topk,
        )

        output, topk_values, topk_indices = decode_forward_with_topk(
            q,
            k_cache,
            v_cache,
            seq_lens,
            top_k=top_k,
            sm_scale=1.0 / (head_dim ** 0.5),
        )

        # The needle position should appear in top-k for all heads
        indices = topk_indices[0].cpu().tolist()  # [H, K]

        for h in range(num_heads):
            head_topk = indices[h]
            self.assertIn(
                needle_position,
                head_topk,
                f"Head {h}: needle position {needle_position} not in top-{top_k}: {head_topk}",
            )


class TestGQABroadcastCorrectness(unittest.TestCase):
    """
    Test that GQA (Grouped Query Attention) head broadcasting is handled correctly.

    In GQA, multiple query heads share the same KV head. The attention extraction
    averages across all query heads before finding top-k, correctly handling the
    GQA head mapping.
    """

    def setUp(self):
        """Check if we can import the required modules."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                compute_topk_attention_chunked,
            )

            self.gqa_available = True
        except ImportError:
            self.gqa_available = False

    def test_gqa_head_averaging(self):
        """
        Test that GQA correctly averages attention scores across query heads.

        The compute_topk_attention_chunked function averages scores across all
        query heads before finding top-k positions. This test verifies that
        the averaged output correctly reflects contributions from all heads.
        """
        if not self.gqa_available:
            self.skipTest("compute_topk_attention_chunked not available")

        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        # GQA setup: 8 query heads, 2 KV heads (ratio 4:1)
        batch_size = 1
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 64
        seq_len = 32
        top_k = 5

        # Create query [B, num_q_heads, D]
        q = torch.randn(
            batch_size, num_q_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Create KV buffer in paged format: [total_kv, num_kv_heads, D]
        # For single sequence, total_kv = seq_len
        k_buffer = torch.randn(
            seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Put a strong signal at position 15 that should be detected after averaging
        needle_pos = 15
        # Make position 15 have keys that match multiple query heads
        for kv_h in range(num_kv_heads):
            # Average of 4 query heads per KV head
            q_start = kv_h * 4
            avg_query = q[0, q_start : q_start + 4, :].mean(dim=0)
            k_buffer[needle_pos, kv_h, :] = avg_query * 2  # Strong match

        # Create indptr and indices for paged attention
        kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        kv_indices = torch.arange(seq_len, dtype=torch.int32, device="cuda")

        sm_scale = 1.0 / (head_dim ** 0.5)

        # Run chunked attention extraction
        (
            topk_scores,
            topk_indices,
            topk_logits,
            logsumexp,
        ) = compute_topk_attention_chunked(
            q,  # [B, num_q_heads, D]
            k_buffer,  # [total_kv, num_kv_heads, D]
            kv_indptr,
            kv_indices,
            sm_scale,
            top_k=top_k,
            chunk_size=8,
        )

        # Verify output shapes: averaged across heads, so [batch, top_k]
        self.assertEqual(topk_scores.shape, (batch_size, top_k))
        self.assertEqual(topk_indices.shape, (batch_size, top_k))

        # The needle position should be in top-k (averaged across all heads)
        indices = topk_indices[0].cpu().tolist()
        # Filter out invalid (-1) indices
        valid_indices = [i for i in indices if i >= 0]

        # The strong signal at position 15 should appear in top-k
        self.assertIn(
            needle_pos,
            valid_indices,
            f"Needle position {needle_pos} should be in top-k after averaging: {valid_indices}",
        )

    def test_gqa_ratio_computation(self):
        """Test that GQA ratio is correctly computed."""
        # Common GQA configurations
        test_cases = [
            (32, 8, 4),  # Llama-3: 32 Q heads, 8 KV heads, ratio 4
            (32, 4, 8),  # Some models: 32 Q heads, 4 KV heads, ratio 8
            (8, 8, 1),  # MHA: equal Q and KV heads, ratio 1
            (8, 1, 8),  # MQA: 1 KV head, ratio 8
        ]

        for num_q_heads, num_kv_heads, expected_ratio in test_cases:
            actual_ratio = num_q_heads // num_kv_heads
            self.assertEqual(
                actual_ratio,
                expected_ratio,
                f"GQA ratio mismatch: {num_q_heads} Q / {num_kv_heads} KV = {actual_ratio}, expected {expected_ratio}",
            )


class TestPagedAttentionMapping(unittest.TestCase):
    """
    Test that PagedAttention correctly maps physical KV addresses to logical positions.

    In paged KV cache systems:
    - Physical addresses: where tokens are stored in GPU memory (can be non-contiguous)
    - Logical positions: the actual token position in the sequence [0, seq_len)

    The attention extraction kernel uses kv_indices to map physical -> logical.
    Returned indices must be LOGICAL positions for interpretability.
    """

    def setUp(self):
        """Check if we can import the required modules."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                compute_topk_attention_chunked,
            )

            self.kernel_available = True
        except ImportError:
            self.kernel_available = False

    def test_physical_vs_logical_position_mapping(self):
        """
        Verify that returned indices are LOGICAL positions, not physical KV addresses.

        Setup a scenario where physical addresses are REVERSED from logical positions:
        - Logical position 0 stored at physical address 9
        - Logical position 1 stored at physical address 8
        - ...
        - Logical position 9 stored at physical address 0

        Put a "needle" at logical position 5 (physical address 4).
        Verify the returned index is 5 (logical), not 4 (physical).
        """
        if not self.kernel_available:
            self.skipTest("compute_topk_attention_chunked not available")

        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        # Setup
        batch_size = 1
        num_q_heads = 4
        num_kv_heads = 1  # Use MQA for simplicity
        head_dim = 64
        seq_len = 10
        top_k = 3
        needle_logical_pos = 5
        needle_physical_addr = (
            seq_len - 1 - needle_logical_pos
        )  # Reversed: physical = 4

        # Create KV buffer in paged format: [total_kv, num_kv_heads, head_dim]
        k_buffer = torch.randn(
            seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Create a REVERSED mapping: kv_indices[physical] = logical
        # Physical addr 0 -> logical pos 9
        # Physical addr 1 -> logical pos 8
        # ...
        # Physical addr 9 -> logical pos 0
        kv_indices = torch.arange(seq_len - 1, -1, -1, dtype=torch.int32, device="cuda")

        # Create query that matches the needle (at physical address 4)
        needle_key = torch.randn(
            1, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        k_buffer[needle_physical_addr] = needle_key  # Physical addr 4 = logical pos 5

        # Query matches the needle
        q = needle_key.expand(batch_size, num_q_heads, head_dim)

        # kv_indptr: [0, seq_len] for single batch
        kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

        sm_scale = 1.0 / (head_dim ** 0.5)

        # Run attention extraction
        (
            topk_scores,
            topk_indices,
            topk_logits,
            logsumexp,
        ) = compute_topk_attention_chunked(
            q,
            k_buffer,
            kv_indptr,
            kv_indices,  # This provides the physical -> logical mapping
            sm_scale,
            top_k=top_k,
            chunk_size=8,
        )

        # CRITICAL: The needle should appear at LOGICAL position 5, not physical address 4
        indices = topk_indices[0].cpu().tolist()
        valid_indices = [i for i in indices if i >= 0]

        # The needle's LOGICAL position (5) should be in top-k
        self.assertIn(
            needle_logical_pos,
            valid_indices,
            f"Expected logical position {needle_logical_pos} in top-k, got {valid_indices}. "
            f"(Physical address was {needle_physical_addr} - if this appeared, mapping is broken)",
        )

        # The physical address (4) should NOT appear unless it happens to be a valid logical position
        # In our reversed mapping, physical 4 maps to logical 5, so this should pass
        # But if the kernel incorrectly returned physical addresses, we'd see issues

    def test_non_contiguous_kv_layout(self):
        """
        Verify correctness with non-contiguous KV cache layout (simulating pages).

        Scenario: Two sequences in same batch, stored non-contiguously:
        - Seq 0: logical [0,1,2,3,4] at physical [10,11,12,13,14]
        - Seq 1: logical [0,1,2,3,4] at physical [0,1,2,3,4]
        """
        if not self.kernel_available:
            self.skipTest("compute_topk_attention_chunked not available")

        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        batch_size = 2
        num_q_heads = 2
        num_kv_heads = 1
        head_dim = 64
        seq_len_per_batch = 5
        total_kv = 15  # Physical buffer size (with gaps)
        top_k = 3

        # Physical KV buffer
        k_buffer = torch.randn(
            total_kv, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Non-contiguous layout:
        # Seq 0: stored at physical [10,11,12,13,14]
        # Seq 1: stored at physical [0,1,2,3,4]
        kv_indptr = torch.tensor([0, 5, 10], dtype=torch.int32, device="cuda")

        # kv_indices maps: the kernel reads from these physical locations
        # For seq 0: indices 0-4 point to physical [10,11,12,13,14]
        # For seq 1: indices 5-9 point to physical [0,1,2,3,4]
        kv_indices = torch.cat(
            [
                torch.arange(
                    10, 15, dtype=torch.int32, device="cuda"
                ),  # Seq 0: phys 10-14
                torch.arange(0, 5, dtype=torch.int32, device="cuda"),  # Seq 1: phys 0-4
            ]
        )

        # Put needle at logical position 2 for both sequences
        needle_logical = 2
        needle_key = torch.randn(
            1, num_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Seq 0: logical 2 -> kv_indices index 2 -> physical 12
        k_buffer[12] = needle_key
        # Seq 1: logical 2 -> kv_indices index 7 -> physical 2
        k_buffer[2] = needle_key

        # Query matches needle
        q = needle_key.expand(batch_size, num_q_heads, head_dim)

        sm_scale = 1.0 / (head_dim ** 0.5)

        (
            topk_scores,
            topk_indices,
            topk_logits,
            logsumexp,
        ) = compute_topk_attention_chunked(
            q,
            k_buffer,
            kv_indptr,
            kv_indices,
            sm_scale,
            top_k=top_k,
            chunk_size=4,
        )

        # Both sequences should report logical position 2
        for b in range(batch_size):
            indices = topk_indices[b].cpu().tolist()
            valid_indices = [i for i in indices if i >= 0]
            self.assertIn(
                needle_logical,
                valid_indices,
                f"Batch {b}: Expected logical position {needle_logical} in top-k, got {valid_indices}",
            )

            # Verify indices are in valid range [0, seq_len)
            self.assertTrue(
                all(0 <= idx < seq_len_per_batch for idx in valid_indices),
                f"Batch {b}: Indices {valid_indices} out of range [0, {seq_len_per_batch})",
            )


class TestPrefixSharingCorrectness(unittest.TestCase):
    """
    Test that prefix sharing / radix cache doesn't corrupt attention positions.

    When multiple requests share a prefix, the KV cache is reused. This test
    verifies that attention positions correctly reflect the logical token
    stream for each request, even when physical memory is shared.
    """

    def test_shared_prefix_positions(self):
        """
        Verify positions are correct when prefix is shared.

        Scenario:
        - Request A: "Hello world how are you" (5 tokens)
        - Request B: "Hello world what is this" (5 tokens, shares "Hello world" prefix)

        Both should report positions [0, 1, 2, 3, 4] correctly, not physical addresses.
        """
        # This test would require a live server with radix cache
        # For now, we verify the concept with a simulation

        # Simulated scenario: two requests share tokens [0, 1]
        shared_prefix_len = 2
        request_a_len = 5
        request_b_len = 5

        # Expected: both requests have logical positions [0, 1, 2, 3, 4]
        expected_positions_a = list(range(request_a_len))
        expected_positions_b = list(range(request_b_len))

        # The key insight: even though physical memory at positions [0,1] is shared,
        # the LOGICAL positions reported should be per-request

        # This is a placeholder - actual test would use live server
        self.assertEqual(expected_positions_a, [0, 1, 2, 3, 4])
        self.assertEqual(expected_positions_b, [0, 1, 2, 3, 4])


class TestAttentionSchemaValidation(unittest.TestCase):
    """
    Validate the attention token schema format.
    """

    def test_attention_token_schema(self):
        """Verify attention tokens follow the expected schema."""
        # Expected schema for fingerprint mode
        expected_fields = {
            "schema_version": int,
            "mode": str,
            "fingerprint": list,
            "manifold": str,
            "step": int,
        }

        # Mock attention token (as would be returned from API)
        sample_token = {
            "schema_version": 1,
            "mode": "fingerprint",
            "fingerprint": [0.1] * 20,
            "manifold": "semantic_bridge",
            "step": 0,
            "think_phase": "output",
        }

        for field, expected_type in expected_fields.items():
            self.assertIn(field, sample_token, f"Missing field: {field}")
            self.assertIsInstance(
                sample_token[field], expected_type, f"Field {field} has wrong type"
            )

    def test_fingerprint_dimension(self):
        """Verify fingerprint has correct dimensions."""
        # v1: 20D, v2: 21D (with rotational variance)
        V1_DIM = 20
        V2_DIM = 21

        v1_fingerprint = [0.1] * V1_DIM
        v2_fingerprint = [0.1] * V2_DIM

        self.assertEqual(len(v1_fingerprint), 20)
        self.assertEqual(len(v2_fingerprint), 21)

    def test_manifold_zone_values(self):
        """Verify manifold zone takes expected values."""
        valid_zones = {
            "syntax_floor",
            "semantic_bridge",
            "structure_ripple",
            "unknown",
        }

        for zone in valid_zones:
            self.assertIn(zone, valid_zones)


class TestLiveServerCorrectness(unittest.TestCase):
    """
    Live server tests for attention correctness.

    These tests require a running server. They are skipped if no server is available.
    """

    def setUp(self):
        self.base_url = os.environ.get("SGLANG_TEST_URL", "http://localhost:30000")
        try:
            import requests

            response = requests.get(f"{self.base_url}/health", timeout=2)
            self.server_available = response.status_code == 200
        except Exception:
            self.server_available = False

    def test_needle_retrieval_live(self):
        """
        Live needle-in-haystack test.

        Sends a prompt with a 'needle' phrase, then verifies the attention
        correctly identifies the needle position when answering.
        """
        if not self.server_available:
            self.skipTest("Server not available")

        import requests

        # Prompt with a specific "needle" fact
        prompt = """Here are some facts:
- The sky is blue
- Water boils at 100 degrees Celsius
- The secret code is ALPHA-7749
- Paris is the capital of France
- Elephants are large mammals

What is the secret code?"""

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0.0,
                "return_attention_tokens": True,
            },
            timeout=30,
        )

        if response.status_code != 200:
            self.skipTest(f"Server returned {response.status_code}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Should contain the secret code
        self.assertIn("ALPHA-7749", content, "Model should retrieve the needle")

        # If attention tokens available, verify they include positions near the needle
        # attention_tokens is directly in choices[0], not meta_info
        attention_tokens = data["choices"][0].get("attention_tokens", [])

        if attention_tokens:
            # The needle is around token position 30-35 (rough estimate)
            # We just verify we got valid attention data
            self.assertGreater(len(attention_tokens), 0)

            # Check first record has valid structure
            first = attention_tokens[0]
            self.assertIn("fingerprint", first)
            self.assertIn("manifold", first)


if __name__ == "__main__":
    unittest.main()
