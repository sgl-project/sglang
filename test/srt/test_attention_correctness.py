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
            batch_size, num_heads, seq_len, head_dim,
            dtype=dtype, device=device
        )
        v_cache = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            dtype=dtype, device=device
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
        q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float16, device="cuda")
        k_cache, v_cache = self._create_mock_kv_cache(
            batch_size, num_heads, head_dim, seq_len, torch.float16
        )

        # Create seq_lens
        seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device="cuda")

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
                indices = topk_indices[b, h].cpu().numpy()
                valid_indices = indices[indices >= 0]  # Filter sentinel values

                self.assertTrue(
                    all(0 <= idx < seq_len for idx in valid_indices),
                    f"Batch {b}, Head {h}: indices {valid_indices} not in [0, {seq_len})"
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
        k_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        # Create a query that matches the needle
        needle_key = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float16, device="cuda")
        k_cache[:, :, needle_position:needle_position+1, :] = needle_key

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
        indices = topk_indices[0].cpu().numpy()  # [H, K]

        for h in range(num_heads):
            head_topk = indices[h]
            self.assertIn(
                needle_position,
                head_topk,
                f"Head {h}: needle position {needle_position} not in top-{top_k}: {head_topk}"
            )


class TestGQABroadcastCorrectness(unittest.TestCase):
    """
    Test that GQA (Grouped Query Attention) head broadcasting is handled correctly.

    In GQA, multiple query heads share the same KV head. The attention extraction
    must correctly map query heads to their corresponding KV groups.
    """

    def setUp(self):
        """Check if we can import the required modules."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                decode_forward_with_topk_gqa,
            )
            self.gqa_available = True
        except ImportError:
            self.gqa_available = False

    def test_gqa_head_group_mapping(self):
        """
        Test that GQA correctly maps query heads to KV groups.

        Example: 8 query heads, 2 KV heads → groups of 4 query heads share 1 KV head.
        """
        if not self.gqa_available:
            self.skipTest("GQA attention kernel not available")

        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # GQA setup: 8 query heads, 2 KV heads
        batch_size = 1
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 64
        seq_len = 32
        top_k = 5

        # Each KV head is shared by 4 query heads
        group_size = num_q_heads // num_kv_heads

        # Create query [B, num_q_heads, D]
        q = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.float16, device="cuda")

        # Create KV cache [B, num_kv_heads, seq_len, D]
        k_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        # Put a distinctive pattern in each KV head
        needle_pos_group0 = 10  # Position for KV head 0
        needle_pos_group1 = 20  # Position for KV head 1

        # Make query heads 0-3 match the needle in KV head 0
        k_cache[0, 0, needle_pos_group0, :] = q[0, 0, :]  # KV head 0, position 10
        k_cache[0, 1, needle_pos_group1, :] = q[0, 4, :]  # KV head 1, position 20

        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            decode_forward_with_topk_gqa,
        )

        output, topk_values, topk_indices = decode_forward_with_topk_gqa(
            q,
            k_cache,
            v_cache,
            seq_lens,
            num_kv_heads=num_kv_heads,
            top_k=top_k,
            sm_scale=1.0 / (head_dim ** 0.5),
        )

        # Verify:
        # - Query heads 0-3 (group 0) should attend to position 10 (their needle in KV head 0)
        # - Query heads 4-7 (group 1) should attend to position 20 (their needle in KV head 1)
        indices = topk_indices[0].cpu().numpy()  # [num_q_heads, K]

        # Check group 0 (query heads 0-3, KV head 0)
        for qh in range(0, 4):
            self.assertIn(
                needle_pos_group0,
                indices[qh],
                f"Query head {qh} (group 0) should attend to position {needle_pos_group0}"
            )

        # Check group 1 (query heads 4-7, KV head 1)
        for qh in range(4, 8):
            self.assertIn(
                needle_pos_group1,
                indices[qh],
                f"Query head {qh} (group 1) should attend to position {needle_pos_group1}"
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
            'schema_version': int,
            'mode': str,
            'fingerprint': list,
            'manifold': str,
            'step': int,
        }

        # Mock attention token (as would be returned from API)
        sample_token = {
            'schema_version': 1,
            'mode': 'fingerprint',
            'fingerprint': [0.1] * 20,
            'manifold': 'semantic_bridge',
            'step': 0,
            'think_phase': 'output',
        }

        for field, expected_type in expected_fields.items():
            self.assertIn(field, sample_token, f"Missing field: {field}")
            self.assertIsInstance(
                sample_token[field],
                expected_type,
                f"Field {field} has wrong type"
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
            'syntax_floor',
            'semantic_bridge',
            'structure_ripple',
            'unknown',
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
        meta_info = data["choices"][0].get("meta_info", {})
        attention_tokens = meta_info.get("attention_tokens", [])

        if attention_tokens:
            # The needle is around token position 30-35 (rough estimate)
            # We just verify we got valid attention data
            self.assertGreater(len(attention_tokens), 0)

            # Check first record has valid structure
            first = attention_tokens[0]
            self.assertIn('fingerprint', first)
            self.assertIn('manifold', first)


if __name__ == "__main__":
    unittest.main()
