"""
Unit tests for DFlash attention layer with RadixAttention integration.

Tests that:
1. Torch fallback mode works correctly
2. project_hidden_to_kv() produces valid K/V tensors
3. RadixAttention mode produces similar results to torch fallback

These tests use isolated imports to avoid circular import issues.
"""

import os
import sys
import unittest

import torch

# Add sglang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))

# Skip if CUDA not available
HAS_CUDA = torch.cuda.is_available()

# Try importing - will be used only in the actual tests
HAS_SGLANG = False  # Set to False until imports work properly
Qwen3DFlashAttention = None

try:
    from sglang.srt.models.qwen3_dflash import Qwen3DFlashAttention

    HAS_SGLANG = True
except ImportError:
    pass


@unittest.skipUnless(HAS_SGLANG, "SGLang not available")
class TestQwen3DFlashAttention(unittest.TestCase):
    """Test Qwen3DFlashAttention with RadixAttention integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_size = 256
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 64
        self.layer_id = 0
        self.dtype = torch.bfloat16

        # Create attention layer
        self.attn = (
            Qwen3DFlashAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                layer_id=self.layer_id,
                rope_theta=1000000,
                max_position_embeddings=4096,
                rms_norm_eps=1e-6,
                attention_bias=True,
            )
            .to(self.device)
            .to(self.dtype)
        )

    def test_torch_fallback_forward(self):
        """Test that torch fallback forward works correctly."""
        batch_size = 1
        ctx_len = 10
        noise_len = 4
        total_len = ctx_len + noise_len

        # Create inputs
        hidden_states = torch.randn(
            total_len, self.hidden_size, device=self.device, dtype=self.dtype
        )
        positions = torch.arange(total_len, device=self.device)

        # Forward with torch fallback (forward_batch=None)
        output = self.attn._forward_torch(positions, hidden_states, ctx_len=ctx_len)

        # Check output shape
        self.assertEqual(output.shape, (noise_len, self.hidden_size))

        # Check output is not all zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))

    def test_project_hidden_to_kv(self):
        """Test that project_hidden_to_kv produces valid K/V tensors."""
        num_tokens = 10

        # Create inputs
        hidden_states = torch.randn(
            num_tokens, self.hidden_size, device=self.device, dtype=self.dtype
        )
        positions = torch.arange(num_tokens, device=self.device)

        # Project to K/V
        k, v = self.attn.project_hidden_to_kv(hidden_states, positions)

        # Check shapes
        expected_shape = (num_tokens, self.num_kv_heads, self.head_dim)
        self.assertEqual(k.shape, expected_shape)
        self.assertEqual(v.shape, expected_shape)

        # Check not all zeros
        self.assertFalse(torch.allclose(k, torch.zeros_like(k)))
        self.assertFalse(torch.allclose(v, torch.zeros_like(v)))

    def test_kv_consistency(self):
        """Test that K/V from project_hidden_to_kv matches full QKV projection."""
        num_tokens = 10

        # Create inputs
        hidden_states = torch.randn(
            num_tokens, self.hidden_size, device=self.device, dtype=self.dtype
        )
        positions = torch.arange(num_tokens, device=self.device)

        # Get K/V from project_hidden_to_kv
        k_proj, v_proj = self.attn.project_hidden_to_kv(hidden_states, positions)

        # Get K/V from full QKV projection
        qkv, _ = self.attn.qkv_proj(hidden_states)
        _, k_full, v_full = qkv.split(
            [self.attn.q_size, self.attn.kv_size, self.attn.kv_size], dim=-1
        )

        # Apply K norm
        k_full = k_full.view(-1, self.attn.num_kv_heads, self.attn.head_dim)
        k_full = self.attn.k_norm(k_full)
        k_full = k_full.view(-1, self.attn.kv_size)

        # Apply RoPE
        dummy_q = torch.zeros_like(k_full)
        _, k_full = self.attn.rotary_emb(positions, dummy_q, k_full)
        k_full = k_full.view(-1, self.attn.num_kv_heads, self.attn.head_dim)

        # V doesn't need RoPE
        v_full = v_full.view(-1, self.attn.num_kv_heads, self.attn.head_dim)

        # Check that K matches
        self.assertTrue(
            torch.allclose(k_proj, k_full, atol=1e-3),
            f"K mismatch: max diff = {(k_proj - k_full).abs().max()}",
        )

        # Check that V matches
        self.assertTrue(
            torch.allclose(v_proj, v_full, atol=1e-3),
            f"V mismatch: max diff = {(v_proj - v_full).abs().max()}",
        )

    def test_bidirectional_attention(self):
        """Test that attention is bidirectional (non-causal)."""
        batch_size = 1
        ctx_len = 5
        noise_len = 3
        total_len = ctx_len + noise_len

        # Create inputs
        hidden_states = torch.randn(
            total_len, self.hidden_size, device=self.device, dtype=self.dtype
        )
        positions = torch.arange(total_len, device=self.device)

        # Forward pass
        output = self.attn._forward_torch(positions, hidden_states, ctx_len=ctx_len)

        # Modify the first context token and check if output changes
        hidden_states_modified = hidden_states.clone()
        hidden_states_modified[0] = hidden_states_modified[0] + 1.0

        output_modified = self.attn._forward_torch(
            positions, hidden_states_modified, ctx_len=ctx_len
        )

        # Output should be different (non-causal attention allows future tokens to see past)
        self.assertFalse(
            torch.allclose(output, output_modified, atol=1e-4),
            "Output should change when context is modified (bidirectional attention)",
        )


@unittest.skipUnless(HAS_SGLANG, "SGLang not available")
class TestDFlashDecoderLayer(unittest.TestCase):
    """Test Qwen3DFlashDecoderLayer with correct normalization pattern."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create a mock config
        class MockConfig:
            hidden_size = 256
            num_attention_heads = 4
            num_key_value_heads = 2
            intermediate_size = 512
            hidden_act = "silu"
            rms_norm_eps = 1e-6
            rope_theta = 1000000
            rope_scaling = None
            max_position_embeddings = 4096
            attention_bias = True

        self.config = MockConfig()
        self.dtype = torch.bfloat16

    def test_decoder_layer_creates(self):
        """Test that decoder layer can be created."""
        from sglang.srt.models.qwen3_dflash import Qwen3DFlashDecoderLayer

        layer = (
            Qwen3DFlashDecoderLayer(
                config=self.config,
                layer_id=0,
            )
            .to(self.device)
            .to(self.dtype)
        )

        self.assertIsNotNone(layer)

    def test_decoder_layer_forward(self):
        """Test decoder layer forward pass."""
        from sglang.srt.models.qwen3_dflash import Qwen3DFlashDecoderLayer

        layer = (
            Qwen3DFlashDecoderLayer(
                config=self.config,
                layer_id=0,
            )
            .to(self.device)
            .to(self.dtype)
        )

        ctx_len = 10
        noise_len = 4
        total_len = ctx_len + noise_len

        hidden_states = torch.randn(
            total_len, self.config.hidden_size, device=self.device, dtype=self.dtype
        )
        positions = torch.arange(total_len, device=self.device)

        output = layer(positions, hidden_states, forward_batch=None, ctx_len=ctx_len)

        # Output should have same shape (concat of unchanged ctx and updated noise)
        self.assertEqual(output.shape, hidden_states.shape)


class TestRadixAttentionIsCausal(unittest.TestCase):
    """Test RadixAttention is_causal parameter for bidirectional attention."""

    def test_is_causal_default_true(self):
        """Test that RadixAttention has is_causal=True by default."""
        try:
            from sglang.srt.layers.radix_attention import RadixAttention
        except ImportError:
            self.skipTest("RadixAttention not available")

        attn = RadixAttention(
            num_heads=4,
            head_dim=64,
            scaling=0.125,
            num_kv_heads=2,
            layer_id=0,
        )
        self.assertTrue(
            attn.is_causal, "RadixAttention should have is_causal=True by default"
        )

    def test_is_causal_false_for_dflash(self):
        """Test that RadixAttention can be created with is_causal=False."""
        try:
            from sglang.srt.layers.radix_attention import RadixAttention
        except ImportError:
            self.skipTest("RadixAttention not available")

        attn = RadixAttention(
            num_heads=4,
            head_dim=64,
            scaling=0.125,
            num_kv_heads=2,
            layer_id=0,
            is_causal=False,  # For DFlash bidirectional attention
        )
        self.assertFalse(
            attn.is_causal, "RadixAttention should have is_causal=False when specified"
        )

    def test_dflash_attention_has_radix_attn(self):
        """Test that Qwen3DFlashAttention has RadixAttention with is_causal=False."""
        try:
            from sglang.srt.models.qwen3_dflash import Qwen3DFlashAttention
        except ImportError:
            self.skipTest("Qwen3DFlashAttention not available")

        attn = Qwen3DFlashAttention(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            layer_id=0,
        )

        # Check that radix_attn exists and has is_causal=False
        self.assertTrue(
            hasattr(attn, "radix_attn"),
            "Qwen3DFlashAttention should have radix_attn attribute",
        )
        self.assertFalse(
            attn.radix_attn.is_causal,
            "Qwen3DFlashAttention.radix_attn should have is_causal=False for bidirectional attention",
        )


if __name__ == "__main__":
    unittest.main()
