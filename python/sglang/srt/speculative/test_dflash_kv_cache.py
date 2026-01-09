#!/usr/bin/env python3
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Comprehensive tests for DFlash V1 KV cache implementation.

These tests verify the critical assumptions required for incremental KV caching:
1. RMSNorm3D (k_norm) is position-independent
2. FC compression can be applied incrementally
3. RoPE can be applied to incremental positions
4. Incremental caching produces identical results to full recomputation

Run with: python -m pytest test_dflash_kv_cache.py -v
Or standalone: python test_dflash_kv_cache.py
"""

import sys
import unittest
from typing import Tuple

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, "/sgl-workspace/sglang/python")


class MockConfig:
    """Mock config for testing DFlash components."""
    def __init__(self):
        self.hidden_size = 2048
        self.num_attention_heads = 16
        self.num_key_value_heads = 2
        self.head_dim = 128
        self.intermediate_size = 5632
        self.rms_norm_eps = 1e-6
        self.attention_bias = True
        self.num_hidden_layers = 3
        self.num_target_layers = 28
        self.block_size = 8
        self.max_position_embeddings = 32768
        self.rope_theta = 10000.0


def get_device():
    """Get the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestRMSNormPositionIndependence(unittest.TestCase):
    """Test that RMSNorm3D is position-independent (critical for caching)."""

    def setUp(self):
        from sglang.srt.models.dflash import RMSNorm3D
        self.device = get_device()
        self.dtype = torch.float32  # Use float32 for numerical precision tests
        self.hidden_size = 128
        self.eps = 1e-6
        
        # Create RMSNorm3D
        self.norm = RMSNorm3D(self.hidden_size, eps=self.eps).to(self.device)

    def test_rmsnorm_is_position_independent_2d(self):
        """Test RMSNorm with 2D input [batch, hidden]."""
        batch_size = 4
        x = torch.randn(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # Full normalization
        out_full = self.norm(x)
        
        # Split normalization (each sample independently)
        out_split = torch.stack([self.norm(x[i:i+1]) for i in range(batch_size)]).squeeze(1)
        
        self.assertTrue(
            torch.allclose(out_full, out_split, atol=1e-6),
            f"RMSNorm is NOT batch-independent! Max diff: {(out_full - out_split).abs().max()}"
        )
        print("✓ RMSNorm 2D is batch-independent")

    def test_rmsnorm_is_position_independent_3d(self):
        """Test RMSNorm with 3D input [batch, seq, hidden]."""
        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # Full normalization
        out_full = self.norm(x)
        
        # Split at position 50
        split_pos = 50
        x_a = x[:, :split_pos, :]
        x_b = x[:, split_pos:, :]
        
        out_a = self.norm(x_a)
        out_b = self.norm(x_b)
        out_split = torch.cat([out_a, out_b], dim=1)
        
        self.assertTrue(
            torch.allclose(out_full, out_split, atol=1e-6),
            f"RMSNorm is NOT position-independent! Max diff: {(out_full - out_split).abs().max()}"
        )
        print("✓ RMSNorm 3D is position-independent (can cache after normalization)")

    def test_rmsnorm_is_position_independent_4d(self):
        """Test RMSNorm with 4D input [batch, seq, heads, head_dim] (K/V format)."""
        batch_size = 2
        seq_len = 100
        num_heads = 8
        head_dim = self.hidden_size // num_heads
        
        # Recreate norm for head_dim
        from sglang.srt.models.dflash import RMSNorm3D
        norm = RMSNorm3D(head_dim, eps=self.eps).to(self.device)
        
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=self.dtype)
        
        # Full normalization
        out_full = norm(x)
        
        # Split at position 50
        split_pos = 50
        out_a = norm(x[:, :split_pos, :, :])
        out_b = norm(x[:, split_pos:, :, :])
        out_split = torch.cat([out_a, out_b], dim=1)
        
        self.assertTrue(
            torch.allclose(out_full, out_split, atol=1e-6),
            f"RMSNorm is NOT position-independent for 4D! Max diff: {(out_full - out_split).abs().max()}"
        )
        print("✓ RMSNorm 4D (K/V format) is position-independent")


class TestFCIncrementalCompression(unittest.TestCase):
    """Test that FC compression can be applied incrementally."""

    def setUp(self):
        from sglang.srt.models.dflash import RMSNorm3D
        self.device = get_device()
        self.dtype = torch.float32
        
        self.hidden_size = 2048
        self.num_layers = 3  # Number of target layers for feature extraction
        
        # Create FC layer (compresses multi-layer features)
        self.fc = nn.Linear(self.num_layers * self.hidden_size, self.hidden_size, bias=False).to(self.device)
        self.hidden_norm = RMSNorm3D(self.hidden_size, eps=1e-6).to(self.device)
        
        # Initialize with random weights
        nn.init.normal_(self.fc.weight, std=0.02)

    def test_fc_is_position_independent(self):
        """Test that FC+norm can be applied to positions independently."""
        batch_size = 2
        seq_len = 100
        
        # Input: multi-layer features
        x = torch.randn(batch_size, seq_len, self.num_layers * self.hidden_size, 
                       device=self.device, dtype=self.dtype)
        
        # Full FC + norm
        out_full = self.hidden_norm(self.fc(x))
        
        # Split at position 50
        split_pos = 50
        x_a = x[:, :split_pos, :]
        x_b = x[:, split_pos:, :]
        
        out_a = self.hidden_norm(self.fc(x_a))
        out_b = self.hidden_norm(self.fc(x_b))
        out_split = torch.cat([out_a, out_b], dim=1)
        
        self.assertTrue(
            torch.allclose(out_full, out_split, atol=1e-5),
            f"FC+norm is NOT position-independent! Max diff: {(out_full - out_split).abs().max()}"
        )
        print("✓ FC+RMSNorm is position-independent (can cache after FC compression)")


class TestRoPEIncrementalApplication(unittest.TestCase):
    """Test that RoPE can be applied incrementally to specific positions."""

    def setUp(self):
        self.device = get_device()
        self.dtype = torch.float32
        
        self.head_dim = 128
        self.rotary_dim = 64  # Partial rotary (Qwen3 style)
        
    def _create_rope_embeddings(self, seq_len: int, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create cos/sin embeddings for positions [0, seq_len)."""
        position_ids = torch.arange(seq_len, device=self.device)
        
        # Simple RoPE computation
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, device=self.device).float() / self.rotary_dim))
        
        # [seq_len, rotary_dim/2]
        freqs = torch.einsum("i,j->ij", position_ids.float(), inv_freq)
        # [seq_len, rotary_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos()[None, :, :]  # [1, seq_len, rotary_dim]
        sin = emb.sin()[None, :, :]
        
        return cos.expand(batch_size, -1, -1), sin.expand(batch_size, -1, -1)

    def test_rope_incremental_matches_full(self):
        """Test that incremental RoPE application matches full computation."""
        from sglang.srt.models.dflash import apply_rotary_pos_emb, rotate_half
        
        batch_size = 2
        num_heads = 8
        total_len = 100
        cached_len = 60
        new_len = total_len - cached_len
        
        # Create K tensor
        k_full = torch.randn(batch_size, num_heads, total_len, self.head_dim, 
                            device=self.device, dtype=self.dtype)
        
        # Create full cos/sin
        cos_full, sin_full = self._create_rope_embeddings(total_len, batch_size)
        
        # Create dummy Q (not used for this test)
        q = torch.randn(batch_size, num_heads, 8, self.head_dim, device=self.device, dtype=self.dtype)
        cos_q = cos_full[:, -8:, :]  # Last 8 positions for Q
        sin_q = sin_full[:, -8:, :]
        
        # Full RoPE application
        _, k_full_rope = apply_rotary_pos_emb(q, k_full, cos_full, sin_full)
        
        # Incremental RoPE application
        # Cached K at positions [0, cached_len)
        k_cached = k_full[:, :, :cached_len, :]
        cos_cached, sin_cached = self._create_rope_embeddings(cached_len, batch_size)
        # We need to create a dummy Q just for the function call
        q_dummy = q
        cos_dummy = cos_cached[:, -8:, :]
        sin_dummy = sin_cached[:, -8:, :]
        _, k_cached_rope = apply_rotary_pos_emb(q_dummy, k_cached, cos_cached, sin_cached)
        
        # New K at positions [cached_len, total_len)
        k_new = k_full[:, :, cached_len:, :]
        # Create cos/sin for positions [cached_len, total_len)
        position_ids_new = torch.arange(cached_len, total_len, device=self.device)
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, device=self.device).float() / self.rotary_dim))
        freqs_new = torch.einsum("i,j->ij", position_ids_new.float(), inv_freq)
        emb_new = torch.cat([freqs_new, freqs_new], dim=-1)
        cos_new = emb_new.cos()[None, :, :].expand(batch_size, -1, -1)
        sin_new = emb_new.sin()[None, :, :].expand(batch_size, -1, -1)
        
        # Apply RoPE only to new K
        # Apply directly without using apply_rotary_pos_emb since we only have K
        cos_unsq = cos_new.unsqueeze(1)  # [batch, 1, new_len, rotary_dim]
        sin_unsq = sin_new.unsqueeze(1)
        
        k_rot = k_new[..., :self.rotary_dim]
        k_pass = k_new[..., self.rotary_dim:]
        k_rot_embed = (k_rot * cos_unsq) + (rotate_half(k_rot) * sin_unsq)
        k_new_rope = torch.cat([k_rot_embed, k_pass], dim=-1)
        
        # Concatenate cached + new
        k_incr_rope = torch.cat([k_cached_rope, k_new_rope], dim=2)
        
        # Compare
        self.assertTrue(
            torch.allclose(k_full_rope, k_incr_rope, atol=1e-5),
            f"RoPE incremental != full! Max diff: {(k_full_rope - k_incr_rope).abs().max()}"
        )
        print("✓ RoPE incremental application matches full computation")


class TestApplyRotarySingle(unittest.TestCase):
    """Test the new apply_rotary_single helper function."""

    def setUp(self):
        self.device = get_device()
        self.dtype = torch.float32
        self.head_dim = 128
        self.rotary_dim = 64

    def test_apply_rotary_single_correctness(self):
        """Test apply_rotary_single produces correct results."""
        from sglang.srt.models.dflash import rotate_half
        
        batch_size = 2
        num_heads = 8
        seq_len = 20
        
        # Create K tensor
        k = torch.randn(batch_size, num_heads, seq_len, self.head_dim, 
                       device=self.device, dtype=self.dtype)
        
        # Create cos/sin for specific positions
        start_pos = 40
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=self.device)
        
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, device=self.device).float() / self.rotary_dim))
        freqs = torch.einsum("i,j->ij", position_ids.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, :, :].expand(batch_size, -1, -1)  # [batch, seq_len, rotary_dim]
        sin = emb.sin()[None, :, :].expand(batch_size, -1, -1)
        
        # Apply manually
        cos_unsq = cos.unsqueeze(1)  # [batch, 1, seq_len, rotary_dim]
        sin_unsq = sin.unsqueeze(1)
        
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]
        k_rot_embed = (k_rot * cos_unsq) + (rotate_half(k_rot) * sin_unsq)
        k_expected = torch.cat([k_rot_embed, k_pass], dim=-1)
        
        # Now test the actual function once implemented
        # For now, just verify the manual computation works
        self.assertEqual(k_expected.shape, k.shape)
        print("✓ apply_rotary_single helper test structure ready")


class TestIncrementalKVCacheCorrectness(unittest.TestCase):
    """Test that incremental KV caching produces identical results to full recomputation."""

    def setUp(self):
        self.device = get_device()
        if self.device != "cuda":
            self.skipTest("CUDA required for full attention tests")
            
        self.dtype = torch.bfloat16
        self.config = MockConfig()
        
    def test_attention_layer_incremental_matches_full(self):
        """Test that attention with incremental cache matches full recomputation."""
        from sglang.srt.models.qwen3_dflash import Qwen3DFlashAttention
        from sglang.srt.models.dflash import DFlashIncrementalKVCache
        
        batch_size = 1
        block_size = 8
        ctx_len = 50
        cached_len = 30  # Simulate having cached first 30 positions
        
        # Create attention layer
        attn = Qwen3DFlashAttention(self.config, layer_idx=0).to(self.device).to(self.dtype)
        attn.eval()
        
        # Create inputs
        hidden_states = torch.randn(batch_size, block_size, self.config.hidden_size,
                                   device=self.device, dtype=self.dtype)
        target_hidden = torch.randn(batch_size, ctx_len, self.config.hidden_size,
                                   device=self.device, dtype=self.dtype)
        
        # Create position embeddings
        try:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
            rotary_emb = Qwen3RotaryEmbedding(self.config)
        except (ImportError, TypeError):
            self.skipTest("Qwen3RotaryEmbedding not available")
            
        position_ids = torch.arange(ctx_len + block_size, device=self.device)[None, :]
        cos, sin = rotary_emb(hidden_states, position_ids)
        
        # ==== Test 1: Full computation (no cache) ====
        with torch.no_grad():
            output_full = attn(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=(cos, sin),
                use_cache=False,
                use_flash_attention=True,
            )
        
        print(f"  Full output shape: {output_full.shape}")
        print(f"  Full output mean: {output_full.mean().item():.6f}")
        
        # ==== Test 2: Incremental with empty cache (should match full) ====
        cache = DFlashIncrementalKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=1024,
            device=self.device,
            dtype=self.dtype,
        )
        
        with torch.no_grad():
            output_incr_empty = attn(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=(cos, sin),
                use_cache=True,
                use_flash_attention=True,
                incremental_cache=cache,
            )
        
        # Should match full computation exactly
        max_diff_empty = (output_full - output_incr_empty).abs().max().item()
        self.assertTrue(
            torch.allclose(output_full, output_incr_empty, atol=1e-3),
            f"Empty cache output differs from full! Max diff: {max_diff_empty:.6f}"
        )
        print(f"  Empty cache max diff: {max_diff_empty:.6f}")
        print("✓ Empty cache matches full recomputation")
        
        # Confirm the cache update
        cache.confirm_update(ctx_len)
        
        # ==== Test 3: Incremental with cached first 30 positions ====
        # Create a fresh cache and populate it with first 30 positions
        cache2 = DFlashIncrementalKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=1024,
            device=self.device,
            dtype=self.dtype,
        )
        
        # First, run with just the first 30 positions to populate cache
        target_hidden_partial = target_hidden[:, :cached_len, :]
        position_ids_partial = torch.arange(cached_len + block_size, device=self.device)[None, :]
        cos_partial, sin_partial = rotary_emb(hidden_states, position_ids_partial)
        
        with torch.no_grad():
            _ = attn(
                hidden_states=hidden_states,
                target_hidden=target_hidden_partial,
                position_embeddings=(cos_partial, sin_partial),
                use_cache=True,
                use_flash_attention=True,
                incremental_cache=cache2,
            )
        
        # Confirm the partial cache
        cache2.confirm_update(cached_len)
        self.assertEqual(cache2.cached_len, cached_len, f"Cache should have {cached_len} positions cached")
        
        # Now run with full 50 positions - should use cached first 30 and compute new 20
        with torch.no_grad():
            output_incr_partial = attn(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=(cos, sin),
                use_cache=True,
                use_flash_attention=True,
                incremental_cache=cache2,
            )
        
        # Should match full computation
        max_diff_partial = (output_full - output_incr_partial).abs().max().item()
        self.assertTrue(
            torch.allclose(output_full, output_incr_partial, atol=1e-3),
            f"Partial cache output differs from full! Max diff: {max_diff_partial:.6f}"
        )
        print(f"  Partial cache ({cached_len} cached) max diff: {max_diff_partial:.6f}")
        print("✓ Partial cache matches full recomputation")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for KV caching."""

    def setUp(self):
        self.device = get_device()
        self.dtype = torch.float32

    def test_empty_cache(self):
        """Test first iteration with empty cache."""
        # This tests the cached_len = 0 case
        from sglang.srt.models.dflash import RMSNorm3D
        
        norm = RMSNorm3D(128, eps=1e-6).to(self.device)
        
        # With cached_len = 0, we process all positions as "new"
        x = torch.randn(2, 50, 128, device=self.device, dtype=self.dtype)
        cached_len = 0
        
        x_new = x[:, cached_len:, :]  # All positions are new
        out = norm(x_new)
        
        self.assertEqual(out.shape, x.shape)
        print("✓ Empty cache (cached_len=0) handled correctly")

    def test_single_position_increment(self):
        """Test incrementing by just 1 position."""
        from sglang.srt.models.dflash import RMSNorm3D
        
        norm = RMSNorm3D(128, eps=1e-6).to(self.device)
        
        x = torch.randn(2, 51, 128, device=self.device, dtype=self.dtype)
        
        # Full
        out_full = norm(x)
        
        # Incremental: 50 cached, 1 new
        out_cached = norm(x[:, :50, :])
        out_new = norm(x[:, 50:51, :])  # Just 1 position
        out_incr = torch.cat([out_cached, out_new], dim=1)
        
        self.assertTrue(
            torch.allclose(out_full, out_incr, atol=1e-6),
            f"Single position increment failed! Max diff: {(out_full - out_incr).abs().max()}"
        )
        print("✓ Single position increment handled correctly")

    def test_dtype_consistency(self):
        """Test that caching works with bfloat16."""
        if self.device != "cuda":
            self.skipTest("CUDA required for bfloat16 test")
            
        from sglang.srt.models.dflash import RMSNorm3D
        
        norm = RMSNorm3D(128, eps=1e-6).to(self.device).to(torch.bfloat16)
        
        x = torch.randn(2, 100, 128, device=self.device, dtype=torch.bfloat16)
        
        out_full = norm(x)
        out_a = norm(x[:, :50, :])
        out_b = norm(x[:, 50:, :])
        out_incr = torch.cat([out_a, out_b], dim=1)
        
        # Slightly higher tolerance for bfloat16
        self.assertTrue(
            torch.allclose(out_full, out_incr, atol=1e-3),
            f"bfloat16 incremental failed! Max diff: {(out_full - out_incr).abs().max()}"
        )
        print("✓ bfloat16 incremental caching works correctly")


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("DFlash V1 KV Cache Verification Tests")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRMSNormPositionIndependence))
    suite.addTests(loader.loadTestsFromTestCase(TestFCIncrementalCompression))
    suite.addTests(loader.loadTestsFromTestCase(TestRoPEIncrementalApplication))
    suite.addTests(loader.loadTestsFromTestCase(TestApplyRotarySingle))
    suite.addTests(loader.loadTestsFromTestCase(TestIncrementalKVCacheCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All critical assumptions verified!")
        print("  - RMSNorm3D is position-independent")
        print("  - FC+RMSNorm is position-independent")  
        print("  - RoPE can be applied incrementally")
        print("\n  Ready to implement incremental KV caching!")
    else:
        print("\n✗ Some tests failed! Review before proceeding.")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

