"""Bit-parity test for the fused RoPE + KV-cache SYCL kernel on XPU.

Compares the fused `apply_rope_inplace_with_kvcache_xpu` against the
reference unfused path (separate RoPE + explicit cache write).
"""

import unittest

import torch


def reference_rope_neox(q, k, cos_sin_cache, positions):
    """Reference RoPE (NeoX style) using pure PyTorch — rotates Q and K in-place."""
    num_tokens = q.shape[0]
    head_dim = q.shape[2]
    rot_dim = cos_sin_cache.shape[1]
    embed_dim = rot_dim // 2

    for t in range(num_tokens):
        pos = positions[t].item()
        cos_vals = cos_sin_cache[pos, :embed_dim].float()
        sin_vals = cos_sin_cache[pos, embed_dim:].float()

        for h in range(q.shape[1]):
            x = q[t, h, :embed_dim].float()
            y = q[t, h, embed_dim:rot_dim].float()
            q[t, h, :embed_dim] = (x * cos_vals - y * sin_vals).to(q.dtype)
            q[t, h, embed_dim:rot_dim] = (x * sin_vals + y * cos_vals).to(q.dtype)

        for h in range(k.shape[1]):
            x = k[t, h, :embed_dim].float()
            y = k[t, h, embed_dim:rot_dim].float()
            k[t, h, :embed_dim] = (x * cos_vals - y * sin_vals).to(k.dtype)
            k[t, h, embed_dim:rot_dim] = (x * sin_vals + y * cos_vals).to(k.dtype)


class TestFusedRopeKVCache(unittest.TestCase):
    def setUp(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        torch.manual_seed(42)

    def _run_parity(self, num_tokens, n_q_heads, n_kv_heads, head_dim, cache_size, is_neox=True):
        rot_dim = head_dim
        max_pos = 8192

        q_ref = torch.randn(num_tokens, n_q_heads, head_dim, dtype=torch.bfloat16, device="xpu")
        k_ref = torch.randn(num_tokens, n_kv_heads, head_dim, dtype=torch.bfloat16, device="xpu")
        v_ref = torch.randn(num_tokens, n_kv_heads, head_dim, dtype=torch.bfloat16, device="xpu")
        cos_sin_cache = torch.randn(max_pos, rot_dim, dtype=torch.float32, device="xpu")
        positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device="xpu")
        out_loc = torch.randperm(cache_size, device="xpu")[:num_tokens].to(torch.int64)

        k_cache_ref = torch.zeros(cache_size, n_kv_heads * head_dim, dtype=torch.bfloat16, device="xpu")
        v_cache_ref = torch.zeros_like(k_cache_ref)
        k_cache_fused = torch.zeros_like(k_cache_ref)
        v_cache_fused = torch.zeros_like(k_cache_ref)

        q_fused = q_ref.clone()
        k_fused = k_ref.clone()
        v_fused = v_ref.clone()

        # Reference: unfused RoPE + explicit cache write
        reference_rope_neox(q_ref, k_ref, cos_sin_cache, positions)
        k_cache_ref[out_loc] = k_ref.view(num_tokens, -1)
        v_cache_ref[out_loc] = v_ref.view(num_tokens, -1)

        # Fused kernel
        from sgl_kernel import apply_rope_inplace_with_kvcache_xpu

        apply_rope_inplace_with_kvcache_xpu(
            q_fused, k_fused, v_fused,
            k_cache_fused, v_cache_fused,
            cos_sin_cache, positions, out_loc,
            is_neox=is_neox,
        )

        torch.testing.assert_close(q_fused, q_ref, rtol=1e-2, atol=1e-3,
                                   msg="Q mismatch between fused and reference")
        torch.testing.assert_close(k_fused, k_ref, rtol=1e-2, atol=1e-3,
                                   msg="K mismatch between fused and reference")
        torch.testing.assert_close(k_cache_fused, k_cache_ref, rtol=1e-2, atol=1e-3,
                                   msg="K-cache mismatch between fused and reference")
        torch.testing.assert_close(v_cache_fused, v_cache_ref, rtol=1e-2, atol=1e-3,
                                   msg="V-cache mismatch between fused and reference")

    def test_typical_gemma4_decode(self):
        """Gemma 4 31B: 16 Q heads, 8 KV heads, head_dim=128, decode bs=1."""
        self._run_parity(num_tokens=1, n_q_heads=16, n_kv_heads=8, head_dim=128, cache_size=2048)

    def test_batch_decode(self):
        """Multiple tokens in a batch (e.g., prefill or batch decode)."""
        self._run_parity(num_tokens=32, n_q_heads=16, n_kv_heads=4, head_dim=128, cache_size=4096)

    def test_large_batch(self):
        """Larger batch to stress coalescing."""
        self._run_parity(num_tokens=128, n_q_heads=32, n_kv_heads=8, head_dim=128, cache_size=8192)

    def test_head_dim_64(self):
        """Smaller head dim (e.g., E2B)."""
        self._run_parity(num_tokens=16, n_q_heads=8, n_kv_heads=4, head_dim=64, cache_size=2048)

    def test_spec_decode_skip(self):
        """Verify that out_loc=-1 tokens are skipped (no cache write)."""
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        torch.manual_seed(42)

        num_tokens, n_q_heads, n_kv_heads, head_dim = 4, 8, 4, 128
        cache_size = 1024
        rot_dim = head_dim
        max_pos = 8192

        q = torch.randn(num_tokens, n_q_heads, head_dim, dtype=torch.bfloat16, device="xpu")
        k = torch.randn(num_tokens, n_kv_heads, head_dim, dtype=torch.bfloat16, device="xpu")
        v = torch.randn(num_tokens, n_kv_heads, head_dim, dtype=torch.bfloat16, device="xpu")
        cos_sin_cache = torch.randn(max_pos, rot_dim, dtype=torch.float32, device="xpu")
        positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device="xpu")

        # Token 1 and 3 are skipped (out_loc=-1)
        out_loc = torch.tensor([10, -1, 20, -1], dtype=torch.int64, device="xpu")

        k_cache = torch.zeros(cache_size, n_kv_heads * head_dim, dtype=torch.bfloat16, device="xpu")
        v_cache = torch.zeros_like(k_cache)

        from sgl_kernel import apply_rope_inplace_with_kvcache_xpu

        apply_rope_inplace_with_kvcache_xpu(
            q, k, v, k_cache, v_cache,
            cos_sin_cache, positions, out_loc, is_neox=True,
        )

        # Skipped slots should remain zero
        zero_row = torch.zeros(n_kv_heads * head_dim, dtype=torch.bfloat16, device="xpu")
        for slot in range(cache_size):
            if slot not in [10, 20]:
                torch.testing.assert_close(k_cache[slot], zero_row)
                torch.testing.assert_close(v_cache[slot], zero_row)

        # Written slots should NOT be zero
        self.assertFalse(torch.all(k_cache[10] == 0).item())
        self.assertFalse(torch.all(v_cache[20] == 0).item())


if __name__ == "__main__":
    unittest.main()
