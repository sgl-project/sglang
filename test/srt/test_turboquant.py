"""
Unit tests for TurboQuant KV cache compression.

Tests:
1. Codebook construction (Lloyd's algorithm, closed-form centroids)
2. WHT rotation correctness (self-inverse property)
3. Quantizer roundtrip quality (MSE, cosine similarity) with WHT
4. Bit packing/unpacking correctness
5. Zero vector and batch consistency

Usage:
    python -m pytest test/srt/test_turboquant.py -v -k "not GPU"   # CPU only
    python -m pytest test/srt/test_turboquant.py -v -k "GPU"       # GPU only
    python -m pytest test/srt/test_turboquant.py -v                # All
"""

import math
import unittest

import numpy as np


class TestCodebook(unittest.TestCase):

    def test_1bit_centroids(self):
        from sglang.srt.layers.quantization.kv_turboquant import build_codebook
        centroids, boundaries = build_codebook(1, head_dim=128)
        self.assertEqual(len(centroids), 2)
        self.assertAlmostEqual(centroids[0], -centroids[1], places=6)
        expected = math.sqrt(2.0 / (math.pi * 128))
        self.assertAlmostEqual(abs(centroids[1]), expected, places=5)

    def test_2bit_centroids(self):
        from sglang.srt.layers.quantization.kv_turboquant import build_codebook
        centroids, boundaries = build_codebook(2, head_dim=128)
        self.assertEqual(len(centroids), 4)
        for i in range(len(centroids) - 1):
            self.assertLess(centroids[i], centroids[i + 1])

    def test_4bit_lloyds(self):
        from sglang.srt.layers.quantization.kv_turboquant import build_codebook
        centroids, boundaries = build_codebook(4, head_dim=128)
        self.assertEqual(len(centroids), 16)


class TestTurboQuantConfig(unittest.TestCase):

    def test_config_creation_cpu(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cfg = TurboQuantConfig(bit_width=4, head_dim=128, device="cpu")
        self.assertEqual(cfg.bit_width, 4)
        self.assertEqual(cfg.k_centroids.shape[0], 16)
        self.assertEqual(cfg.signs1.shape[0], 128)
        self.assertEqual(cfg.signs2.shape[0], 128)
        self.assertEqual(cfg.k_packed_dim, 64)

    def test_signs_are_pm1(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cfg = TurboQuantConfig(bit_width=4, head_dim=128, device="cpu")
        self.assertTrue((cfg.signs1.abs() == 1.0).all())
        self.assertTrue((cfg.signs2.abs() == 1.0).all())

    def test_packed_dim_2bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cfg = TurboQuantConfig(bit_width=2, head_dim=128, device="cpu")
        self.assertEqual(cfg.k_packed_dim, 32)

    def test_packed_dim_4bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cfg = TurboQuantConfig(bit_width=4, head_dim=128, device="cpu")
        self.assertEqual(cfg.k_packed_dim, 64)


try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@unittest.skipUnless(HAS_CUDA, "CUDA not available")
class TestTurboQuantGPU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cls.device = "cuda"
        cls.configs = {}
        for bits in [2, 4]:
            cls.configs[bits] = TurboQuantConfig(
                bit_width=bits, head_dim=128, device=cls.device
            )

    def _roundtrip(self, bits, tokens=64, heads=4):
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_quantize,
        )
        cfg = self.configs[bits]
        torch.manual_seed(42)
        x = torch.randn(tokens, heads, 128, device=self.device, dtype=torch.bfloat16)

        packed, norms, quant_norms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, cfg.bit_width
        )
        # dequantize includes inverse WHT — returns in original domain
        x_hat = batched_dequantize(
            packed, norms, cfg.k_centroids, cfg.bit_width, cfg.signs1, cfg.signs2
        )

        mse = ((x.float() - x_hat.float()) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            x.float().reshape(-1), x_hat.float().reshape(-1), dim=0
        ).item()
        return mse, cos

    def test_2bit_roundtrip(self):
        mse, cos = self._roundtrip(2)
        self.assertGreater(cos, 0.7, f"2-bit cosine too low: {cos:.4f}")

    def test_4bit_roundtrip(self):
        mse, cos = self._roundtrip(4)
        self.assertGreater(cos, 0.95, f"4-bit cosine too low: {cos:.4f}")

    def test_wht_self_inverse(self):
        """Normalized WHT applied twice should return the original."""
        from sglang.jit_kernel.hadamard import hadamard_transform
        import math
        torch.manual_seed(0)
        x = torch.randn(8, 4, 128, device=self.device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(128)
        y = hadamard_transform(x, scale=scale)
        x_back = hadamard_transform(y, scale=scale)
        err = (x - x_back).abs().max().item()
        self.assertLess(err, 1e-4, f"WHT not self-inverse: max error {err}")

    def test_packing_2bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import batched_quantize
        cfg = self.configs[2]
        x = torch.randn(8, 4, 128, device=self.device, dtype=torch.bfloat16)
        packed, norms, quant_norms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 2
        )
        self.assertEqual(packed.shape, (8, 4, 32))
        self.assertEqual(packed.dtype, torch.uint8)

    def test_packing_4bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import batched_quantize
        cfg = self.configs[4]
        x = torch.randn(8, 4, 128, device=self.device, dtype=torch.bfloat16)
        packed, norms, quant_norms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        self.assertEqual(packed.shape, (8, 4, 64))

    def test_zero_vector(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_quantize,
        )
        cfg = self.configs[4]
        x = torch.zeros(1, 1, 128, device=self.device, dtype=torch.bfloat16)
        packed, norms, quant_norms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        x_hat = batched_dequantize(
            packed, norms, cfg.k_centroids, 4, cfg.signs1, cfg.signs2
        )
        self.assertAlmostEqual(x_hat.abs().max().item(), 0.0, places=3)

    def test_batch_consistency(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_quantize,
        )
        cfg = self.configs[4]
        torch.manual_seed(42)
        x = torch.randn(4, 2, 128, device=self.device, dtype=torch.bfloat16)

        packed_b, norms_b, _ = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        hat_b = batched_dequantize(packed_b, norms_b, cfg.k_centroids, 4, cfg.signs1, cfg.signs2)

        for i in range(4):
            packed_i, norms_i, _ = batched_quantize(
                x[i:i+1], cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
            )
            hat_i = batched_dequantize(packed_i, norms_i, cfg.k_centroids, 4, cfg.signs1, cfg.signs2)
            torch.testing.assert_close(hat_b[i:i+1], hat_i, atol=1e-4, rtol=0)

    def test_attention_score_preservation(self):
        """Verify Q@K^T scores are well-preserved after quantization."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_quantize,
        )
        cfg = self.configs[4]
        torch.manual_seed(42)
        Q = torch.randn(1, 4, 128, device=self.device, dtype=torch.bfloat16)
        K = torch.randn(32, 4, 128, device=self.device, dtype=torch.bfloat16)

        scores_orig = torch.matmul(Q.float(), K.float().transpose(-2, -1))

        k_packed, k_norms, _ = batched_quantize(
            K, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        K_hat = batched_dequantize(k_packed, k_norms, cfg.k_centroids, 4, cfg.signs1, cfg.signs2)
        scores_quant = torch.matmul(Q.float(), K_hat.float().transpose(-2, -1))

        cos = torch.nn.functional.cosine_similarity(
            scores_orig.flatten(), scores_quant.flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.9, f"Attention scores diverged: cos={cos:.4f}")

    def test_query_rotation_equivalence(self):
        """Verify Q_rot @ K_rotspace produces same scores as Q @ K_dequant."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_dequantize_rotspace, batched_quantize,
        )
        for bits in [2, 4]:
            cfg = self.configs[bits]
            torch.manual_seed(42)
            Q = torch.randn(4, 8, 128, device=self.device, dtype=torch.bfloat16)
            K = torch.randn(32, 8, 128, device=self.device, dtype=torch.bfloat16)

            # Quantize K
            k_packed, k_norms, k_quant_norms = batched_quantize(
                K, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
            )

            # Method A: Traditional dequant (with inverse WHT)
            K_hat = batched_dequantize(
                k_packed, k_norms, cfg.k_centroids, bits, cfg.signs1, cfg.signs2
            )
            scores_a = torch.einsum(
                "qhd,khd->hqk", Q.float(), K_hat.float()
            )

            # Method B: Query Rotation + rotspace dequant (no inverse WHT)
            Q_rot = cfg.rotate_query(Q)
            safe_k_qnorms = torch.where(k_quant_norms > 1e-10, k_quant_norms, torch.ones_like(k_quant_norms))
            k_dequant_scale = k_norms / safe_k_qnorms
            K_rotspace = batched_dequantize_rotspace(
                k_packed, k_dequant_scale, cfg.k_centroids, bits, head_dim=128
            )
            scores_b = torch.einsum(
                "qhd,khd->hqk", Q_rot.float(), K_rotspace.float()
            )

            max_diff = (scores_a - scores_b).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                scores_a.flatten(), scores_b.flatten(), dim=0
            ).item()
            self.assertLess(
                max_diff, 0.2,
                f"{bits}-bit: Query rotation scores diverge: max_diff={max_diff:.4f}"
            )
            self.assertGreater(
                cos, 0.99,
                f"{bits}-bit: Query rotation scores diverge: cos={cos:.6f}"
            )

    def test_output_inverse_rotation(self):
        """Verify inverse_rotate(rotate(x)) == x (bf16 precision)."""
        cfg = self.configs[4]
        torch.manual_seed(42)
        x = torch.randn(4, 8, 128, device=self.device, dtype=torch.bfloat16)
        x_rot = cfg.rotate_query(x)
        x_back = cfg.inverse_rotate_output(x_rot)
        max_err = (x.float() - x_back.float()).abs().max().item()
        self.assertLess(max_err, 0.02, f"Rotation roundtrip error: {max_err}")

    def test_rotspace_dequant_v_output_equivalence(self):
        """Verify attention output equivalence for V side:
        D1@H@D2 @ sum(attn_i * V_rotspace_i) == sum(attn_i * V_dequant_i)"""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_dequantize_rotspace, batched_quantize,
        )
        cfg = self.configs[4]
        torch.manual_seed(42)
        V = torch.randn(32, 4, 128, device=self.device, dtype=torch.bfloat16)
        # attn_weights: (1, 4, 32) — 1 query, 4 heads, 32 KV tokens
        attn_weights = torch.softmax(
            torch.randn(1, 4, 32, device=self.device), dim=-1
        )

        v_packed, v_norms, v_quant_norms = batched_quantize(
            V, cfg.signs1, cfg.signs2, cfg.v_centroids, cfg.v_boundaries, 4
        )

        # Method A: traditional dequant
        V_hat = batched_dequantize(
            v_packed, v_norms, cfg.v_centroids, 4, cfg.signs1, cfg.signs2
        )
        # V_hat: (32, 4, 128) → need (heads, kv_tokens, dim) for einsum
        o_a = torch.einsum("qhk,khd->qhd", attn_weights, V_hat.float())

        # Method B: rotspace + inverse rotation on output
        safe_v_qnorms = torch.where(v_quant_norms > 1e-10, v_quant_norms, torch.ones_like(v_quant_norms))
        v_dequant_scale = v_norms / safe_v_qnorms
        V_rotspace = batched_dequantize_rotspace(
            v_packed, v_dequant_scale, cfg.v_centroids, 4, head_dim=128
        )
        o_rotspace = torch.einsum("qhk,khd->qhd", attn_weights, V_rotspace.float())
        o_b = cfg.inverse_rotate_output(o_rotspace.to(torch.bfloat16)).float()

        max_diff = (o_a - o_b).abs().max().item()
        self.assertLess(
            max_diff, 0.1,
            f"V output rotspace equivalence failed: max_diff={max_diff:.4f}"
        )

    def test_fused_decode_kernel_correctness(self):
        """Verify fused TQ decode kernel matches PyTorch reference implementation."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize_rotspace, batched_quantize,
        )
        from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
            tq_decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )

        torch.manual_seed(42)
        batch = 2
        q_heads = 32
        kv_heads = 8
        head_dim = 128
        seq_lens = [64, 48]
        total_kv = sum(seq_lens)
        max_kv_splits = 4

        for bits in [2, 4]:  # fused kernel supports both 2-bit and 4-bit
            cfg = self.configs[bits]

            # Generate Q (already rotated) and raw K/V
            Q = torch.randn(batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16)
            K_raw = torch.randn(total_kv, kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)
            V_raw = torch.randn(total_kv, kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)

            # Quantize K and V
            k_packed, k_norms, k_qnorms = batched_quantize(
                K_raw, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
            )
            v_packed, v_norms, v_qnorms = batched_quantize(
                V_raw, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
            )

            # Precompute dequant scales
            safe_k_qn = torch.where(k_qnorms > 1e-10, k_qnorms, torch.ones_like(k_qnorms))
            safe_v_qn = torch.where(v_qnorms > 1e-10, v_qnorms, torch.ones_like(v_qnorms))
            k_dscale = (k_norms / safe_k_qn).to(torch.bfloat16)
            v_dscale = (v_norms / safe_v_qn).to(torch.bfloat16)

            # Build kv_indptr and kv_indices (identity mapping: slot i = position i)
            kv_indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=self.device)
            kv_indices = torch.arange(total_kv, dtype=torch.int64, device=self.device)

            # num_kv_splits
            num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=self.device)

            # --- Method A: Fused kernel ---
            o_fused = torch.zeros(batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16)
            attn_logits_a = torch.zeros(batch, q_heads, max_kv_splits, head_dim, device=self.device, dtype=torch.float32)
            attn_lse_a = torch.zeros(batch, q_heads, max_kv_splits, device=self.device, dtype=torch.float32)

            tq_decode_attention_fwd(
                Q, k_packed, v_packed,
                k_dscale, v_dscale,
                cfg.k_centroids,
                o_fused, kv_indptr, kv_indices,
                attn_logits_a, attn_lse_a,
                num_kv_splits, max_kv_splits,
                sm_scale=1.0 / (head_dim ** 0.5),
                bit_width=bits,
            )

            # --- Method B: PyTorch reference (rotspace dequant + standard attention) ---
            K_dequant = batched_dequantize_rotspace(
                k_packed, k_dscale, cfg.k_centroids, bits, head_dim=128
            )
            V_dequant = batched_dequantize_rotspace(
                v_packed, v_dscale, cfg.k_centroids, bits, head_dim=128
            )
            o_ref = torch.zeros(batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16)
            attn_logits_b = torch.zeros(batch, q_heads, max_kv_splits, head_dim, device=self.device, dtype=torch.float32)
            attn_lse_b = torch.zeros(batch, q_heads, max_kv_splits, device=self.device, dtype=torch.float32)

            decode_attention_fwd(
                Q, K_dequant, V_dequant, o_ref,
                kv_indptr, kv_indices,
                attn_logits_b, attn_lse_b,
                num_kv_splits, max_kv_splits,
                sm_scale=1.0 / (head_dim ** 0.5),
                k_scale=1.0, v_scale=1.0,
            )

            # Compare
            max_diff = (o_fused.float() - o_ref.float()).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                o_fused.float().flatten(), o_ref.float().flatten(), dim=0
            ).item()
            self.assertLess(
                max_diff, 0.1,
                f"{bits}-bit fused kernel output diverged: max_diff={max_diff:.4f}"
            )
            self.assertGreater(
                cos, 0.999,
                f"{bits}-bit fused kernel output diverged: cos={cos:.6f}"
            )

    def test_asymmetric_k4v2_roundtrip(self):
        """Verify K=4bit V=2bit asymmetric quantization roundtrip."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            TurboQuantConfig, batched_quantize, batched_dequantize,
        )
        cfg = TurboQuantConfig(bit_width=4, head_dim=128, device=self.device,
                               k_bit_width=4, v_bit_width=2)
        self.assertEqual(cfg.k_packed_dim, 64)   # 4-bit: dim//2
        self.assertEqual(cfg.v_packed_dim, 32)    # 2-bit: dim//4
        self.assertEqual(cfg.k_centroids.shape[0], 16)
        self.assertEqual(cfg.v_centroids.shape[0], 4)

        torch.manual_seed(42)
        K = torch.randn(16, 4, 128, device=self.device, dtype=torch.bfloat16)
        V = torch.randn(16, 4, 128, device=self.device, dtype=torch.bfloat16)

        k_packed, k_norms, k_qnorms = batched_quantize(
            K, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, cfg.k_bit_width)
        v_packed, v_norms, v_qnorms = batched_quantize(
            V, cfg.signs1, cfg.signs2, cfg.v_centroids, cfg.v_boundaries, cfg.v_bit_width)

        self.assertEqual(k_packed.shape[-1], 64)
        self.assertEqual(v_packed.shape[-1], 32)

        # Use full dequant (with inverse WHT) to compare in original domain
        K_hat = batched_dequantize(
            k_packed, k_norms, cfg.k_centroids, cfg.k_bit_width, cfg.signs1, cfg.signs2)
        V_hat = batched_dequantize(
            v_packed, v_norms, cfg.v_centroids, cfg.v_bit_width, cfg.signs1, cfg.signs2)

        k_cos = torch.nn.functional.cosine_similarity(
            K.float().reshape(-1), K_hat.float().reshape(-1), dim=0).item()
        v_cos = torch.nn.functional.cosine_similarity(
            V.float().reshape(-1), V_hat.float().reshape(-1), dim=0).item()
        self.assertGreater(k_cos, 0.9, f"K 4-bit roundtrip cos too low: {k_cos:.4f}")
        self.assertGreater(v_cos, 0.5, f"V 2-bit roundtrip cos too low: {v_cos:.4f}")

    def test_move_kv_cache_dequant_correctness(self):
        """Verify data is correct after move_kv_cache."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            TurboQuantConfig, batched_quantize, batched_dequantize_rotspace,
        )
        cfg = self.configs[4]
        torch.manual_seed(42)
        x = torch.randn(4, 8, 128, device=self.device, dtype=torch.bfloat16)

        packed, norms, qnorms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, cfg.k_bit_width)

        # Simulate pool: write to positions 0-3, move to 10-13
        pool_packed = torch.zeros(20, 8, 64, dtype=torch.uint8, device=self.device)
        safe_qnorms = torch.where(qnorms > 1e-10, qnorms, torch.ones_like(qnorms))
        dequant_scale = (norms / safe_qnorms).to(torch.bfloat16)
        pool_dscale = torch.zeros(20, 8, dtype=torch.bfloat16, device=self.device)

        src = torch.arange(4, device=self.device)
        tgt = torch.arange(10, 14, device=self.device)

        pool_packed[src] = packed
        pool_dscale[src] = dequant_scale

        # Move
        pool_packed[tgt] = pool_packed[src]
        pool_dscale[tgt] = pool_dscale[src]

        # Dequant from moved positions
        hat_src = batched_dequantize_rotspace(
            pool_packed[src], pool_dscale[src], cfg.k_centroids, 4, head_dim=128)
        hat_tgt = batched_dequantize_rotspace(
            pool_packed[tgt], pool_dscale[tgt], cfg.k_centroids, 4, head_dim=128)

        torch.testing.assert_close(hat_src, hat_tgt, atol=1e-6, rtol=0)

    def test_non_128_head_dim(self):
        """Verify TurboQuant works with head_dim=64 and head_dim=256."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            TurboQuantConfig, batched_quantize, batched_dequantize, batched_dequantize_rotspace,
        )
        for dim in [64, 256]:
            for bits in [2, 4]:
                cfg = TurboQuantConfig(bit_width=bits, head_dim=dim, device=self.device)
                torch.manual_seed(42)
                x = torch.randn(8, 4, dim, device=self.device, dtype=torch.bfloat16)

                packed, norms, qnorms = batched_quantize(
                    x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits)

                # Full dequant roundtrip
                x_hat = batched_dequantize(
                    packed, norms, cfg.k_centroids, bits, cfg.signs1, cfg.signs2)
                self.assertEqual(x_hat.shape, x.shape,
                    f"dim={dim} bits={bits}: shape mismatch {x_hat.shape} vs {x.shape}")

                cos = torch.nn.functional.cosine_similarity(
                    x.float().reshape(-1), x_hat.float().reshape(-1), dim=0).item()
                min_cos = 0.5 if bits == 2 else 0.93
                self.assertGreater(cos, min_cos,
                    f"dim={dim} bits={bits}: cos={cos:.4f} < {min_cos}")

                # Rotspace dequant
                safe_qn = torch.where(qnorms > 1e-10, qnorms, torch.ones_like(qnorms))
                ds = (norms / safe_qn).to(torch.bfloat16)
                rs = batched_dequantize_rotspace(
                    packed, ds, cfg.k_centroids, bits, head_dim=dim)
                self.assertEqual(rs.shape[-1], dim)


if __name__ == "__main__":
    unittest.main()
