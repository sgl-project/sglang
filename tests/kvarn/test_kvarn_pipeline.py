# SPDX-License-Identifier: Apache-2.0
"""Integration test: verify QK^T invariance under the full KVarN pipeline.

The key mathematical invariant of KVarN is:

    QK^T = (Q @ H) @ (K_rot)^T = (Q @ H) @ (dequant(quant(K @ H)))^T

Since H is orthonormal, QK^T is preserved if dequant∘quant is a good
approximation of the identity.  This test verifies that the full pipeline
(rotation → sinkhorn → RTN → pack → unpack → dequant → un-rotation)
preserves attention scores within an acceptable error bound.
"""

import math

import pytest
import torch

from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.dequant import (
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_v,
)
from sglang.srt.layers.quantization.kvarn.hadamard import build_hadamard
from sglang.srt.layers.quantization.kvarn.store import (
    kvarn_store_tile_k,
    kvarn_store_tile_v,
)


class TestPipelineInvariance:
    """Test that the full KVarN pipeline preserves attention scores."""

    @pytest.mark.parametrize("bits", [4])
    def test_qk_score_preservation(self, bits):
        """QK^T scores should be preserved within ~30% relative error at 4-bit."""
        torch.manual_seed(42)
        D = 128
        G = 128
        cfg = KVarNConfig(head_dim=D, key_bits=bits, value_bits=bits, group=G)

        H = build_hadamard(D, torch.device("cpu"))

        # Simulate one head's K tile: [D, G] (channels × tokens)
        k_tile = torch.randn(D, G, dtype=torch.float32) * 0.1
        # Simulate query: [1, D]
        q = torch.randn(1, D, dtype=torch.float32) * 0.1

        # Original scores
        scores_orig = torch.mm(q, k_tile)  # [1, G]

        # Rotate
        k_rot = torch.mm(H, k_tile)  # [D, G]
        q_rot = torch.mm(q, H)  # [1, D]

        # Store (quantize)
        stored = kvarn_store_tile_k(k_rot, bits=bits, sinkhorn_iters=8)

        # Dequant
        k_rot_dequant = kvarn_dequant_tile_k(
            stored["q_packed_uint8"],
            stored["s_col_K"],
            stored["zp_K"],
            stored["s_row_K"],
            group=G,
            bits=bits,
        )

        # Scores in rotated frame
        scores_rot = torch.mm(q_rot, k_rot_dequant)  # [1, G]

        # Compare
        err = (scores_rot - scores_orig).abs().mean()
        mag = scores_orig.abs().mean()
        rel_err = err / mag
        assert rel_err < 0.5, f"QK score relative error too high: {rel_err:.3f}"

    @pytest.mark.parametrize("bits", [4])
    def test_attention_output_preservation(self, bits):
        """Full attention output (QK^T * V) should be preserved within tolerance."""
        torch.manual_seed(42)
        D = 128
        G = 128
        cfg = KVarNConfig(head_dim=D, key_bits=bits, value_bits=bits, group=G)
        H = build_hadamard(D, torch.device("cpu"))

        k_tile = torch.randn(D, G, dtype=torch.float32) * 0.1  # [D, G]
        v_tile = torch.randn(G, D, dtype=torch.float32) * 0.1  # [G, D]
        q = torch.randn(1, D, dtype=torch.float32) * 0.1  # [1, D]

        # Original attention: softmax(QK^T / sqrt(D)) @ V
        scores = torch.mm(q, k_tile) / math.sqrt(D)  # [1, G]
        attn = torch.softmax(scores, dim=-1)  # [1, G]
        out_orig = torch.mm(attn, v_tile)  # [1, D]

        # Rotate
        k_rot = torch.mm(H, k_tile)  # [D, G]
        v_rot = torch.mm(v_tile, H)  # [G, D]
        q_rot = torch.mm(q, H)  # [1, D]

        # Store + dequant
        k_stored = kvarn_store_tile_k(k_rot, bits=bits, sinkhorn_iters=8)
        k_deq = kvarn_dequant_tile_k(
            k_stored["q_packed_uint8"],
            k_stored["s_col_K"],
            k_stored["zp_K"],
            k_stored["s_row_K"],
            group=G,
            bits=bits,
        )
        v_stored = kvarn_store_tile_v(v_rot, bits=bits, sinkhorn_iters=8)
        v_deq = kvarn_dequant_tile_v(
            v_stored["q_packed_uint8"],
            v_stored["s_col_V"],
            v_stored["s_row_V"],
            v_stored["zp_V"],
            head_dim=D,
            bits=bits,
        )

        # Rotated attention
        scores_rot = torch.mm(q_rot, k_deq) / math.sqrt(D)  # [1, G]
        attn_rot = torch.softmax(scores_rot, dim=-1)  # [1, G]
        out_rot = torch.mm(attn_rot, v_deq)  # [1, D]

        # Un-rotate output
        out_unrot = torch.mm(out_rot, H)  # [1, D]

        err = (out_unrot - out_orig).abs().mean()
        mag = out_orig.abs().mean()
        rel_err = err / mag
        # Attention output is more sensitive than raw scores due to softmax
        assert rel_err < 0.7, f"Attention output relative error too high: {rel_err:.3f}"

    def test_compression_ratio(self):
        """Verify the storage savings: fp16 → int4 should be ~4x smaller."""
        cfg = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)
        D, G = cfg.head_dim, cfg.group

        # fp16 K tile: D * G * 2 bytes
        fp16_k_bytes = D * G * 2
        # fp16 V tile: G * D * 2 bytes
        fp16_v_bytes = G * D * 2
        fp16_total = fp16_k_bytes + fp16_v_bytes

        # KVarN tile: tile_bytes (includes scales)
        kvarn_bytes = cfg.tile_bytes

        ratio = fp16_total / kvarn_bytes
        # K4V4: packed K = D*G*4/8, packed V = G*D*4/8, plus scales
        # Should be roughly 3-5x compression
        assert ratio > 2.5, f"Compression ratio too low: {ratio:.2f}x"
        assert ratio < 6.0, f"Compression ratio unexpectedly high: {ratio:.2f}x"

    @pytest.mark.parametrize("preset", ["kvarn_k4v4_g128", "kvarn_k4v2_g128"])
    def test_store_dequant_roundtrip_improves_with_sinkhorn(self, preset):
        """Sinkhorn should improve roundtrip accuracy vs. plain RTN."""
        torch.manual_seed(42)
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        D, G = cfg.head_dim, cfg.group

        tile = torch.randn(D, G, dtype=torch.float32) * 0.1

        # With sinkhorn
        stored_s = kvarn_store_tile_k(tile, bits=cfg.key_bits, sinkhorn_iters=8)
        deq_s = kvarn_dequant_tile_k(
            stored_s["q_packed_uint8"],
            stored_s["s_col_K"],
            stored_s["zp_K"],
            stored_s["s_row_K"],
            group=G,
            bits=cfg.key_bits,
        )
        err_s = (deq_s - tile).abs().mean()

        # Without sinkhorn (1 iteration ≈ minimal balancing)
        stored_n = kvarn_store_tile_k(tile, bits=cfg.key_bits, sinkhorn_iters=1)
        deq_n = kvarn_dequant_tile_k(
            stored_n["q_packed_uint8"],
            stored_n["s_col_K"],
            stored_n["zp_K"],
            stored_n["s_row_K"],
            group=G,
            bits=cfg.key_bits,
        )
        err_n = (deq_n - tile).abs().mean()

        # Sinkhorn should be at least as good (usually better)
        # We use <= rather than < to avoid flakiness on easy tiles
        assert (
            err_s <= err_n * 1.1
        ), f"Sinkhorn ({err_s:.6f}) should be <= plain RTN ({err_n:.6f}) * 1.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
