# SPDX-License-Identifier: Apache-2.0
"""GPU tests for KVarN Triton kernels.

These tests require a CUDA GPU and will be skipped if no GPU is available.

Run:
    python -m pytest tests/kvarn/test_kvarn_triton_gpu.py -v --tb=short
"""

import pytest
import torch

# Skip all tests in this module if no GPU is available
if not torch.cuda.is_available():
    pytest.skip("No CUDA GPU available", allow_module_level=True)

from sglang.srt.layers.attention.kvarn_ops.triton_sinkhorn import (
    kvarn_sinkhorn_triton,
)
from sglang.srt.layers.quantization.kvarn.dequant import (
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_v,
)
from sglang.srt.layers.quantization.kvarn.hadamard import build_hadamard
from sglang.srt.layers.quantization.kvarn.sinkhorn import (
    variance_normalize_batched,
)
from sglang.srt.layers.quantization.kvarn.store import (
    kvarn_store_tile_k,
    kvarn_store_tile_v,
)

DEVICE = torch.device("cuda")
D = 128
G = 128


class TestTritonSinkhornGPU:
    """Test the Triton Sinkhorn kernel against the PyTorch reference."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_triton_matches_reference(self, batch_size):
        """Triton Sinkhorn output should match PyTorch reference (within fp32 tolerance)."""
        torch.manual_seed(42)
        tiles = torch.randn(batch_size, D, G, device=DEVICE, dtype=torch.float32)

        # PyTorch reference
        bal_ref, sc_ref, sr_ref = variance_normalize_batched(tiles, iterations=16)

        # Triton
        bal_tri, sc_tri, sr_tri = kvarn_sinkhorn_triton(tiles, iterations=16)

        # Triton returns [N, C] and [N, R], PyTorch returns [N, 1, C] and [N, R, 1]
        torch.testing.assert_close(bal_tri, bal_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(sc_tri, sc_ref.squeeze(1), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(sr_tri, sr_ref.squeeze(-1), rtol=1e-2, atol=1e-2)

    def test_reduces_imbalance_gpu(self):
        """Triton Sinkhorn should reduce the imbalance metric."""
        torch.manual_seed(0)
        tile = torch.randn(D, G, device=DEVICE, dtype=torch.float32)
        scales = torch.linspace(0.1, 10.0, D, device=DEVICE).unsqueeze(1)
        tile = tile * scales

        def imbalance(t):
            sc = t.std(dim=-2)
            sr = t.std(dim=-1)
            return (
                sc.max() / sc.min().clamp_min(1e-8)
                + sr.max() / sr.min().clamp_min(1e-8)
            ).item()

        orig_imb = imbalance(tile)
        tiles = tile.unsqueeze(0)
        bal, _, _ = kvarn_sinkhorn_triton(tiles, iterations=16)
        new_imb = imbalance(bal[0])
        assert new_imb < orig_imb

    def test_reconstruction_gpu(self):
        """tile ≈ balanced * s_col * s_row"""
        torch.manual_seed(42)
        tiles = torch.randn(4, D, G, device=DEVICE, dtype=torch.float32)
        bal, sc, sr = kvarn_sinkhorn_triton(tiles, iterations=16)
        reconstructed = bal * sc.unsqueeze(1) * sr.unsqueeze(2)
        torch.testing.assert_close(reconstructed, tiles, rtol=1e-2, atol=1e-2)


class TestStoreDequantGPU:
    """Test store/dequant roundtrip on GPU."""

    def test_k_roundtrip_gpu(self):
        torch.manual_seed(42)
        tile = torch.randn(D, G, device=DEVICE, dtype=torch.float32)
        result = kvarn_store_tile_k(tile, bits=4, sinkhorn_iters=16)
        dequant = kvarn_dequant_tile_k(
            result["q_packed_uint8"],
            result["s_col_K"],
            result["zp_K"],
            result["s_row_K"],
            group=G,
            bits=4,
        )
        err = (dequant - tile).abs().mean()
        mag = tile.abs().mean()
        assert err / mag < 0.3

    def test_v_roundtrip_gpu(self):
        torch.manual_seed(42)
        tile = torch.randn(G, D, device=DEVICE, dtype=torch.float32)
        result = kvarn_store_tile_v(tile, bits=4, sinkhorn_iters=16)
        dequant = kvarn_dequant_tile_v(
            result["q_packed_uint8"],
            result["s_col_V"],
            result["s_row_V"],
            result["zp_V"],
            head_dim=D,
            bits=4,
        )
        err = (dequant - tile).abs().mean()
        mag = tile.abs().mean()
        assert err / mag < 0.3


class TestHadamardRotationGPU:
    """Test Hadamard rotation on GPU."""

    def test_orthonormal_gpu(self):
        H = build_hadamard(D, DEVICE)
        I = torch.mm(H, H.t())
        torch.testing.assert_close(I, torch.eye(D, device=DEVICE), rtol=1e-4, atol=1e-4)

    def test_rotation_preserves_dot_product_gpu(self):
        torch.manual_seed(42)
        H = build_hadamard(D, DEVICE)
        q = torch.randn(1, D, device=DEVICE)
        k = torch.randn(4, D, device=DEVICE)
        score_orig = torch.mm(q, k.t())
        q_rot = torch.mm(q, H)
        k_rot = torch.mm(k, H)
        score_rot = torch.mm(q_rot, k_rot.t())
        torch.testing.assert_close(score_rot, score_orig, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
