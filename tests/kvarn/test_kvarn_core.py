# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVarN config, sinkhorn, store, and dequant (CPU-only).

Run:
    python -m pytest tests/kvarn/test_kvarn_core.py -v
"""

import pytest
import torch

from sglang.srt.layers.quantization.kvarn.config import (
    KVARN_PRESETS,
    KVarNConfig,
    is_kvarn_dtype,
)
from sglang.srt.layers.quantization.kvarn.dequant import (
    _unpack_lowbit,
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_v,
)
from sglang.srt.layers.quantization.kvarn.sinkhorn import (
    variance_normalize,
    variance_normalize_batched,
)
from sglang.srt.layers.quantization.kvarn.store import (
    _pack_4bit,
    kvarn_store_tile_k,
    kvarn_store_tile_v,
)

# ─── Config ──────────────────────────────────────────────────────────────────


class TestKVarNConfig:
    def test_presets_exist(self):
        assert "kvarn_k4v4_g128" in KVARN_PRESETS
        assert "kvarn_k4v2_g128" in KVARN_PRESETS

    def test_is_kvarn_dtype(self):
        assert is_kvarn_dtype("kvarn_k4v4_g128")
        assert is_kvarn_dtype("kvarn_k4v2_g64")
        assert not is_kvarn_dtype("fp8_e4m3")
        assert not is_kvarn_dtype("auto")

    def test_from_cache_dtype(self):
        cfg = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)
        assert cfg.key_bits == 4
        assert cfg.value_bits == 4
        assert cfg.group == 128
        assert cfg.head_dim == 128

    def test_from_cache_dtype_invalid(self):
        with pytest.raises(ValueError, match="Unknown KVarN"):
            KVarNConfig.from_cache_dtype("invalid", head_dim=128)

    def test_tile_bytes_positive(self):
        cfg = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)
        assert cfg.tile_bytes > 0
        assert cfg.tile_bytes_aligned >= cfg.tile_bytes

    def test_tile_bytes_alignment(self):
        cfg = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)
        assert cfg.tile_bytes_aligned % 8 == 0

    @pytest.mark.parametrize("preset", list(KVARN_PRESETS.keys()))
    def test_all_presets(self, preset):
        cfg = KVarNConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.group in (64, 128)
        assert cfg.key_bits in (2, 4)
        assert cfg.value_bits in (2, 4)

    def test_slot_offsets_consistent(self):
        cfg = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)
        assert cfg.k_s_col_offset == cfg.k_packed_offset + cfg.k_packed_bytes
        assert cfg.k_zp_offset == cfg.k_s_col_offset + cfg.head_dim * 2
        assert cfg.k_s_row_offset == cfg.k_zp_offset + cfg.head_dim * 2
        assert cfg.v_packed_offset == cfg.k_s_row_offset + cfg.group * 2
        assert cfg.v_s_col_offset == cfg.v_packed_offset + cfg.v_packed_bytes
        assert cfg.v_s_row_offset == cfg.v_s_col_offset + cfg.head_dim * 2
        assert cfg.v_zp_offset == cfg.v_s_row_offset + cfg.group * 2
        assert cfg.v_zp_offset + cfg.group * 2 == cfg.tile_bytes


# ─── Sinkhorn ────────────────────────────────────────────────────────────────


class TestSinkhorn:
    def test_identity_tile(self):
        """An already-balanced tile should remain near-identity."""
        torch.manual_seed(42)
        tile = torch.randn(128, 128) * 0.1 + 1.0  # roughly uniform scale
        balanced, s_col, s_row = variance_normalize(tile, iterations=16)
        assert balanced.shape == tile.shape
        # The balanced tile should have lower max/min std ratio than original
        orig_col_std = tile.std(dim=0)
        bal_col_std = balanced.std(dim=0)
        assert (
            bal_col_std.max() / bal_col_std.min()
            <= orig_col_std.max() / orig_col_std.min() + 0.1
        )

    def test_reduces_imbalance(self):
        """Sinkhorn should reduce the imbalance metric."""
        torch.manual_seed(0)
        # Create an imbalanced tile: scale rows by varying amounts
        tile = torch.randn(128, 128)
        scales = torch.linspace(0.1, 10.0, 128).unsqueeze(1)
        tile = tile * scales

        def imbalance(t):
            sc = t.std(dim=-2)
            sr = t.std(dim=-1)
            return sc.max() / sc.min().clamp_min(1e-8) + sr.max() / sr.min().clamp_min(
                1e-8
            )

        orig_imb = imbalance(tile.float())
        balanced, _, _ = variance_normalize(tile, iterations=16)
        new_imb = imbalance(balanced)
        assert new_imb < orig_imb

    def test_batched_matches_single(self):
        torch.manual_seed(123)
        tiles = torch.randn(4, 128, 128)
        bal_b, sc_b, sr_b = variance_normalize_batched(tiles, iterations=8)
        for i in range(4):
            bal_s, sc_s, sr_s = variance_normalize(tiles[i], iterations=8)
            torch.testing.assert_close(bal_b[i], bal_s, rtol=1e-4, atol=1e-4)

    def test_reconstruction(self):
        """tile ≈ balanced * s_col * s_row"""
        torch.manual_seed(42)
        tile = torch.randn(128, 128)
        balanced, s_col, s_row = variance_normalize(tile, iterations=16)
        reconstructed = balanced * s_col * s_row
        torch.testing.assert_close(reconstructed, tile, rtol=1e-3, atol=1e-3)


# ─── Store + Dequant roundtrip ───────────────────────────────────────────────


class TestStoreDequant:
    @pytest.mark.parametrize("bits", [4])
    def test_k_roundtrip(self, bits):
        torch.manual_seed(42)
        D, G = 128, 128
        tile = torch.randn(D, G, dtype=torch.float32)

        result = kvarn_store_tile_k(tile, bits=bits, sinkhorn_iters=8)
        dequant = kvarn_dequant_tile_k(
            result["q_packed_uint8"],
            result["s_col_K"],
            result["zp_K"],
            result["s_row_K"],
            group=G,
            bits=bits,
        )

        # 4-bit quantization should give reasonable SNR
        err = (dequant - tile).abs().mean()
        mag = tile.abs().mean()
        assert err / mag < 0.3, f"K roundtrip error too high: {err/mag}"

    @pytest.mark.parametrize("bits", [4])
    def test_v_roundtrip(self, bits):
        torch.manual_seed(42)
        G, D = 128, 128
        tile = torch.randn(G, D, dtype=torch.float32)

        result = kvarn_store_tile_v(tile, bits=bits, sinkhorn_iters=8)
        dequant = kvarn_dequant_tile_v(
            result["q_packed_uint8"],
            result["s_col_V"],
            result["s_row_V"],
            result["zp_V"],
            head_dim=D,
            bits=bits,
        )

        err = (dequant - tile).abs().mean()
        mag = tile.abs().mean()
        assert err / mag < 0.3, f"V roundtrip error too high: {err/mag}"

    def test_pack4_unpack4(self):
        """4-bit pack/unpack roundtrips correctly."""
        torch.manual_seed(0)
        q = torch.randint(0, 16, (4, 128), dtype=torch.int32)
        packed = _pack_4bit(q)
        unpacked = _unpack_lowbit(packed, 128, bits=4)
        torch.testing.assert_close(unpacked, q.to(torch.uint8))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
