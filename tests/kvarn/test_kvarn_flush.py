# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVarN flush manager (CPU-only)."""

import pytest
import torch

from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.flush_manager import KVarNFlushManager


@pytest.fixture
def cfg():
    return KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)


@pytest.fixture
def manager(cfg):
    return KVarNFlushManager(
        cfg=cfg,
        num_layers=2,
        num_kv_heads=4,
        head_dim=128,
        v_head_dim=128,
        sink_blocks=1,
    )


def _make_tail_pools(num_layers, pool_slots, group, hk, d, device="cpu"):
    tail_K = [
        torch.randn(pool_slots, group, hk, d, dtype=torch.float16, device=device)
        for _ in range(num_layers)
    ]
    tail_V = [
        torch.randn(pool_slots, group, hk, d, dtype=torch.float16, device=device)
        for _ in range(num_layers)
    ]
    return tail_K, tail_V


def _make_compressed_cache(num_layers, num_blocks, hk, tile_bytes, device="cpu"):
    return [
        torch.zeros(num_blocks, hk, tile_bytes, dtype=torch.uint8, device=device)
        for _ in range(num_layers)
    ]


class TestFlushManager:
    def test_init(self, manager, cfg):
        assert manager.num_layers == 2
        assert manager.num_kv_heads == 4
        assert manager.head_dim == 128
        assert manager.group == cfg.group
        assert manager.tile_bytes > 0

    def test_flush_writes_nonzero(self, manager, cfg):
        G, Hk, D = cfg.group, 4, 128
        pool_slots = 4
        tail_K, tail_V = _make_tail_pools(2, pool_slots, G, Hk, D)
        cache = _make_compressed_cache(2, 8, Hk, manager.tile_bytes)

        assert cache[0][0, 0].sum().item() == 0
        manager.flush_block(0, tail_K, tail_V, slot=0, compressed_cache=cache)
        assert cache[0][0, 0].sum().item() > 0

    def test_flush_batched_writes_nonzero(self, manager, cfg):
        G, Hk, D = cfg.group, 4, 128
        pool_slots = 4
        tail_K, tail_V = _make_tail_pools(2, pool_slots, G, Hk, D)
        cache = _make_compressed_cache(2, 8, Hk, manager.tile_bytes)

        assert cache[0][0, 0].sum().item() == 0
        manager.flush_blocks_batched([0, 1], tail_K, tail_V, [0, 1], cache)
        assert cache[0][0, 0].sum().item() > 0
        assert cache[0][1, 0].sum().item() > 0

    def test_dequant_roundtrip(self, manager, cfg):
        """Flush then dequant should approximately recover the original data."""
        G, Hk, D = cfg.group, 4, 128
        pool_slots = 2
        tail_K, tail_V = _make_tail_pools(2, pool_slots, G, Hk, D)
        cache = _make_compressed_cache(2, 8, Hk, manager.tile_bytes)

        # Save originals for comparison
        orig_k = tail_K[0][0].clone()  # [G, Hk, D]
        orig_v = tail_V[0][0].clone()

        # Flush block 0 to int4
        manager.flush_block(0, tail_K, tail_V, slot=0, compressed_cache=cache)

        # Dequant back
        K_deq, V_deq = manager.dequant_block(0, cache, layer_id=0)
        # K_deq, V_deq are [G, Hk, D] in rotated frame

        # Check shapes
        assert K_deq.shape == (G, Hk, D)
        assert V_deq.shape == (G, Hk, D)

        # Check approximate recovery (within 4-bit quantization error)
        k_err = (K_deq.float() - orig_k.float()).abs().mean()
        k_mag = orig_k.float().abs().mean()
        rel_err = k_err / k_mag
        assert rel_err < 0.5, f"K dequant roundtrip error too high: {rel_err:.3f}"

    def test_no_flush_empty(self, manager, cfg):
        G, Hk, D = cfg.group, 4, 128
        pool_slots = 2
        tail_K, tail_V = _make_tail_pools(2, pool_slots, G, Hk, D)
        cache = _make_compressed_cache(2, 8, Hk, manager.tile_bytes)
        # Just verify it doesn't crash with empty batch
        manager.flush_blocks_batched([], tail_K, tail_V, [], cache)
        assert cache[0][0, 0].sum().item() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
