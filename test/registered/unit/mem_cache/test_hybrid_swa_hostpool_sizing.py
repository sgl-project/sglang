"""
Unit guard for hybrid-SWA HiCache host-pool sizing.

A fixed ``--hicache-size`` is a *total* host budget. A hybrid-SWA stack builds
two host pools (full + SWA); each used to receive the full budget, so the stack
allocated ~2x the requested host memory (and hung cudaHostRegister at larger
sizes). The fix splits the budget across the pools by bytes/token.

These cases pin the split math:
  * the per-pool shares sum to the requested total (no doubling), and
  * shares are proportional to bytes/token, i.e. every pool gets ~equal token
    capacity.
A regression that reverts to handing each pool the full budget makes the
"sum == total" assertion fail.
"""

import unittest

import torch

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _mha_host_bytes_per_token,
    _split_hicache_size_by_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeMHAPool:
    """Minimal stand-in exposing the fields _mha_host_bytes_per_token reads."""

    def __init__(self, layer_num: int, head_num: int = 8, head_dim: int = 128):
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.store_dtype = torch.bfloat16


class TestHybridSWAHostPoolSizing(CustomTestCase):
    def test_split_sums_to_total_budget(self):
        # The whole point of the fix: shares add up to the requested budget
        # rather than each pool taking the full amount.
        shares = _split_hicache_size_by_weight(total_size_gb=60.0, weights=[3, 1])
        self.assertAlmostEqual(sum(shares), 60.0, places=6)

    def test_split_is_proportional_to_bytes_per_token(self):
        # Weighting by bytes/token yields equal token capacity per pool:
        # share_i / bytes_per_token_i is constant across pools.
        full = _FakeMHAPool(layer_num=34)
        swa = _FakeMHAPool(layer_num=14)
        w_full = _mha_host_bytes_per_token(full)
        w_swa = _mha_host_bytes_per_token(swa)

        full_gb, swa_gb = _split_hicache_size_by_weight(
            total_size_gb=48.0, weights=[w_full, w_swa]
        )
        self.assertAlmostEqual(full_gb + swa_gb, 48.0, places=6)
        # Equal token capacity => shares track the layer-count ratio.
        self.assertAlmostEqual(full_gb / swa_gb, w_full / w_swa, places=6)
        self.assertAlmostEqual(full_gb / swa_gb, 34 / 14, places=6)

    def test_bytes_per_token_matches_host_pool_formula(self):
        # Guards that the split weight stays in sync with
        # MHATokenToKVPoolHost.get_size_per_token (K and V across all layers).
        pool = _FakeMHAPool(layer_num=24, head_num=8, head_dim=128)
        expected = 128 * 8 * 24 * torch.bfloat16.itemsize * 2
        self.assertEqual(_mha_host_bytes_per_token(pool), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
