"""Unit test for hybrid HiCache fixed-size budget splitting."""

import unittest

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _split_hicache_size,
    _split_swa_hicache_size,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Pool:
    def __init__(self, kv_bytes):
        self._kv_bytes = kv_bytes
        self.size = kv_bytes

    def get_kv_size_bytes(self):
        return self._kv_bytes


class TestSplitHicacheSize(CustomTestCase):
    def test_unset_swa_ratio_follows_device_pool_token_ratio(self):
        pools = (_Pool(100), _Pool(50))
        full_size, swa_size = _split_swa_hicache_size(100, pools, (6, 4))

        self.assertAlmostEqual((swa_size / 4) / (full_size / 6), 0.5)
        self.assertEqual(full_size + swa_size, 100)

    def test_swa_full_tokens_ratio_overrides_the_device_proportion(self):
        pools = (_Pool(100), _Pool(50))
        full_size, swa_size = _split_swa_hicache_size(100, pools, (6, 4), 0.25)

        full_tokens = full_size / 6
        swa_tokens = swa_size / 4
        self.assertAlmostEqual(swa_tokens / full_tokens, 0.25)
        self.assertEqual(full_size + swa_size, 100)

    def test_splits_total_budget_by_device_bytes(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(75 * 10**9), _Pool((15 * 10**9, 10 * 10**9)))
        )
        self.assertEqual(shares, (75.0, 25.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled


class TestHostBytesPerToken(CustomTestCase):
    def test_mha_includes_mtp_draft_layers(self):
        class _DevicePool:
            head_dim = 4
            v_head_dim = 2
            head_num = 3
            layer_num = 5
            store_dtype = type("DType", (), {"itemsize": 2})()

        size = MHATokenToKVPoolHost.get_size_per_token_for_device_pool(
            _DevicePool(), (object(), object())
        )

        self.assertEqual(size, (4 + 2) * 3 * (5 + 2) * 2)

    def test_mla_includes_sharding_and_mtp_draft_layers(self):
        class _DevicePool:
            layer_num = 10
            layer_shard_enabled = True
            layer_shard_size = 4
            kv_lora_rank = 3
            qk_rope_head_dim = 1
            store_dtype = type("DType", (), {"itemsize": 2})()

        size = MLATokenToKVPoolHost.get_size_per_token_for_device_pool(
            _DevicePool(), mtp_draft_device_pools=(object(), object())
        )

        self.assertEqual(size, (3 + 1) * (3 + 2) * 2)


if __name__ == "__main__":
    unittest.main()
