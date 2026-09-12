import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_info import get_hicache_info
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiCacheInfo(unittest.TestCase):
    def test_uses_primary_logical_host_capacity(self):
        anchor_pool = SimpleNamespace(
            layout="layer_first",
            page_size=64,
            device="cpu",
            size=1024,
            logical_size=8192,
            can_use_write_back_jit=False,
        )
        host_pool_group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=anchor_pool,
                    device_pool=None,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=True,
                )
            ]
        )
        tree_cache = SimpleNamespace(
            token_to_kv_pool_host=host_pool_group,
            full_kv_pool_host=SimpleNamespace(logical_size=4096),
        )

        self.assertEqual(get_hicache_info(tree_cache).host_total_tokens, 8192)

    def test_supports_full_pool_fallback(self):
        tree_cache = SimpleNamespace(
            token_to_kv_pool_host=None,
            full_kv_pool_host=SimpleNamespace(logical_size=4096),
        )

        self.assertEqual(get_hicache_info(tree_cache).host_total_tokens, 4096)

    def test_missing_host_pool_fails_fast(self):
        with self.assertRaisesRegex(RuntimeError, "no host pool"):
            get_hicache_info(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
