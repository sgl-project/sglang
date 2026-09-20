"""
Unit tests for the HiCache L3 (mooncake) object-key layout reported to external
KV routers via GET /get_hicache_l3_cache_layout.

python3 -m unittest hicache.test_hicache_l3_object_layout.TestHiCacheL3ObjectLayout
"""

import unittest

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sidecar(pool_name: PoolName, hit_policy: PoolHitPolicy) -> SidecarPoolSpec:
    indices_from_pool = (
        PoolName.SWA if hit_policy is PoolHitPolicy.TRAILING_PAGES else PoolName.KV
    )
    return SidecarPoolSpec(
        pool_name=pool_name,
        indices_from_pool=indices_from_pool,
        hit_policy=hit_policy,
    )


class TestHiCacheL3ObjectLayout(CustomTestCase):
    def _store(self, *, mla_suffix="", pp_rank=0, pp_size=1, registered=()):
        store = MooncakeStore.__new__(MooncakeStore)
        # A real store needs a live mooncake master; the host pools are
        # sentinels because the DeepSeek V4 branch only checks registration.
        store.registered_pools = {name: object() for name in registered}
        store.mla_suffix = mla_suffix
        store.mha_suffix = "0"
        store.is_mla_backend = True
        store.pp_rank = pp_rank
        store.pp_size = pp_size
        store.config_prefix = "tag_my-model"
        return store

    def test_layout_reports_suffixes_prefix_and_hit_policy(self):
        store = self._store(
            registered=(PoolName.DEEPSEEK_V4_C4, PoolName.DEEPSEEK_V4_C4_STATE)
        )

        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=[
                _sidecar(PoolName.DEEPSEEK_V4_C4, PoolHitPolicy.ALL_PAGES),
                _sidecar(PoolName.DEEPSEEK_V4_C4_STATE, PoolHitPolicy.TRAILING_PAGES),
            ],
            swa_trailing_pages=2,
            page_size=64,
        )

        self.assertEqual(layout["backend"], "mooncake")
        self.assertEqual(layout["page_size"], 64)

        object_layout = layout["hicache_object_layout"]
        self.assertEqual(object_layout["key_prefix"], "tag_my-model")
        self.assertTrue(object_layout["is_mla_backend"])
        self.assertEqual(object_layout["pp_rank"], 0)
        self.assertEqual(object_layout["pp_size"], 1)
        self.assertEqual(
            object_layout["pools"],
            [
                {
                    "pool": "deepseek_v4_c4",
                    "suffixes": ["__deepseek_v4_c4"],
                    "hit_policy": "all_pages",
                },
                {
                    "pool": "deepseek_v4_c4_state",
                    "suffixes": ["__deepseek_v4_c4_state"],
                    "hit_policy": "trailing_pages",
                    "trailing_pages": 2,
                },
            ],
        )

    def test_layout_skips_pools_that_are_not_registered(self):
        store = self._store(registered=(PoolName.DEEPSEEK_V4_C4,))

        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=[
                _sidecar(PoolName.DEEPSEEK_V4_C4, PoolHitPolicy.ALL_PAGES),
                _sidecar(PoolName.DEEPSEEK_V4_C128, PoolHitPolicy.ALL_PAGES),
            ],
            swa_trailing_pages=None,
            page_size=64,
        )

        pools = layout["hicache_object_layout"]["pools"]
        self.assertEqual([entry["pool"] for entry in pools], ["deepseek_v4_c4"])

    def test_layout_suffixes_embed_the_pp_rank(self):
        store = self._store(
            mla_suffix="1",
            pp_rank=1,
            pp_size=2,
            registered=(PoolName.DEEPSEEK_V4_C4,),
        )

        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=[
                _sidecar(PoolName.DEEPSEEK_V4_C4, PoolHitPolicy.ALL_PAGES)
            ],
            swa_trailing_pages=None,
            page_size=64,
        )

        object_layout = layout["hicache_object_layout"]
        self.assertEqual(object_layout["pools"][0]["suffixes"], ["_1_deepseek_v4_c4"])
        self.assertEqual(object_layout["pp_rank"], 1)
        self.assertEqual(object_layout["pp_size"], 2)


if __name__ == "__main__":
    unittest.main()
