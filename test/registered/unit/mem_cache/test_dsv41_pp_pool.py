import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.utils import build_kv_layer_ids
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.pool_configurator import (
    resolve_dsv4_local_pool_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _pool(start, end):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._stage_start = start
    pool._stage_end = end
    pool.compression_ratios = [0, 0] + [2] * 18 + [1] * 20
    pool.kv_source_layers = [2, 8, 14, 20]
    return pool


class TestDeepSeekV41PPPool(unittest.TestCase):
    def test_pool_layout_accounts_for_preceding_sources(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20
        sources = [2, 8, 14, 20]

        expected_sources = ({2, 8}, {8, 14}, {20}, {20})
        for stage, expected in enumerate(expected_sources):
            local_ratios, local_sources = resolve_dsv4_local_pool_layout(
                ratios,
                sources,
                tuple(range(stage * 10, (stage + 1) * 10)),
            )
            self.assertEqual(local_ratios, ratios[stage * 10 : (stage + 1) * 10])
            self.assertEqual(local_sources, expected)

    def test_consumer_stage_allocates_preceding_source(self):
        pool = _pool(30, 40)

        self.assertEqual(
            DeepSeekV4TokenToKVPool._collect_sources_by_ratio(pool),
            {1: [20]},
        )

    def test_pd_registration_excludes_consumer_replica(self):
        pool = _pool(30, 40)
        pool.sources_by_ratio = {1: [20]}
        pool.kv_pools = {1: object()}
        pool.index_pools = {}

        self.assertEqual(pool.get_kv_layer_ids(), [])
        self.assertEqual(
            build_kv_layer_ids(
                token_to_kv_pool=pool,
                draft_token_to_kv_pool=None,
                num_draft_entries=0,
                num_hidden_layers=40,
            ),
            [],
        )

    def test_pd_registration_reports_producer_layer(self):
        pool = _pool(20, 30)
        pool.sources_by_ratio = {1: [20]}
        pool.kv_pools = {1: object()}
        pool.index_pools = {}

        self.assertEqual(pool.get_kv_layer_ids(), [20])

    def test_state_layer_ids_follow_registered_buffers(self):
        pool = _pool(0, 10)
        pool.swa_kv_pool = object()
        pool._unified_kv = False
        pool.compress_state_pools = [
            None,
            None,
            SimpleNamespace(request_scoped=False),
            SimpleNamespace(request_scoped=True),
        ]
        pool.indexer_compress_state_pools = [
            None,
            SimpleNamespace(request_scoped=False),
        ]

        self.assertEqual(pool.get_state_layer_ids(), list(range(10)) + [2, 1])
        self.assertEqual(pool.get_request_state_layer_ids(), [3])
        self.assertEqual(pool.get_unified_swa_ring_layer_ids(), [])

    def test_request_state_pools_share_transfer_indices(self):
        pool = _pool(0, 10)
        indices = np.array([7], dtype=np.int32)
        pool.compress_state_pools = [
            SimpleNamespace(
                request_scoped=True,
                ratio=2,
                online=False,
                ring_size=2,
                transfer_indices=lambda req, seq: indices,
            )
            for _ in range(3)
        ]

        actual = pool.request_state_transfer_indices(7, 17)

        np.testing.assert_array_equal(actual, indices)

    def test_request_state_pools_reject_different_transfer_indices(self):
        pool = _pool(0, 10)
        pool.compress_state_pools = [
            SimpleNamespace(
                request_scoped=True,
                ratio=2,
                online=False,
                ring_size=2,
                transfer_indices=lambda req, seq: np.array([7], dtype=np.int32),
            ),
            SimpleNamespace(
                request_scoped=True,
                ratio=2,
                online=False,
                ring_size=4,
                transfer_indices=lambda req, seq: np.array([8], dtype=np.int32),
            ),
        ]

        with self.assertRaisesRegex(AssertionError, "one ring layout"):
            pool.request_state_transfer_indices(7, 17)


if __name__ == "__main__":
    unittest.main()
