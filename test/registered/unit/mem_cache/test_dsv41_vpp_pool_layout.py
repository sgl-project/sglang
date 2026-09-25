import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.pool_configurator import (
    resolve_dsv4_local_pool_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepSeekV41VPPPoolLayout(unittest.TestCase):
    def _pool(self, layer_ids):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.layer_ids = tuple(layer_ids)
        pool._local_layer_index = {
            layer_id: index for index, layer_id in enumerate(layer_ids)
        }
        pool.compression_ratios = [0, 0] + [2] * 18 + [1] * 20
        pool.kv_source_layers = [2, 8, 14, 20]
        return pool

    def test_consumer_stage_registers_remote_low_ratio_source(self):
        pool = self._pool(range(25, 30))

        sources = pool._collect_sources_by_ratio()

        self.assertEqual(sources, {1: [20]})
        pool.sources_by_ratio = sources
        self.assertEqual(pool.source_layer_of(29), 20)

    def test_stage_starting_after_source_maps_to_replicated_source(self):
        pool = self._pool(range(9, 20))
        pool.sources_by_ratio = pool._collect_sources_by_ratio()
        pool.kv_pools = {2: object()}

        pool._init_compressed_layer_mapping()

        self.assertEqual(pool.sources_by_ratio, {2: [8, 14]})
        self.assertEqual(pool.layer_mapping[9].compress_layer_id, 0)
        self.assertEqual(pool.layer_mapping[14].compress_layer_id, 1)

    def test_swa_layer_ids_map_to_dense_local_indices(self):
        pool = self._pool(tuple(range(0, 5)) + tuple(range(20, 25)))

        self.assertEqual(pool._swa_local_layer_id(0), 0)
        self.assertEqual(pool._swa_local_layer_id(20), 5)
        self.assertEqual(pool._swa_local_layer_id(24), 9)

    def test_pd_manifest_excludes_consumer_replica(self):
        pool = self._pool(range(25, 30))
        pool.sources_by_ratio = {1: [20]}
        pool.index_pools = {
            1: SimpleNamespace(contiguous_page_row_buffers=lambda: [object()])
        }

        self.assertEqual(pool.get_kv_layer_ids(), [])

    def test_pd_manifest_includes_producer_kv_and_index_entries(self):
        pool = self._pool(range(20, 25))
        pool.sources_by_ratio = {1: [20]}
        pool.index_pools = {
            1: SimpleNamespace(contiguous_page_row_buffers=lambda: [object()])
        }

        self.assertEqual(pool.get_kv_layer_ids(), [20, 20])

    def test_pool_budget_uses_only_owned_layers_and_required_sources(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20
        layer_ids = tuple(range(0, 5)) + tuple(range(20, 25))

        local_ratios, low_ratio_sources = resolve_dsv4_local_pool_layout(
            ratios,
            [2, 8, 14, 20],
            layer_ids,
        )

        self.assertEqual(local_ratios, [0, 0, 2, 2, 2] + [1] * 5)
        self.assertEqual(low_ratio_sources, {2, 20})

    def test_state_manifest_uses_absolute_non_contiguous_layer_ids(self):
        pool = self._pool(tuple(range(0, 5)) + tuple(range(20, 25)))
        pool.swa_kv_pool = SimpleNamespace()
        ratio4_state = SimpleNamespace(ratio=4)
        pool.compress_state_pools = [None] * 40
        pool.indexer_compress_state_pools = [None] * 40
        pool.compress_state_pools[20] = ratio4_state
        pool.indexer_compress_state_pools[20] = ratio4_state

        self.assertEqual(
            pool.get_state_layer_ids(),
            list(range(0, 5)) + list(range(20, 25)) + [20, 20],
        )

    def test_c128_state_manifest_contains_only_producer_layers(self):
        pool = self._pool(range(20, 25))
        pool.compress_state_pools = [None] * 40
        pool.compress_state_pools[20] = SimpleNamespace(ratio=2)

        self.assertEqual(pool.get_c128_state_layer_ids(), [20])

    def test_source_pages_install_into_consumer_local_page_ids(self):
        producer = self._pool(range(20, 25))
        producer.sources_by_ratio = {1: [20]}
        producer.index_pools = {}
        producer.kv_pools = {
            1: SimpleNamespace(kv_buffer=[torch.arange(30).reshape(6, 5)])
        }
        payload = producer.export_source_pages(
            20, torch.tensor([1, 3], dtype=torch.int64)
        )

        consumer = self._pool(range(25, 30))
        consumer.sources_by_ratio = {1: [20]}
        consumer.index_pools = {}
        consumer_buffer = torch.zeros((6, 5), dtype=torch.int64)
        consumer.kv_pools = {1: SimpleNamespace(kv_buffer=[consumer_buffer])}
        consumer.install_source_pages(
            20,
            torch.tensor([4, 2], dtype=torch.int64),
            payload,
        )

        self.assertTrue(
            torch.equal(consumer_buffer[4], producer.kv_pools[1].kv_buffer[0][1])
        )
        self.assertTrue(
            torch.equal(consumer_buffer[2], producer.kv_pools[1].kv_buffer[0][3])
        )

    def test_source_index_pages_follow_consumer_page_remap(self):
        producer = self._pool(range(20, 25))
        producer.page_size = 8
        producer.sources_by_ratio = {1: [20]}
        producer.kv_pools = {
            1: SimpleNamespace(kv_buffer=[torch.arange(30).reshape(6, 5)])
        }
        producer_index = torch.arange(48).reshape(24, 2)
        producer.index_pools = {
            1: SimpleNamespace(
                page_size=2,
                index_k_with_scale_buffer=[producer_index],
            )
        }
        payload = producer.export_source_pages(20, torch.tensor([1], dtype=torch.int64))

        consumer = self._pool(range(25, 30))
        consumer.page_size = 8
        consumer.sources_by_ratio = {1: [20]}
        consumer.kv_pools = {
            1: SimpleNamespace(kv_buffer=[torch.zeros((6, 5), dtype=torch.int64)])
        }
        consumer_index = torch.zeros((24, 2), dtype=torch.int64)
        consumer.index_pools = {
            1: SimpleNamespace(
                page_size=2,
                index_k_with_scale_buffer=[consumer_index],
            )
        }

        consumer.install_source_pages(
            20,
            torch.tensor([3], dtype=torch.int64),
            payload,
        )

        self.assertTrue(torch.equal(consumer_index[12:16], producer_index[4:8]))


if __name__ == "__main__":
    unittest.main()
