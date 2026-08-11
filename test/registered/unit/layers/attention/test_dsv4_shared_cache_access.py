import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.dsv4.shared_cache_access import (
    DSV4SharedCacheAccess,
    DSV4SharedPageLayout,
    get_dsv4_shared_cache_access,
)
from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4SharedPageLayout(CustomTestCase):
    def test_page_adapter_preserves_packed_page_and_slot_semantics(self):
        owner_layout = OwnerShardedLayout(
            cp_size=4,
            ownership_granule=256,
            logical_rows=10 * 256,
            physical_blocks_per_rank=8,
        )
        layout = DSV4SharedPageLayout(owner_layout)

        pages = torch.tensor([-1, 0, 1, 2, 3, 4], dtype=torch.int32)
        slots = torch.tensor([-1, 0, 257, 512, 768, 1027], dtype=torch.int64)

        self.assertEqual(layout.cp_size, 4)
        self.assertEqual(layout.page_size, 256)
        self.assertEqual(layout.pages_per_rank, 8)
        self.assertEqual(layout.owner_layout.minimum_blocks_per_rank, 3)
        self.assertEqual(
            layout.translate_pages(pages).tolist(),
            [-1, 0, 8, 16, 24, 1],
        )
        self.assertEqual(
            layout.translate_slots(slots).tolist(),
            [-1, 0, 2049, 4096, 6144, 259],
        )

    def test_rank_relative_and_owner_local_aliases_delegate_to_neutral_layout(self):
        owner_layout = OwnerShardedLayout(
            cp_size=4,
            ownership_granule=4,
            logical_rows=40,
            physical_blocks_per_rank=8,
        )
        layout = DSV4SharedPageLayout(owner_layout)
        slots = torch.tensor([-1, 0, 4, 8, 12, 20], dtype=torch.int64)

        self.assertEqual(
            layout.translate_slots_for_rank(slots, rank=2).tolist(),
            [-1, 64, 96, 0, 32, 100],
        )
        self.assertEqual(
            layout.owner_layout.owner_local_rows(slots).tolist(),
            [-1, 0, 0, 0, 0, 4],
        )
        self.assertEqual(
            layout.owner_layout.owned_row_mask(slots, rank=1).tolist(),
            [False, False, True, False, False, True],
        )

    def test_dsv4_shared_pool_does_not_import_the_dsa_model_module(self):
        from sglang.srt.mem_cache import deepseek_v4_shared

        self.assertNotIn("dsa_cache_shared", inspect.getsource(deepseek_v4_shared))

    def test_base_pool_has_no_shared_adapter(self):
        self.assertIsNone(get_dsv4_shared_cache_access(SimpleNamespace()))

    def test_shared_adapter_has_explicit_release_surface(self):
        pool = MagicMock()
        access = DSV4SharedCacheAccess(pool)
        pool.shared_cache_access = access

        self.assertIs(get_dsv4_shared_cache_access(pool), access)
        expected = {
            "publish_writes",
            "flashmla_prefill_demand_kwargs",
            "translate_slots",
            "kv_owner_write_target",
            "compressor_state_layout",
        }
        self.assertTrue(expected.issubset(set(dir(access))))
        self.assertNotIn("__getattr__", DSV4SharedCacheAccess.__dict__)

    def test_shared_adapter_omits_unconsumed_page_compatibility_surface(self):
        obsolete_access = {
            "translate_pages",
            "plan_flashmla_kv_read",
            "stage_sparse_pages",
            "prepare_swa_pages",
            "stage_swa_pages",
            "prepare_extra_pages",
            "stage_extra_pages",
        }
        obsolete_layout = {
            "local_pages_per_layer",
            "translate_pages_for_rank",
            "translate_local_slots",
            "owned_slot_mask",
        }

        self.assertTrue(obsolete_access.isdisjoint(DSV4SharedCacheAccess.__dict__))
        self.assertTrue(obsolete_layout.isdisjoint(DSV4SharedPageLayout.__dict__))

    def test_adapter_publication_and_owner_write_delegate_explicitly(self):
        pool = MagicMock()
        pool.get_compressor_write_info.return_value = ("cache", 3, 8)
        access = DSV4SharedCacheAccess(pool)

        access.publish_writes()
        result = access.kv_owner_write_target(17, is_indexer=True)

        pool.synchronize_shared_writes.assert_called_once_with()
        pool.get_compressor_write_info.assert_called_once_with(17, is_indexer=True)
        self.assertEqual(result, ("cache", 3, 8))

    def test_flashmla_prefill_demand_kwargs_are_built_by_the_adapter(self):
        pool = MagicMock()
        pool.prefill_demand_cache.next_view.return_value = SimpleNamespace(
            rows="rows",
            tags="tags",
            stats="stats",
            epoch=17,
            ways=1,
        )
        pool.get_flashmla_demand_layout.return_value = (3, 8, 256, 64, 2, 32)
        pool.prefill_demand_cache_direct_slots = False
        access = DSV4SharedCacheAccess(pool)

        self.assertEqual(
            access.flashmla_prefill_demand_kwargs(11),
            {
                "shared_kv_row_cache": "rows",
                "shared_kv_cache_tags": "tags",
                "shared_kv_cache_stats": "stats",
                "shared_kv_cache_epoch": 17,
                "shared_kv_cache_ways": 1,
                "shared_kv_cache_direct_slots": False,
                "shared_kv_rank": 3,
                "shared_kv_size": 8,
                "shared_swa_page_size": 256,
                "shared_swa_pages_per_rank": 64,
                "shared_extra_page_size": 2,
                "shared_extra_pages_per_rank": 32,
            },
        )

    def test_adapter_uses_local_layout_for_replicated_indexer_state(self):
        state_pool = SimpleNamespace()

        self.assertEqual(
            DSV4SharedCacheAccess.compressor_state_layout(state_pool),
            (0, 1, 0),
        )

    def test_attention_and_model_call_sites_do_not_probe_shared_pool_methods(self):
        attention_dir = Path(__file__).parents[5] / "python/sglang/srt/layers/attention"
        production_files = (
            attention_dir / "deepseek_v4_backend.py",
            attention_dir / "dsv4/indexer.py",
            attention_dir / "dsv4/compressor_v2.py",
            attention_dir / "dsv4/metadata.py",
            attention_dir.parent.parent / "models/deepseek_v4.py",
        )
        legacy_capability_names = (
            '"synchronize_shared_writes"',
            '"prepare_compressed_pages_for_read"',
            "prepare_shared_flashmla_kv_read",
            "_dsv4_shared_indexer_stage_plan",
            '"translate_swa_pages_for_read"',
            '"get_compressor_write_info"',
            '"get_shared_state_layout"',
        )

        for production_file in production_files:
            source = production_file.read_text()
            for capability in legacy_capability_names:
                with self.subTest(file=production_file.name, capability=capability):
                    self.assertNotIn(capability, source)


if __name__ == "__main__":
    unittest.main()
