import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import (
    build_dspark_swa_page_indices,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    DraftSWASidecarLayout,
    use_dsv4_dspark_draft_swa_sidecar,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    resolve_pool_hit_boundary,
)
from sglang.srt.mem_cache.unified_cache.components.draft_swa_sidecar_component import (
    DraftSWASidecarComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4DraftSWASidecar(unittest.TestCase):
    def test_all_pages_keeps_partial_prefix_semantics(self):
        transfer = PoolTransfer(
            name=PoolName.DRAFT_SWA,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )

        boundary, sidecar_hit = resolve_pool_hit_boundary(
            kv_pages=10,
            transfer=transfer,
            has_component=lambda i: i < 6,
        )

        self.assertEqual(boundary, 6)
        self.assertTrue(sidecar_hit)

    def test_storage_miss_recomputes_one_trailing_page(self):
        transfer = PoolTransfer(
            name=PoolName.DRAFT_SWA,
            keys=["tail"],
            hit_policy=PoolHitPolicy.RECOMPUTE_TRAILING,
        )

        boundary, sidecar_hit = resolve_pool_hit_boundary(
            kv_pages=10,
            transfer=transfer,
            has_component=lambda _: False,
        )

        self.assertEqual(boundary, 9)
        self.assertFalse(sidecar_hit)

    def test_storage_hit_reuses_full_prefix(self):
        transfer = PoolTransfer(
            name=PoolName.DRAFT_SWA,
            keys=["tail"],
            hit_policy=PoolHitPolicy.RECOMPUTE_TRAILING,
        )

        boundary, sidecar_hit = resolve_pool_hit_boundary(
            kv_pages=10,
            transfer=transfer,
            has_component=lambda i: i == 9,
        )

        self.assertEqual(boundary, 10)
        self.assertTrue(sidecar_hit)

    def test_dependent_component_tracks_coverage_without_owning_slots(self):
        root = SimpleNamespace(parent=None, component_data=[None, None, None])
        swa_data = SimpleNamespace(
            metadata={},
            value=torch.arange(256),
            host_value=torch.arange(256),
        )
        node = SimpleNamespace(
            parent=root,
            component_data=[None, swa_data, None],
        )
        cache = SimpleNamespace(
            page_size=256,
            sliding_window_size=128,
            tree_core=SimpleNamespace(root_node=root),
        )
        component = DraftSWASidecarComponent(cache)

        component.mark_device(node)
        component.mark_host(node)

        self.assertTrue(component.has_device_window(node))
        self.assertTrue(component.has_host_window(node))
        component.clear_device(node)
        self.assertFalse(component.has_device_window(node))
        self.assertIsNotNone(swa_data.value)

    def test_missing_sidecar_match_requires_recomputable_tail(self):
        root = SimpleNamespace(parent=None, component_data=[None, None, None])
        swa_data = SimpleNamespace(
            metadata={},
            value=torch.arange(256),
            host_value=None,
        )
        node = SimpleNamespace(
            parent=root,
            key=list(range(256)),
            component_data=[None, swa_data, None],
        )
        cache = SimpleNamespace(
            page_size=256,
            sliding_window_size=128,
            tree_core=SimpleNamespace(root_node=root),
        )
        sidecar = DraftSWASidecarComponent(cache)
        swa = object.__new__(SWAComponent)
        swa.sliding_window_size = 128
        swa.tree_core = SimpleNamespace(
            has_swa_host_pool=False,
            enable_hicache=False,
        )
        swa.draft_swa_sidecar = sidecar

        exact_match = swa.create_match_validator_for_key(
            match_device_only=True,
            match_key_len=256,
        )
        match_with_tail = swa.create_match_validator_for_key(
            match_device_only=True,
            match_key_len=512,
        )

        self.assertFalse(exact_match(node))
        self.assertTrue(match_with_tail(node))
        sidecar.mark_device(node)
        covered_exact_match = swa.create_match_validator_for_key(
            match_device_only=True,
            match_key_len=256,
        )
        self.assertTrue(covered_exact_match(node))

    def test_rollout_defaults_on_and_keeps_explicit_fallback(self):
        spec = SimpleNamespace(is_dspark=lambda: True)
        with (
            patch(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool.is_npu",
                return_value=False,
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate."
                "is_unified_kv_triton",
                return_value=False,
            ),
        ):
            self.assertTrue(
                use_dsv4_dspark_draft_swa_sidecar(
                    SimpleNamespace(speculative_dspark_draft_swa_sidecar=None), spec
                )
            )
            self.assertFalse(
                use_dsv4_dspark_draft_swa_sidecar(
                    SimpleNamespace(speculative_dspark_draft_swa_sidecar=False), spec
                )
            )

    def test_explicit_enable_rejects_request_scoped_unified_kv(self):
        spec = SimpleNamespace(is_dspark=lambda: True)
        with (
            patch(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool.is_npu",
                return_value=False,
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate."
                "is_unified_kv_triton",
                return_value=True,
            ),
            self.assertRaisesRegex(NotImplementedError, "request-scoped SWA ring"),
        ):
            use_dsv4_dspark_draft_swa_sidecar(
                SimpleNamespace(speculative_dspark_draft_swa_sidecar=True), spec
            )

    def test_scratch_rows_are_outside_committed_sidecar(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.draft_swa_layout = DraftSWASidecarLayout.build(
            committed_size=1024,
            scratch_width=5,
            num_req_slots=4,
            page_size=256,
        )

        locs = pool.get_draft_swa_scratch_locs(
            torch.tensor([0, 3], dtype=torch.int64), block_size=5
        )

        self.assertEqual(
            locs.tolist(),
            [1280, 1281, 1282, 1283, 1284, 1295, 1296, 1297, 1298, 1299],
        )
        self.assertEqual(pool.draft_swa_layout.committed_pages, 5)
        self.assertEqual(pool.draft_swa_layout.physical_size, 1299)
        self.assertTrue(torch.all(locs > pool.draft_swa_layout.committed_size))
        with self.assertRaisesRegex(ValueError, "exceeds scratch width"):
            pool.get_draft_swa_scratch_locs(
                torch.tensor([0], dtype=torch.int64), block_size=6
            )

    def test_pd_buffer_range_excludes_scratch_pages(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.draft_swa_layout = DraftSWASidecarLayout.build(
            committed_size=1024,
            scratch_width=5,
            num_req_slots=4,
            page_size=256,
        )
        pool.swa_page_size = 256
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty((10, 8), dtype=torch.uint8)]
        )
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []

        _, data_lens, item_lens = pool.get_state_buf_infos()

        self.assertEqual(item_lens, [8])
        self.assertEqual(data_lens, [5 * 8])
        self.assertLess(data_lens[0], pool.swa_kv_pool.kv_buffer[0].nbytes)

    def test_hicache_view_excludes_scratch_pages(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.draft_swa_layout = DraftSWASidecarLayout.build(
            committed_size=1024,
            scratch_width=5,
            num_req_slots=4,
            page_size=256,
        )
        backing = torch.empty((10, 8), dtype=torch.uint8)
        pool.swa_kv_pool = SimpleNamespace(kv_buffer=[backing])

        committed = pool.get_draft_swa_committed_buffers()

        self.assertEqual(committed[0].shape, (5, 8))
        self.assertEqual(committed[0].data_ptr(), backing.data_ptr())
        self.assertLess(committed[0].nbytes, backing.nbytes)

    def test_page_indices_read_direct_scratch_rows(self):
        req_to_token = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
        full_to_swa = torch.arange(64, dtype=torch.int64) + 20

        indices, lengths = build_dspark_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa,
            req_pool_indices_per_request=torch.tensor([0]),
            offsets=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            invalid=torch.zeros((1, 4), dtype=torch.bool),
            out_loc=torch.tensor([50, 51], dtype=torch.int64),
            context_lens=torch.tensor([3], dtype=torch.int32),
            block_size=2,
            swa_window=4,
            page_index_aligned_size=8,
            block_swa_locs=torch.tensor([100, 101], dtype=torch.int64),
        )

        self.assertEqual(indices.shape, (2, 8))
        self.assertEqual(indices[0, :5].tolist(), [31, 32, 33, 100, 101])
        self.assertTrue(torch.equal(indices[0], indices[1]))
        self.assertEqual(lengths.tolist(), [5, 5])


if __name__ == "__main__":
    unittest.main()
