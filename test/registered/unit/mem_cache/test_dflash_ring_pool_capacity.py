"""Draft ring capacity is independent of the target allocator's virtual IDs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_pool_rows,
    draft_kv_ring_rows_per_req,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestDFlashRingPoolCapacity(CustomTestCase):
    def _init_pools(self, *, ring, shrink=0):
        # Real page-interleaved address-space property, without GPU storage.
        alloc = PageInterleavePoolAllocator.__new__(PageInterleavePoolAllocator)
        alloc.size, alloc.shard_size, alloc.page_size = 131072, 8, 64
        self.assertEqual(alloc.draft_virtual_id_space, 131520)
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.is_draft_worker = True
        cfg.spec_algorithm = SpeculativeAlgorithm.DFLASH
        cfg.is_hybrid_swa_mtp_draft = False
        cfg.kv_shard_rank = None
        cfg.model_config = SimpleNamespace(hf_config=None)
        req = SimpleNamespace(req_to_token=torch.empty((17, 1), dtype=torch.int32))
        sizes = kcc._PoolSizes(
            max_total_num_tokens=16384,
            max_running_requests=16,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
            c4_max_total_num_tokens=0,
            c128_max_total_num_tokens=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            c4_state_dtype=None,
            c128_state_dtype=None,
        )
        with (
            envs.SGLANG_DFLASH_DRAFT_KV_RING.override(ring),
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(enable_unified_memory=False),
            ),
            patch.object(
                kcc, "get_parallel", return_value=SimpleNamespace(attn_dcp_size=1)
            ),
            patch.object(
                kcc,
                "get_schedule",
                return_value=SimpleNamespace(
                    page_size=64, prefill_only_disable_kv_cache=False
                ),
            ),
            patch.object(
                kcc,
                "get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=6),
            ),
            patch(
                "sglang.srt.speculative.dflash_draft_ring.draft_kv_ring_window_left",
                return_value=1023,
            ),
            patch.object(kcc, "is_deepseek_dsa", return_value=False),
            patch.object(kcc, "is_deepseek_v4", return_value=False),
            patch.object(
                kcc.KVCacheConfigurator,
                "_validate_prefill_only_disable_kv_cache_pool_family",
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_build_token_to_kv_pool",
                side_effect=lambda *, sizes, **kw: SimpleNamespace(
                    size=sizes.max_total_num_tokens - shrink
                ),
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_build_token_to_kv_pool_allocator",
                return_value=alloc,
            ),
        ):
            return cfg._init_pools(
                sizes=sizes, req_to_token_pool=req, token_to_kv_pool_allocator=alloc
            )

    def test_ring_uses_request_rows_instead_of_target_virtual_space(self):
        pools = self._init_pools(ring=True)
        # Production failure: 17 request slots * 1031 rows + sink, page-aligned.
        self.assertEqual(pools.token_to_kv_pool.size, 17536)
        self.assertLess(
            pools.token_to_kv_pool.size,
            pools.token_to_kv_pool_allocator.draft_virtual_id_space,
        )

    def test_ring_rejects_an_undersized_pool(self):
        with self.assertRaisesRegex(AssertionError, "draft KV ring"):
            self._init_pools(ring=True, shrink=64)

    def test_non_ring_still_spans_all_target_virtual_ids(self):
        self.assertEqual(self._init_pools(ring=False).token_to_kv_pool.size, 131520)
        with self.assertRaisesRegex(AssertionError, "virtual-id space"):
            self._init_pools(ring=False, shrink=64)

    def test_ring_writes_fit_pool_for_high_positions_and_last_request(self):
        for dcp in (1, 8):
            with self.subTest(dcp=dcp):
                rows = draft_kv_ring_rows_per_req(
                    window_left=1023, block_size=6, dcp_size=dcp
                )
                ring = DFlashDraftKVRing(
                    rows_per_req=rows, dcp_size=dcp, req_pool_rows=17, device="cpu"
                )
                pool_size = draft_kv_ring_pool_rows(
                    req_pool_rows=17, rows_per_req=rows, page_size=64
                )
                pos = torch.arange(131072, 131072 + rows * dcp * 2)
                locs = ring.locs(torch.full_like(pos, 16), pos)
                self.assertGreater(int((locs // dcp).min()), 0)
                self.assertLess(int((locs // dcp).max()), pool_size)


if __name__ == "__main__":
    unittest.main()
