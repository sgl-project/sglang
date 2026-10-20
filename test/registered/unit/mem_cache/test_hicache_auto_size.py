import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiCacheAutoSize(CustomTestCase):
    def test_v4_capacity_counts_host_pages_and_keeps_draft_physical(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.size = 8
        pool.swa_size = 8
        pool.swa_page_size = 2
        pool.start_layer, pool.end_layer = 0, 2
        pool._unified_kv = False
        pool.layer_mapping = [
            SimpleNamespace(compress_ratio=4, compress_layer_id=0),
            SimpleNamespace(compress_ratio=128, compress_layer_id=0),
        ]
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[None, None], bytes_per_page_padded=12
        )
        pool.c4_kv_pool = SimpleNamespace(kv_buffer=[None], bytes_per_page_padded=20)
        pool.c128_kv_pool = SimpleNamespace(kv_buffer=[None], bytes_per_page_padded=30)
        pool.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[torch.zeros((4, 8), dtype=torch.uint8)],
        )
        state = SimpleNamespace(
            ring_size=2,
            kv_score_buffer=SimpleNamespace(
                kv_score=torch.zeros((8, 3), dtype=torch.int16)
            ),
        )
        pool.compress_state_pools = [state, None]
        pool.indexer_compress_state_pools = [state, None]
        pool.sources_by_ratio = {2: [1]}
        pool.kv_pools = {
            2: SimpleNamespace(kv_buffer=[torch.zeros((4, 16), dtype=torch.uint8)])
        }
        pool.index_pools = {
            2: SimpleNamespace(
                page_size=2,
                index_k_with_scale_buffer=None,
                index_k_payload_buffer=[torch.zeros((4, 8), dtype=torch.uint8)],
                index_k_scale_buffer=[torch.zeros((4, 4), dtype=torch.uint8)],
            )
        }
        draft = SimpleNamespace(
            swa_kv_pool=SimpleNamespace(kv_buffer=[None], bytes_per_page_padded=5)
        )
        params = SimpleNamespace(
            page_size=4,
            req_to_token_pool=None,
            mtp_draft_device_pools=(draft,),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: pool, size_full=8
            ),
        )
        # Target: SWA 96 + C4 40 + C128 60 + indexer 16 + states 96
        # + C2 KV 32 + C2 indexer/scale 24 = 364. Draft adds 20.
        self.assertEqual(sizing._estimate_hicache_bytes(params, None), 384)
        for rank, expected in ((0, 384), (1, 20)):
            with (
                patch.object(
                    sizing,
                    "get_memory",
                    return_value=SimpleNamespace(enable_mla_hicache_host_dedup=True),
                ),
                patch.object(sizing, "mla_dedup_rank_and_size", return_value=(rank, 2)),
                patch.object(sizing, "mla_host_dedup_eligible", return_value=True),
            ):
                self.assertEqual(
                    sizing._estimate_hicache_bytes(params, None, physical=True),
                    expected,
                )

    def test_physical_estimate_keeps_rank_local_draft_and_state(self):
        target = Mock(
            spec=MLATokenToKVPool,
            size=100,
            host_capacity_bytes=None,
            get_kv_size_bytes=Mock(return_value=1000),
        )
        draft = Mock(
            spec=MLATokenToKVPool,
            size=50,
            host_capacity_bytes=None,
            get_kv_size_bytes=Mock(return_value=100),
        )
        state = Mock(host_capacity_bytes=None, get_kv_size_bytes=Mock(return_value=300))
        for hybrid in (False, True):
            pool = (
                Mock(spec=HybridLinearKVPool, full_kv_pool=target, size=100)
                if hybrid
                else target
            )
            params = CacheInitParams(
                disable=False,
                req_to_token_pool=(
                    Mock(spec=HybridReqToTokenPool, mamba_pool=state)
                    if hybrid
                    else None
                ),
                token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=pool)),
                page_size=2,
                mtp_draft_device_pools=(draft,),
            )
            for rank in (0, 1):
                with (
                    self.subTest(hybrid=hybrid, rank=rank),
                    patch.object(
                        sizing,
                        "get_memory",
                        return_value=SimpleNamespace(
                            enable_mla_hicache_host_dedup=True
                        ),
                    ),
                    patch.object(
                        sizing, "mla_dedup_rank_and_size", return_value=(rank, 4)
                    ),
                    patch.object(sizing, "mla_host_dedup_eligible", return_value=True),
                ):
                    independent = 200 + (300 if hybrid else 0)
                    self.assertEqual(
                        sizing._estimate_hicache_bytes(params, None, physical=True),
                        independent + (1000 if rank == 0 else 0),
                    )
                    self.assertEqual(
                        sizing._estimate_hicache_bytes(params, None), 1000 + independent
                    )

    def test_dedup_budget_counts_owners_and_independent_pools_per_host(self):
        cases = [
            # node, host allowance, physical bytes at ratio 1
            ([(0, 400, 100), (0, 400, 0), (0, 400, 0), (0, 400, 0)], 0, 4, 400),
            ([(0, 400, 100), (0, 400, 100), (0, 400, 0), (0, 400, 0)], 0, 2, 200),
            (
                [(0, 400, 110), (0, 400, 10), (0, 400, 10), (0, 400, 10)],
                1,
                400 / 140,
                28,
            ),
            # Different snapshots on one host use the smaller allowance.
            ([(0, 400, 100), (0, 300, 0)], 0, 3, 300),
            # Cross-node group: a dummy-only host does not cap the owner.
            ([(0, 400, 100), (1, 200, 0)], 1, float("inf"), 0),
            # PP/DP owners on another node consume that node's budget.
            ([(0, 400, 100), (1, 200, 200)], 1, 1, 200),
            # Real pool sizes can be NumPy scalars. TiB host budgets overflow
            # int64 when multiplied by even a modest physical pool size.
            (
                [(0, 2**41, np.int64(2**29)), (0, 2**41, 0)],
                0,
                4096,
                2**41,
            ),
        ]
        for records, rank, expected_ratio, expected_budget in cases:
            node, allowance, demand = records[rank]
            local_ranks = sum(row[0] == node for row in records)

            def gather(outputs, local, *, group):
                self.assertEqual(local.tolist(), [node, allowance, demand])
                for output, row in zip(outputs, records):
                    output.copy_(torch.tensor(row))

            with (
                self.subTest(records=records, rank=rank),
                patch.object(
                    sizing,
                    "get_parallel",
                    return_value=SimpleNamespace(
                        node_rank=node, world_group=SimpleNamespace(cpu_group="world")
                    ),
                ),
                patch.object(sizing, "ranks_per_host", return_value=local_ranks),
                patch.object(
                    sizing.torch.distributed, "is_initialized", return_value=True
                ),
                patch.object(
                    sizing.torch.distributed,
                    "get_world_size",
                    return_value=len(records),
                ),
                patch.object(
                    sizing.torch.distributed, "all_gather", side_effect=gather
                ),
            ):
                ratio, budget = sizing._dedup_host_budget(
                    demand, allowance // local_ranks
                )
                self.assertEqual(ratio, expected_ratio)
                self.assertEqual(budget, expected_budget)

    def test_auto_sizing_books_the_redistributed_owner_budget(self):
        memory = SimpleNamespace(
            hicache_host_memory_fraction=0.8,
            hicache_ratio=8,
            enable_mla_hicache_host_dedup=True,
        )
        context = Mock()
        with (
            patch.object(sizing, "get_memory", return_value=memory),
            patch.object(sizing, "get_context", return_value=context),
            patch.object(sizing, "_estimate_hicache_bytes", return_value=100),
            patch.object(sizing, "host_memory_budget_bytes", return_value=100),
            patch.object(
                sizing, "_dedup_host_budget", return_value=(3.2, 320)
            ) as share,
            patch.object(sizing, "ranks_per_host", return_value=4),
            patch.object(
                sizing.torch.distributed, "is_initialized", return_value=False
            ),
            sizing.auto_size_hicache(Mock(), None, enabled=True),
        ):
            share.assert_called_once_with(100, 80)
            context.override.assert_called_once_with(
                "hicache.auto_size", hicache_ratio=3.2 * 0.95
            )
            # The owner's real allocations can exceed its former equal-rank
            # allowance, but independent later pools still debit the snapshot.
            self.assertEqual(base.host_memory_budget_bytes(300), 320)
            self.assertEqual(base.host_memory_budget_bytes(21), 20)
            self.assertEqual(base._host_memory_budget.get(), 20)
        self.assertIsNone(base._host_memory_budget.get())

    def test_explicit_dedup_ratio_keeps_capacity_and_checks_budget(self):
        for requested in (4, 6):
            memory = SimpleNamespace(
                hicache_host_memory_fraction=None,
                hicache_ratio=requested,
                hicache_size=0,
                enable_mla_hicache_host_dedup=True,
            )
            context = Mock()
            with (
                self.subTest(requested=requested),
                patch.object(sizing, "get_memory", return_value=memory),
                patch.object(sizing, "get_context", return_value=context),
                patch.object(sizing, "_estimate_hicache_bytes", return_value=100),
                patch.object(
                    sizing, "host_memory_budget_bytes", return_value=125
                ) as available,
                patch.object(
                    sizing, "_dedup_host_budget", return_value=(5, 500)
                ) as share,
                patch.object(
                    sizing.torch.distributed, "is_initialized", return_value=False
                ),
            ):
                if requested == 4:
                    with sizing.auto_size_hicache(Mock(), None, enabled=True):
                        self.assertEqual(base.host_memory_budget_bytes(400), 500)
                        self.assertEqual(base._host_memory_budget.get(), 100)
                else:
                    with self.assertRaisesRegex(ValueError, "explicit HiCache ratio 6"):
                        with sizing.auto_size_hicache(Mock(), None, enabled=True):
                            self.fail("An oversized explicit ratio reached allocation")
                available.assert_called_once_with(auto_size=False)
                share.assert_called_once_with(100, 125)
                context.override.assert_not_called()
                self.assertIsNone(base._host_memory_budget.get())

    def test_hybrid_target_draft_uses_its_sidecar_slot_capacity(self):
        # EAGLE3 can pair a hybrid target with a plain MHA draft. Full drafts
        # follow full target slots; SWA drafts follow the smaller SWA capacity.
        target = Mock(
            spec=SWAKVPool,
            size=128,
            full_kv_pool=Mock(
                size=128,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=4096),
            ),
            swa_kv_pool=Mock(
                size=32,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=1024),
            ),
        )
        draft_mha = Mock(
            spec=MHATokenToKVPool,
            size=16,
            host_capacity_bytes=None,
            get_kv_size_bytes=Mock(return_value=(128, 128)),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=2,
        )
        for draft, expected_sidecar_bytes in (
            (draft_mha, 2048),
            (Mock(spec=BaseSWAKVPool, swa_kv_pool=draft_mha), 512),
        ):
            with self.subTest(
                draft_type=type(draft).__name__, bytes=expected_sidecar_bytes
            ):
                plan = Mock(mode="sidecar", device_pools=(draft,))
                self.assertEqual(
                    sizing._estimate_hicache_bytes(params, plan),
                    4096 + 1024 + expected_sidecar_bytes,
                )

    def test_unified_views_are_sized_from_host_capacity(self):
        """Unified sub-pools answer get_kv_size_bytes with zero (UnifiedKVPool
        logs the shared buffer once) and publish host_capacity_bytes instead,
        the weight the explicit host-size split already uses. An estimate taken
        from get_kv_size_bytes is zero device bytes, and the default ratio then
        divides by it."""
        target = Mock(
            spec=SWAKVPool,
            size=128,
            full_kv_pool=Mock(
                size=128,
                host_capacity_bytes=4096,
                get_kv_size_bytes=Mock(return_value=(0, 0)),
            ),
            swa_kv_pool=Mock(
                size=32,
                host_capacity_bytes=1024,
                get_kv_size_bytes=Mock(return_value=(0, 0)),
            ),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=2,
        )
        self.assertEqual(sizing._estimate_hicache_bytes(params, None), 4096 + 1024)

    def test_default_ratio_fits_host_budget_and_pools_book_it(self):
        """With only --enable-hierarchical-cache the default ratio shrinks to the
        per-rank budget, pools book one snapshot, and an explicit ratio opts out."""
        pool = MHATokenToKVPool(
            size=128,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=pool)),
            page_size=2,
        )
        rank_budget = 10_000
        # Four ranks per host (e.g. TP8 over two 4-GPU nodes) share what psutil reports.
        host_free = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 4 * rank_budget
        with (
            get_context().override_server_args(enable_hierarchical_cache=True),
            patch.object(base, "ranks_per_host", return_value=4),
            patch.object(base, "available_host_memory_bytes", return_value=host_free),
            sizing.auto_size_hicache(params, None, enabled=True),
        ):
            ratio = get_memory().hicache_ratio
            self.assertLess(ratio, 2.0)
            host = MHATokenToKVPoolHost(
                pool.get_device_pool_infos()[0].buffer_info,
                ratio,
                0,
                2,
                "layer_first",
                pin_memory=False,
                device="cpu",
                device_capacity_tokens=pool.size
                if pool.host_capacity_tokens is None
                else pool.host_capacity_tokens,
                start_layer=pool.start_layer,
                end_layer=pool.end_layer,
                write_back_staging=pool.hicache_write_back_staging,
            )
            self.assertLessEqual(host.size * host.size_per_token, 0.8 * rank_budget)
            with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                MHATokenToKVPoolHost(
                    pool.get_device_pool_infos()[0].buffer_info,
                    ratio,
                    0,
                    2,
                    "layer_first",
                    pin_memory=False,
                    device="cpu",
                    device_capacity_tokens=pool.size
                    if pool.host_capacity_tokens is None
                    else pool.host_capacity_tokens,
                    start_layer=pool.start_layer,
                    end_layer=pool.end_layer,
                    write_back_staging=pool.hicache_write_back_staging,
                )
        self.assertIsNone(base._host_memory_budget.get())

        explicit = ServerArgs(model_path="dummy", hicache_ratio=2.0)
        explicit.resolve_once()
        self.assertIsNone(
            resolution_result(explicit, "hicache_host_memory_fraction", 0.8)
        )


if __name__ == "__main__":
    unittest.main()
