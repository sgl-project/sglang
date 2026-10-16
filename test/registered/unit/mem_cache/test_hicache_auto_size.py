import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiCacheAutoSize(CustomTestCase):
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
                pool, ratio, 0, 2, "layer_first", pin_memory=False, device="cpu"
            )
            self.assertLessEqual(host.size * host.size_per_token, 0.8 * rank_budget)
            with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                MHATokenToKVPoolHost(
                    pool, ratio, 0, 2, "layer_first", pin_memory=False, device="cpu"
                )
        self.assertIsNone(base._host_memory_budget.get())

        explicit = ServerArgs(model_path="dummy", hicache_ratio=2.0)
        explicit.resolve_once()
        self.assertIsNone(
            resolution_result(explicit, "hicache_host_memory_fraction", 0.8)
        )


class TestMLADedupHostBudget(unittest.TestCase):
    def setUp(self):
        self.target = Mock(
            spec=MLATokenToKVPool,
            size=8,
            host_capacity_bytes=192,
        )
        self.params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=self.target)),
            page_size=2,
        )

    @contextmanager
    def _runtime(
        self,
        weights,
        *,
        rank=0,
        local_ranks=8,
        dedup_size=8,
        available=8000,
        fraction=None,
        enabled=True,
        eligible=True,
    ):
        memory = SimpleNamespace(
            enable_mla_hicache_host_dedup=enabled,
            hicache_storage_backend=None,
            hicache_host_memory_fraction=fraction,
            hicache_ratio=2.0,
        )
        world = SimpleNamespace(
            world_size=len(weights), rank_in_group=rank, cpu_group=object()
        )
        available = (
            [available] * len(weights) if isinstance(available, int) else available
        )

        def gather(outputs, value, **kwargs):
            self.assertIs(kwargs["group"], world.cpu_group)
            self.assertEqual(value.tolist(), [weights[rank], available[rank]])
            for output, weight, free in zip(outputs, weights, available):
                output.copy_(torch.tensor([weight, free]))

        def override(_source, **kwargs):
            memory.hicache_ratio = kwargs["hicache_ratio"]

        with ExitStack() as stack:
            patches = (
                patch.object(sizing, "get_memory", return_value=memory),
                patch.object(
                    sizing, "get_context", return_value=Mock(override=override)
                ),
                patch.object(
                    sizing,
                    "get_parallel",
                    return_value=SimpleNamespace(world_group=world),
                ),
                patch.object(sizing, "ranks_per_host", return_value=local_ranks),
                patch.object(base, "ranks_per_host", return_value=local_ranks),
                patch.object(
                    base,
                    "available_host_memory_bytes",
                    return_value=base.HICACHE_HOST_MEMORY_RESERVE_BYTES
                    + available[rank],
                ),
                patch.object(
                    sizing,
                    "mla_dedup_rank_and_size",
                    return_value=(rank % dedup_size, dedup_size),
                ),
                patch(
                    "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                    return_value=(rank % dedup_size, dedup_size),
                ),
                patch(
                    "sglang.srt.mem_cache.mla_host_dedup.mla_host_dedup_eligible",
                    return_value=eligible,
                ),
                patch.object(torch.distributed, "is_initialized", return_value=True),
                patch.object(torch.distributed, "all_reduce"),
            )
            for item in patches:
                stack.enter_context(item)
            all_gather = stack.enter_context(
                patch.object(torch.distributed, "all_gather", side_effect=gather)
            )
            yield memory, all_gather
        self.assertIsNone(base._host_memory_budget.get())

    def test_explicit_source_can_exceed_the_old_per_rank_budget(self):
        with (
            self._runtime([192] + [0] * 7),
            sizing.auto_size_hicache(self.params, None, enabled=True),
        ):
            self.assertEqual(base.host_memory_budget_bytes(6000), 8000)
            self.assertEqual(base.host_memory_budget_bytes(), 2000)
            # A failed request does not consume the snapshot.
            self.assertEqual(base.host_memory_budget_bytes(2001), 2000)
            self.assertEqual(base.host_memory_budget_bytes(), 2000)

    def test_rank_local_drafts_keep_a_budget_on_every_rank(self):
        draft = Mock(size=4, host_capacity_bytes=12)
        plan = Mock(mode="sidecar", device_pools=(draft,))
        weights = [192 + 24] + [24] * 7
        for rank in (0, 1, 7):
            with (
                self.subTest(rank=rank),
                self._runtime(weights, rank=rank),
                sizing.auto_size_hicache(self.params, plan, enabled=True),
            ):
                self.assertEqual(
                    base.host_memory_budget_bytes(), 4500 if rank == 0 else 500
                )

    def test_multiple_dedup_groups_share_the_host_budget(self):
        # DP2 x attention-TP4: two physical target copies, not eight.
        weights = [192, 0, 0, 0] * 2
        for rank in (0, 4):
            with (
                self.subTest(rank=rank),
                self._runtime(weights, rank=rank, dedup_size=4),
                sizing.auto_size_hicache(self.params, None, enabled=True),
            ):
                self.assertEqual(base.host_memory_budget_bytes(), 4000)

    def test_multinode_uses_only_colocated_allocations_and_headroom(self):
        weights = [192] + [0] * 7 + [384] + [0] * 7
        # Different PP stages may have different bytes/token and host headroom.
        self.target.host_capacity_bytes = 384
        with (
            self._runtime(weights, rank=8, available=[1000] * 8 + [8000, 7000] * 4),
            sizing.auto_size_hicache(self.params, None, enabled=True),
        ):
            self.assertEqual(base.host_memory_budget_bytes(), 7000)

    def test_cross_node_dummy_host_needs_no_payload_budget(self):
        with (
            self._runtime([192] + [0] * 15, rank=8, dedup_size=16, fraction=0.8) as (
                memory,
                _,
            ),
            sizing.auto_size_hicache(self.params, None, enabled=True),
        ):
            self.assertEqual(base.host_memory_budget_bytes(), 0)
            self.assertEqual(memory.hicache_ratio, 2.0)

    def test_auto_size_uses_physical_bytes_and_books_the_snapshot(self):
        with (
            self._runtime([192] + [0] * 7, available=400, fraction=0.8) as (
                memory,
                all_gather,
            ),
            sizing.auto_size_hicache(self.params, None, enabled=True),
        ):
            self.assertAlmostEqual(memory.hicache_ratio, 320 * 0.95 / 192)
            self.assertEqual(base.host_memory_budget_bytes(300), 320)
            self.assertEqual(base.host_memory_budget_bytes(), 20)
            all_gather.assert_called_once()

    def test_large_numpy_byte_counts_do_not_overflow(self):
        self.target.host_capacity_bytes = np.int64(1 << 36)
        with (
            self._runtime([1 << 36] + [0] * 7, available=1 << 40),
            sizing.auto_size_hicache(self.params, None, enabled=True),
        ):
            self.assertEqual(base.host_memory_budget_bytes(), 1 << 40)

    def test_auto_size_preserves_draft_budget_on_source_and_peers(self):
        plan = Mock(
            mode="sidecar", device_pools=(Mock(size=4, host_capacity_bytes=12),)
        )
        weights = [216] + [24] * 7
        for rank in (0, 1):
            with (
                self.subTest(rank=rank),
                self._runtime(weights, rank=rank, available=800, fraction=0.8) as (
                    memory,
                    _,
                ),
                sizing.auto_size_hicache(self.params, plan, enabled=True),
            ):
                self.assertAlmostEqual(memory.hicache_ratio, 640 * 0.95 / 384)
                self.assertEqual(
                    base.host_memory_budget_bytes(), 360 if rank == 0 else 40
                )

    def test_disabled_or_ineligible_dedup_preserves_rank_split(self):
        for gate in ({"enabled": False}, {"eligible": False}, {"dedup_size": 1}):
            with (
                self.subTest(gate=gate),
                self._runtime([192] * 8, **gate) as (_, all_gather),
                sizing.auto_size_hicache(self.params, None, enabled=True),
            ):
                self.assertEqual(base.host_memory_budget_bytes(), 1000)
                self.assertIsNone(base._host_memory_budget.get())
                all_gather.assert_not_called()

    def test_mla_and_dsa_pools_share_one_physical_budget(self):
        target = MLATokenToKVPool(
            size=8,
            page_size=2,
            dtype=torch.float16,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self.params.token_to_kv_pool_allocator.get_kvcache.return_value = target
        target_bytes = sizing._pool_bytes(target)
        indexer_device = SimpleNamespace(
            layer_num=2,
            layer_shard_enabled=False,
            store_dtype=torch.float16,
            size=8,
            start_layer=0,
            end_layer=2,
            index_head_dim=8,
            quant_block_size=4,
            page_size=2,
            index_page_size=2,
            skip_topk_layers=[False] * 2,
        )
        kwargs = dict(
            host_to_device_ratio=2,
            host_size=0,
            page_size=2,
            layout="page_first",
            pin_memory=False,
        )

        def allocate(dims, dtype, device, **_kwargs):
            return torch.empty(dims, dtype=dtype, device=device)

        with (
            self._runtime([target_bytes] + [0] * 7, available=1100),
            patch.dict(
                "sglang.srt.mem_cache.pool_host.mla.ALLOC_MEMORY_FUNCS",
                {"cpu": allocate},
            ),
            patch.object(DSAIndexerPoolHost, "init_kv_buffer"),
            patch.object(DSAIndexerPoolHost, "_init_write_back_staging_buffers"),
        ):
            # MLA and indexer guards each exceed the old 137-byte rank share.
            with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                MLATokenToKVPoolHost(target, **kwargs)
            with sizing.auto_size_hicache(self.params, None, enabled=True):
                host = MLATokenToKVPoolHost(target, **kwargs)
                self.addCleanup(host.destroy)
                indexer = DSAIndexerPoolHost(
                    make_dsa_indexer_pool_decl(indexer_device), host, pin_memory=False
                )
                self.assertEqual(
                    host.kv_buffer.numel() * host.kv_buffer.element_size(), 432
                )
                self.assertFalse(indexer._is_dummy)
                self.assertEqual(base.host_memory_budget_bytes(), 92)
                with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                    MLATokenToKVPoolHost(target, **kwargs)


if __name__ == "__main__":
    unittest.main()
