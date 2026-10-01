import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHiSparseRequestRelease(CustomTestCase):
    def check_release(self, page_size, *, buffered=True, extra=True, need_sort=False):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )

        # Only the storage/streams are doubled. Both physical and logical page
        # allocation/free use the real allocator, including page deduplication.
        pool = SimpleNamespace()
        pool.register_mapping = lambda mapping: setattr(
            pool, "full_to_hisparse_device_index_mapping", mapping
        )
        pool._translate_loc_to_hisparse_device = lambda loc: (
            pool.full_to_hisparse_device_index_mapping[loc]
        )
        pool.translate_loc_from_full_to_compressed = lambda loc: loc
        allocator = HiSparseTokenToKVPoolAllocator(
            8 * page_size, page_size, torch.float32, "cpu", pool, need_sort
        )
        device = allocator.hisparse_attn_allocator
        device.debug_mode = True
        logical = allocator.logical_attn_allocator.alloc(2 * page_size)
        other_logical = allocator.logical_attn_allocator.alloc(page_size)
        other_device = device.alloc(page_size)
        mapping = pool.full_to_hisparse_device_index_mapping
        mapping[other_logical] = other_device
        side = (
            device.alloc(page_size) if buffered else torch.empty(0, dtype=torch.int64)
        )
        if buffered:
            # Different indices on the same page also appear in the side buffer.
            mapping[logical[:page_size]] = side
        if extra:
            # Speculative alloc_extend can reserve physical slots outside the
            # side buffer, including beyond the last committed token.
            mapping[logical[page_size:]] = device.alloc(page_size)

        host_free = MagicMock()
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.decode_producer_stream = None
        coordinator.wait_for_pending_backup = MagicMock()
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.mem_pool_device = pool
        coordinator.req_device_buffer_size = torch.tensor([len(side)])
        coordinator.req_to_device_buffer = torch.zeros(
            (1, page_size), dtype=torch.int64
        )
        coordinator.req_to_device_buffer[0, : len(side)] = side
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=logical.reshape(1, -1)
        )
        coordinator.mem_pool_host = SimpleNamespace(
            allocated_host_indices=lambda *args: torch.tensor([7]), free=host_free
        )
        coordinator.req_to_host_pool = torch.zeros(
            (1, 2 * page_size), dtype=torch.int64
        )
        coordinator.req_to_host_pool_allocated_len = torch.tensor([2 * page_size])
        coordinator.req_device_buffer_tokens = torch.zeros((1, 1, page_size))
        coordinator.req_device_buffer_token_locs = torch.zeros((1, 1, page_size))
        coordinator.lru_slots = torch.zeros((1, 1, page_size))
        coordinator._lru_init = torch.arange(page_size)
        coordinator._skip_first_backup = [True]
        req = SimpleNamespace(
            kv=SimpleNamespace(
                req_pool_idx=0,
                kv_committed_len=page_size,
                kv_allocated_len=2 * page_size,
            )
        )

        coordinator.request_finished(req)
        self.assertEqual(device.available_size(), 7 * page_size)
        self.assertEqual(len(torch.unique(device.get_all_free_pages())), 7)
        self.assertTrue(torch.all(mapping[logical] == 0))
        torch.testing.assert_close(mapping[other_logical], other_device)
        self.assertFalse(
            torch.isin(other_device // page_size, device.get_all_free_pages()).any()
        )
        host_free.assert_called_once()
        coordinator.wait_for_pending_backup.assert_called_once()

        # Normal release follows coordinator cleanup; it must not free the same
        # physical page again through stale logical-to-device mapping entries.
        allocator.free(logical)
        self.assertEqual(device.available_size(), 7 * page_size)
        self.assertEqual(
            allocator.logical_attn_allocator.available_size(), 15 * page_size
        )
        allocator.free(other_logical)
        self.assertEqual(device.available_size(), 8 * page_size)
        self.assertEqual(
            allocator.logical_attn_allocator.available_size(), 16 * page_size
        )

    def test_releases_speculative_pages_outside_device_buffer(self):
        for page_size in (1, 64):
            for need_sort in (False, True):
                with self.subTest(page_size=page_size, need_sort=need_sort):
                    self.check_release(page_size, need_sort=need_sort)

    def test_mapping_only_request_release(self):
        for page_size in (1, 64):
            with self.subTest(page_size=page_size):
                self.check_release(page_size, buffered=False)

    def test_buffer_aliases_are_not_double_freed(self):
        for page_size in (1, 64):
            with self.subTest(page_size=page_size):
                self.check_release(page_size, extra=False)

    def test_empty_device_ownership(self):
        self.check_release(64, buffered=False, extra=False)


class TestHiSparseDecodeRemap(CustomTestCase):
    def test_page_size_one_reclaims_temporary_device_slot(self):
        """Decode remapping must reclaim its temporary slot without freeing the live slot."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

        pool = MiniMaxSparseKVPool(
            size=8,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=8,
            idx_head_dim=16,
            dense_layer_ids=[0],
            sparse_layer_ids=[1],
            disable_value_sparse_layer_ids=[1],
            device="cpu",
            start_layer=0,
            end_layer=2,
            enable_hisparse=True,
        )
        allocator = HiSparseTokenToKVPoolAllocator(
            size=pool.size,
            page_size=1,
            dtype=pool.dtype,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_dsv4_hisparse = False
        coordinator.mem_pool_device = pool.main_pool
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.device_buffer_size = 2
        coordinator.req_to_device_buffer = allocator.hisparse_attn_allocator.alloc(
            3
        ).reshape(1, 3)
        coordinator.req_device_buffer_size = torch.tensor([3])
        coordinator.req_device_buffer_token_locs = torch.zeros(
            (1, 1, 3), dtype=torch.int32
        )
        coordinator._skip_first_backup = [True]
        out_loc = allocator.alloc(1)
        with patch("sglang.srt.managers.hisparse_coordinator._is_hip", False):
            for _ in range(2):
                coordinator._skip_first_backup[0] = True
                coordinator.map_last_loc_to_buffer(
                    seq_lens=torch.tensor([3]),
                    out_cache_loc=out_loc,
                    req_pool_indices=torch.tensor([0]),
                    seq_lens_cpu=torch.tensor([3]),
                    req_pool_indices_cpu=torch.tensor([0]),
                )
                self.assertEqual(
                    allocator.hisparse_attn_allocator.available_size(), pool.size - 3
                )
                torch.testing.assert_close(
                    allocator.full_to_hisparse_device_index_mapping[out_loc],
                    coordinator.req_to_device_buffer[:, 2],
                )


class TestDeepSeekV4HiSparseAllocator(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_free_returns_c4_slots(self):
        """Freeing a request's logical row must return the C4 HiSparse slots its
        compressed tokens were mapped to."""
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        page_size, full_size = 256, 4 * 256
        override = get_context().override_server_args(page_size=page_size)
        override.install()
        self.addCleanup(override.restore)
        pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=4,
            swa_size=full_size,
            c4_size=full_size // 4,
            c128_size=2 * full_size // 128,  # sizes the C4 logical index space
            c4_state_pool_size=16,
            c128_state_pool_size=0,
            page_size=page_size,
            swa_page_size=page_size,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[0, 4],
            full_size=full_size,
            enable_hisparse=True,
        )
        logical = PagedTokenToKVPoolAllocator(
            full_size, page_size, pool.dtype, "cpu", pool, need_sort=False
        )
        allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator(logical)
        c4_free = allocator.hisparse_attn_allocator.available_size()

        # What alloc_extend leaves behind for a one-page request.
        locs = logical.alloc(page_size)
        compressed = allocator.hisparse_kvcache.translate_loc_from_full_to_compressed(
            locs
        )
        slots = allocator.hisparse_attn_allocator.alloc(len(compressed))
        allocator.full_to_hisparse_device_index_mapping[compressed] = slots

        allocator.free(locs)

        self.assertEqual(allocator.hisparse_attn_allocator.available_size(), c4_free)
        self.assertFalse(
            allocator.full_to_hisparse_device_index_mapping[compressed].any()
        )

    def test_forwards_swa_tail_allocation_to_logical_allocator(self):
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        logical_allocator = MagicMock(spec=["alloc_extend_swa_tail"])
        allocator.logical_attn_allocator = logical_allocator

        expected = torch.tensor([8, 9, 10], dtype=torch.int64)
        logical_allocator.alloc_extend_swa_tail.return_value = expected

        prefix_lens = torch.tensor([0], dtype=torch.int64)
        prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([512], dtype=torch.int64)
        seq_lens_cpu = torch.tensor([512], dtype=torch.int64)
        last_loc = torch.tensor([-1], dtype=torch.int64)

        result = allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=512,
            swa_tail_len=128,
        )

        self.assertIs(result, expected)
        logical_allocator.alloc_extend_swa_tail.assert_called_once()
        _, kwargs = logical_allocator.alloc_extend_swa_tail.call_args
        self.assertIs(kwargs["prefix_lens"], prefix_lens)
        self.assertIs(kwargs["prefix_lens_cpu"], prefix_lens_cpu)
        self.assertIs(kwargs["seq_lens"], seq_lens)
        self.assertIs(kwargs["seq_lens_cpu"], seq_lens_cpu)
        self.assertIs(kwargs["last_loc"], last_loc)
        self.assertEqual(kwargs["extend_num_tokens"], 512)
        self.assertEqual(kwargs["swa_tail_len"], 128)

    def test_forwards_prealloc_reclaim_to_logical_allocator(self):
        """PD decode preallocation must not crash on the HiSparse composite."""
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        logical_allocator = MagicMock(spec=["reclaim_for_prealloc"])
        allocator.logical_attn_allocator = logical_allocator
        logical_allocator.reclaim_for_prealloc.return_value = None

        tree_cache = object()
        self.assertIsNone(allocator.reclaim_for_prealloc(tree_cache, 512, 256))
        logical_allocator.reclaim_for_prealloc.assert_called_once_with(
            tree_cache, 512, 256
        )

        logical_allocator.reclaim_for_prealloc.return_value = "SWA eviction short"
        self.assertEqual(
            allocator.reclaim_for_prealloc(tree_cache, 512, 256),
            "SWA eviction short",
        )

    def test_hisparse_budget_uses_full_logical_capacity_for_swa_tail(self):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        logical_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=32),
            full_available_size=MagicMock(return_value=512),
        )
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            logical_attn_allocator=logical_allocator
        )
        queue.scheduler = SimpleNamespace(enable_hisparse=True, last_batch=None)
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._need_space_for_single_req = MagicMock(return_value=0)
        queue._active_reserved_tokens = MagicMock(return_value=0)

        budget = queue._allocatable_token_budgets()

        self.assertEqual(budget, 512)
        logical_allocator.full_available_size.assert_called_once_with()
        logical_allocator.available_size.assert_not_called()

    def test_hisparse_prealloc_uses_swa_tail_for_direct_host_path(self):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        fill_len = 512
        sliding_window_size = 200
        # _swa_tail_len floors the window start to a page boundary:
        # floor_align(512 - 200, 256) = 256, so the tail is one whole page.
        swa_tail_len = 256
        kv_loc = torch.arange(512, 512 + fill_len, dtype=torch.int64)
        host_indices = torch.arange(1000, 1128, dtype=torch.int64)

        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=list(range(fill_len)),
            output_ids=[],
            kv=ReqKvInfo(),
        )

        class ReqToTokenPool:
            def __init__(self):
                self.writes = []

            def alloc(self, reqs):
                for item in reqs:
                    item.kv.req_pool_idx = 0
                return torch.tensor([0], dtype=torch.int64)

            def write(self, indices, values):
                self.writes.append((indices, values))

        req_to_token_pool = ReqToTokenPool()
        allocator = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=256,
            available_size=MagicMock(return_value=fill_len),
            swa_available_size=MagicMock(return_value=swa_tail_len),
            alloc_extend_swa_tail=MagicMock(return_value=kv_loc),
            alloc_logical_only=MagicMock(return_value=kv_loc),
        )
        regular_host_alloc = MagicMock(return_value=host_indices)
        coordinator = SimpleNamespace(
            mem_pool_host=SimpleNamespace(alloc_paged_token_slots=regular_host_alloc),
            req_to_host_pool=object(),
            req_to_host_pool_allocated_len=object(),
            host_token_len=MagicMock(side_effect=lambda token_len: token_len // 4),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = req_to_token_pool
        queue.token_to_kv_pool_allocator = allocator
        queue.tree_cache = SimpleNamespace(
            evictable_size=MagicMock(return_value=0),
            protected_size=MagicMock(return_value=0),
            maybe_hand_to_session=lambda req: None,
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
            sliding_window_size=sliding_window_size,
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)

        result = queue._pre_alloc(req)

        self.assertTrue(torch.equal(result, host_indices))
        allocator.alloc_extend_swa_tail.assert_called_once()
        allocator.alloc_logical_only.assert_not_called()
        _, kwargs = allocator.alloc_extend_swa_tail.call_args
        self.assertEqual(kwargs["extend_num_tokens"], fill_len)
        self.assertEqual(kwargs["swa_tail_len"], swa_tail_len)
        self.assertEqual(
            req.kv.get_evicted_seqlen(ComponentType.SWA), fill_len - swa_tail_len
        )
        self.assertEqual(req.kv.kv_allocated_len, fill_len)
        self.assertEqual(req.kv.kv_committed_len, fill_len)
        self.assertEqual(req.extend_end, fill_len)
        self.assertEqual(len(req_to_token_pool.writes), 1)
        coordinator.host_token_len.assert_called_once_with(fill_len)
        regular_host_alloc.assert_called_once_with(
            coordinator.req_to_host_pool,
            coordinator.req_to_host_pool_allocated_len,
            req.kv.req_pool_idx,
            0,
            len(host_indices),
        )
        self.assertTrue(torch.equal(req_to_token_pool.writes[0][1], kv_loc))

        # C4 indexer/C128 use the logical allocator's full-page IDs. They do not
        # use either the independently allocated host pages or C4 sparse slots.
        np.testing.assert_array_equal(
            np.unique(kv_loc.numpy() // allocator.page_size),
            np.array([2, 3]),
        )

    def test_mooncake_uses_separate_host_and_device_page_indices(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = False
        manager.max_transfer_batch_indices = 0
        manager._transfer_data = MagicMock(return_value=0)

        with ThreadPoolExecutor(max_workers=1) as executor:
            ret = manager._send_kvcache_generic(
                mooncake_session_id="session",
                src_data_ptrs=[1000, 2000, 3000],
                dst_data_ptrs=[10000, 20000, 30000],
                item_lens=[100, 100, 100],
                prefill_data_indices=np.array([1, 2], dtype=np.int32),
                dst_data_indices=np.array([7, 8], dtype=np.int32),
                executor=executor,
                dst_device_data_indices=np.array([21, 22], dtype=np.int32),
                dst_device_data_ptrs={20000, 30000},
            )

        self.assertEqual(ret, 0)
        manager._transfer_data.assert_called_once_with(
            "session",
            [
                (1100, 10700, 200),
                (2100, 22100, 200),
                (3100, 32100, 200),
            ],
        )

    def test_mooncake_derives_device_buffers_from_local_pp_layout(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000, 3000],
            kv_item_lens=[100, 100, 100],
            kv_layer_ids=[],
            mla_compression_ratios=[4, 128, 4, 128],
            prefill_start_layer=0,
            prefill_end_layer=2,
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        executor = MagicMock()

        # send_kvcache reads the memory bag (the unified-memory envelope-layout
        # check), so the context has to be published. This is the non-unified
        # path -- pin that explicitly rather than leaning on the default.
        with get_context().override_server_args(enable_unified_memory=False):
            manager.send_kvcache(
                "session",
                np.array([1], dtype=np.int32),
                [10000, 20000, 30000],
                np.array([7], dtype=np.int32),
                executor,
                dst_device_kv_indices=np.array([21], dtype=np.int32),
            )

        kwargs = manager._send_kvcache_generic.call_args.kwargs
        self.assertEqual(kwargs["dst_device_data_ptrs"], {20000, 30000})

    def test_mooncake_transfer_metadata_carries_device_page_indices(self):
        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        host_pages = np.array([7, 8], dtype=np.int32)
        device_pages = np.array([21, 22], dtype=np.int32)
        info = TransferInfo.from_zmq(
            [
                b"9",
                b"127.0.0.1",
                b"12345",
                b"session",
                host_pages.tobytes(),
                b"0",
                b"",
                b"1",
                b"0",
                device_pages.tobytes(),
            ]
        )

        np.testing.assert_array_equal(info.dst_kv_indices, host_pages)
        np.testing.assert_array_equal(info.dst_device_kv_indices, device_pages)


if __name__ == "__main__":
    unittest.main()
