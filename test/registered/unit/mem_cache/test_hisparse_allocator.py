import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV4HiSparseAllocator(CustomTestCase):
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
        swa_tail_len = 128
        kv_loc = torch.arange(512, 512 + fill_len, dtype=torch.int64)
        host_indices = torch.arange(1000, 1128, dtype=torch.int64)

        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=list(range(fill_len)),
            output_ids=[],
            kv=None,
        )

        def set_extend_range(start, end):
            req.extend_range = SimpleNamespace(start=start, end=end, length=end - start)

        req.set_extend_range = set_extend_range

        class ReqToTokenPool:
            def __init__(self):
                self.writes = []

            def alloc(self, reqs):
                for item in reqs:
                    item.req_pool_idx = 0
                return torch.tensor([0], dtype=torch.int64)

            def write(self, indices, values):
                self.writes.append((indices, values))

        req_to_token_pool = ReqToTokenPool()
        allocator = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=256,
            available_size=MagicMock(return_value=fill_len),
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
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_tail_len = MagicMock(return_value=swa_tail_len)

        result = queue._pre_alloc(req)

        self.assertTrue(torch.equal(result, host_indices))
        allocator.alloc_extend_swa_tail.assert_called_once()
        allocator.alloc_logical_only.assert_not_called()
        _, kwargs = allocator.alloc_extend_swa_tail.call_args
        self.assertEqual(kwargs["extend_num_tokens"], fill_len)
        self.assertEqual(kwargs["swa_tail_len"], swa_tail_len)
        self.assertEqual(req.kv.swa_evicted_seqlen, fill_len - swa_tail_len)
        self.assertEqual(req.kv.kv_allocated_len, fill_len)
        self.assertEqual(req.kv_committed_len, fill_len)
        self.assertEqual(req.extend_range.length, fill_len)
        self.assertEqual(len(req_to_token_pool.writes), 1)
        coordinator.host_token_len.assert_called_once_with(fill_len)
        regular_host_alloc.assert_called_once_with(
            coordinator.req_to_host_pool,
            coordinator.req_to_host_pool_allocated_len,
            req.req_pool_idx,
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
            mla_compression_ratios=[4, 128, 4, 128],
            prefill_start_layer=0,
            prefill_end_layer=2,
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        executor = MagicMock()

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
