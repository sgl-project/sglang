import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV4HiSparseAllocator(CustomTestCase):
    def test_available_size_reports_logical_capacity(self):
        """A small device working set must not look like logical KV usage."""
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = 4
        allocator.logical_attn_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=5_445_376),
            full_available_size=MagicMock(return_value=5_445_376),
        )
        allocator.hisparse_attn_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=272_268)
        )

        self.assertEqual(allocator.available_size(), 5_445_376)
        self.assertEqual(allocator.full_available_size(), 5_445_376)
        allocator.hisparse_attn_allocator.available_size.assert_not_called()

    def test_empty_pool_does_not_report_eighty_percent_full_usage(self):
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )

        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = 4
        allocator.logical_attn_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=5_445_376),
            full_available_size=MagicMock(return_value=5_445_376),
            swa_available_size=MagicMock(return_value=326_656),
        )
        allocator.hisparse_attn_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=272_268)
        )

        observer = SchedulerPoolStatsObserver(
            tree_cache=SimpleNamespace(
                full_evictable_size=MagicMock(return_value=0),
                swa_evictable_size=MagicMock(return_value=0),
            ),
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=None,
            session_controller=None,
            hisparse_coordinator=None,
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            enable_hisparse=True,
            full_tokens_per_layer=5_445_376,
            swa_tokens_per_layer=326_656,
            max_total_num_tokens=5_445_376,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )

        stats = observer._get_swa_token_info()

        self.assertEqual(stats.full_num_used, 0)
        self.assertEqual(stats.full_token_usage, 0.0)
        self.assertEqual(stats.swa_num_used, 0)
        self.assertEqual(stats.swa_token_usage, 0.0)

    def test_device_capacity_is_still_checked_by_alloc_extend(self):
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = 4
        allocator.page_size = 256
        allocator.hisparse_page_size = 64
        allocator.logical_attn_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=1_000_000)
        )
        allocator.hisparse_attn_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=0)
        )

        prefix_lens = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([256], dtype=torch.int64)

        result = allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=256,
        )

        self.assertIsNone(result)

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
        kv_loc = torch.arange(fill_len, dtype=torch.int64)
        host_indices = torch.arange(1000, 1000 + fill_len, dtype=torch.int64)

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
            page_size=64,
            available_size=MagicMock(return_value=fill_len),
            alloc_extend_swa_tail=MagicMock(return_value=kv_loc),
            alloc_logical_only=MagicMock(return_value=kv_loc),
        )
        coordinator = SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                alloc_paged_token_slots=MagicMock(return_value=host_indices)
            ),
            req_to_host_pool=object(),
            req_to_host_pool_allocated_len=object(),
            host_token_len=MagicMock(side_effect=lambda length: length),
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

        self.assertIs(result, host_indices)
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
        coordinator.mem_pool_host.alloc_paged_token_slots.assert_called_once()


if __name__ == "__main__":
    unittest.main()
