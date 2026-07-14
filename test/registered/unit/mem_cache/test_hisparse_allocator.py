import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
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
            swa_tail_end=512,
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
        self.assertEqual(kwargs["swa_tail_end"], 512)

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

        fill_len = 510
        allocated_len = 512
        swa_tail_len = 126
        kv_loc = torch.arange(allocated_len, dtype=torch.int64)
        host_indices = torch.arange(1000, 1000 + allocated_len, dtype=torch.int64)

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
            available_size=MagicMock(return_value=allocated_len),
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
        self.assertEqual(kwargs["extend_num_tokens"], allocated_len)
        self.assertEqual(kwargs["swa_tail_len"], swa_tail_len)
        self.assertEqual(kwargs["swa_tail_end"], fill_len)
        self.assertEqual(req.kv.swa_evicted_seqlen, fill_len - swa_tail_len)
        self.assertEqual(req.kv.kv_allocated_len, allocated_len)
        self.assertEqual(req.kv_committed_len, fill_len)
        self.assertEqual(req.extend_range.length, fill_len)
        self.assertEqual(len(req_to_token_pool.writes), 1)
        coordinator.mem_pool_host.alloc_paged_token_slots.assert_called_once()


class _FakeChildAllocator:
    def __init__(
        self,
        *,
        allocation: torch.Tensor | None,
    ) -> None:
        self.allocation = allocation
        self.alloc_sizes: list[int] = []
        self.freed: list[torch.Tensor] = []

    def available_size(self) -> int:
        return 64

    def alloc(self, need_size: int) -> torch.Tensor | None:
        self.alloc_sizes.append(need_size)
        return self.allocation

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())


class _FakeC4Pool:
    @staticmethod
    def translate_loc_from_full_to_compressed(
        full_indices: torch.Tensor,
    ) -> torch.Tensor:
        return full_indices[(full_indices + 1) % 4 == 0] // 4


def _make_direct_allocator(
    *,
    dsv4: bool,
    device_allocation: torch.Tensor | None,
) -> HiSparseTokenToKVPoolAllocator | DeepSeekV4HiSparseTokenToKVPoolAllocator:
    allocator_class = (
        DeepSeekV4HiSparseTokenToKVPoolAllocator
        if dsv4
        else HiSparseTokenToKVPoolAllocator
    )
    allocator = object.__new__(allocator_class)
    allocator.page_size = 8
    allocator.compress_ratio = 4 if dsv4 else 1
    allocator.hisparse_page_size = 2 if dsv4 else 8
    allocator.device = "cpu"
    allocator.is_not_in_free_group = True
    allocator.logical_attn_allocator = _FakeChildAllocator(
        allocation=torch.arange(8, 16, dtype=torch.int64)
    )
    allocator.hisparse_attn_allocator = _FakeChildAllocator(
        allocation=device_allocation
    )
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        64,
        dtype=torch.int64,
    )
    if dsv4:
        allocator.hisparse_kvcache = _FakeC4Pool()
    return allocator


class TestHiSparseDirectAllocator(unittest.TestCase):
    def test_direct_alloc_publishes_generic_and_dsv4_mappings(self) -> None:
        """Direct allocation publishes both generic and compressed mappings."""
        cases = (
            (False, torch.arange(24, 32, dtype=torch.int64)),
            (True, torch.arange(6, 8, dtype=torch.int64)),
        )

        for dsv4, device_indices in cases:
            with self.subTest(dsv4=dsv4):
                allocator = _make_direct_allocator(
                    dsv4=dsv4,
                    device_allocation=device_indices,
                )

                result = allocator.alloc(8)
                mapping_indices = (
                    torch.tensor([2, 3], dtype=torch.int64)
                    if dsv4
                    else torch.arange(8, 16, dtype=torch.int64)
                )

                self.assertTrue(
                    torch.equal(result, torch.arange(8, 16, dtype=torch.int64))
                )
                self.assertTrue(
                    torch.equal(
                        allocator.full_to_hisparse_device_index_mapping[
                            mapping_indices
                        ],
                        device_indices,
                    )
                )

    def test_direct_alloc_rolls_back_on_second_child_failure(self) -> None:
        """A failed device-child allocation leaves no logical or mapping owner."""
        for dsv4 in (False, True):
            with self.subTest(dsv4=dsv4):
                allocator = _make_direct_allocator(
                    dsv4=dsv4,
                    device_allocation=None,
                )

                result = allocator.alloc(8)

                self.assertIsNone(result)
                self.assertEqual(len(allocator.logical_attn_allocator.freed), 1)
                self.assertTrue(
                    torch.all(
                        allocator.full_to_hisparse_device_index_mapping == 0
                    )
                )


if __name__ == "__main__":
    unittest.main()
