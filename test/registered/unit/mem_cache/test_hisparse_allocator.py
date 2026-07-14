import unittest

import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.multi_ended_allocator import (
    MultiEndedAllocator,
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeChildAllocator:
    def __init__(
        self,
        *,
        available: int,
        allocation: torch.Tensor | None,
    ) -> None:
        self.available = available
        self.allocation = allocation
        self.alloc_sizes: list[int] = []
        self.freed: list[torch.Tensor] = []

    def available_size(self) -> int:
        return self.available

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


def _make_generic_allocator(
    *,
    logical_allocation: torch.Tensor | None,
    device_allocation: torch.Tensor | None,
) -> HiSparseTokenToKVPoolAllocator:
    allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.page_size = 4
    allocator.device = "cpu"
    allocator.is_not_in_free_group = True
    allocator.logical_attn_allocator = _FakeChildAllocator(
        available=64,
        allocation=logical_allocation,
    )
    allocator.hisparse_attn_allocator = _FakeChildAllocator(
        available=64,
        allocation=device_allocation,
    )
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        64,
        dtype=torch.int64,
    )
    return allocator


def _make_dsv4_allocator(
    *,
    logical_allocation: torch.Tensor | None,
    device_allocation: torch.Tensor | None,
) -> DeepSeekV4HiSparseTokenToKVPoolAllocator:
    allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    allocator.page_size = 8
    allocator.compress_ratio = 4
    allocator.hisparse_page_size = 2
    allocator.device = "cpu"
    allocator.is_not_in_free_group = True
    allocator.logical_attn_allocator = _FakeChildAllocator(
        available=64,
        allocation=logical_allocation,
    )
    allocator.hisparse_attn_allocator = _FakeChildAllocator(
        available=64,
        allocation=device_allocation,
    )
    allocator.hisparse_kvcache = _FakeC4Pool()
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        64,
        dtype=torch.int64,
    )
    return allocator


class TestHiSparseDirectAllocator(unittest.TestCase):
    def test_page_aligned_spec_capability_matrix_is_explicit(self) -> None:
        """Spec direct capability remains disabled only for unsupported allocators."""
        supported_classes = (
            PagedTokenToKVPoolAllocator,
            MultiEndedAllocator,
            UnifiedMambaTokenToKVPoolAllocator,
            SWATokenToKVPoolAllocator,
            UnifiedSWATokenToKVPoolAllocator,
        )
        for allocator_class in supported_classes:
            self.assertTrue(allocator_class.supports_spec_page_aligned_alloc)

        unsupported_classes = (
            PureSWATokenToKVPoolAllocator,
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        )
        for allocator_class in unsupported_classes:
            self.assertFalse(allocator_class.supports_spec_page_aligned_alloc)

    def test_generic_direct_alloc_publishes_complete_page_mapping(self) -> None:
        """Generic direct allocation publishes matching logical and device pages."""
        logical_indices = torch.arange(4, 8, dtype=torch.int64)
        device_indices = torch.arange(8, 12, dtype=torch.int64)
        allocator = _make_generic_allocator(
            logical_allocation=logical_indices,
            device_allocation=device_indices,
        )

        result = allocator.alloc(4)

        self.assertTrue(torch.equal(result, logical_indices))
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[logical_indices],
                device_indices,
            )
        )

    def test_generic_second_child_failure_rolls_back_logical_page(self) -> None:
        """Generic direct allocation rolls back when the device child fails."""
        logical_indices = torch.arange(4, 8, dtype=torch.int64)
        allocator = _make_generic_allocator(
            logical_allocation=logical_indices,
            device_allocation=None,
        )

        result = allocator.alloc(4)

        self.assertIsNone(result)
        self.assertEqual(len(allocator.logical_attn_allocator.freed), 1)
        self.assertTrue(
            torch.equal(
                allocator.logical_attn_allocator.freed[0],
                logical_indices,
            )
        )
        self.assertTrue(torch.all(allocator.full_to_hisparse_device_index_mapping == 0))

    def test_dsv4_direct_alloc_uses_c4_count_and_translated_keys(self) -> None:
        """DSV4 direct allocation maps translated C4 keys to C4 device slots."""
        logical_indices = torch.arange(8, 16, dtype=torch.int64)
        device_indices = torch.arange(4, 6, dtype=torch.int64)
        allocator = _make_dsv4_allocator(
            logical_allocation=logical_indices,
            device_allocation=device_indices,
        )

        result = allocator.alloc(8)
        compressed_indices = torch.tensor([2, 3], dtype=torch.int64)

        self.assertTrue(torch.equal(result, logical_indices))
        self.assertEqual(allocator.hisparse_attn_allocator.alloc_sizes, [2])
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[compressed_indices],
                device_indices,
            )
        )

    def test_dsv4_second_child_failure_rolls_back_full_logical_page(self) -> None:
        """DSV4 direct allocation rolls back the full logical page on C4 OOM."""
        logical_indices = torch.arange(8, 16, dtype=torch.int64)
        allocator = _make_dsv4_allocator(
            logical_allocation=logical_indices,
            device_allocation=None,
        )

        result = allocator.alloc(8)

        self.assertIsNone(result)
        self.assertTrue(
            torch.equal(
                allocator.logical_attn_allocator.freed[0],
                logical_indices,
            )
        )
        self.assertTrue(
            torch.all(allocator.full_to_hisparse_device_index_mapping == 0)
        )

    def test_zero_direct_alloc_returns_empty_without_child_mutation(self) -> None:
        """Zero-sized direct allocation returns an empty int64 tensor."""
        allocator = _make_generic_allocator(
            logical_allocation=torch.arange(4, 8, dtype=torch.int64),
            device_allocation=torch.arange(8, 12, dtype=torch.int64),
        )

        result = allocator.alloc(0)

        self.assertEqual(result.dtype, torch.int64)
        self.assertEqual(result.numel(), 0)
        self.assertEqual(allocator.logical_attn_allocator.alloc_sizes, [])
        self.assertEqual(allocator.hisparse_attn_allocator.alloc_sizes, [])


if __name__ == "__main__":
    unittest.main()
