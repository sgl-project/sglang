import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import (
    HiSparseTokenToKVPoolAllocator,
    _HiSparsePageOwnership,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ChildAllocator:
    def __init__(self, allocation: torch.Tensor | None) -> None:
        self.allocation = allocation
        self.is_not_in_free_group = True
        self.alloc_sizes: list[int] = []
        self.freed: list[torch.Tensor] = []

    def alloc(self, need_size: int) -> torch.Tensor | None:
        self.alloc_sizes.append(need_size)
        return self.allocation

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())


def _make_coordinator(
    *,
    rows: torch.Tensor,
    capacities: torch.Tensor,
    mapping: torch.Tensor,
    extra_allocation: torch.Tensor | None,
) -> tuple[HiSparseCoordinator, _ChildAllocator]:
    child_allocator = _ChildAllocator(extra_allocation)
    allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.page_size = 4
    allocator.compress_ratio = 1
    allocator.hisparse_attn_allocator = child_allocator
    allocator.full_to_hisparse_device_index_mapping = mapping
    allocator._page_ownership = _HiSparsePageOwnership(
        mapping=mapping,
        child_allocator=child_allocator,
        page_size=4,
    )

    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.token_to_kv_pool_allocator = allocator
    coordinator.mem_pool_device = SimpleNamespace(
        full_to_hisparse_device_index_mapping=mapping
    )
    coordinator.is_dsv4_hisparse = False
    coordinator.compress_ratio = 1
    coordinator.device_buffer_size = 8
    coordinator.padded_buffer_size = 12
    coordinator.req_to_device_buffer = rows
    coordinator.req_device_buffer_size = capacities
    coordinator.req_device_buffer_token_locs = torch.zeros(
        (1, rows.shape[0], rows.shape[1]), dtype=torch.int32
    )
    return coordinator, child_allocator


class TestHiSparseDecodeTransaction(unittest.TestCase):
    def test_in_page_decode_skips_owner_transaction(self) -> None:
        """An in-page decode reuses its published destination without allocation."""
        mapping = torch.zeros(64, dtype=torch.int64)
        rows = torch.arange(40, 52, dtype=torch.int64).reshape(1, 12)
        mapping[9] = rows[0, 1]
        coordinator, child_allocator = _make_coordinator(
            rows=rows,
            capacities=torch.tensor([12], dtype=torch.int64),
            mapping=mapping,
            extra_allocation=torch.arange(24, 28, dtype=torch.int64),
        )
        mapping_before = mapping.clone()

        coordinator._rehome_page_boundary_owners(
            seq_lens=torch.tensor([2], dtype=torch.int64),
            out_cache_loc=torch.tensor([9], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([2], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
        )

        self.assertTrue(torch.equal(mapping, mapping_before))
        self.assertEqual(child_allocator.alloc_sizes, [])
        self.assertEqual(child_allocator.freed, [])

    def test_boundary_transaction_transfers_growth_and_releases_surplus(self) -> None:
        """A page boundary transfers a grow owner and releases a surplus page."""
        mapping = torch.zeros(64, dtype=torch.int64)
        mapping[12:16] = torch.arange(20, 24, dtype=torch.int64)
        mapping[16:20] = torch.arange(32, 36, dtype=torch.int64)
        rows = torch.zeros((2, 12), dtype=torch.int64)
        rows[0, :4] = torch.arange(4, 8, dtype=torch.int64)
        rows[1] = torch.arange(40, 52, dtype=torch.int64)
        coordinator, child_allocator = _make_coordinator(
            rows=rows,
            capacities=torch.tensor([4, 12], dtype=torch.int64),
            mapping=mapping,
            extra_allocation=torch.arange(24, 28, dtype=torch.int64),
        )

        coordinator._rehome_page_boundary_owners(
            seq_lens=torch.tensor([5, 5], dtype=torch.int64),
            out_cache_loc=torch.tensor([12, 16], dtype=torch.int64),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5, 5], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0, 1], dtype=torch.int64),
        )

        self.assertTrue(
            torch.equal(mapping[12:16], torch.arange(20, 24, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(mapping[16:20], torch.arange(44, 48, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(rows[0, 4:12], torch.arange(20, 28, dtype=torch.int64))
        )
        self.assertEqual(coordinator.req_device_buffer_size.tolist(), [12, 12])
        self.assertEqual(child_allocator.alloc_sizes, [4])
        self.assertTrue(
            torch.equal(
                child_allocator.freed[0], torch.arange(32, 36, dtype=torch.int64)
            )
        )

    def test_net_extra_failure_preserves_all_owners(self) -> None:
        """A failed net-extra allocation leaves mapping, rows, and capacity intact."""
        mapping = torch.zeros(64, dtype=torch.int64)
        mapping[12:16] = torch.arange(20, 24, dtype=torch.int64)
        rows = torch.zeros((1, 12), dtype=torch.int64)
        rows[0, :4] = torch.arange(4, 8, dtype=torch.int64)
        capacities = torch.tensor([4], dtype=torch.int64)
        coordinator, child_allocator = _make_coordinator(
            rows=rows,
            capacities=capacities,
            mapping=mapping,
            extra_allocation=None,
        )
        mapping_before = mapping.clone()
        rows_before = rows.clone()

        with self.assertRaisesRegex(RuntimeError, "net allocation failed"):
            coordinator._rehome_page_boundary_owners(
                seq_lens=torch.tensor([5], dtype=torch.int64),
                out_cache_loc=torch.tensor([12], dtype=torch.int64),
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
                req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            )

        self.assertTrue(torch.equal(mapping, mapping_before))
        self.assertTrue(torch.equal(rows, rows_before))
        self.assertEqual(capacities.tolist(), [4])
        self.assertEqual(child_allocator.freed, [])


if __name__ == "__main__":
    unittest.main()
