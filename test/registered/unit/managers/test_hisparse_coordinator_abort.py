import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator  # noqa: E402
from sglang.srt.mem_cache.allocator.hisparse import (  # noqa: E402
    HiSparseTokenToKVPoolAllocator,
)

_DEV = "cpu"
_PAGE_SIZE = 64
_HISPARSE_SIZE = 64
_ROW_WIDTH = 128
_POOL_IDX = 0


class _StubHiSparseKvCache:
    def register_mapping(self, mapping: torch.Tensor) -> None:
        self.full_to_hisparse_device_index_mapping = mapping

    def _translate_loc_to_hisparse_device(
        self, compressed_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.full_to_hisparse_device_index_mapping[compressed_indices]


def _make_allocator() -> HiSparseTokenToKVPoolAllocator:
    return HiSparseTokenToKVPoolAllocator(
        _HISPARSE_SIZE,
        _PAGE_SIZE,
        torch.float16,
        torch.device(_DEV),
        _StubHiSparseKvCache(),
        False,
    )


def _make_coordinator(
    allocator: HiSparseTokenToKVPoolAllocator, row: torch.Tensor
) -> HiSparseCoordinator:
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.ack_staging_queue = []
    coordinator.write_staging_stream = MagicMock()
    coordinator.req_to_token_pool = SimpleNamespace(req_to_token=row)
    coordinator.token_to_kv_pool_allocator = allocator
    coordinator.mem_pool_host = SimpleNamespace(
        allocated_host_indices=MagicMock(
            return_value=torch.empty((0,), dtype=torch.int64)
        ),
        free=MagicMock(),
    )
    coordinator.req_to_host_pool = torch.full((1, 4), -1, dtype=torch.int64)
    coordinator.req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)
    coordinator._skip_first_backup = torch.zeros(1, dtype=torch.bool)
    return coordinator


class TestAbortStagingRequestFreesTheWholeAllocation(CustomTestCase):
    def test_abort_then_release_never_frees_a_hisparse_page_twice(self):
        """Abort used to clear mappings only up to extend_range.end, so the padding slot of a page-rounded allocation kept its mapping and the later release freed the same hisparse page a second time."""
        allocator = _make_allocator()
        row = torch.zeros((1, _ROW_WIDTH), dtype=torch.int64, device=_DEV)

        allocated_len = _PAGE_SIZE
        prefill_len = _PAGE_SIZE - 1
        locs = allocator.alloc(allocated_len)
        self.assertIsNotNone(locs)
        row[_POOL_IDX, :allocated_len] = locs

        req = SimpleNamespace(
            req_pool_idx=_POOL_IDX,
            extend_range=SimpleNamespace(end=prefill_len),
            kv=SimpleNamespace(kv_allocated_len=allocated_len),
            hisparse_staging=True,
        )
        coordinator = _make_coordinator(allocator, row)

        coordinator.abort_staging_request(req)
        allocator.free(row[_POOL_IDX, :allocated_len])

        hisparse_free_pages = allocator.hisparse_attn_allocator.free_pages.tolist()
        self.assertEqual(
            len(hisparse_free_pages),
            len(set(hisparse_free_pages)),
            "a hisparse page was returned to the free list twice (double free)",
        )
        self.assertEqual(
            len(hisparse_free_pages), allocator.hisparse_attn_allocator.num_pages
        )

    def test_abort_clears_every_mapping_of_the_allocation(self):
        """Any surviving mapping entry re-frees its page on the release path, so [0, kv_allocated_len) must be fully cleared."""
        allocator = _make_allocator()
        row = torch.zeros((1, _ROW_WIDTH), dtype=torch.int64, device=_DEV)

        allocated_len = _PAGE_SIZE
        locs = allocator.alloc(allocated_len)
        self.assertIsNotNone(locs)
        row[_POOL_IDX, :allocated_len] = locs

        req = SimpleNamespace(
            req_pool_idx=_POOL_IDX,
            extend_range=SimpleNamespace(end=_PAGE_SIZE - 1),
            kv=SimpleNamespace(kv_allocated_len=allocated_len),
            hisparse_staging=True,
        )
        coordinator = _make_coordinator(allocator, row)

        coordinator.abort_staging_request(req)

        mapping = allocator.full_to_hisparse_device_index_mapping
        self.assertEqual(int(mapping[locs].count_nonzero()), 0)


if __name__ == "__main__":
    unittest.main()
