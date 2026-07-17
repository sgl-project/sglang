import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator  # noqa: E402
from sglang.srt.mem_cache.allocator.hisparse import (  # noqa: E402
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (  # noqa: E402
    HiSparseC4DevicePool,
)
from sglang.srt.utils.common import ceil_align  # noqa: E402

_DEV = "cpu"
_LOGICAL_PAGE = 256
_HISPARSE_PAGE = 64
_DEVICE_BUFFER_SIZE = 64
_PADDED_BUFFER_SIZE = _DEVICE_BUFFER_SIZE + _HISPARSE_PAGE
_HISPARSE_POOL_SIZE = 4096
_MAPPING_SIZE = 1024
_POOL_IDX = 0


def _make_dsv4_allocator() -> DeepSeekV4HiSparseTokenToKVPoolAllocator:
    allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    allocator.compress_ratio = 4
    allocator.page_size = _LOGICAL_PAGE
    allocator.hisparse_page_size = _HISPARSE_PAGE
    allocator.device = torch.device(_DEV)

    kvcache = SimpleNamespace(compress_ratio=4)
    kvcache.translate_loc_from_full_to_compressed = (
        HiSparseC4DevicePool.translate_loc_from_full_to_compressed.__get__(kvcache)
    )
    allocator.hisparse_kvcache = kvcache

    allocator.logical_attn_allocator = SimpleNamespace(
        alloc=lambda need_size: torch.arange(need_size, dtype=torch.int64),
    )
    allocator.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
        _HISPARSE_POOL_SIZE,
        _HISPARSE_PAGE,
        torch.float16,
        torch.device(_DEV),
        SimpleNamespace(),
        False,
    )
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        _MAPPING_SIZE, dtype=torch.int64
    )
    return allocator


def _make_coordinator(
    allocator: DeepSeekV4HiSparseTokenToKVPoolAllocator, row: torch.Tensor
) -> HiSparseCoordinator:
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.is_dsv4_hisparse = True
    coordinator.device_buffer_size = _DEVICE_BUFFER_SIZE
    coordinator.padded_buffer_size = _PADDED_BUFFER_SIZE
    coordinator.mem_pool_device = allocator.hisparse_kvcache
    coordinator.req_to_token_pool = SimpleNamespace(req_to_token=row)
    coordinator.token_to_kv_pool_allocator = allocator
    coordinator.req_to_device_buffer = torch.zeros(
        (1, _PADDED_BUFFER_SIZE), dtype=torch.int64
    )
    coordinator.req_device_buffer_size = torch.zeros(1, dtype=torch.int64)
    coordinator.req_device_buffer_tokens = torch.full(
        (1, 1, _PADDED_BUFFER_SIZE), -1, dtype=torch.int32
    )
    coordinator.req_device_buffer_token_locs = torch.full(
        (1, 1, _PADDED_BUFFER_SIZE), -1, dtype=torch.int32
    )
    coordinator._device_buffer_arange_i32 = torch.arange(
        _DEVICE_BUFFER_SIZE, dtype=torch.int32
    )
    return coordinator


def _stage_request(seq_len: int) -> SimpleNamespace:
    allocator = _make_dsv4_allocator()
    kv_allocated_len = ceil_align(seq_len, _LOGICAL_PAGE)

    logical_locs = allocator.alloc(kv_allocated_len)
    row = torch.zeros((1, kv_allocated_len), dtype=torch.int64, device=_DEV)
    row[_POOL_IDX, :kv_allocated_len] = logical_locs

    coordinator = _make_coordinator(allocator, row)
    mapping_before = allocator.full_to_hisparse_device_index_mapping.clone()
    req = SimpleNamespace(
        req_pool_idx=_POOL_IDX,
        extend_range=SimpleNamespace(end=seq_len),
        kv=SimpleNamespace(kv_allocated_len=kv_allocated_len),
    )

    coordinator.alloc_device_buffer(req)

    return SimpleNamespace(
        allocator=allocator,
        coordinator=coordinator,
        mapping_before=mapping_before,
    )


class TestDsv4StagingHarvestsTheWholeAllocation(CustomTestCase):
    def test_staging_then_release_returns_every_hisparse_page_exactly_once(self):
        """Harvesting only extend_range.end left the padding C4 entries of the page-rounded allocation mapped but unreachable, so their page was never freed and leaked for the process lifetime."""
        for seq_len in (256, 512, 513, 514, 515, 516):
            with self.subTest(seq_len=seq_len):
                fixture = _stage_request(seq_len)

                buffer_row = fixture.coordinator.req_to_device_buffer[_POOL_IDX]
                fixture.allocator.free_hisparse_indices(torch.unique(buffer_row))

                hisparse_allocator = fixture.allocator.hisparse_attn_allocator
                free_pages = hisparse_allocator.free_pages.tolist()
                self.assertEqual(
                    len(free_pages),
                    len(set(free_pages)),
                    "a hisparse page was returned to the free list twice",
                )
                self.assertEqual(len(free_pages), hisparse_allocator.num_pages)

    def test_reserve_slot_holds_the_last_real_compressed_token(self):
        """The reserved slot feeds the first decode step, so the newest-entry swap must select the last real C4 entry rather than a padding entry harvested from the aligned tail."""
        seq_len = 513
        fixture = _stage_request(seq_len)

        last_real_compressed_index = seq_len // 4 - 1
        expected_reserve = int(fixture.mapping_before[last_real_compressed_index])
        expected_prefix = fixture.mapping_before[:_DEVICE_BUFFER_SIZE].tolist()

        buffer_row = fixture.coordinator.req_to_device_buffer[_POOL_IDX]
        self.assertEqual(int(buffer_row[_DEVICE_BUFFER_SIZE]), expected_reserve)
        self.assertEqual(buffer_row[:_DEVICE_BUFFER_SIZE].tolist(), expected_prefix)


if __name__ == "__main__":
    unittest.main()
