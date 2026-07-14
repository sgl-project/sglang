import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
from sglang.srt.utils.common import ceil_align

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingAllocator:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.freed: list[torch.Tensor] = []
        self.freed_swa: list[torch.Tensor] = []

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.detach().cpu().clone())

    def free_swa(self, indices: torch.Tensor) -> None:
        self.freed_swa.append(indices.detach().cpu().clone())


def _make_req(swa_evict_floor: int) -> SimpleNamespace:
    return SimpleNamespace(
        req_pool_idx=0,
        origin_input_ids=list(range(10)),
        output_ids=[],
        cache_protected_len=0,
        swa_evict_floor=swa_evict_floor,
        kv=SimpleNamespace(swa_evicted_seqlen=0),
        extra_key=None,
        last_node=None,
    )


def _make_pure_swa_radix_cache(
    allocator: _RecordingAllocator,
) -> PureSWARadixCache:
    cache = PureSWARadixCache.__new__(PureSWARadixCache)
    cache.disable_finished_insert = False
    cache.disable = False
    cache.page_size = allocator.page_size
    cache.is_eagle = False
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(12, dtype=torch.int64).unsqueeze(0)
    )
    cache.token_to_kv_pool_allocator = allocator
    return cache


class TestSWAEvictFloorAlignment(unittest.TestCase):
    def test_scheduler_aligns_floor_to_concrete_allocator_page_size(self):
        """The scheduler persists one canonical floor for every admitted request."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(page_size=4)
        reqs = [
            SimpleNamespace(extend_range=SimpleNamespace(end=end))
            for end in (0, 1, 4, 5)
        ]

        scheduler._set_swa_evict_floors(reqs)

        self.assertEqual([req.swa_evict_floor for req in reqs], [0, 4, 4, 8])

    def test_common_reader_matches_old_local_ceil_result(self):
        """The common reader consumes the same boundary formerly produced locally."""
        page_size = 4
        old_unaligned_floor = 5
        canonical_floor = ceil_align(old_unaligned_floor, page_size)
        req = _make_req(canonical_floor)
        allocator = _RecordingAllocator(page_size)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(12, dtype=torch.int64).unsqueeze(0)
        )

        free_swa_out_of_window_slots(
            req,
            pre_len=7,
            sliding_window_size=8,
            page_size=page_size,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertEqual(req.kv.swa_evicted_seqlen, canonical_floor)
        self.assertEqual(allocator.freed_swa, [])

    def test_common_reader_rejects_misaligned_floor_before_free(self):
        """The common reader fails before allocator mutation on a malformed floor."""
        req = _make_req(swa_evict_floor=5)
        allocator = _RecordingAllocator(page_size=4)
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(12, dtype=torch.int64).unsqueeze(0)
        )

        with self.assertRaisesRegex(AssertionError, "swa_evict_floor"):
            free_swa_out_of_window_slots(
                req,
                pre_len=7,
                sliding_window_size=8,
                page_size=4,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
            )

        self.assertEqual(req.kv.swa_evicted_seqlen, 0)
        self.assertEqual(allocator.freed_swa, [])

    def test_pure_swa_radix_reader_matches_old_local_ceil_result(self):
        """The PureSWA radix reader preserves the old locally rounded boundary."""
        page_size = 4
        canonical_floor = ceil_align(5, page_size)
        allocator = _RecordingAllocator(page_size)
        cache = _make_pure_swa_radix_cache(allocator)
        req = _make_req(canonical_floor)
        req.kv.swa_evicted_seqlen = canonical_floor

        result = cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=10)
        allocator.free(
            cache.req_to_token_pool.req_to_token[
                req.req_pool_idx, result.unhandled_kv_start : 10
            ]
        )

        self.assertEqual(result.unhandled_kv_start, 8)
        self.assertEqual(len(allocator.freed), 2)
        self.assertTrue(torch.equal(torch.cat(allocator.freed), torch.arange(10)))

    def test_pure_swa_radix_reader_rejects_misaligned_floor_before_free(self):
        """The PureSWA radix reader fails before freeing a malformed boundary."""
        allocator = _RecordingAllocator(page_size=4)
        cache = _make_pure_swa_radix_cache(allocator)
        req = _make_req(swa_evict_floor=5)

        with self.assertRaisesRegex(AssertionError, "swa_evict_floor"):
            cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=10)

        self.assertEqual(allocator.freed, [])


if __name__ == "__main__":
    unittest.main()
