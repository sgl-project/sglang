"""Regression for all-SWA KV allocation at page_size > 1.

PureSWATokenToKVPoolAllocator asserted page_size == 1 in __init__ and raised
NotImplementedError from alloc_extend/alloc_decode, so an all-SWA model died with
a bare AssertionError on any backend whose kernels need a paged KV layout (the
Intel XPU backend rewrites page_size to 128). The single pool is addressed
through an identity full->SWA mapping, so the paged allocator can drive it
directly -- no SWA peer to allocate and no mapping to write.
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import PureSWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 128
POOL_PAGES = 8
POOL_TOKENS = PAGE_SIZE * POOL_PAGES


class _FakeSWAKVPool(BaseSWAKVPool):
    """Minimal BaseSWAKVPool: the allocator only stores it and registers the mapping."""

    def __init__(self):
        self.swa_kv_pool = object()
        self.registered_mapping = None

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.registered_mapping = full_to_swa_index_mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return kv_indices

    def get_state_buf_infos(self):
        return [], [], []

    def get_key_buffer(self, layer_id: int):
        raise NotImplementedError

    def get_value_buffer(self, layer_id: int):
        raise NotImplementedError

    def get_kv_buffer(self, layer_id: int):
        raise NotImplementedError

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError


class _RecordingPagedAllocator:
    """Stands in for the paged inner allocator so the delegation is observable
    without the triton alloc_extend/alloc_decode kernels, which need a GPU."""

    def __init__(self, *, free_pages: int):
        self.size = free_pages * PAGE_SIZE
        self.page_size = PAGE_SIZE
        self._free_pages = free_pages
        self.extend_calls = []
        self.decode_calls = []

    def available_size(self) -> int:
        return self._free_pages * PAGE_SIZE

    def alloc_extend(self, *args, num_new_pages=None, **kwargs):
        self.extend_calls.append(num_new_pages)
        return torch.arange(args[5] if len(args) > 5 else 0, dtype=torch.int64)

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        self.decode_calls.append(int(seq_lens_cpu.numel()))
        return torch.zeros(seq_lens_cpu.numel(), dtype=torch.int64)


def _make_allocator(page_size: int) -> PureSWATokenToKVPoolAllocator:
    return PureSWATokenToKVPoolAllocator(
        POOL_TOKENS,
        page_size,
        torch.bfloat16,
        "cpu",
        _FakeSWAKVPool(),
        need_sort=False,
    )


class TestPureSWAPagedAllocator(CustomTestCase):
    def test_paged_page_size_is_accepted(self):
        """page_size > 1 must construct and pick the paged inner allocator."""
        alloc = _make_allocator(PAGE_SIZE)

        self.assertIsInstance(alloc.swa_attn_allocator, PagedTokenToKVPoolAllocator)
        # Single pool: the full half must be the very same allocator object, or
        # the two halves would double-book the same slots.
        self.assertIs(alloc.full_attn_allocator, alloc.swa_attn_allocator)
        self.assertEqual(alloc.available_size(), POOL_TOKENS)

    def test_page_size_one_still_unpaged(self):
        """The page_size == 1 path must keep the token allocator it always used."""
        alloc = _make_allocator(1)

        self.assertIsInstance(alloc.swa_attn_allocator, TokenToKVPoolAllocator)

    def test_mapping_covers_every_addressable_slot(self):
        """Paged slots run one page past size_swa, so the identity mapping must
        too; a short mapping would index out of bounds on the last page."""
        alloc = _make_allocator(PAGE_SIZE)
        mapping = alloc.full_to_swa_index_mapping

        highest_slot = (POOL_PAGES + 1) * PAGE_SIZE - 1
        self.assertGreater(mapping.numel() - 1, highest_slot)
        self.assertEqual(int(mapping[highest_slot]), highest_slot)

    def test_free_round_trip_conserves_pages(self):
        """Freeing paged slots must return the pool to full, or the all-SWA pool
        leaks a page per request. Pages come from the inner allocator because the
        wrapper's alloc() is the unpaged API."""
        alloc = _make_allocator(PAGE_SIZE)

        indices = alloc.swa_attn_allocator.alloc(2 * PAGE_SIZE)
        self.assertIsNotNone(indices)
        # Page-aligned: the XPU kernels index the page table by index // page_size.
        self.assertEqual(int(indices[0]) % PAGE_SIZE, 0)
        self.assertEqual(alloc.available_size(), POOL_TOKENS - 2 * PAGE_SIZE)

        alloc.free(indices)
        self.assertEqual(alloc.available_size(), POOL_TOKENS)

    def test_free_is_page_deduplicated(self):
        """Freeing the slots of one page must credit that page once, not once per
        slot -- over-crediting hands the same page out twice."""
        alloc = _make_allocator(PAGE_SIZE)

        indices = alloc.swa_attn_allocator.alloc(PAGE_SIZE)
        alloc.free(indices)

        self.assertEqual(alloc.available_size(), POOL_TOKENS)

    def test_alloc_is_still_the_unpaged_api(self):
        """alloc() stays page_size == 1 only, as in the parent: the paged
        scheduler path must keep using alloc_extend/alloc_decode."""
        alloc = _make_allocator(PAGE_SIZE)

        with self.assertRaises(AssertionError):
            alloc.alloc(PAGE_SIZE)

    def test_alloc_extend_gates_on_capacity_before_delegating(self):
        """An extend that needs more pages than remain must return None instead
        of letting the inner allocator hand out slots it does not have."""
        alloc = _make_allocator(PAGE_SIZE)
        alloc.swa_attn_allocator = alloc.full_attn_allocator = _RecordingPagedAllocator(
            free_pages=1
        )

        out = alloc.alloc_extend(
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            torch.tensor([3 * PAGE_SIZE], dtype=torch.int64),
            torch.tensor([3 * PAGE_SIZE], dtype=torch.int64),
            torch.tensor([-1], dtype=torch.int64),
            3 * PAGE_SIZE,
        )

        self.assertIsNone(out)
        self.assertEqual(alloc.swa_attn_allocator.extend_calls, [])

    def test_alloc_extend_delegates_once_with_page_count(self):
        """The single pool must be paged once per extend -- a second call (the
        hybrid parent's SWA peer) would consume the pages twice."""
        alloc = _make_allocator(PAGE_SIZE)
        alloc.swa_attn_allocator = alloc.full_attn_allocator = _RecordingPagedAllocator(
            free_pages=POOL_PAGES
        )

        alloc.alloc_extend(
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            torch.tensor([2 * PAGE_SIZE], dtype=torch.int64),
            torch.tensor([2 * PAGE_SIZE], dtype=torch.int64),
            torch.tensor([-1], dtype=torch.int64),
            2 * PAGE_SIZE,
        )

        self.assertEqual(alloc.swa_attn_allocator.extend_calls, [2])

    def test_alloc_decode_delegates_once(self):
        """Same one-pool contract on the decode step."""
        alloc = _make_allocator(PAGE_SIZE)
        alloc.swa_attn_allocator = alloc.full_attn_allocator = _RecordingPagedAllocator(
            free_pages=POOL_PAGES
        )

        alloc.alloc_decode(
            torch.tensor([PAGE_SIZE + 1], dtype=torch.int64),
            torch.tensor([PAGE_SIZE + 1], dtype=torch.int64),
            torch.tensor([PAGE_SIZE - 1], dtype=torch.int64),
        )

        self.assertEqual(alloc.swa_attn_allocator.decode_calls, [1])


if __name__ == "__main__":
    unittest.main()
