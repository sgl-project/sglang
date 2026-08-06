"""free_swa_segment vs free_swa: differential equivalence over page sizes and
segment lengths, whole-page mapping clearing, contract asserts, and the opt-outs
that must not inherit the fast path. See SWATokenToKVPoolAllocator.free_swa_segment
for why the data-dependent ops are avoided.

    python -m pytest test/registered/unit/mem_cache/test_swa_free_segment.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.multi_ended_allocator import UnifiedSWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SIZE_FULL = 256
SIZE_SWA = 128


class _FakeSWAKVPool(BaseSWAKVPool):
    """The allocator only needs isinstance() plus register_mapping; the buffer
    methods are never reached on the free path."""

    def __init__(self):
        self.swa_kv_pool = None
        self.mapping = None

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.mapping = full_to_swa_index_mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return self.mapping[kv_indices]

    def get_state_buf_infos(self):
        raise NotImplementedError

    def get_key_buffer(self, layer_id):
        raise NotImplementedError

    def get_value_buffer(self, layer_id):
        raise NotImplementedError

    def get_kv_buffer(self, layer_id):
        raise NotImplementedError

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError


def _make_allocator(page_size):
    return SWATokenToKVPoolAllocator(
        size=SIZE_FULL,
        size_swa=SIZE_SWA,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=_FakeSWAKVPool(),
        need_sort=False,
    )


def _make_pure_allocator():
    return PureSWATokenToKVPoolAllocator(
        size_swa=SIZE_SWA,
        page_size=1,
        dtype=torch.float16,
        device="cpu",
        kvcache=_FakeSWAKVPool(),
        need_sort=False,
    )


def _alloc_row(alloc, num_tokens):
    """Mirror alloc_extend without its triton kernel: one page-aligned full and
    swa allocation driven by the same length, so token t's swa slot is
    swa_page[t // ps] * ps + t % ps -- the invariant the fast path relies on."""
    ps = alloc.page_size
    padded = -(num_tokens // -ps) * ps
    full = alloc.full_attn_allocator.alloc(padded)
    swa = alloc.swa_attn_allocator.alloc(padded)
    alloc.set_full_to_swa_mapping(full, swa)
    return full[:num_tokens]


def _swa_free_pages(alloc):
    inner = alloc.swa_attn_allocator
    return torch.sort(inner.free_pages)[0]


class TestSWAFreeSegment(unittest.TestCase):
    def test_matches_legacy_free_swa(self):
        # Sweep page sizes and segment lengths, including a trailing partial
        # page, at every page-aligned start the contract allows.
        for page_size in (1, 2, 4):
            for num_tokens in (1, page_size, page_size + 1, 3 * page_size - 1):
                for start in range(0, num_tokens, page_size):
                    legacy, fast = _make_allocator(page_size), _make_allocator(
                        page_size
                    )
                    legacy_row, fast_row = (
                        _alloc_row(legacy, num_tokens),
                        _alloc_row(fast, num_tokens),
                    )

                    legacy.free_swa(legacy_row[start:])
                    fast.free_swa_segment(fast_row[start:], start_pos=start)

                    label = f"{page_size=} {num_tokens=} {start=}"
                    self.assertTrue(
                        torch.equal(_swa_free_pages(legacy), _swa_free_pages(fast)),
                        f"swa pages differ: {label}",
                    )
                    self.assertTrue(
                        torch.equal(
                            legacy.full_to_swa_index_mapping,
                            fast.full_to_swa_index_mapping,
                        ),
                        f"mapping differs: {label}",
                    )

    def test_clears_whole_page_of_a_partial_tail(self):
        # A page only partly covered by the segment still gets its whole mapping
        # row zeroed, matching _expand_to_full_pages.
        alloc = _make_allocator(4)
        row = _alloc_row(alloc, 6)
        alloc.free_swa_segment(row, start_pos=0)
        page_of_tail = (row[4].item() // 4) * 4
        cleared = alloc.full_to_swa_index_mapping[page_of_tail : page_of_tail + 4]
        self.assertTrue(torch.equal(cleared, torch.zeros_like(cleared)))

    def test_empty_segment_is_noop(self):
        alloc = _make_allocator(4)
        row = _alloc_row(alloc, 4)
        before = _swa_free_pages(alloc)
        alloc.free_swa_segment(row[:0], start_pos=0)
        self.assertTrue(torch.equal(_swa_free_pages(alloc), before))

    def test_unaligned_start_pos_rejected(self):
        alloc = _make_allocator(4)
        row = _alloc_row(alloc, 8)
        with self.assertRaises(AssertionError):
            alloc.free_swa_segment(row[1:], start_pos=1)

    def test_debug_mode_catches_already_dead_mapping(self):
        # Freeing the same range twice is exactly what the contract forbids, and
        # what the legacy `> 0` filter used to absorb silently.
        alloc = _make_allocator(4)
        alloc.debug_mode = True
        row = _alloc_row(alloc, 8)
        alloc.free_swa_segment(row, start_pos=0)
        with self.assertRaises(AssertionError):
            alloc.free_swa_segment(row, start_pos=0)

    def test_inside_a_free_group_matches_free_swa(self):
        # Neither path participates in the free group -- both run immediately.
        # Deferring would mean reading the caller's req_to_token view after the
        # call returns, which only holds while nobody rewrites that row.
        for page_size in (1, 4):
            legacy, fast = _make_allocator(page_size), _make_allocator(page_size)
            legacy_row = _alloc_row(legacy, 2 * page_size)
            fast_row = _alloc_row(fast, 2 * page_size)

            for alloc in (legacy, fast):
                alloc.free_group_begin()
            legacy.free_swa(legacy_row)
            fast.free_swa_segment(fast_row, start_pos=0)
            for alloc in (legacy, fast):
                alloc.free_group_end()

            self.assertTrue(
                torch.equal(_swa_free_pages(legacy), _swa_free_pages(fast)),
                f"swa pages differ: {page_size=}",
            )
            self.assertTrue(
                torch.equal(
                    legacy.full_to_swa_index_mapping, fast.full_to_swa_index_mapping
                ),
                f"mapping differs: {page_size=}",
            )


class TestPureSWAFreeSegment(unittest.TestCase):
    def test_identity_mapping_survives(self):
        # full == swa here and the mapping is a constant identity table: zeroing
        # it would break every later translate.
        alloc = _make_pure_allocator()
        before = alloc.full_to_swa_index_mapping.clone()
        indices = alloc.alloc(4)
        alloc.free_swa_segment(indices, start_pos=0)
        self.assertTrue(torch.equal(alloc.full_to_swa_index_mapping, before))

    def test_matches_legacy_free_swa(self):
        legacy, fast = _make_pure_allocator(), _make_pure_allocator()
        legacy_indices, fast_indices = legacy.alloc(4), fast.alloc(4)
        legacy.free_swa(legacy_indices)
        fast.free_swa_segment(fast_indices, start_pos=0)
        self.assertTrue(
            torch.equal(
                torch.sort(legacy.swa_attn_allocator.free_pages)[0],
                torch.sort(fast.swa_attn_allocator.free_pages)[0],
            )
        )


class TestFastPathOptOuts(unittest.TestCase):
    def test_unified_swa_does_not_inherit_the_fast_path(self):
        # Shared mode has no full_to_swa_index_mapping table (the swa v2p IS the
        # mapping) and tombstones with -1, so inheriting the parent's gather and
        # page clearing would free live pages. Guards against the override being
        # dropped, which fails silently.
        self.assertIsNot(
            UnifiedSWATokenToKVPoolAllocator.free_swa_segment,
            SWATokenToKVPoolAllocator.free_swa_segment,
        )

    def test_pure_swa_does_not_inherit_the_fast_path(self):
        self.assertIsNot(
            PureSWATokenToKVPoolAllocator.free_swa_segment,
            SWATokenToKVPoolAllocator.free_swa_segment,
        )


if __name__ == "__main__":
    unittest.main()
