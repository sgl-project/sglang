"""free_swa_segment vs an independent oracle and vs free_swa: stride page
extraction over segment alignments, need_sort routing, contract asserts, and the
opt-outs that must not inherit the fast path. See
SWATokenToKVPoolAllocator.free_swa_segment for why the data-dependent ops are
avoided.

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

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

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


def _make_allocator(page_size, need_sort=False):
    return SWATokenToKVPoolAllocator(
        size=SIZE_FULL,
        size_swa=SIZE_SWA,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=_FakeSWAKVPool(),
        need_sort=need_sort,
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
    swa_page[t // ps] * ps + t % ps -- the invariant the fast path relies on.
    The GPU suite covers the real alloc_extend path."""
    ps = alloc.page_size
    padded = -(num_tokens // -ps) * ps
    full = alloc.full_attn_allocator.alloc(padded)
    swa = alloc.swa_attn_allocator.alloc(padded)
    alloc.set_full_to_swa_mapping(full, swa)
    return full[:num_tokens]


def _swa_free_pages(alloc):
    # Span both containers: need_sort (PD disagg) routes frees into release_pages.
    inner = alloc.swa_attn_allocator
    return torch.sort(torch.cat((inner.free_pages, inner.release_pages)))[0]


def _oracle(row, start, end, page_size, mapping):
    """Independent reference, computed in plain Python from the row and a
    pre-free snapshot of the mapping -- never through an allocator free path.

    Returns (swa page ids that must be released, mapping slots that must be 0).
    """
    ps = page_size
    full_pages = sorted({row[i].item() // ps for i in range(start, end)})
    touched = [p * ps + off for p in full_pages for off in range(ps)]
    swa_pages = sorted(
        {mapping[t].item() // ps for t in touched if mapping[t].item() > 0}
    )
    return swa_pages, touched


class TestSWAFreeSegment(unittest.TestCase):
    def test_sweep_against_oracle_and_free_swa(self):
        # Segment alignments (aligned start per the contract, aligned and partial
        # tails, interior segments as the real caller emits) x need_sort routing.
        for page_size in (1, 2, 4):
            for need_sort in (False, True):
                for num_tokens in (page_size, page_size + 1, 3 * page_size - 1):
                    for start in range(0, num_tokens, page_size):
                        for end in range(start + 1, num_tokens + 1):
                            self._check_one(
                                page_size, need_sort, num_tokens, start, end
                            )

    def _check_one(self, page_size, need_sort, num_tokens, start, end):
        label = f"{page_size=} {need_sort=} {num_tokens=} {start=} {end=}"

        fast = _make_allocator(page_size, need_sort=need_sort)
        fast_row = _alloc_row(fast, num_tokens)
        expected_swa_pages, expected_cleared = _oracle(
            fast_row, start, end, page_size, fast.full_to_swa_index_mapping.clone()
        )
        before = _swa_free_pages(fast)

        fast.free_swa_segment(fast_row[start:end], start_pos=start)

        released = sorted(set(_swa_free_pages(fast).tolist()) - set(before.tolist()))
        self.assertEqual(released, expected_swa_pages, f"vs oracle: {label}")
        cleared = fast.full_to_swa_index_mapping[expected_cleared]
        self.assertTrue(
            torch.equal(cleared, torch.zeros_like(cleared)), f"mapping: {label}"
        )

        legacy = _make_allocator(page_size, need_sort=need_sort)
        legacy_row = _alloc_row(legacy, num_tokens)
        legacy.free_swa(legacy_row[start:end])
        self.assertTrue(
            torch.equal(_swa_free_pages(legacy), _swa_free_pages(fast)),
            f"vs free_swa: {label}",
        )
        self.assertTrue(
            torch.equal(
                legacy.full_to_swa_index_mapping, fast.full_to_swa_index_mapping
            ),
            f"vs free_swa mapping: {label}",
        )

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
        # Guards the decision that the fast path stays out of the free group:
        # deferring would mean reading the caller's req_to_token view after the
        # call returns. Not swept across the other dimensions -- neither path
        # reads is_not_in_free_group, so the dimension gates nothing.
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
                f"{page_size=}",
            )
            self.assertTrue(
                torch.equal(
                    legacy.full_to_swa_index_mapping, fast.full_to_swa_index_mapping
                ),
                f"{page_size=}",
            )


class TestPureSWAFreeSegment(unittest.TestCase):
    def test_identity_mapping_survives(self):
        # full == swa here and the mapping is a constant identity table: zeroing
        # it would break every later translate. Also the behavioral guard that
        # PureSWA keeps its own free_swa_segment.
        alloc = _make_pure_allocator()
        before = alloc.full_to_swa_index_mapping.clone()
        indices = alloc.alloc(4)
        alloc.free_swa_segment(indices, start_pos=0)
        self.assertTrue(torch.equal(alloc.full_to_swa_index_mapping, before))
        self.assertEqual(
            sorted(
                set(indices.tolist())
                & set(alloc.swa_attn_allocator.free_pages.tolist())
            ),
            sorted(indices.tolist()),
        )


class TestFastPathOptOuts(unittest.TestCase):
    def test_unified_swa_does_not_inherit_the_fast_path(self):
        # Shared mode has no full_to_swa_index_mapping table (the swa v2p IS the
        # mapping) and tombstones with -1, so inheriting the parent's gather and
        # page clearing would free live pages. No behavioral test here -- the
        # allocator is too heavy to build on CPU -- so this is the only guard,
        # and dropping the override fails silently otherwise.
        self.assertIsNot(
            UnifiedSWATokenToKVPoolAllocator.free_swa_segment,
            SWATokenToKVPoolAllocator.free_swa_segment,
        )


if __name__ == "__main__":
    unittest.main()
