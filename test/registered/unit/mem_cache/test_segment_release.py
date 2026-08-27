"""Unit tests for the sync-free, positionally-named KV page release.

``free()`` dedups page ids with ``torch.unique`` and ``free_swa()`` additionally
reads ``full_to_swa_index_mapping`` back (``unique`` plus a ``> 0`` mask) to
discover which pages carry SWA state. Both shapes are data-dependent, so both
sync the device -- and behind the WAR-fenced schedule stream a sync costs a whole
forward. The new API lets the caller name the pages instead:

  - ``PagedTokenToKVPoolAllocator.free_page_reps()`` / ``free_segment()``
  - ``SWATokenToKVPoolAllocator.free_full_segment()`` / ``free_swa_segment()`` /
    ``free_swa_page_reps()``

Every test here pins the new path against the old one: same freed page set, same
mapping, same free-group behaviour. Real pools and allocators, CPU tensors --
only index bookkeeping is under test, never the KV math.

    python -m pytest test/registered/unit/mem_cache/test_segment_release.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_DTYPE = torch.bfloat16
_HEAD_NUM = 1
_HEAD_DIM = 8
_DEVICE = "cpu"
_ROWS = 4


def _build(page_size: int, *, cols_in_pages: int = 8, need_sort: bool = False):
    """One SWA allocator with every full slot mapped to an SWA slot.

    Returns the allocator and a ``[_ROWS, cols]`` view of the allocated full-pool
    indices, standing in for ``req_to_token``.
    """
    cols = cols_in_pages * page_size
    kv_size = (_ROWS * cols_in_pages + 4) * page_size
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size,
        page_size=page_size,
        dtype=_DTYPE,
        head_num=_HEAD_NUM,
        head_dim=_HEAD_DIM,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device=_DEVICE,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size,
        page_size=page_size,
        dtype=_DTYPE,
        device=_DEVICE,
        kvcache=kv_pool,
        need_sort=need_sort,
    )
    need = _ROWS * cols
    full = allocator.full_attn_allocator.alloc(need)
    swa = allocator.swa_attn_allocator.alloc(need)
    assert full is not None and swa is not None, "test pool too small"
    allocator.full_to_swa_index_mapping[full] = swa
    return allocator, full.reshape(_ROWS, cols)


def _free_page_set(sub_allocator) -> set:
    """Pages currently released, spanning both containers.

    ``need_sort`` (the PD decode configuration) routes frees into
    ``release_pages`` instead of ``free_pages``.
    """
    pages = torch.cat((sub_allocator.free_pages, sub_allocator.release_pages))
    return set(pages.tolist())


class TestPagedFreePageReps(CustomTestCase):
    """PagedTokenToKVPoolAllocator.free_page_reps / free_segment."""

    PAGE = 64

    def test_page_reps_free_same_pages_as_free(self):
        for start_page, num_pages in ((0, 1), (0, 4), (2, 3), (5, 2)):
            ref, ref_rows = _build(self.PAGE)
            new, new_rows = _build(self.PAGE)
            start = start_page * self.PAGE
            end = start + num_pages * self.PAGE

            ref_before = _free_page_set(ref.full_attn_allocator)
            ref.full_attn_allocator.free(ref_rows[0, start:end])
            ref_freed = _free_page_set(ref.full_attn_allocator) - ref_before

            new_before = _free_page_set(new.full_attn_allocator)
            new.full_attn_allocator.free_page_reps(new_rows[0, start : end : self.PAGE])
            new_freed = _free_page_set(new.full_attn_allocator) - new_before

            self.assertEqual(num_pages, len(ref_freed))
            self.assertEqual(ref_freed, new_freed, f"{start_page=} {num_pages=}")

    def test_empty_page_reps_is_a_noop(self):
        allocator, rows = _build(self.PAGE)
        before = _free_page_set(allocator.full_attn_allocator)

        allocator.full_attn_allocator.free_page_reps(rows[0, 0:0])

        self.assertEqual(before, _free_page_set(allocator.full_attn_allocator))

    def test_page_reps_inside_free_group_are_deferred(self):
        allocator, rows = _build(self.PAGE)
        sub = allocator.full_attn_allocator
        before = _free_page_set(sub)

        sub.free_group_begin()
        sub.free_page_reps(rows[0, 0 : 2 * self.PAGE : self.PAGE])
        self.assertEqual(before, _free_page_set(sub), "must not release inside a group")

        sub.free_group_end()
        self.assertEqual(2, len(_free_page_set(sub) - before))

    def test_group_owns_deferred_page_reps(self):
        # free() and free_segment() hand the deferred tensor to
        # _copy_for_free_group(); free_page_reps() must too. The reps are a
        # strided view of the caller's kv row, so without the copy a caller
        # that rewrites the row between begin/end silently changes which
        # pages free_group_end releases.
        allocator, rows = _build(self.PAGE)
        sub = allocator.full_attn_allocator
        reps = rows[0, 0 : 2 * self.PAGE : self.PAGE]
        expected = set((reps // self.PAGE).tolist())
        before = _free_page_set(sub)

        sub.free_group_begin()
        sub.free_page_reps(reps)
        rows[0].zero_()
        sub.free_group_end()

        self.assertEqual(expected, _free_page_set(sub) - before)

    def test_free_segment_matches_free_for_aligned_and_unaligned_starts(self):
        # start_pos == 0 takes the stride-slice branch; a non-aligned start_pos
        # additionally emits the partial head page.
        for start_pos, num_tokens in (
            (0, 2 * self.PAGE),
            (self.PAGE, 3 * self.PAGE),
            (self.PAGE // 2, 2 * self.PAGE),
            (self.PAGE + 7, self.PAGE),
        ):
            ref, ref_rows = _build(self.PAGE)
            new, new_rows = _build(self.PAGE)
            end = start_pos + num_tokens

            ref_before = _free_page_set(ref.full_attn_allocator)
            ref.full_attn_allocator.free(ref_rows[0, start_pos:end])
            ref_freed = _free_page_set(ref.full_attn_allocator) - ref_before

            new_before = _free_page_set(new.full_attn_allocator)
            new.full_attn_allocator.free_segment(
                new_rows[0, start_pos:end], start_pos=start_pos
            )
            new_freed = _free_page_set(new.full_attn_allocator) - new_before

            self.assertEqual(ref_freed, new_freed, f"{start_pos=} {num_tokens=}")

    def test_free_segment_with_need_sort_routes_to_release_pages(self):
        allocator, rows = _build(self.PAGE, need_sort=True)
        sub = allocator.full_attn_allocator
        released_before = sub.release_pages.numel()

        sub.free_segment(rows[0, 0 : 2 * self.PAGE], start_pos=0)

        self.assertEqual(released_before + 2, sub.release_pages.numel())


class TestSWASegmentRelease(CustomTestCase):
    """SWATokenToKVPoolAllocator.free_full_segment / free_swa_segment / reps."""

    PAGE = 64

    def _ranges_cases(self):
        p = self.PAGE
        return (
            [(0, 0, p)],
            [(0, 0, 4 * p)],
            [(0, 0, 2 * p), (1, p, 3 * p)],
            [(0, 0, 4 * p), (2, p, 2 * p), (3, 2 * p, 5 * p)],
            [(i, i * p, (i + 2) * p) for i in range(_ROWS)],
            # non-zero but still page-aligned starts
            [(i, 3 * p, 6 * p) for i in range(_ROWS)],
        )

    def test_page_reps_match_free_swa_pages_and_mapping(self):
        for ranges in self._ranges_cases():
            ref, ref_rows = _build(self.PAGE)
            new, new_rows = _build(self.PAGE)

            ref_before = _free_page_set(ref.swa_attn_allocator)
            for row, start, end in ranges:
                ref.free_swa(ref_rows[row, start:end])
            ref_freed = _free_page_set(ref.swa_attn_allocator) - ref_before

            new_before = _free_page_set(new.swa_attn_allocator)
            reps = torch.cat(
                [new_rows[row, start : end : self.PAGE] for row, start, end in ranges]
            )
            new.free_swa_page_reps(reps)
            new_freed = _free_page_set(new.swa_attn_allocator) - new_before

            self.assertEqual(ref_freed, new_freed, f"{ranges=}")
            self.assertTrue(
                torch.equal(
                    ref.full_to_swa_index_mapping, new.full_to_swa_index_mapping
                ),
                f"mapping differs for {ranges=}",
            )

    def test_freed_pages_are_reusable(self):
        allocator, rows = _build(self.PAGE)
        avail_before = allocator.swa_available_size()

        allocator.free_swa_page_reps(rows[0, 0 : 3 * self.PAGE : self.PAGE])

        self.assertEqual(avail_before + 3 * self.PAGE, allocator.swa_available_size())
        self.assertIsNotNone(allocator.swa_attn_allocator.alloc(3 * self.PAGE))

    def test_empty_reps_is_a_noop(self):
        allocator, rows = _build(self.PAGE)
        mapping_before = allocator.full_to_swa_index_mapping.clone()
        avail_before = allocator.swa_available_size()

        allocator.free_swa_page_reps(rows[0, 0:0])

        self.assertEqual(avail_before, allocator.swa_available_size())
        self.assertTrue(
            torch.equal(mapping_before, allocator.full_to_swa_index_mapping)
        )

    def test_swa_segment_delegates_to_reps_and_matches_free_swa(self):
        for start_pos, num_pages in ((0, 2), (self.PAGE, 3), (3 * self.PAGE, 1)):
            ref, ref_rows = _build(self.PAGE)
            new, new_rows = _build(self.PAGE)
            end = start_pos + num_pages * self.PAGE

            ref_before = _free_page_set(ref.swa_attn_allocator)
            ref.free_swa(ref_rows[0, start_pos:end])
            ref_freed = _free_page_set(ref.swa_attn_allocator) - ref_before

            new_before = _free_page_set(new.swa_attn_allocator)
            new.free_swa_segment(new_rows[0, start_pos:end], start_pos=start_pos)
            new_freed = _free_page_set(new.swa_attn_allocator) - new_before

            self.assertEqual(num_pages, len(ref_freed))
            self.assertEqual(ref_freed, new_freed, f"{start_pos=} {num_pages=}")
            self.assertTrue(
                torch.equal(
                    ref.full_to_swa_index_mapping, new.full_to_swa_index_mapping
                )
            )

    def test_swa_segment_page_size_one_falls_back_to_free_swa(self):
        # page_size == 1 has no page structure to name, so free_swa_segment must
        # delegate to free_swa; an unaligned start is legal there.
        start, end = 2, 8
        ref, ref_rows = _build(1)
        new, new_rows = _build(1)

        ref_before = _free_page_set(ref.swa_attn_allocator)
        ref.free_swa(ref_rows[0, start:end])
        ref_freed = _free_page_set(ref.swa_attn_allocator) - ref_before

        new_before = _free_page_set(new.swa_attn_allocator)
        new.free_swa_segment(new_rows[0, start:end], start_pos=start)
        new_freed = _free_page_set(new.swa_attn_allocator) - new_before

        self.assertEqual(end - start, len(ref_freed))
        self.assertEqual(ref_freed, new_freed)
        self.assertTrue(
            torch.equal(ref.full_to_swa_index_mapping, new.full_to_swa_index_mapping)
        )

    def test_swa_segment_empty_is_a_noop(self):
        allocator, rows = _build(self.PAGE)
        avail_before = allocator.swa_available_size()

        allocator.free_swa_segment(rows[0, 0:0], start_pos=0)

        self.assertEqual(avail_before, allocator.swa_available_size())

    def test_swa_segment_rejects_unaligned_start(self):
        allocator, rows = _build(self.PAGE)

        with self.assertRaises(AssertionError):
            allocator.free_swa_segment(rows[0, 1 : 1 + self.PAGE], start_pos=1)

    def test_full_segment_frees_only_the_full_pool(self):
        allocator, rows = _build(self.PAGE)
        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()

        allocator.free_full_segment(rows[0, 0 : 2 * self.PAGE], start_pos=0)

        self.assertEqual(full_before + 2 * self.PAGE, allocator.full_available_size())
        self.assertEqual(swa_before, allocator.swa_available_size())

    def test_split_release_frees_both_pools_exactly_once(self):
        """What SWAChunkCache.cache_finished_req does: name each pool's range."""
        ref, ref_rows = _build(self.PAGE)
        new, new_rows = _build(self.PAGE)
        end = 4 * self.PAGE

        ref.free(ref_rows[0, 0:end])
        new.free_full_segment(new_rows[0, 0:end], start_pos=0)
        new.free_swa_segment(new_rows[0, 0:end], start_pos=0)

        self.assertEqual(ref.full_available_size(), new.full_available_size())
        self.assertEqual(ref.swa_available_size(), new.swa_available_size())
        self.assertTrue(
            torch.equal(ref.full_to_swa_index_mapping, new.full_to_swa_index_mapping)
        )


if __name__ == "__main__":
    unittest.main()
