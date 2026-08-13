"""free_swa_segment vs an independent oracle: stride page extraction over segment
alignments and need_sort routing, plus the free-group contract. See
SWATokenToKVPoolAllocator.free_swa_segment for why the data-dependent ops are avoided.

    python -m pytest test/registered/unit/mem_cache/test_swa_free_segment.py -v
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.multi_ended_allocator import UnifiedSWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SIZE = 256


def _pool():
    # The free path never reaches the pool, so isinstance plus swa_kv_pool is all
    # the allocator needs from it.
    pool = MagicMock(spec=BaseSWAKVPool)
    pool.swa_kv_pool = None
    return pool


def _alloc(page_size, need_sort=False):
    return SWATokenToKVPoolAllocator(
        size=SIZE,
        size_swa=SIZE,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=_pool(),
        need_sort=need_sort,
    )


def _row(alloc, num_tokens):
    """alloc_extend without its triton kernel: one page-aligned full and swa
    allocation of the same length, so token t's swa slot is
    swa_page[t // ps] * ps + t % ps. The GPU suite covers the real path."""
    ps = alloc.page_size
    padded = -(num_tokens // -ps) * ps
    full = alloc.full_attn_allocator.alloc(padded)
    alloc.set_full_to_swa_mapping(full, alloc.swa_attn_allocator.alloc(padded))
    return full[:num_tokens]


def _freed(alloc):
    # need_sort (PD disagg) routes frees into release_pages, so span both.
    inner = alloc.swa_attn_allocator
    return set(torch.cat((inner.free_pages, inner.release_pages)).tolist())


def _oracle(row, start, end, ps, mapping):
    """Independent reference in plain Python -- never an allocator free path.
    Returns (swa page ids to release, mapping slots that must be zero)."""
    pages = sorted({row[i].item() // ps for i in range(start, end)})
    touched = [p * ps + off for p in pages for off in range(ps)]
    swa = sorted({mapping[t].item() // ps for t in touched if mapping[t].item() > 0})
    return swa, touched


class TestSWAFreeSegment(unittest.TestCase):
    def test_sweep_against_oracle(self):
        # Aligned start per the contract; aligned, partial and interior segments.
        for ps in (1, 2, 4):
            for need_sort in (False, True):
                for n in (ps, ps + 1, 3 * ps - 1):
                    for start in range(0, n, ps):
                        for end in range(start + 1, n + 1):
                            a = _alloc(ps, need_sort=need_sort)
                            row = _row(a, n)
                            want_swa, want_zero = _oracle(
                                row, start, end, ps, a.full_to_swa_index_mapping.clone()
                            )
                            before = _freed(a)
                            a.free_swa_segment(
                                row[start:end], start_pos=start, swa_alive_from=start
                            )

                            label = f"{ps=} {need_sort=} {n=} {start=} {end=}"
                            self.assertEqual(
                                sorted(_freed(a) - before), want_swa, label
                            )
                            got = a.full_to_swa_index_mapping[want_zero]
                            self.assertTrue(
                                torch.equal(got, torch.zeros_like(got)), label
                            )

    def test_group_defers_and_queues_owned_values(self):
        # Queueing the caller's view would make the flush read whatever a remap
        # wrote into that row in between.
        for ps in (1, 4):
            a = _alloc(ps)
            row = _row(a, 2 * ps)
            want_swa, want_zero = _oracle(
                row, 0, row.numel(), ps, a.full_to_swa_index_mapping.clone()
            )
            before = _freed(a)

            a.free_group_begin()
            a.free_swa_segment(row, start_pos=0, swa_alive_from=0)
            self.assertEqual(_freed(a), before, f"not deferred: {ps=}")
            row.fill_(1)  # stand in for the row being remapped mid-group
            a.free_group_end()

            self.assertEqual(sorted(_freed(a) - before), want_swa, f"{ps=}")
            got = a.full_to_swa_index_mapping[want_zero]
            self.assertTrue(torch.equal(got, torch.zeros_like(got)), f"{ps=}")

    def test_contract_asserts(self):
        a = _alloc(4)
        a.debug_mode = True
        row = _row(a, 8)
        with self.assertRaises(AssertionError):  # start_pos not page aligned
            a.free_swa_segment(row[1:], start_pos=1, swa_alive_from=1)
        a.free_swa_segment(row, start_pos=0, swa_alive_from=0)
        with self.assertRaises(AssertionError):  # range already freed
            a.free_swa_segment(row, start_pos=0, swa_alive_from=0)

    def test_dead_prefix_is_rejected_without_debug_mode(self):
        """The liveness precondition is host-side, so it holds without
        debug_mode -- reading a dead page would release page 0 and leak."""
        ps = 4
        a = _alloc(ps)
        a.debug_mode = False
        row = _row(a, 3 * ps)
        # Front page already evicted: the request is live only from ps onward.
        a.free_swa_segment(row[:ps], start_pos=0, swa_alive_from=0)
        with self.assertRaises(AssertionError):
            a.free_swa_segment(row, start_pos=0, swa_alive_from=ps)
        # Stating the frontier correctly frees only what is still mapped.
        a.free_swa_segment(row[ps:], start_pos=ps, swa_alive_from=ps)
        self.assertNotIn(0, _freed(a))


class TestOptOuts(unittest.TestCase):
    def test_pure_swa_keeps_its_identity_mapping(self):
        # full == swa and the mapping is a constant identity table; zeroing it
        # would break every later translate.
        a = PureSWATokenToKVPoolAllocator(
            size_swa=SIZE,
            page_size=1,
            dtype=torch.float16,
            device="cpu",
            kvcache=_pool(),
            need_sort=False,
        )
        before = a.full_to_swa_index_mapping.clone()
        a.free_swa_segment(a.alloc(4), start_pos=0, swa_alive_from=0)
        self.assertTrue(torch.equal(a.full_to_swa_index_mapping, before))

    def test_unified_swa_does_not_inherit_the_fast_path(self):
        # Shared mode has no mapping table (the swa v2p IS the mapping) and
        # tombstones with -1. Too heavy to build on CPU, so this is the only guard
        # against the override being dropped, which fails silently.
        self.assertIsNot(
            UnifiedSWATokenToKVPoolAllocator.free_swa_segment,
            SWATokenToKVPoolAllocator.free_swa_segment,
        )


if __name__ == "__main__":
    unittest.main()
