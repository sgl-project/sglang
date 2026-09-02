"""Unit tests for the fixed-shape in-place
out-of-window SWA free must leave allocator + mapping state BITWISE identical
to eager free_swa — same call sites, same timing (in-group frees still land at
free_group_end) — and must fall back to the legacy path for inputs its
contract does not cover (partial mappings, unaligned frontiers, eager unified
compaction)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 128
FULL_PAGES = 128
SWA_PAGES = 64


def _make_allocator(page_size=PAGE) -> SWATokenToKVPoolAllocator:
    kvcache = MagicMock(spec=BaseSWAKVPool)
    kvcache.full_kv_pool = None
    kvcache.swa_kv_pool = None
    return SWATokenToKVPoolAllocator(
        size=FULL_PAGES * page_size,
        size_swa=SWA_PAGES * page_size,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=kvcache,
        need_sort=False,
    )


def _segments(seed: int, n_segments: int, page_size=PAGE, pages_per_seg=2):
    """Disjoint page-aligned full-index segments (whole pages, offset 0) —
    the shape free_swa_out_of_window_slots produces."""
    g = torch.Generator().manual_seed(seed)
    pages = torch.randperm(FULL_PAGES - 1, generator=g)[: n_segments * pages_per_seg]
    pages = pages + 1
    segs = []
    for i in range(n_segments):
        p = pages[pages_per_seg * i : pages_per_seg * (i + 1)]
        idx = (p[:, None] * page_size + torch.arange(page_size)[None, :]).reshape(-1)
        segs.append(idx.to(torch.int64))
    return segs


def _install(alloc, segs, page_size=PAGE, mapped_fraction=1.0):
    """Install swa mappings the way paired paged allocation would: each full
    page of a segment maps onto one whole swa page with matching offsets."""
    for seg in segs:
        n = int(seg.numel() * mapped_fraction) // page_size * page_size
        if n:
            swa_indices = alloc.swa_attn_allocator.alloc(n)
            assert swa_indices is not None
            alloc.set_full_to_swa_mapping(seg[:n], swa_indices)


def _bitwise_state(alloc):
    """Exact allocator + mapping state: free-list tensors bit-for-bit (order
    matters — allocation pops from the front), the mapping tensor, and the
    availability the scheduler reads."""
    sub = alloc.swa_attn_allocator
    return (
        sub.free_pages.clone(),
        sub.release_pages.clone() if sub.release_pages is not None else None,
        alloc.full_to_swa_index_mapping.clone(),
        alloc.swa_available_size(),
    )


def _assert_states_equal(test, a, b):
    test.assertTrue(torch.equal(a[0], b[0]), f"free_pages differ:\n{a[0]}\n{b[0]}")
    if a[1] is not None or b[1] is not None:
        test.assertTrue(torch.equal(a[1], b[1]), "release_pages differ")
    test.assertTrue(torch.equal(a[2], b[2]), "mapping differs")
    test.assertEqual(a[3], b[3], "availability differs")


class TestSyncFreeSwaInplaceStatic(CustomTestCase):
    """Static-partition SWATokenToKVPoolAllocator (what galileo d36 runs)."""

    def _run(
        self,
        segs,
        use_inplace,
        *,
        page_size=PAGE,
        group=False,
        mixed_seg=None,
        mapped_fraction=1.0,
        partial_flag=False,
        start_positions=None,
    ):
        alloc = _make_allocator(page_size)
        install_segs = segs + ([mixed_seg] if mixed_seg is not None else [])
        _install(alloc, install_segs, page_size, mapped_fraction)
        if partial_flag:
            alloc._swa_mapping_may_be_partial = True
        if group:
            alloc.free_group_begin()
        for i, seg in enumerate(segs):
            start_pos = start_positions[i] if start_positions is not None else 0
            if use_inplace:
                alloc.free_swa_segment_inplace(seg, start_pos=start_pos)
            else:
                alloc.free_swa(seg)
        if mixed_seg is not None:
            alloc.free_swa(mixed_seg)
        if group:
            alloc.free_group_end()
        return _bitwise_state(alloc)

    def test_inplace_equals_eager_bitwise(self):
        segs = _segments(seed=3, n_segments=5)
        a = self._run(segs, use_inplace=False)
        b = self._run(segs, use_inplace=True)
        _assert_states_equal(self, a, b)

    def test_inplace_in_free_group_equals_eager(self):
        segs = _segments(seed=7, n_segments=4)
        a = self._run(segs, use_inplace=False, group=True)
        b = self._run(segs, use_inplace=True, group=True)
        _assert_states_equal(self, a, b)

    def test_mixed_group_equals_eager(self):
        segs = _segments(seed=11, n_segments=4)
        mixed = segs[-1]
        segs = segs[:-1]
        a = self._run(segs, use_inplace=False, group=True, mixed_seg=mixed)
        b = self._run(segs, use_inplace=True, group=True, mixed_seg=mixed)
        _assert_states_equal(self, a, b)

    def test_partial_mapping_flag_falls_back(self):
        segs = _segments(seed=5, n_segments=3)
        a = self._run(segs, use_inplace=False, mapped_fraction=0.5, partial_flag=True)
        b = self._run(segs, use_inplace=True, mapped_fraction=0.5, partial_flag=True)
        _assert_states_equal(self, a, b)

    def test_unaligned_start_falls_back(self):
        segs = _segments(seed=13, n_segments=2)
        a = self._run(segs, use_inplace=False)
        b = self._run(segs, use_inplace=True, start_positions=[1, 1])
        _assert_states_equal(self, a, b)

    def test_tail_alloc_branchless_flag_set_semantics(self):
        segs = _segments(seed=17, n_segments=2)
        a = self._run(segs, use_inplace=False, partial_flag=True)
        b = self._run(segs, use_inplace=True, partial_flag=True)
        _assert_states_equal(self, a, b)

    def test_page_size_one_inplace_equals_eager(self):
        segs = []
        g = torch.Generator().manual_seed(23)
        perm = torch.randperm(FULL_PAGES - 1, generator=g) + 1
        segs = [perm[:16].to(torch.int64), perm[16:48].to(torch.int64)]
        a = self._run(segs, use_inplace=False, page_size=1)
        b = self._run(segs, use_inplace=True, page_size=1)
        _assert_states_equal(self, a, b)

    def test_page_size_one_in_group(self):
        g = torch.Generator().manual_seed(29)
        perm = torch.randperm(FULL_PAGES - 1, generator=g) + 1
        segs = [perm[:16].to(torch.int64), perm[16:48].to(torch.int64)]
        a = self._run(segs, use_inplace=False, page_size=1, group=True)
        b = self._run(segs, use_inplace=True, page_size=1, group=True)
        _assert_states_equal(self, a, b)

    def test_out_of_window_helper_matches_eager(self):
        def setup():
            alloc = _make_allocator()
            req_to_token = torch.zeros(1, 512, dtype=torch.int64)
            row = torch.arange(512, dtype=torch.int64) + 512
            req_to_token[0] = row
            alloc.set_full_to_swa_mapping(row, alloc.swa_attn_allocator.alloc(512))
            return alloc, req_to_token, row

        alloc_a, _, row = setup()
        alloc_a.free_swa(row[:256])
        a = _bitwise_state(alloc_a)

        alloc_b, req_to_token, _ = setup()
        req = SimpleNamespace(kv=ReqKvInfo(req_pool_idx=0, cache_protected_len=0))
        free_swa_out_of_window_slots(
            req,
            511,
            sliding_window_size=PAGE,
            page_size=PAGE,
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=alloc_b,
        )
        self.assertEqual(req.kv.swa_evicted_seqlen, 256)
        b = _bitwise_state(alloc_b)
        _assert_states_equal(self, a, b)

    def test_pure_swa_inplace(self):
        kvcache = MagicMock(spec=BaseSWAKVPool)
        kvcache.swa_kv_pool = None

        def build():
            return PureSWATokenToKVPoolAllocator(
                size_swa=SWA_PAGES,
                page_size=1,
                dtype=torch.float16,
                device="cpu",
                kvcache=kvcache,
                need_sort=False,
            )

        def run(inplace):
            alloc = build()
            idx = alloc.alloc(8)
            if inplace:
                alloc.free_swa_segment_inplace(idx, start_pos=0)
            else:
                alloc.free_swa(idx)
            return (
                alloc.swa_attn_allocator.free_pages.clone(),
                alloc.swa_attn_allocator.available_size(),
            )

        fp_a, av_a = run(False)
        fp_b, av_b = run(True)
        self.assertTrue(torch.equal(fp_a, fp_b))
        self.assertEqual(av_a, av_b)


class TestSyncFreeSwaInplaceUnified(CustomTestCase):
    """UnifiedSWATokenToKVPoolAllocator / MultiEndedAllocator coverage."""

    def _build_composite(self, page_size=1, lazy=True, n_full_pages=32, n_swa_pages=16):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedSWATokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.unified_memory_pool import (
            MHASubPoolSpec,
            UnifiedKVPool,
        )

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        total = (
            n_full_pages * page_size * full_spec.entry_bytes()
            + n_swa_pages * page_size * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device="cpu",
            enable_memory_saver=False,
        )

        class _SubKV:
            def __init__(self, max_slots):
                self.buf = torch.full((max_slots,), -1, dtype=torch.int64)
                self.allocator = None

            def move_kv_cache(self, dst_loc, src_loc):
                self.buf[dst_loc] = self.buf[src_loc].clone()

            def attach_allocator(self, allocator):
                self.allocator = allocator

        kvcache = SimpleNamespace(
            full_kv_pool=_SubKV(pool.max_slots("full")),
            swa_kv_pool=_SubKV(pool.max_slots("swa")),
            attach_allocators=lambda *, full_allocator, swa_allocator: None,
        )
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device="cpu",
            full_max_total_num_tokens=n_full_pages * page_size,
            swa_max_total_num_tokens=n_swa_pages * page_size,
            page_size=page_size,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy,
        )
        return allocator

    @staticmethod
    def _swa_state(allocator):
        sa = allocator.swa_attn_allocator
        return (
            sa.virtual_to_physical.clone(),
            sa.physical_to_virtual.clone(),
            sa._free_phys_pages.clone(),
            sa.free_virtual_ids.clone() if sa.free_virtual_ids is not None else None,
            sa.live_page_count,
        )

    def _assert_swa_state_equal(self, a, b):
        names = ("v2p", "p2v", "_free_phys_pages", "free_virtual_ids")
        for name, ta, tb in zip(names, a[:4], b[:4]):
            if ta is None or tb is None:
                self.assertIsNone(ta)
                self.assertIsNone(tb)
                continue
            self.assertTrue(torch.equal(ta, tb), f"{name} differs:\n{ta}\n{tb}")
        self.assertEqual(a[4], b[4], "live_page_count differs")

    def test_unified_ps1_inplace_equals_free_swa(self):
        def run(inplace):
            allocator = self._build_composite(page_size=1, lazy=True)
            v = allocator.alloc(8)
            self.assertIsNotNone(v)
            seg = v[2:6]
            if inplace:
                allocator.free_swa_segment_inplace(seg, start_pos=2)
            else:
                allocator.free_swa(seg)
            return self._swa_state(allocator)

        self._assert_swa_state_equal(run(False), run(True))

    def test_unified_eager_compaction_falls_back(self):
        def run(inplace):
            allocator = self._build_composite(page_size=1, lazy=False)
            v = allocator.alloc(8)
            self.assertIsNotNone(v)
            seg = v[0:4]
            if inplace:
                allocator.free_swa_segment_inplace(seg, start_pos=0)
            else:
                allocator.free_swa(seg)
            sa = allocator.swa_attn_allocator
            return (
                sa.virtual_to_physical.clone(),
                sa.physical_to_virtual.clone(),
                sa.allocated_count(),
            )

        a, b = run(False), run(True)
        for ta, tb in zip(a[:2], b[:2]):
            self.assertTrue(torch.equal(ta, tb))
        self.assertEqual(a[2], b[2])

    def test_unified_paged_inplace_equals_free_swa(self):
        def run(inplace):
            allocator = self._build_composite(
                page_size=4, lazy=True, n_full_pages=16, n_swa_pages=16
            )
            v = allocator.alloc(16)
            self.assertIsNotNone(v)
            seg = v[4:12]
            if inplace:
                allocator.free_swa_segment_inplace(seg, start_pos=4)
            else:
                allocator.free_swa(seg)
            return self._swa_state(allocator)

        self._assert_swa_state_equal(run(False), run(True))


if __name__ == "__main__":
    unittest.main()
