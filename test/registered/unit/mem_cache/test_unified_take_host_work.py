"""A take of free pages does no host work in proportion to the holes left.

After a flush has read a band's holes on the host, a run of one-page takes
must not re-sort or copy that host list on every take: draining H holes
would cost time quadratic in H on the scheduler thread.
"""

import gc
import time
import unittest
import weakref
from unittest import mock

from sglang.srt.mem_cache.kv.alloc import band as band_module
from sglang.srt.mem_cache.kv.alloc import host_free_list as host_free_list_module
from sglang.srt.mem_cache.kv.alloc.signals import StreamTransfer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=20, stage="weekly", runner_config="cpu")


def _chain(tokens):
    setup_allocator_context()
    return build_unified_bundle(
        UnifiedGeometry(
            chain="tri",
            full_layer_ids=(0,),
            swa_layer_ids=(1,),
            state_layer_ids=(2,),
            full_tokens=tokens,
            swa_tokens=tokens,
            context_len=tokens,
        )
    ).token_to_kv_pool_allocator


def _holes_in_end(holes):
    """The owner end with ``holes`` interior holes, its host view read."""
    owner = _chain(2 * holes + 64).full_attn_allocator
    ids = owner.alloc(2 * holes)
    owner.free(ids[1::2])
    owner.flush_opportunistic()
    return owner


def _holes_in_float(holes):
    """The window float with ``holes`` interior holes, its host view read."""
    flt = _chain(2 * holes + 64).swa_attn_allocator
    pages = flt.take_physical_pages(2 * holes)
    flt.free_physical(pages[1::2])
    flt.flush_opportunistic()
    return flt


def _drain(band, takes):
    start = time.perf_counter()
    for _ in range(takes):
        assert band.take_physical_pages(1) is not None
    return time.perf_counter() - start


class TestTakesDoNoHostWorkPerHole(CustomTestCase):
    def test_draining_scales_with_the_takes(self):
        for build in (_holes_in_end, _holes_in_float):
            with self.subTest(band=build.__name__):
                small = _drain(build(4096), 4096)
                large = _drain(build(16384), 16384)
                # Linear: four times the takes; quadratic would be sixteen.
                self.assertLess(large, 8 * small + 0.5)
                self.assertLess(large, 10.0)


class TestATinyTakeAmongManyCopyFences(CustomTestCase):
    """Fences of host copies still reading pages a move vacated order no
    choice of holes: a one-page take from a large fragmented free list takes
    the nearest hole without walking the fences once per hole."""

    def test_a_one_page_take_among_many_fences(self):
        owner = _holes_in_end(16384)
        epp = owner.entry_bytes_per_page
        holes = sorted(owner._free_pages_snapshot())
        end = owner._hole_end()
        # Copy fences on every other hole but the ones a take reaches first.
        far = holes[64:-64]
        for page in far[::16]:
            owner.usage.vacate(
                StreamTransfer(_Unfired()), [(page, page + 1)], page_bytes=epp
            )
        self.assertGreaterEqual(len(owner.usage.uses), 1000)
        start = time.perf_counter()
        taken = owner.take_physical_pages(1)
        elapsed = time.perf_counter() - start
        self.assertIsNotNone(taken)
        expected = holes[0] if end == "low" else holes[-1]
        self.assertEqual(taken.tolist(), [expected])
        self.assertLess(elapsed, 1.0)


class _Unfired:
    def query(self):
        return False


class _Fired:
    def query(self):
        return True


class _CountingHostFreeList(host_free_list_module.HostFreeList):
    """Every host copy made while counting, so their work can be summed."""

    __slots__ = ()
    made = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _CountingHostFreeList.made.append(self)


class TestClearAndParkDropTheHostCopy(CustomTestCase):
    """Clearing a band, or parking a float, empties its free list and must
    drop the old list's host copy with it, not keep it alive."""

    def _read_view(self, band):
        band._free_pages_snapshot()
        self.assertTrue(band._free_pages_host_known())
        return weakref.ref(band._free_phys_pages)

    def test_clear(self):
        owner = _holes_in_end(64)
        old = self._read_view(owner)
        owner.clear()
        gc.collect()
        self.assertIsNone(old())

    def test_park(self):
        flt = _chain(256).swa_attn_allocator
        pages = flt.take_physical_pages(8)
        flt.free_physical(pages[:4])
        old = self._read_view(flt)
        flt.free_physical(pages[4:])  # the last live pages: it parks
        self.assertTrue(flt._is_frontier_transparent())
        gc.collect()
        self.assertIsNone(old())


class TestHostFreeListStorage(CustomTestCase):
    """The host copy keeps what a list of its pages would hold, in a backing
    list within about twice the pages it holds, whichever end the takes
    consume (the high end too, which leaves its consumed tail behind)."""

    def test_takes_from_either_end_keep_the_backing_bounded(self):
        for end in ("low", "high"):
            with self.subTest(end=end):
                pages = list(range(2048))
                host = host_free_list_module.HostFreeList(list(pages), ascending=True)
                while len(host) > 1:
                    n = min(7, len(host) - 1)
                    taken = host.peek(n, end)
                    expected = pages[:n] if end == "low" else pages[-n:][::-1]
                    self.assertEqual(taken, expected)
                    host.take(n, end)
                    pages = pages[n:] if end == "low" else pages[:-n]
                    self.assertEqual(host.to_list(), pages)
                    self.assertLessEqual(len(host._pages), 2 * len(host) + 8)


class TestSmallTakesUnderAFenceScaleWithTheTakes(CustomTestCase):
    """With a fence recorded in the chain, a one-page take touches only the
    pages it takes (and, choosing among hazarded holes, those it passes), so
    draining holes one at a time stays linear in them; a host copy of the free
    list is kept only for a fence that can still order a take here."""

    TAKES = 256

    def _owner_with_a_fence(self, holes, signal):
        owner = _holes_in_end(holes)
        epp = owner.entry_bytes_per_page
        free = sorted(owner._free_pages_snapshot())
        # A fence on a hole far from the end the takes draw from.
        page = free[len(free) // 2]
        owner.usage.vacate(signal, [(page, page + 1)], page_bytes=epp)
        return owner

    def _work(self, owner):
        """The host-list work, device reads of the free list and host sorts
        of more pages than a take takes, over ``TAKES`` one-page takes after a
        first one (which may sort the host copy the flush read, once)."""
        self.assertIsNotNone(owner.take_physical_pages(1))
        _CountingHostFreeList.made = []
        if owner._host_free is not None:
            # The copy the flush read, made before counting started.
            owner._host_free.__class__ = _CountingHostFreeList
            owner._host_free.touched = 0
            _CountingHostFreeList.made.append(owner._host_free)
        reads, sorts = [], []
        real_snapshot = band_module.Band._free_pages_snapshot
        real_sorted = sorted
        with (
            mock.patch.object(band_module, "HostFreeList", _CountingHostFreeList),
            mock.patch.object(
                band_module.Band,
                "_free_pages_snapshot",
                lambda band, *a, **k: reads.append(1) or real_snapshot(band, *a, **k),
            ),
            mock.patch.object(
                band_module,
                "sorted",
                lambda pages, *a, **k: (
                    (len(pages) > 8 and sorts.append(len(pages)))
                    or real_sorted(pages, *a, **k)
                ),
                create=True,
            ),
            mock.patch.object(
                host_free_list_module,
                "sorted",
                lambda pages, *a, **k: (
                    sorts.append(len(pages)) or real_sorted(pages, *a, **k)
                ),
                create=True,
            ),
        ):
            for _ in range(self.TAKES):
                self.assertIsNotNone(owner.take_physical_pages(1))
        touched = sum(host.touched for host in _CountingHostFreeList.made)
        return touched, len(reads), len(sorts)

    def test_an_outstanding_copy_fence(self):
        work = {}
        for holes in (4096, 16384):
            owner = self._owner_with_a_fence(holes, StreamTransfer(_Unfired()))
            self.assertTrue(owner._fence_may_order_takes())
            work[holes] = self._work(owner)
            # The fence can still order a take: the host copy is kept.
            self.assertTrue(owner._free_pages_host_known())
        # The same work whatever the holes left, a few pages per take.
        self.assertEqual(work[4096], work[16384])
        touched, reads, sorts = work[16384]
        self.assertLessEqual(touched, 4 * self.TAKES)
        self.assertEqual((reads, sorts), (0, 0))

    def test_an_outstanding_forward_fence_chooses_without_a_full_walk(self):
        """A fence of an unretired forward orders the choice (quiet holes
        first): the walk from the end stops at the pages it takes."""
        work = {}
        for holes in (4096, 16384):
            event = object()
            owner = _holes_in_end(holes)
            owner.usage.register_forward(event)
            epp = owner.entry_bytes_per_page
            free = sorted(owner._free_pages_snapshot())
            page = free[len(free) // 2]
            owner.usage.vacate(event, [(page, page + 1)], page_bytes=epp)
            work[holes] = self._work(owner)
        self.assertEqual(work[4096], work[16384])
        touched, reads, sorts = work[16384]
        self.assertLessEqual(touched, 4 * self.TAKES)
        self.assertEqual((reads, sorts), (0, 0))

    def test_a_completed_fence_keeps_no_host_copy(self):
        owner = self._owner_with_a_fence(16384, StreamTransfer(_Fired()))
        self.assertTrue(owner.usage.has_fences())  # still recorded
        self.assertFalse(owner._fence_may_order_takes())
        touched, reads, sorts = self._work(owner)
        self.assertEqual((reads, sorts), (0, 0))
        self.assertLessEqual(touched, 4 * self.TAKES)
        self.assertFalse(owner._free_pages_host_known())

    def test_a_fence_out_of_reach_keeps_no_host_copy(self):
        """A fence on bytes the band cannot span before a move (the state
        end's, at the far end of the chain) orders nothing here."""
        allocator = _chain(2 * 4096 + 64)
        owner = allocator.full_attn_allocator
        ids = owner.alloc(2 * 4096)
        owner.free(ids[1::2])
        owner.flush_opportunistic()
        self.assertIsNotNone(allocator.alloc_state_slots(1))
        state = next(
            band
            for band in owner._chain_bands()
            if band is not owner and band.grow_direction != "float"
        )
        lo, hi = owner._reach_bytes()
        epp = state.entry_bytes_per_page
        first, end = state._span_range()
        self.assertLess(first, end)
        page = first
        self.assertFalse(lo < (page + 1) * epp and page * epp < hi)
        state.usage.vacate(
            StreamTransfer(_Unfired()), [(page, page + 1)], page_bytes=epp
        )
        self.assertTrue(state.usage.has_fences())
        self.assertFalse(owner._fence_may_order_takes())
        self._work(owner)
        self.assertFalse(owner._free_pages_host_known())


if __name__ == "__main__":
    unittest.main()
