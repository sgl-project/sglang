"""A take of free pages does no host work in proportion to the holes left.

After a flush has read a band's holes on the host, a run of one-page takes
must not re-sort or copy that host list on every take: draining H holes
would cost time quadratic in H on the scheduler thread.
"""

import gc
import time
import unittest
import weakref

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


class TestClearAndParkDropTheHostCopy(CustomTestCase):
    """Clearing a band, or parking a float, empties its free list and must
    drop the old list's host copy with it, not keep it alive."""

    def _read_view(self, band):
        band._free_pages_host_view()
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


if __name__ == "__main__":
    unittest.main()
