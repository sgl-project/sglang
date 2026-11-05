"""When a physical page of the unified pool may be written, and by whom.

A page's hazards (unfinished work that may still touch it) are recorded when
it is freed or vacated; whoever writes it next waits only what its own
ordering does not already imply (`page_usage.CLEARANCE`). Every path that hands
out a page goes through `Band.acquire`, every path that returns one through
`Band.release`, and a float's on-demand moves (`make_room`, `compact_holes`)
clear their destinations the same way. Events here are host-controlled
stand-ins, and stream waits are recorded instead of issued, so the tests run on
the CPU.
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.alloc import band as band_module
from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.page_usage import (
    CLEARANCE,
    Consumer,
    Hazard,
    PageUsage,
    Wait,
)
from sglang.srt.mem_cache.kv.alloc.signals import signals_of
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    FakeKVCache,
    FakeUnifiedSWAKVPool,
    arena_of,
    setup_allocator_context,
    stand_in_views,
    tri_sub_pool_specs,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_ROLES = ChainRoles(owner="full", window="swa", state="mamba")

C, H, W = Consumer, Hazard, Wait

# The rule, written out: what each consumer waits for each hazard.
_EXPECTED = {
    C.FORWARD: {
        H.FORWARD_WRITE: W.NONE,
        H.FORWARD_READ: W.NONE,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
    C.SCHEDULE_COPY: {
        H.FORWARD_WRITE: W.STREAM,
        H.FORWARD_READ: W.STREAM,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
    C.ZERO: {
        H.FORWARD_WRITE: W.NONE,
        H.FORWARD_READ: W.STREAM,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
    C.HOST_LOAD: {
        H.FORWARD_WRITE: W.NONE,
        H.FORWARD_READ: W.NONE,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
    C.NETWORK_RECEIVE: {
        H.FORWARD_WRITE: W.AFTER_RETIRE,
        H.FORWARD_READ: W.AFTER_RETIRE,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
}


class _Event:
    """A CUDA event stand-in whose completion the test controls."""

    def __init__(self):
        self.fired = False

    def query(self):
        return self.fired


class _Waits:
    """Records the signals a band waits, in place of the stream waits."""

    def __init__(self):
        self.keys = []

    def __call__(self, key, stream=None):
        for signal in signals_of(key):
            if not signal.query():
                self.keys.append(signal)


# Hazards a fence records here; a network transfer is recorded by PD's
# transfers, and a fixed page is not a fence.
_FENCED = (H.FORWARD_WRITE, H.FORWARD_READ, H.COPY_READ, H.COPY_WRITE)


def _fence(usage: PageUsage, hazard: Hazard):
    """One fence of ``hazard`` over bytes [0, 8) of ``usage``; its key."""
    key = _Event()
    if hazard in (H.FORWARD_WRITE, H.FORWARD_READ):
        usage.register_forward(key)
    writes = hazard in (H.FORWARD_WRITE, H.COPY_WRITE)
    usage.vacate(key, [(0, 1)], page_bytes=8, writes=writes)
    return key


def _chain(owner_grows="down"):
    """A full / window float / state chain over 48 + 32 + 8 entries. Call it
    inside any runtime override the chain must see."""
    full, swa, mamba = tri_sub_pool_specs(head_num=1, head_dim=8)
    specs = [full, swa, mamba]
    if owner_grows == "up":
        full = dataclasses.replace(full, grow_direction="up")
        mamba = dataclasses.replace(mamba, grow_direction="down")
        specs = [mamba, swa, full]
    pool = arena_of(
        total_bytes=48 * full.entry_bytes()
        + 32 * swa.entry_bytes()
        + 8 * mamba.entry_bytes(),
        sub_pool_specs=specs,
        device="cpu",
    )
    kvcache = FakeUnifiedSWAKVPool(pool)
    state = FakeKVCache(pool, "mamba")
    return ChainAllocator(
        roles=_ROLES,
        arena=pool,
        kvcache=kvcache,
        views=stand_in_views(_ROLES, kvcache, state),
        device="cpu",
        full_max_total_num_tokens=48,
        swa_max_total_num_tokens=32,
        need_sort=False,
    )


class _Bands(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.waits = _Waits()
        patcher = mock.patch.object(band_module, "wait_signals", self.waits)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _overlapped_free(self, alloc, band, virtual, page_ids):
        """``band`` frees ``virtual`` while a forward that writes it runs (an
        overlapped batch); returns the forward and the freed physical page."""
        forward = _Event()
        page = int(band.ids.virtual_to_physical[page_ids][0])
        alloc.set_latest_forward_done_event(forward)
        band.set_inflight_forward(forward, virtual)
        return forward, page


class TestClearanceTable(CustomTestCase):
    def test_the_rule_is_the_table(self):
        self.assertEqual(
            {c: dict(row) for c, row in CLEARANCE.items()},
            _EXPECTED,
        )

    def test_every_consumer_and_hazard(self):
        """Each pair, read through the ledger: what is waited where, and
        whether the fence stays for the next writer."""
        for consumer, row in _EXPECTED.items():
            for hazard, wait in row.items():
                if hazard not in _FENCED:
                    continue
                with self.subTest(consumer=consumer, hazard=hazard):
                    usage = PageUsage()
                    key = _fence(usage, hazard)
                    clearance = usage.take_clearance(0, 8, (consumer,))
                    observed = {
                        W.STREAM: clearance.stream,
                        W.HOST: clearance.host,
                        W.AFTER_RETIRE: clearance.after,
                    }
                    for kind, keys in observed.items():
                        self.assertEqual(keys, [key] if kind is wait else [])
                    self.assertFalse(clearance.excluded)
                    kept = usage.has_fences()
                    self.assertEqual(kept, wait in (W.NONE, W.AFTER_RETIRE))
                    # Bytes the fence does not meet need nothing.
                    self.assertTrue(usage.take_clearance(8, 16, (consumer,)).quiet)

    def test_a_retired_forward_needs_no_wait_before_a_receive(self):
        usage = PageUsage()
        key = _fence(usage, H.FORWARD_WRITE)
        usage.retire_forward(key)
        clearance = usage.take_clearance(0, 8, (C.NETWORK_RECEIVE,))
        self.assertEqual(clearance.after, [])

    def test_fixed_pages_are_excluded_for_every_consumer(self):
        usage = PageUsage()
        usage.fix("test", [3])
        for consumer in C:
            with self.subTest(consumer=consumer):
                self.assertTrue(
                    usage.clearance([(3, 4)], (consumer,), page_bytes=8).excluded
                )
                self.assertTrue(
                    usage.clearance([(4, 5)], (consumer,), page_bytes=8).quiet
                )

    def test_the_pure_read_matches_the_consuming_one(self):
        for consumer, row in _EXPECTED.items():
            for hazard in row:
                if hazard not in _FENCED:
                    continue
                usage = PageUsage()
                _fence(usage, hazard)
                read = usage.clearance([(0, 1)], (consumer,), page_bytes=8)
                taken = usage.take_clearance(0, 8, (consumer,))
                self.assertEqual(
                    (read.stream, read.host, read.after),
                    (taken.stream, taken.host, taken.after),
                )


class TestHoleTakesUnderOverlap(_Bands):
    """BUG REGRESSION. A hole take checked nothing: a page freed while an
    overlapped forward still wrote it went to whoever asked, so a write on
    the schedule stream raced that forward's late write."""

    def _recent_hole(self):
        alloc = _chain()
        full = alloc.full_attn_allocator
        ids = alloc.alloc(12)
        slot = ids[3:4]
        forward, page = self._overlapped_free(alloc, full, slot, slot)
        alloc.free(slot)
        self.assertEqual(full._free_pages_snapshot(), [page])
        return alloc, full, forward, page

    def test_each_consumer_of_a_hole_a_running_forward_writes(self):
        waits_forward = {
            C.FORWARD: False,
            C.ZERO: False,
            C.HOST_LOAD: False,
            C.SCHEDULE_COPY: True,
            C.NETWORK_RECEIVE: True,
        }
        for consumer, waits in waits_forward.items():
            with self.subTest(consumer=consumer):
                self.waits.keys.clear()
                _, full, forward, page = self._recent_hole()
                pages = full.acquire(1, consumer)
                self.assertEqual(pages.tolist(), [page])
                self.assertEqual(self.waits.keys, [forward] if waits else [])

    def test_a_copy_prefers_a_quiet_hole(self):
        alloc = _chain()
        full = alloc.full_attn_allocator
        ids = alloc.alloc(12)
        quiet = ids[8:9]
        quiet_page = int(full.ids.virtual_to_physical[quiet][0])
        alloc.free(quiet)  # no forward runs: a quiet hole
        slot = ids[3:4]
        forward, _ = self._overlapped_free(alloc, full, slot, slot)
        alloc.free(slot)
        with mock.patch.object(
            full, "_free_pages_snapshot", side_effect=AssertionError("host read")
        ):
            pages = full.acquire(1, C.SCHEDULE_COPY)
        self.assertEqual(pages.tolist(), [quiet_page])
        self.assertEqual(self.waits.keys, [])
        # The hazarded hole is next, and is waited for.
        full.acquire(1, C.SCHEDULE_COPY)
        self.assertEqual(self.waits.keys, [forward])

    def test_a_state_slot_freed_under_a_running_forward(self):
        """A new state slot may be written on the schedule stream (a state
        copy) before any forward runs: it waits the forward that still writes
        the slot it reuses."""
        alloc = _chain()
        state = alloc.mamba_allocator
        slots = alloc.alloc_state_slots(2)
        slot = slots[:1]
        forward, _ = self._overlapped_free(alloc, state, slot, slot)
        state.free(slot)
        self.assertIsNotNone(alloc.alloc_state_slots(1))
        self.assertEqual(self.waits.keys, [forward])

    def test_a_retired_forward_leaves_the_hole_quiet(self):
        alloc, full, forward, page = self._recent_hole()
        alloc.retire_forward(forward)
        self.assertEqual(full.acquire(1, C.SCHEDULE_COPY).tolist(), [page])
        self.assertEqual(self.waits.keys, [])


class TestFreedSlotCompactedInto(_Bands):
    def test_a_compaction_into_a_slot_a_running_forward_writes_waits_it(self):
        """BUG REGRESSION. An allocation's compaction copied a live page into
        a hole whose slot an overlapped forward still wrote."""
        alloc = _chain()
        full = alloc.full_attn_allocator
        ids = alloc.alloc(12)
        phys = full.ids.virtual_to_physical[ids].tolist()
        # The page farthest from the frontier: once a hole, a copy fills it.
        far = phys.index(max(phys))
        slot = ids[far : far + 1]
        forward, page = self._overlapped_free(alloc, full, slot, slot)
        alloc.free(slot)
        moves = full.flush_for_allocation()
        self.assertGreater(moves, 0)
        self.assertIn(forward, self.waits.keys)
        self.assertNotIn(page, full._free_pages_snapshot())


class TestWidenedPagesUnderDcp(_Bands):
    """BUG REGRESSION. Under decode context parallelism the alloc surface
    hands out widened ids, `dcp_size` per physical row. A forward's write set
    divided them by the DCP width twice, so a page freed while that forward
    still wrote it was not seen as written, and compaction copied a live page
    into it without waiting the forward."""

    def test_a_compaction_into_a_page_a_running_forward_writes_waits_it(self):
        for owner_grows in ("down", "up"):
            with (
                self.subTest(owner_grows=owner_grows),
                get_parallel().override(attn_dcp_size=2),
            ):
                self.waits.keys.clear()
                alloc = _chain(owner_grows)
                full = alloc.full_attn_allocator
                self.assertEqual(full.grow_direction, owner_grows)
                self.assertEqual((full.page_size, full.pool_page_size), (2, 1))
                ids = full.alloc(24)
                pages = ids[::2] // full.page_size
                phys = full.ids.virtual_to_physical[pages].tolist()
                # The page farthest from the frontier: once a hole, a copy
                # fills it.
                far = phys.index(max(phys) if owner_grows == "down" else min(phys))
                # Page 0 is the only one the double division leaves in place.
                self.assertGreater(int(pages[far]), 0)
                slot = ids[2 * far : 2 * far + 2]
                forward, page = self._overlapped_free(
                    alloc, full, slot, pages[far : far + 1]
                )
                full.free(slot)
                self.assertTrue(full.usage.has_recent_holes())
                moves = full.flush_for_allocation()
                self.assertGreater(moves, 0)
                self.assertIn(forward, self.waits.keys)
                self.assertNotIn(page, full._free_pages_snapshot())

    def test_frees_and_write_sets_name_the_same_pages(self):
        with get_parallel().override(attn_dcp_size=2):
            alloc = _chain()
            full = alloc.full_attn_allocator
            ids = alloc.alloc(8)
            forward = _Event()
            alloc.set_latest_forward_done_event(forward)
            full.set_inflight_forward(forward, ids)
            # The pages `free` would record for these ids.
            self.assertEqual(
                full._forward_written_pages(),
                set((ids // full.page_size).tolist()),
            )
            self.assertEqual(len(full._forward_written_pages()), 4)


class TestFloatMoves(_Bands):
    """A window float moves its pages only on demand (`make_room`,
    `compact_holes`). The copy writes holes and gap pages, which unfinished
    work may still touch, and leaves pages a forward may still read."""

    def _float_with_a_recent_hole(self, rank=1):
        """A float whose ``rank``-th lowest page was freed while a forward
        that writes it runs: a hole inside the span, below the sources a
        high-side ask moves."""
        alloc = _chain()
        flt = alloc.swa_attn_allocator
        ids = alloc.alloc(8)
        pages = flt.ids.virtual_to_physical[ids].tolist()
        low = sorted(pages)[rank]
        slot = ids[pages.index(low) : pages.index(low) + 1]
        forward, page = self._overlapped_free(alloc, flt, slot, slot)
        alloc.free_swa(slot)
        return alloc, flt, forward, page

    def test_a_copy_into_a_hole_a_running_forward_writes_waits_it(self):
        alloc, flt, forward, page = self._float_with_a_recent_hole()
        _, gap_high = flt._gap_pages()
        epp = flt.entry_bytes_per_page
        moved_into = []
        real = flt._move_pages_and_rebind

        def record(src, dst):
            moved_into.extend(dst.tolist())
            return real(src, dst)

        with mock.patch.object(flt, "_move_pages_and_rebind", side_effect=record):
            opened = flt.make_room(side="high", min_bytes=(gap_high + 1) * epp)
        self.assertGreaterEqual(opened, (gap_high + 1) * epp)
        self.assertIn(page, moved_into, "the geometry copied into no hole")
        self.assertEqual(self.waits.keys, [forward])
        self.assertFalse(alloc.verify_byte_accounting())

    def test_a_move_waits_no_forward_that_only_reads_its_sources(self):
        """The copy waits only a forward that writes a page it moves; the
        pages it vacates are fenced, and a later copy into them waits."""
        alloc = _chain()
        flt = alloc.swa_attn_allocator
        alloc.alloc(8)
        reader = _Event()
        alloc.set_latest_forward_done_event(reader)
        low, high = flt.low_wm_page, flt.high_wm_page
        _, gap_high = flt._gap_pages()
        epp = flt.entry_bytes_per_page
        flt.make_room(side="high", min_bytes=(gap_high + 1) * epp)
        self.assertEqual(self.waits.keys, [])
        self.assertLess(flt.high_wm_page, high)
        vacated = range(max(flt.high_wm_page, low), high)
        self.assertTrue(vacated, "the geometry vacated no page")
        flt._wait_fences(vacated[0], vacated[-1] + 1)
        self.assertEqual(self.waits.keys, [reader])

    def test_an_interior_shift_waits_every_running_forward(self):
        """Packing that copies into sources vacated earlier in the same pass
        orders every copy after the running forwards."""
        # The two pages below the hole shift up by one: the lower one is
        # copied into the page the upper one just left.
        alloc, flt, forward, _ = self._float_with_a_recent_hole(rank=2)
        self.assertEqual(flt.compact_holes(retreat_side="low"), 2)
        self.assertIn(forward, self.waits.keys)
        self.assertEqual(flt._hole_pages(), 0)
        self.assertFalse(alloc.verify_byte_accounting())


class TestForwardTakes(_Bands):
    def test_a_forward_take_that_needs_no_wait_keeps_the_fence(self):
        """A take whose writer is ordered after the fence's forward does not
        wait, and leaves the fence for the next writer of those bytes."""
        alloc = _chain()
        state = alloc.mamba_allocator
        forward = _Event()
        alloc.set_latest_forward_done_event(forward)
        first, _ = state._span_range()
        epp = state.entry_bytes_per_page
        state.usage.vacate(forward, [(first, first + 1)], page_bytes=epp)
        clearance = state._range_clearance(first, first + 1, (C.FORWARD,))
        self.assertTrue(clearance.quiet)
        self.assertTrue(
            state.usage.fences_need(first * epp, (first + 1) * epp, (C.ZERO,))
        )


class TestFloatHandBack(_Bands):
    def test_a_recent_hole_absorbed_into_the_gap_is_fenced(self):
        """BUG REGRESSION. A float handed recent holes on its boundary back to
        the gap unresolved, where a neighbour's write saw no hazard."""
        alloc = _chain()
        flt = alloc.swa_attn_allocator
        ids = alloc.alloc(8)
        pages = flt.ids.virtual_to_physical[ids].tolist()
        edge = pages.index(min(pages))
        slot = ids[edge : edge + 1]
        forward, page = self._overlapped_free(alloc, flt, slot, slot)
        alloc.free_swa(slot)
        self.assertEqual(flt.low_wm_page, page)
        self.assertGreater(flt.flush_opportunistic(), 0)
        self.assertGreater(flt.low_wm_page, page)
        epp = flt.entry_bytes_per_page
        lo, hi = page * epp, (page + 1) * epp
        self.assertTrue(flt.usage.fences_need(lo, hi, (C.SCHEDULE_COPY,)))
        self.assertFalse(flt.usage.fences_need(lo, hi, (C.FORWARD, C.ZERO)))

    def test_a_parked_float_fences_its_recent_holes(self):
        alloc = _chain()
        flt = alloc.swa_attn_allocator
        ids = alloc.alloc(4)
        low, high = flt.low_wm_page, flt.high_wm_page
        forward, _ = self._overlapped_free(alloc, flt, ids, ids[:1])
        alloc.free_swa(ids)
        self.assertTrue(flt._is_frontier_transparent())
        epp = flt.entry_bytes_per_page
        self.assertTrue(
            flt.usage.fences_need(low * epp, high * epp, (C.SCHEDULE_COPY,))
        )
        self.assertFalse(flt.usage.has_recent_holes())


class TestEndBandsCompactLazily(CustomTestCase):
    def test_an_eager_chain_is_refused(self):
        with self.assertRaises(AssertionError):
            setup_allocator_context()
            full, swa, mamba = tri_sub_pool_specs(head_num=1, head_dim=8)
            pool = arena_of(
                total_bytes=48 * full.entry_bytes()
                + 32 * swa.entry_bytes()
                + 8 * mamba.entry_bytes(),
                sub_pool_specs=[full, swa, mamba],
                device="cpu",
            )
            kvcache = FakeUnifiedSWAKVPool(pool)
            ChainAllocator(
                roles=_ROLES,
                arena=pool,
                kvcache=kvcache,
                views=stand_in_views(_ROLES, kvcache, FakeKVCache(pool, "mamba")),
                device="cpu",
                lazy_compaction=False,
            )

    def test_startup_refuses_eager_compaction(self):
        """BUG REGRESSION. Eager compaction ignored fixed pages, so with
        HiCache it moved a page reserved for a host load. Unified memory now
        refuses the eager mode at startup, with or without HiCache."""
        import msgspec

        from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.server_args import ServerArgs

        def run(hicache):
            sa = ServerArgs(model_path="dummy")
            for name, value in {
                "enable_unified_memory": True,
                "enable_two_batch_overlap": False,
                "disaggregation_mode": "null",
                "speculative_algorithm": None,
                "speculative_eagle_topk": None,
                "enable_hierarchical_cache": hicache,
                "enable_lmcache": False,
                "dcp_size": 1,
                "cuda_graph_config": SimpleNamespace(
                    prefill=SimpleNamespace(backend=Backend.DISABLED),
                    decode=SimpleNamespace(backend=Backend.FULL),
                ),
                "cuda_graph_backend_prefill": Backend.DISABLED,
            }.items():
                msgspec.Struct.__setattr__(sa, name, value)
            handle_unified_memory_pool(sa)

        with envs.SGLANG_DISABLE_LAZY_COMPACTION.override(True):
            for hicache in (False, True):
                with self.subTest(hicache=hicache):
                    with self.assertRaises(AssertionError) as ctx:
                        run(hicache)
                    self.assertIn("lazy compaction", str(ctx.exception))
        run(False)


if __name__ == "__main__":
    unittest.main()
