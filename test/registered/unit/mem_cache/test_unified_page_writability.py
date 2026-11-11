"""When a physical page of the unified pool may be written, and by whom.

A page's hazards (unfinished work that may still touch it) are recorded when a
move vacates it or a transfer touches it; its next writer waits only what its
own ordering does not already imply (`page_usage.CLEARANCE`). A freed page is
quiet: it returns only once no unprocessed forward includes it. Events are
host-controlled stand-ins and stream waits are recorded instead of issued, so
these run on the CPU.
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.mem_cache.kv.alloc import band as band_module
from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.page_usage import (
    CLEARANCE,
    Consumer,
    Hazard,
    PageUsage,
    Wait,
)
from sglang.srt.mem_cache.kv.alloc.pricing import PLAN_STEPS, feasible, plan
from sglang.srt.mem_cache.kv.alloc.signals import (
    NetworkTransfer,
    StreamTransfer,
    signals_of,
)
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
        H.FORWARD: W.NONE,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
    C.SCHEDULE_COPY: {
        H.FORWARD: W.STREAM,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
    C.ZERO: {
        H.FORWARD: W.STREAM,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
    C.HOST_LOAD: {
        H.FORWARD: W.NONE,
        H.COPY_READ: W.STREAM,
        H.COPY_WRITE: W.STREAM,
        H.NETWORK: W.HOST,
        H.FIXED: W.EXCLUDED,
    },
    C.NETWORK_RECEIVE: {
        H.FORWARD: W.AFTER_RETIRE,
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

    def __call__(self, key, stream=None, *, network_waiter=None):
        for signal in signals_of(key):
            if isinstance(signal, NetworkTransfer):
                if not signal.released:
                    self.keys.append(signal)
                    signal.released = True
            elif not signal.query():
                self.keys.append(signal)


def _fence(usage: PageUsage, hazard: Hazard):
    """One fence of ``hazard`` over bytes [0, 8) of ``usage``; its key."""
    if hazard is H.FORWARD:
        key = _Event()
        usage.register_forward(key)
    elif hazard is H.NETWORK:
        key = NetworkTransfer()
    else:
        key = StreamTransfer(_Event())
    writes = hazard is H.COPY_WRITE
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

    def _running_forward(self, alloc, band, written):
        """A forward that writes ``written`` of ``band`` runs; returns it."""
        forward = _Event()
        alloc.set_latest_forward_done_event(forward)
        band.set_inflight_forward(forward, written)
        return forward


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
                if hazard is H.FIXED:
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
        key = _fence(usage, H.FORWARD)
        usage.retire_forward(key)
        clearance = usage.take_clearance(0, 8, (C.NETWORK_RECEIVE,))
        self.assertEqual(clearance.after, [])

    def test_a_transfer_key_is_never_a_forward(self):
        """A stream copy's key is not in the forward map, which reads a
        forgotten forward as retired: the forward question refuses it, and
        a release key holding one asks each member its own question."""
        usage = PageUsage()
        usage.record_transfer(
            _Event(), [0], writes=False, release=None, unit_tokens=None
        )
        (copy,) = usage.transfers()
        forward, network = _Event(), NetworkTransfer()
        usage.register_forward(forward)
        for key in (copy.signal, network, (forward, copy.signal)):
            with self.subTest(key=key), self.assertRaises(TypeError):
                usage.forward_retired(key)
        self.assertTrue(usage.ended(copy.signal))
        self.assertFalse(usage.ended((forward, copy.signal)))
        self.assertFalse(usage.ended(network))
        usage.retire_forward(forward)
        network.released = True
        self.assertTrue(usage.ended((forward, copy.signal, network)))
        with self.assertRaises(TypeError):
            usage.transfer_ended(forward)

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
                if hazard is H.FIXED:
                    continue
                usage = PageUsage()
                _fence(usage, hazard)
                read = usage.clearance([(0, 1)], (consumer,), page_bytes=8)
                taken = usage.take_clearance(0, 8, (consumer,))
                self.assertEqual(
                    (read.stream, read.host, read.after),
                    (taken.stream, taken.host, taken.after),
                )


class TestFencesSplitByReaders(CustomTestCase):
    def test_a_wait_for_part_of_a_fence_keeps_the_rest(self):
        """A wait for part of a fence keeps the rest fenced alike on every
        rank, whether a host copy still reading one byte split the fence."""
        guarded = []
        for split in (False, True):
            usage = PageUsage()
            forward = _Event()
            usage.register_forward(forward)
            usage.latest_forward = forward
            if split:
                usage.vacate(forward, [(0, 5)], page_bytes=1)
                copy = StreamTransfer(_Event())
                usage.vacate((forward, copy), [(5, 6)], page_bytes=1)
            else:
                usage.vacate(forward, [(0, 6)], page_bytes=1)
            usage.take_clearance(0, 2, (Consumer.SCHEDULE_COPY,))
            guarded.append(
                [usage.fences_need(b, b + 1, (Consumer.ZERO,)) for b in range(6)]
            )
        self.assertEqual(guarded[0], guarded[1])
        self.assertEqual(guarded[0], [False, False, True, True, True, True])


class TestFreedPagesAreQuiet(_Bands):
    """A page freed while a recorded forward runs stays pending until that
    forward is processed, so it reaches the holes quiet: no consumer waits for
    it. Freeing a page a host load has reserved trips the invariant check."""

    def test_a_freed_page_is_a_quiet_hole_for_every_consumer(self):
        for consumer in C:
            with self.subTest(consumer=consumer):
                self.waits.keys.clear()
                alloc = _chain()
                full = alloc.full_attn_allocator
                ids = alloc.alloc(12)
                self._running_forward(alloc, full, ids[11:12])
                slot = ids[3:4]
                page = int(full.ids.virtual_to_physical[slot][0])
                alloc.free(slot)
                self.assertEqual(full.acquire(1, consumer).tolist(), [page])
                self.assertEqual(self.waits.keys, [])

    def test_a_free_of_a_reserved_page_trips_the_check(self):
        alloc = _chain()
        full = alloc.full_attn_allocator
        ids = alloc.alloc(12)
        page = int(full.ids.virtual_to_physical[ids[3:4]][0])
        full.fix_pages(band_module._LOAD_RESERVATION, [page])
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            alloc.free(ids[4:5])  # another page: fine
            with self.assertRaises(RuntimeError):
                alloc.free(ids[3:4])


class TestHostLoadReservations(_Bands):
    """A host load reserves pages (`Band.alloc_physical`) and keeps them fixed
    until the tree binds them; the pages it takes are named exactly."""

    def _reserved(self, band):
        return sorted(band.usage.fixed.get(band_module._LOAD_RESERVATION, ()))

    def test_a_reservation_from_pending_pages_stays_fixed_through_relief(self):
        alloc = _chain()
        full, steps = alloc.full_attn_allocator, alloc.forward_steps
        ids = full.alloc(full.available_size())
        launch = steps.forward_launched()
        steps.forward_recorded(_Event())
        full.free(ids[:4])
        pending = sorted(torch.cat(full._pending_until_retire.pages()).tolist())
        reserved = full.alloc_physical(4)
        # Settled before the snapshots: the reservation names those pages.
        self.assertEqual(sorted(reserved.tolist()), pending)
        self.assertEqual(self._reserved(full), pending)
        self.assertFalse(full._pending_until_retire)
        full.flush_for_allocation()  # relief before the copy is submitted
        self.assertEqual(self._reserved(full), pending)
        self.assertEqual(sorted(reserved.tolist()), pending)
        full.cancel_physical_reservation(reserved)
        self.assertEqual(self._reserved(full), [])
        steps.forward_processed(launch)
        self.assertEqual(full._byte_accounting_violations(), [])

    def test_a_reservation_of_drained_holes_reads_ahead_of_any_wait(self):
        """Freed during a forward (device only), drained when it is processed
        (the host copy unknown), then reserved from holes: the pages are
        exact, and every host read precedes every wait."""
        alloc = _chain()
        full, steps = alloc.full_attn_allocator, alloc.forward_steps
        ids = full.alloc(full.available_size())
        launch = steps.forward_launched()
        steps.forward_recorded(_Event())
        freed = sorted(full.ids.virtual_to_physical[ids[:4]].tolist())
        full.free(ids[:4])
        steps.forward_processed(launch)
        self.assertIsNone(full._free_pages_host)
        events = []
        real_tolist = torch.Tensor.tolist
        with (
            mock.patch.object(
                band_module,
                "wait_signals",
                side_effect=lambda *a, **k: events.append("wait"),
            ),
            mock.patch.object(
                torch.Tensor,
                "tolist",
                lambda t: events.append("read") or real_tolist(t),
            ),
        ):
            reserved = full.alloc_physical(4)
        self.assertEqual(sorted(reserved.tolist()), freed)
        self.assertEqual(self._reserved(full), freed)
        if "wait" in events:
            self.assertNotIn("read", events[events.index("wait") :])

    def test_a_load_cancelled_after_submission_keeps_its_copy_hazard(self):
        """Submitted on the rows the tree bound, then cancelled while the copy
        still runs: the freed pages keep the copy's write, which their next
        writer waits."""
        alloc = _chain()
        full = alloc.full_attn_allocator
        reserved = full.alloc_physical(4)
        pages = sorted(reserved.tolist())
        rows = full.free_virtual_ids[:4].clone()
        full.free_virtual_ids = full.free_virtual_ids[4:]
        full.ids.bind(rows, reserved)
        full.end_load_reservation()
        copy = _Event()
        full.record_transfer(copy, rows, writes=True)
        full.free_physical(reserved)
        self.waits.keys.clear()
        taken = full.acquire(
            int(full._free_phys_pages.shape[0]), Consumer.SCHEDULE_COPY
        )
        self.assertTrue(set(pages) <= set(taken.tolist()))
        self.assertIn(copy, self.waits.keys)


class TestCompactionWaitsTheWritersOfItsSources(_Bands):
    def test_a_move_of_a_page_a_running_forward_writes_waits_it(self):
        """A copy reads its source once: a forward still writing the source
        would leave its write behind at the old place."""
        alloc = _chain()
        full = alloc.full_attn_allocator
        ids = alloc.alloc(12)
        phys = full.ids.virtual_to_physical[ids].tolist()
        # The page nearest the frontier moves into the hole the farthest
        # page leaves.
        near = phys.index(min(phys))
        far = phys.index(max(phys))
        alloc.free(ids[far : far + 1])
        forward = self._running_forward(alloc, full, ids[near : near + 1])
        self.assertGreater(full.flush_for_allocation(), 0)
        self.assertIn(forward, self.waits.keys)


class TestRelocationHoles(_Bands):
    def test_holes_a_relocation_creates_keep_their_fences(self):
        """Regression: holes a relocation leaves in its new block keep their fences
        until the take that writes one clears it."""
        alloc = _chain()
        flt = alloc.swa_attn_allocator
        ids = alloc.alloc(12)
        self.assertIsNotNone(ids)
        low, high = flt.low_wm_page, flt.high_wm_page
        gap_low, _ = flt._gap_pages()
        self.assertGreater(gap_low, 0, "the geometry left no low gap")
        reader = _Event()
        full = alloc.full_attn_allocator
        alloc.set_latest_forward_done_event(reader)
        # A neighbour's page there was vacated while the forward still reads it.
        full.usage.vacate(reader, [(low - 1, low)], page_bytes=flt.entry_bytes_per_page)
        flt._holes_dirty = False
        self.assertEqual(flt.relocate_into(low - 1, high), 0)
        self.assertEqual(self.waits.keys, [])
        self.assertTrue(flt._holes_dirty)
        self.assertIn(low - 1, flt._free_pages_snapshot())
        # A forward's take (zeroed on hand-out) must not overwrite the bytes.
        holes = len(flt._free_pages_snapshot())
        pages = flt.acquire(holes, C.FORWARD).tolist()
        self.assertIn(low - 1, pages)
        self.assertEqual(self.waits.keys, [reader])


class TestWidenedPagesUnderDcp(_Bands):
    """Under decode context parallelism the alloc surface hands out widened
    ids, `dcp_size` per physical row; a forward's write set must name the
    same physical pages, or compaction moves a page the forward still writes
    without waiting for it."""

    def test_a_move_of_a_page_a_running_forward_writes_waits_it(self):
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
                far = phys.index(max(phys) if owner_grows == "down" else min(phys))
                near = phys.index(min(phys) if owner_grows == "down" else max(phys))
                # Page 0 maps to itself even when the DCP width is divided
                # out twice, so it would hide the error.
                self.assertGreater(int(pages[near]), 0)
                full.free(ids[2 * far : 2 * far + 2])
                forward = self._running_forward(
                    alloc, full, ids[2 * near : 2 * near + 2]
                )
                self.assertGreater(full.flush_for_allocation(), 0)
                self.assertIn(forward, self.waits.keys)


class TestTakesReadNoEventCompletion(_Bands):
    """Ranks that differ only in local event completion bind the same
    physical pages and end with the same free lists and capacities."""

    def _run(self, copy_fired: bool):
        """One sequence of allocations and frees; only the host copy's event
        differs between runs. Returns everything a rank decides from."""
        alloc = _chain()
        full, swa = alloc.full_attn_allocator, alloc.swa_attn_allocator
        # A grow-down owner takes holes from the high end of its list.
        self.assertEqual(full._hole_end(), "high")
        ids = alloc.alloc(12)
        # A fence of a forward that ended and retired, and of a host copy,
        # on one live page of each band: what is left after a forward take
        # (it need not wait the forward) over a page a backup still read.
        forward, copy = _Event(), _Event()
        forward.fired, copy.fired = True, copy_fired
        alloc.set_latest_forward_done_event(forward)
        for band in (full, swa):
            page = int(band.ids.virtual_to_physical[ids[0:1]][0])
            band.usage.vacate(
                (forward, StreamTransfer(copy)),
                [(page, page + 1)],
                page_bytes=band.entry_bytes_per_page,
            )
        alloc.retire_forward(forward)
        seen = {}
        for slot in (2, 5, 8):
            alloc.free(ids[slot : slot + 1])
        part = alloc.alloc(2)  # two of three holes
        seen["part"] = [
            band.ids.virtual_to_physical[part].tolist() for band in (full, swa)
        ]
        alloc.free(ids[9:11])
        whole = alloc.alloc(3)  # every hole
        seen["whole"] = [
            band.ids.virtual_to_physical[whole].tolist() for band in (full, swa)
        ]
        # Two holes while a forward that writes neither runs.
        alloc.free(ids[3:4])
        running = _Event()
        alloc.set_latest_forward_done_event(running)
        full.set_inflight_forward(running, ids[11:12])
        alloc.free(ids[6:7])
        seen["copy"] = full.acquire(1, C.SCHEDULE_COPY).tolist()
        # One page of each take goes back: which physical page that is sets
        # the frontiers, the float's span and so its neighbours' room.
        alloc.free(part[:1])
        alloc.free(whole[:1])
        alloc.flush_opportunistic()
        seen["frontiers"] = (full.watermark_physical, swa.low_wm_page, swa.high_wm_page)
        seen["available"] = [member.available_size() for member in alloc.members]
        seen["holes"] = [sorted(b._free_pages_snapshot()) for b in (full, swa)]
        return seen

    def test_takes_do_not_depend_on_event_completion(self):
        fired, pending = self._run(True), self._run(False)
        self.assertEqual(fired, pending)
        # One order: from the take's end, nearest first.
        full_part, _ = fired["part"]
        self.assertEqual(full_part, sorted(full_part, reverse=True))
        full_whole, _ = fired["whole"]
        self.assertEqual(full_whole, sorted(full_whole, reverse=True))


class TestReservationRelease(_Bands):
    def test_a_cancelled_reservation_fences_the_transfer_on_it(self):
        """A reservation given back while a host copy still writes it: the
        free records the copy, and the next writer of the page waits it."""
        alloc = _chain()
        flt = alloc.swa_attn_allocator
        alloc.alloc(4)
        reserved = flt.alloc_physical(2)
        pages = sorted(set((reserved // flt.page_size).tolist()))
        copy = _Event()
        flt.record_transfer(
            copy, torch.tensor(pages), writes=True, unit_tokens=flt.page_size
        )
        flt.cancel_physical_reservation(reserved)
        self.assertEqual(self.waits.keys, [])
        clearance = flt._range_clearance(pages[0], pages[0] + 1, (C.FORWARD,))
        self.assertEqual([key.events for key in clearance.stream], [copy])


class TestStateSlotsFusedTake(_Bands):
    def test_frontier_slots_are_taken_and_bound_in_one_kernel(self):
        """BUG REGRESSION. State slots lost the fused take-and-bind path."""
        alloc = _chain()
        real = band_module.alloc_bind_inplace
        with mock.patch.object(
            band_module, "alloc_bind_inplace", side_effect=real
        ) as fused:
            slots = alloc.alloc_state_slots(2)
        self.assertEqual(fused.call_count, 1)
        state = alloc.mamba_allocator
        self.assertTrue(bool((state.ids.virtual_to_physical[slots] >= 0).all()))


class TestSendRowsMovedOutOfTheBand(_Bands):
    """Regression: after a relief or compaction moved a send's rows out of the
    band, later plans keep only in-band send pages and never extend into its bytes."""

    def _run(self, owner_grows, history):
        """Replay ``history`` on a fresh chain. ``("beyond", None)`` checks
        that an unreleased send then names a page outside the owner."""
        alloc = _chain(owner_grows)
        full = alloc.full_attn_allocator
        rows, slots, sends = [], [], []
        for op, arg in history:
            if op == "alloc":
                ids = alloc.alloc(arg)
                if ids is not None:
                    rows.append(ids)
            elif op == "free":
                ids = rows.pop(arg % len(rows))
                # A row is released only once the sends of it ended.
                for send, sent, _ in sends:
                    if sent is ids:
                        send.released = True
                alloc.free(ids)
            elif op == "send":
                # As the storage records a send: by physical rows.
                ids = rows[arg % len(rows)]
                pages = full.ids.virtual_to_physical[ids // full.page_size]
                send = NetworkTransfer()
                full.record_transfer(
                    send,
                    pages,
                    writes=False,
                    release=send,
                    unit_tokens=full.pool_page_size,
                )
                sends.append((send, ids, pages.tolist()))
            elif op == "state":
                slot = alloc.alloc_state_slots(1)
                if slot is not None:
                    slots.append(slot)
            elif op == "state_free":
                alloc.mamba_allocator.free(slots.pop(0))
            elif op == "relief":
                alloc.lay_out_for(arg)
            elif op == "compact":
                full.flush_for_allocation()
            elif op == "flush":
                alloc.flush_opportunistic()
            elif op == "beyond":
                first, end = full._span_range()
                self.assertTrue(
                    any(
                        not send.released and not all(first <= p < end for p in pages)
                        for send, _, pages in sends
                    ),
                    "no unreleased send names a page beyond the owner",
                )
        self.assertEqual(alloc.verify_byte_accounting(), [])
        return alloc

    # A relief that could not keep the sends in place moved one out.
    _AFTER_A_RELIEF = [
        ("alloc", 3),
        ("alloc", 10),
        ("alloc", 6),
        ("free", 1),
        ("alloc", 10),
        ("alloc", 3),
        ("alloc", 10),
        ("free", 1),
        ("alloc", 3),
        ("alloc", 2),
        ("send", 3),
        ("alloc", 10),
        ("free", 5),
        ("free", 4),
        ("send", 4),
        ("free", 2),
        ("free", 0),
        ("relief", {"mamba": 24}),
        ("beyond", None),
        ("alloc", 10),
    ]

    def test_the_next_take_lays_out_around_a_send_a_relief_moved(self):
        for owner_grows in ("down", "up"):
            with self.subTest(owner_grows=owner_grows):
                self._run(owner_grows, self._AFTER_A_RELIEF)

    def test_the_next_take_lays_out_around_a_send_a_compaction_moved(self):
        self._run(
            "up",
            [
                ("alloc", 1),
                ("alloc", 3),
                ("alloc", 6),
                ("state", None),
                ("alloc", 4),
                ("alloc", 10),
                ("alloc", 6),
                ("alloc", 3),
                ("alloc", 4),
                ("send", 7),
                ("alloc", 6),
                ("free", 6),
                ("state_free", None),
                ("state", None),
                ("free", 5),
                ("state", None),
                ("compact", None),
                ("beyond", None),
                ("state", None),
            ],
        )

    def test_a_flush_after_it_moves_only_live_pages(self):
        self._run(
            "down",
            [
                ("alloc", 1),
                ("alloc", 4),
                ("alloc", 10),
                ("alloc", 6),
                ("alloc", 6),
                ("alloc", 1),
                ("alloc", 10),
                ("alloc", 1),
                ("alloc", 1),
                ("alloc", 3),
                ("send", 8),
                ("state", None),
                ("free", 1),
                ("compact", None),
                ("beyond", None),
                ("free", 3),
                ("state", None),
                ("flush", None),
            ],
        )

    def test_the_bytes_a_send_reads_bound_every_plan_but_the_last_resort(self):
        history = self._AFTER_A_RELIEF[: self._AFTER_A_RELIEF.index(("beyond", None))]
        alloc = self._run("down", history)
        held = alloc.network_held_bytes()
        self.assertTrue(held, "the moved send's bytes are not held")
        snapshot = alloc.held_snapshot()
        self.assertEqual(snapshot.held, held)
        # Every band's pages stay inside its span: none of them is fixed
        # outside it for a plan to reach.
        for band in alloc.bands:
            first, end = band._span_range()
            self.assertTrue(all(first <= p < end for p in band.network_rows()))
        for name in ("full", "swa", "mamba"):
            for pages in range(1, 25):
                for step in PLAN_STEPS:
                    found = plan(snapshot, {name: pages}, step)
                    if found is None:
                        continue
                    for band_name, take in found.takes.items():
                        e = snapshot.band(band_name).page_bytes
                        if take.pages is not None:
                            lo, hi = take.pages[0] * e, take.pages[1] * e
                            for h_lo, h_hi in held:
                                self.assertFalse(lo < h_hi and h_lo < hi, found)
                    for band_name, (first, end) in found.targets.items():
                        e = snapshot.band(band_name).page_bytes
                        for h_lo, h_hi in held:
                            self.assertFalse(first * e < h_hi and h_lo < end * e, found)
                # Admission counts every byte: a layout through the held bytes
                # waits for the send, the last resort.
                self.assertEqual(
                    feasible(snapshot, {name: pages}),
                    feasible(alloc.pricing_snapshot(), {name: pages}),
                )


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
        """Eager compaction ignores fixed pages (it could move a page reserved
        for a host load), so unified memory refuses it at startup, with or
        without HiCache."""
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
