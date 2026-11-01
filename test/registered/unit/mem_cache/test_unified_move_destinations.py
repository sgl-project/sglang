"""What a page move may overwrite, and what it may wait for.

A request's row can go back to the free list while a forward launched before
the release still writes its next slot (an overlapped batch): a copy into that
hole must wait the forward. A row a network peer reads or writes is moved only
when no other layout serves the allocation, since moving it waits the peer's
completion flag on the GPU. A relief reads everything it needs on the host
before its first stream wait.
"""

import time
import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.signals import CompletionFlags, TransferRelease
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    FakeKVCache,
    FakeUnifiedSWAKVPool,
    arena_of,
    setup_allocator_context,
    stand_in_views,
    tri_sub_pool_specs,
)

register_cuda_ci(est_time=12, stage="extra-a", runner_config="1-gpu-small")

_ROLES = ChainRoles(owner="full", window="swa", state="mamba")


class _Chain(CustomTestCase):
    def _chain(self, tokens):
        setup_allocator_context()
        full, swa, mamba = tri_sub_pool_specs(head_num=1, head_dim=8)
        pool = arena_of(
            total_bytes=48 * full.entry_bytes()
            + 32 * swa.entry_bytes()
            + 8 * mamba.entry_bytes(),
            sub_pool_specs=[full, swa, mamba],
            device="cuda",
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        state = FakeKVCache(pool, "mamba")
        for fake in (kvcache.full_kv_pool, kvcache.swa_kv_pool, state):
            fake.buf = fake.buf.cuda()
        alloc = ChainAllocator(
            roles=_ROLES,
            arena=pool,
            kvcache=kvcache,
            views=stand_in_views(_ROLES, kvcache, state),
            device="cuda",
            full_max_total_num_tokens=48,
            swa_max_total_num_tokens=32,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=True,
        )
        ids = alloc.alloc(tokens)
        self.assertIsNotNone(ids)
        schedule, side = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(schedule):
            # A running scheduler has loaded these kernels and pinned buffers.
            for band in alloc.members:
                band.prewarm_compaction()
                band._to_device([0])
                for part in (ids[:1], ids[1:2], ids[:10], ids):
                    band.ids.translate(part)
            torch.cat([ids[:1], ids[1:2]]).tolist()
            torch.empty(1, device="cuda")
            torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        return pool, alloc, ids, schedule, side

    def _slot_and_survivor(self, full, ids):
        """The slot farthest from the owner's frontier, and the survivor
        nearest it: once the slot is a hole, a flush copies the survivor in."""
        phys = full.ids.virtual_to_physical[ids].tolist()
        down = full.grow_direction == "down"
        far = (max if down else min)(range(len(phys)), key=lambda i: phys[i])
        near = (min if down else max)(range(len(phys)), key=lambda i: phys[i])
        return ids[far : far + 1], phys[far], ids[near : near + 1], phys[near]


_DELAY = 500_000_000  # GPU cycles a delayed actor sleeps


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMoveDestinations(_Chain):
    def _overlapped_release(self, delay):
        """A forward launched before the release of its slot's row writes the
        slot late; the row is released while it runs."""
        pool, alloc, ids, schedule, side = self._chain(12)
        full = alloc.full_attn_allocator
        epp = full.entry_bytes_per_page
        slot, slot_page, survivor, survivor_page = self._slot_and_survivor(full, ids)
        original = pool.raw[survivor_page * epp : (survivor_page + 1) * epp].clone()
        done = torch.cuda.Event()
        with torch.cuda.stream(side):
            torch.cuda._sleep(delay)
            pool.raw[slot_page * epp : (slot_page + 1) * epp].fill_(0xEE)
            done.record()
        full.set_latest_forward_done_event(done)
        full.set_inflight_forward(done, slot)
        alloc.free(slot)
        return pool, full, epp, slot_page, survivor, original, done, schedule

    def _flush_into_the_slot(self, delay):
        pool, full, epp, slot_page, survivor, original, done, schedule = (
            self._overlapped_release(delay)
        )
        with torch.cuda.stream(schedule):
            moves = full.flush_for_allocation()
        returned_early = not done.query()
        torch.cuda.synchronize()
        moved_to = int(full.ids.virtual_to_physical[survivor][0])
        intact = torch.equal(pool.raw[moved_to * epp : (moved_to + 1) * epp], original)
        return moves, moved_to, slot_page, returned_early, intact

    def test_a_copy_into_a_slot_a_running_forward_writes_waits_it(self):
        """REGRESSION: an allocation's flush copied a live page into a hole
        whose slot an overlapped forward still wrote, so the forward's late
        write landed on the moved page. The copy must wait the forward on the
        GPU, without blocking the host."""
        self._flush_into_the_slot(0)  # loads every kernel the flush launches
        moves, moved_to, slot_page, returned_early, intact = self._flush_into_the_slot(
            _DELAY
        )
        self.assertGreater(moves, 0, "the geometry moved no page")
        self.assertEqual(moved_to, slot_page, "the survivor went elsewhere")
        self.assertTrue(returned_early, "the flush blocked the host on the forward")
        self.assertTrue(intact, "the forward's late write overwrote the moved page")

    def test_an_idle_flush_leaves_a_slot_a_running_forward_writes(self):
        """REGRESSION: the per-step flush, which never waits, copied a live
        page into that hole. It must leave the hole alone."""
        pool, full, epp, slot_page, survivor, original, done, schedule = (
            self._overlapped_release(_DELAY)
        )
        with torch.cuda.stream(schedule):
            moves = full.flush_opportunistic()
        torch.cuda.synchronize()
        self.assertEqual(moves, 0, "an idle flush filled a hole a forward writes")
        page = int(full.ids.virtual_to_physical[survivor][0])
        self.assertTrue(torch.equal(pool.raw[page * epp : (page + 1) * epp], original))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestReliefsAroundNetworkRows(_Chain):
    """The owner's end holds a row a network peer touches near its frontier
    and holes far from it; the state end below needs room the owner's
    compaction gives. Compaction that keeps the row in place gives less."""

    # State pages the layout must give: around the held row, or only with the
    # row moved.
    AROUND_THE_ROW = 53
    ONLY_WITH_THE_ROW_MOVED = 56

    def _owner_with_holes(self):
        pool, alloc, ids, schedule, side = self._chain(30)
        full = alloc.full_attn_allocator
        self.assertEqual(full.grow_direction, "down")
        # The window float empties, so the state end meets the owner's end.
        alloc.free_swa(ids)
        phys = full.ids.virtual_to_physical[ids].tolist()
        by_page = {p: ids[i : i + 1] for i, p in enumerate(phys)}
        low = min(phys)
        full.free(torch.cat([by_page[p] for p in range(low + 20, low + 25)]))
        row_page = low + 2
        torch.cuda.synchronize()
        return pool, alloc, full, by_page[row_page], row_page, schedule, side

    def _flag(self, *, published):
        flags = CompletionFlags(num_slots=4)
        flag = flags.acquire()
        # A failed check must not leave a stream waiting the flag forever.
        self.addCleanup(flag.publish)
        if published:
            flag.publish()
        return flag

    def _record(self, full, page, flag, *, writes):
        full.record_transfer(
            flag,
            [page],
            writes=writes,
            release=TransferRelease(),
            unit_tokens=full.pool_page_size,
        )

    def _lay_out(self, alloc, schedule, pages):
        with torch.cuda.stream(schedule):
            self.assertTrue(alloc.lay_out_for({"mamba": pages}))
            laid = torch.cuda.Event()
            laid.record()
        return laid

    def _receive_then_lay_out(self, pages, *, published):
        pool, alloc, full, row, row_page, schedule, side = self._owner_with_holes()
        flag = self._flag(published=published)
        self._record(full, row_page, flag, writes=True)
        laid = self._lay_out(alloc, schedule, pages)
        return pool, full, row, row_page, flag, laid, side

    def test_a_relief_keeps_a_received_row_in_place_when_it_can(self):
        """REGRESSION: an allocation's flush copied a row a receive still
        wrote, so the schedule stream, and every forward after it, waited the
        remote peer. When a layout serves the allocation around the row, the
        row stays where it is and nothing waits its flag."""
        self._receive_then_lay_out(self.AROUND_THE_ROW, published=True)
        torch.cuda.synchronize()
        pool, full, row, row_page, flag, laid, side = self._receive_then_lay_out(
            self.AROUND_THE_ROW, published=False
        )
        time.sleep(0.05)
        self.assertTrue(laid.query(), "the relief waited the receive's flag")
        self.assertEqual(int(full.ids.virtual_to_physical[row][0]), row_page)
        self.assertFalse(full.fixed_pages(), "the relief left the row fixed")

    def test_a_relief_moves_a_received_row_after_its_flag_when_it_must(self):
        """When only moving the row serves the allocation, the relief moves it
        after the flag, and the moved row carries the receive's bytes."""
        self._receive_then_lay_out(self.ONLY_WITH_THE_ROW_MOVED, published=True)
        torch.cuda.synchronize()
        pool, alloc, full, row, row_page, schedule, side = self._owner_with_holes()
        epp = full.entry_bytes_per_page
        flag = self._flag(published=False)
        written = torch.cuda.Event()
        with torch.cuda.stream(side):
            torch.cuda._sleep(_DELAY // 2)
            pool.raw[row_page * epp : (row_page + 1) * epp].fill_(42)
            written.record()
        self._record(full, row_page, flag, writes=True)
        laid = self._lay_out(alloc, schedule, self.ONLY_WITH_THE_ROW_MOVED)
        written.synchronize()
        time.sleep(0.05)
        self.assertFalse(laid.query(), "the row was copied before the flag")
        flag.publish()
        laid.synchronize()
        moved_to = int(full.ids.virtual_to_physical[row][0])
        self.assertNotEqual(moved_to, row_page)
        self.assertTrue(
            bool((pool.raw[moved_to * epp : (moved_to + 1) * epp] == 42).all())
        )

    def _send_moved_then_lay_out(self, *, published):
        pool, alloc, full, row, row_page, schedule, side = self._owner_with_holes()
        flag = self._flag(published=published)
        self._record(full, row_page, flag, writes=False)
        with torch.cuda.stream(schedule):
            self.assertGreater(full.flush_opportunistic(), 0)
        torch.cuda.synchronize()
        pending = set(full.usage.pending_pages)
        laid = self._lay_out(alloc, schedule, self.AROUND_THE_ROW)
        return full, row_page, pending, laid

    def test_a_relief_does_not_reuse_a_row_a_send_still_reads(self):
        """REGRESSION: a row an idle flush moved away from a send stayed
        pending until the send's release, yet the planner counted it free and
        an allocation's flush waited the send's flag to reuse it; the moved
        rows the send does not read were held with it. When a layout serves
        the allocation without the row, it stays pending and nothing waits."""
        self._send_moved_then_lay_out(published=True)
        torch.cuda.synchronize()
        full, row_page, pending, laid = self._send_moved_then_lay_out(published=False)
        self.assertEqual(pending, {row_page}, "pending beyond the row the send reads")
        time.sleep(0.05)
        self.assertTrue(laid.query(), "the relief waited the send's flag")
        self.assertIn(row_page, full.usage.pending_pages)
        self.assertNotIn(row_page, full._free_pages_snapshot())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestReliefHostReads(_Chain):
    def _relief(self, delay):
        pool, alloc, ids, schedule, side = self._chain(30)
        alloc.free(ids[10:20])
        done = torch.cuda.Event()
        with torch.cuda.stream(side):
            torch.cuda._sleep(delay)
            done.record()
        alloc.set_latest_forward_done_event(done)
        alloc.set_inflight_forward(done, torch.cat([ids[:10], ids[20:]]))
        window = alloc.swa_attn_allocator
        span_before = window._span_range()
        with torch.cuda.stream(schedule):
            self.assertTrue(alloc.lay_out_for({"mamba": 20}))
        returned_early = not done.query()
        torch.cuda.synchronize()
        return returned_early, window._span_range() != span_before

    def test_a_relief_reads_every_band_before_its_first_wait(self):
        """REGRESSION: a relief that compacted the owner's end, waiting a
        running forward on the GPU, then read the window float's write sets
        on the host, which parked the scheduler until the forward ended."""
        self._relief(0)
        returned_early, relocated = self._relief(_DELAY)
        self.assertTrue(relocated, "the float stayed")
        self.assertTrue(returned_early, "the relief blocked the host on the forward")


if __name__ == "__main__":
    unittest.main()
