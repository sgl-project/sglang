"""Page moves against copies that may still run: a move waits only for a copy
that writes a page it moves; a page it vacates that a copy still reads is not
overwritten before that copy ends; a copy that touches no moved page holds
nothing. Host-cache copies are waited on the GPU; a network transfer, which runs
outside every stream, is waited for on the host, never on the GPU."""

import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.page_usage import Consumer
from sglang.srt.mem_cache.kv.alloc.signals import NetworkTransfer
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    FakeKVCache,
    FakeUnifiedSWAKVPool,
    arena_of,
    open_float_side,
    setup_allocator_context,
    stand_in_views,
    tri_sub_pool_specs,
)

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")

_ROLES = ChainRoles(owner="full", window="swa", state="mamba")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMovesAgainstRunningCopies(CustomTestCase):
    def _chain(self):
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
        ids = alloc.alloc(12)
        self.assertIsNotNone(ids)
        alloc.free(ids[:4])
        live = ids[4:]
        schedule, copier = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(schedule):
            # A running scheduler has loaded these kernels and pinned buffers.
            for band in alloc.members:
                band.prewarm_compaction()
                band._to_device([0])
                for part in (live[:1], live[1:2], live):
                    band.ids.translate(part)
            torch.empty(1, device="cuda")
            torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        return pool, alloc, live, schedule, copier

    def _ask(self, flt):
        gap_low, _ = flt._gap_pages()
        return (gap_low + 2) * flt.entry_bytes_per_page

    def test_float_move_waits_a_load_that_writes_a_moved_page(self):
        """REGRESSION: a window float relocated pages a host load was still
        writing: only an end flush waited host copies. A delayed load rewrites
        every live window page; after the float moves, the moved pages must
        carry the load's bytes, and the move must return before the load ends."""
        pool, alloc, live, schedule, copier = self._chain()
        flt = alloc.swa_attn_allocator
        epp = flt.entry_bytes_per_page
        pages = flt.ids.virtual_to_physical[live].tolist()
        ask = self._ask(flt)
        done = torch.cuda.Event()
        with torch.cuda.stream(copier):
            torch.cuda._sleep(500_000_000)
            for k, page in enumerate(pages):
                pool.raw[page * epp : (page + 1) * epp].fill_(100 + k)
            done.record()
        with torch.cuda.stream(schedule):
            # Recorded on the stream moves run on, as the storage requires.
            flt.record_transfer(done, live, writes=True)
            opened = open_float_side(flt, "low", ask)
            self.assertFalse(done.query(), "the move blocked the host on the load")
        torch.cuda.synchronize()
        self.assertGreaterEqual(opened, ask)
        moved = flt.ids.virtual_to_physical[live].tolist()
        self.assertNotEqual(moved, pages, "the geometry asked for no relocation")
        raw = pool.raw.cpu()
        for k, page in enumerate(moved):
            self.assertTrue(
                bool((raw[page * epp : (page + 1) * epp] == 100 + k).all()),
                f"window page {k} lost the load's write",
            )

    def test_float_page_a_backup_reads_stays_until_the_backup_ends(self):
        """REGRESSION: a page a window float vacated was fenced only against
        forwards, so a host backup still reading it saw the next owner's bytes.
        The move itself must not wait the backup (it only reads); the vacated
        pages are overwritten only after it."""
        pool, alloc, live, schedule, copier = self._chain()
        flt = alloc.swa_attn_allocator
        epp = flt.entry_bytes_per_page
        pages = flt.ids.virtual_to_physical[live].tolist()
        original = pool.raw.clone()
        seen = torch.empty((len(pages), epp), dtype=torch.uint8, device="cuda")
        done = torch.cuda.Event()
        with torch.cuda.stream(copier):
            torch.cuda._sleep(500_000_000)
            for k, page in enumerate(pages):
                seen[k].copy_(pool.raw[page * epp : (page + 1) * epp])
            done.record()
        with torch.cuda.stream(schedule):
            flt.record_transfer(done, live, writes=False)
            open_float_side(flt, "low", self._ask(flt))
            copied = torch.cuda.Event()
            copied.record()
        copied.synchronize()
        self.assertFalse(done.query(), "the move waited for a reading copy")
        moved = flt.ids.virtual_to_physical[live].tolist()
        vacated = sorted(set(pages) - set(moved))
        self.assertTrue(vacated, "the geometry vacated no page")
        with torch.cuda.stream(schedule):
            first, end = vacated[0], vacated[-1] + 1
            flt._wait_fences(first, end)
            pool.raw[first * epp : end * epp].fill_(7)
        self.assertFalse(done.query(), "the copy ended before the overwrite")
        torch.cuda.synchronize()
        expected = torch.stack(
            [original[p * epp : (p + 1) * epp] for p in pages]
        ).cuda()
        self.assertTrue(torch.equal(seen, expected), "the backup read a reused page")

    def test_float_move_waits_on_the_host_for_a_receive_that_writes_a_moved_page(
        self,
    ):
        """A transfer outside every CUDA stream is waited for on the host. A
        delayed writer rewrites every live window page; the waiter returns only
        once it ended. The move copies after the waiter, so the moved pages
        carry the writer's bytes."""
        pool, alloc, live, schedule, copier = self._chain()
        flt = alloc.swa_attn_allocator
        epp = flt.entry_bytes_per_page
        pages = flt.ids.virtual_to_physical[live].tolist()
        ask = self._ask(flt)
        written = torch.cuda.Event()
        with torch.cuda.stream(copier):
            torch.cuda._sleep(500_000_000)
            for k, page in enumerate(pages):
                pool.raw[page * epp : (page + 1) * epp].fill_(100 + k)
            written.record()
        waited = _EngineWaiter(written)
        alloc.set_network_waiter(waited)
        transfer = NetworkTransfer("receive")
        with torch.cuda.stream(schedule):
            flt.record_transfer(
                transfer, live, writes=True, release=transfer, unit_tokens=None
            )
            opened = open_float_side(flt, "low", ask)
        self.assertEqual(waited.calls, [["receive"]])
        self.assertTrue(transfer.released)
        torch.cuda.synchronize()
        self.assertGreaterEqual(opened, ask)
        moved = flt.ids.virtual_to_physical[live].tolist()
        self.assertNotEqual(moved, pages, "the geometry asked for no relocation")
        raw = pool.raw.cpu()
        for k, page in enumerate(moved):
            self.assertTrue(
                bool((raw[page * epp : (page + 1) * epp] == 100 + k).all()),
                f"window page {k} was copied before the receive ended",
            )

    def test_end_flush_does_not_wait_a_copy_of_pages_it_keeps(self):
        """REGRESSION: an end flush waited every outstanding host copy before
        moving anything. A delayed load writes one owner page the flush keeps;
        after its delay it also stamps a page the flush moves, unrecorded, as a
        clock: if the copy waited the load, the moved page carries the stamp."""
        pool, alloc, live, schedule, copier = self._chain()
        full = alloc.full_attn_allocator
        epp = full.entry_bytes_per_page
        # Interior holes on the owner end: its survivors nearest the frontier
        # move into them; the one farthest from it stays.
        alloc.free(live[2:6])
        alive = live[full.ids.virtual_to_physical[live] >= 0]
        phys = full.ids.virtual_to_physical[alive]
        down = full.grow_direction == "down"
        far = int(phys.argmax() if down else phys.argmin())
        near = int(phys.argmin() if down else phys.argmax())
        kept, stamped = alive[far : far + 1], alive[near : near + 1]
        kept_page, stamped_page = int(phys[far]), int(phys[near])
        original = pool.raw[stamped_page * epp : (stamped_page + 1) * epp].clone()
        torch.cuda.synchronize()
        done = torch.cuda.Event()
        with torch.cuda.stream(copier):
            torch.cuda._sleep(500_000_000)
            pool.raw[kept_page * epp : (kept_page + 1) * epp].fill_(100)
            pool.raw[stamped_page * epp : (stamped_page + 1) * epp].fill_(55)
            done.record()
        # Registered as the host-cache controller does on submission.
        alloc.end_load_reservations()
        with torch.cuda.stream(schedule):
            full.record_transfer(done, kept, writes=True)
            moves = full.flush_for_allocation()
        self.assertFalse(done.query(), "the flush blocked the host on the load")
        torch.cuda.synchronize()
        self.assertGreater(moves, 0, "the geometry moved no page")
        moved_to = int(full.ids.virtual_to_physical[stamped][0])
        self.assertNotEqual(moved_to, stamped_page, "the stamped page did not move")
        raw = pool.raw
        self.assertTrue(
            torch.equal(raw[moved_to * epp : (moved_to + 1) * epp], original),
            "the flush waited a load that writes only a page it keeps",
        )
        self.assertTrue(
            bool((raw[kept_page * epp : (kept_page + 1) * epp] == 100).all())
        )

    def _end_flush_against_backup(self, forward: str):
        """An end band with interior holes, every survivor read by a delayed
        backup. ``forward``: ``"none"``, ``"retired"`` (a forward retires
        before the flush, so its release holds both) or ``"pending"`` (the
        flush runs under that forward, which retires before the next flush
        drains the vacated pages)."""
        pool, alloc, live, schedule, copier = self._chain()
        full = alloc.full_attn_allocator
        epp = full.entry_bytes_per_page
        alloc.end_load_reservations()
        # Interior holes on the owner end: the survivors nearest the frontier
        # move into them.
        alloc.free(live[2:6])
        live = live[full.ids.virtual_to_physical[live] >= 0]
        down = full.grow_direction == "down"
        pages = full.ids.virtual_to_physical[live].tolist()
        original = pool.raw.clone()
        seen = torch.empty((len(pages), epp), dtype=torch.uint8, device="cuda")
        fwd = torch.cuda.Event()
        fwd.record()
        torch.cuda.synchronize()
        done = torch.cuda.Event()
        with torch.cuda.stream(copier):
            torch.cuda._sleep(500_000_000)
            for k, page in enumerate(pages):
                seen[k].copy_(pool.raw[page * epp : (page + 1) * epp])
            done.record()
        with torch.cuda.stream(schedule):
            full.record_transfer(done, live, writes=False)
            if forward != "none":
                full.set_latest_forward_done_event(fwd)
            if forward == "retired":
                full.retire_forward(fwd)
            moves = full.flush_opportunistic()
            if forward == "pending":
                self.assertTrue(full.usage.has_pending())
                full.retire_forward(fwd)
                full.flush_opportunistic()
            flushed = torch.cuda.Event()
            flushed.record()
        flushed.synchronize()
        self.assertFalse(
            done.query(), "the schedule stream waited a backup of vacated pages"
        )
        self.assertGreater(moves, 0, "the geometry moved no page")
        moved = full.ids.virtual_to_physical[live].tolist()
        vacated = sorted(set(pages) - set(moved))
        self.assertTrue(vacated, "the geometry vacated no page")
        self.assertFalse(full.usage.has_pending())
        # A vacated page is a hole, or beyond the frontier once a later flush
        # absorbed it: take every hole, and the frontier through it.
        free = full._free_pages_snapshot()
        beyond = [p for p in vacated if p not in free]
        need = len(free)
        if beyond:
            wm = full.watermark_physical
            need += wm - min(beyond) + 1 if down else max(beyond) - wm + 1
        with torch.cuda.stream(schedule):
            got = full.acquire(need, Consumer.SCHEDULE_COPY)
            for page in vacated:
                pool.raw[page * epp : (page + 1) * epp].fill_(7)
        self.assertFalse(done.query(), "the backup ended before the overwrite")
        torch.cuda.synchronize()
        self.assertTrue(set(vacated) <= set(got.tolist()))
        expected = torch.stack(
            [original[p * epp : (p + 1) * epp] for p in pages]
        ).cuda()
        self.assertTrue(torch.equal(seen, expected), "the backup read a reused page")

    def test_opportunistic_flush_leaves_a_backup_to_the_next_write(self):
        """REGRESSION: a non-urgent end flush that vacated a page a host backup
        still read waited that backup on the schedule stream, so the next
        forward inherited it. The vacated page returns as a hole fenced by the
        backup: the stream runs on, and only a write into the page waits it."""
        self._end_flush_against_backup("none")

    def test_opportunistic_flush_fences_a_retired_forward_with_its_backup(self):
        """As above, with the newest forward retired before the flush: the
        release holds the forward and the backup."""
        self._end_flush_against_backup("retired")

    def test_pending_drain_leaves_a_backup_to_the_next_write(self):
        """As above, for pages vacated under a running forward: once it
        retires, the next flush returns them to the free list fenced by the
        backup, without waiting it on the schedule stream."""
        self._end_flush_against_backup("pending")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestChainFlushReadsBeforeAnyWait(CustomTestCase):
    """Allocation pressure flushes the owner end, then the window end. A wait
    the first queues for a running load must not park the host when the
    second reads its own copies' pages: every band of the chain reads what it
    needs before the first wait."""

    def _chain(self):
        setup_allocator_context()
        roles = ChainRoles(owner="full", window="swa")
        full, swa = (
            MHASubPoolSpec(
                name=name,
                layer_num=1,
                head_num=1,
                head_dim=8,
                store_dtype=torch.float16,
                grow_direction=direction,
            )
            for name, direction in (("full", "up"), ("swa", "down"))
        )
        pool = arena_of(
            total_bytes=24 * full.entry_bytes() + 24 * swa.entry_bytes(),
            sub_pool_specs=[full, swa],
            device="cuda",
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        for fake in (kvcache.full_kv_pool, kvcache.swa_kv_pool):
            fake.buf = fake.buf.cuda()
        alloc = ChainAllocator(
            roles=roles,
            arena=pool,
            kvcache=kvcache,
            views=stand_in_views(roles, kvcache),
            device="cuda",
            full_max_total_num_tokens=24,
            swa_max_total_num_tokens=24,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=True,
        )
        ids = alloc.alloc(12)
        self.assertIsNotNone(ids)
        # Interior holes in both ends: the survivors past them move.
        alloc.free(ids[2:6])
        live = torch.cat([ids[:2], ids[6:]])
        schedule = torch.cuda.Stream()
        with torch.cuda.stream(schedule):
            # A running scheduler has loaded these kernels and pinned buffers.
            for band in alloc.members:
                band.prewarm_compaction()
                band._to_device([0])
                band.ids.translate(live)
            torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        return pool, alloc, live, schedule

    def test_allocation_pressure_flush_returns_while_the_load_runs(self):
        pool, alloc, live, schedule = self._chain()
        bands = (alloc.full_attn_allocator, alloc.swa_attn_allocator)
        before = [band.ids.virtual_to_physical[live].tolist() for band in bands]
        done = torch.cuda.Event()
        with torch.cuda.stream(torch.cuda.Stream()):
            torch.cuda._sleep(500_000_000)
            for band, pages in zip(bands, before):
                epp = band.entry_bytes_per_page
                for k, page in enumerate(pages):
                    pool.raw[page * epp : (page + 1) * epp].fill_(100 + k)
            done.record()
        alloc.end_load_reservations()
        with torch.cuda.stream(schedule):
            # A host load writes every live page of both ends, registered as
            # the host-cache controller does on submission.
            for band in bands:
                band.record_transfer(done, live, writes=True)
            moves = [band.flush_for_allocation() for band in bands]
            self.assertFalse(done.query(), "the flush blocked the host on the load")
        torch.cuda.synchronize()
        self.assertTrue(all(moves), f"the geometry moved no page: {moves}")
        self.assertEqual(alloc.verify_byte_accounting(), [])
        raw = pool.raw.cpu()
        for band, pages in zip(bands, before):
            moved = band.ids.virtual_to_physical[live].tolist()
            self.assertNotEqual(moved, pages)
            self.assertEqual(
                band.ids.physical_to_virtual[torch.tensor(moved)].tolist(),
                live.tolist(),
            )
            epp = band.entry_bytes_per_page
            for k, page in enumerate(moved):
                self.assertTrue(
                    bool((raw[page * epp : (page + 1) * epp] == 100 + k).all()),
                    f"{band.sub_pool_name} page {k} lost the load's write",
                )


class _EngineWaiter:
    """A network waiter standing in for a transfer engine: it returns once
    ``done`` (the engine's last touch of the rows) fired, recording each wait."""

    def __init__(self, done):
        self.done = done
        self.calls = []

    def __call__(self, transfers):
        self.calls.append([t.peer for t in transfers])
        self.done.synchronize()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMovesAgainstNetworkTransfers(TestMovesAgainstRunningCopies):
    """A network engine addresses rows by the physical ids it was handed, so a
    record keeps those rows, whatever moves later; only the host learns that it
    ended, so a move that cannot avoid it waits on the host."""

    def _owner_end(self):
        pool, alloc, live, schedule, copier = self._chain()
        full = alloc.full_attn_allocator
        # Interior holes; the survivor nearest the frontier moves first.
        alloc.free(live[2:6])
        alive = live[full.ids.virtual_to_physical[live] >= 0]
        phys = full.ids.virtual_to_physical[alive]
        near = int(phys.argmin() if full.grow_direction == "down" else phys.argmax())
        moved, page = alive[near : near + 1], int(phys[near])
        torch.cuda.synchronize()
        return pool, alloc, full, moved, page, schedule, copier

    def _record(self, full, page, peer, *, writes):
        transfer = NetworkTransfer(peer)
        full.record_transfer(
            transfer,
            [page],
            writes=writes,
            release=transfer,
            unit_tokens=full.pool_page_size,
        )
        return transfer

    def test_a_move_without_a_waiter_refuses_rather_than_waiting_on_the_gpu(self):
        """No GPU wait stands in for a network transfer: with no waiter a move
        that must wait for one raises instead of proceeding."""
        pool, alloc, full, moved, page, schedule, copier = self._owner_end()
        self._record(full, page, "receive", writes=True)
        with torch.cuda.stream(schedule):
            with self.assertRaisesRegex(RuntimeError, "no network waiter"):
                full.flush_for_allocation()


if __name__ == "__main__":
    unittest.main()
