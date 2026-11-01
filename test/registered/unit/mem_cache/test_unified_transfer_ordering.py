"""Page moves against host-cache copies that may still run: a move waits, on
the GPU, only for a copy that writes a page it moves; a page it vacates that a
copy still reads is not overwritten before that copy ends; a copy that touches
no moved page holds nothing."""

import threading
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
        flt.record_transfer(done, live, writes=True)

        with torch.cuda.stream(schedule):
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
        flt.record_transfer(done, live, writes=False)

        with torch.cuda.stream(schedule):
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

    def test_float_move_waits_a_flag_published_by_the_host(self):
        """A transfer outside every CUDA stream signals with a completion flag
        the host publishes. A delayed writer rewrites every live window page;
        a host thread publishes the flag only after it ends. The move must
        return at once, and the moved pages must carry the writer's bytes."""
        pool, alloc, live, schedule, copier = self._chain()
        flt = alloc.swa_attn_allocator
        epp = flt.entry_bytes_per_page
        pages = flt.ids.virtual_to_physical[live].tolist()
        ask = self._ask(flt)
        flags = CompletionFlags(num_slots=4)
        # A reused slot: the earlier flag's waits stay satisfied.
        flags.release(flags.acquire())
        flag = flags.acquire()
        written = torch.cuda.Event()
        with torch.cuda.stream(copier):
            torch.cuda._sleep(500_000_000)
            for k, page in enumerate(pages):
                pool.raw[page * epp : (page + 1) * epp].fill_(100 + k)
            written.record()
        publisher = threading.Thread(
            target=lambda: (written.synchronize(), flag.publish())
        )
        flt.record_transfer(flag, live, writes=True)

        with torch.cuda.stream(schedule):
            opened = open_float_side(flt, "low", ask)
            self.assertFalse(flag.query(), "the move blocked the host on the flag")
        publisher.start()
        publisher.join()
        torch.cuda.synchronize()
        self.assertTrue(flag.query())
        self.assertGreaterEqual(opened, ask)
        moved = flt.ids.virtual_to_physical[live].tolist()
        self.assertNotEqual(moved, pages, "the geometry asked for no relocation")
        raw = pool.raw.cpu()
        for k, page in enumerate(moved):
            self.assertTrue(
                bool((raw[page * epp : (page + 1) * epp] == 100 + k).all()),
                f"window page {k} was copied before the flag",
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
        alloc.set_hicache_transfer_done_event(("test", "load"), done)
        full.record_transfer(done, kept, writes=True)

        with torch.cuda.stream(schedule):
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMovesAgainstNetworkTransfers(TestMovesAgainstRunningCopies):
    """A network engine addresses rows by the physical ids it was handed, so a
    record keeps those rows, whatever moves later; its completion is a flag a
    host thread publishes."""

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
        return pool, full, moved, page, schedule, copier

    def _record(self, full, page, flag, *, writes):
        full.record_transfer(
            flag,
            [page],
            writes=writes,
            release=TransferRelease(),
            unit_tokens=full.pool_page_size,
        )

    def test_a_row_a_send_reads_moves_and_is_reused_only_after_the_flag(self):
        """A send reads its row by physical id until the engine is done. The
        row moves at once; the row it vacates is overwritten only after the
        flag, so the engine reads the original bytes."""
        pool, full, moved, page, schedule, copier = self._owner_end()
        epp = full.entry_bytes_per_page
        original = pool.raw[page * epp : (page + 1) * epp].clone()
        flags = CompletionFlags(num_slots=4)
        flag = flags.acquire()
        seen = torch.empty(epp, dtype=torch.uint8, device="cuda")
        read = torch.cuda.Event()
        with torch.cuda.stream(copier):
            torch.cuda._sleep(500_000_000)
            seen.copy_(pool.raw[page * epp : (page + 1) * epp])
            read.record()
        publisher = threading.Thread(
            target=lambda: (read.synchronize(), flag.publish())
        )
        self._record(full, page, flag, writes=False)

        with torch.cuda.stream(schedule):
            self.assertGreater(full.flush_for_allocation(), 0)
            self.assertFalse(flag.query(), "the move waited for a reading send")
            new_page = int(full.ids.virtual_to_physical[moved][0])
            self.assertNotEqual(new_page, page, "the read row did not move")
            full._wait_fences(page, page + 1)
            pool.raw[page * epp : (page + 1) * epp].fill_(7)
        publisher.start()
        publisher.join()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(seen, original), "the send read a reused row")
        self.assertTrue(
            torch.equal(pool.raw[new_page * epp : (new_page + 1) * epp], original)
        )

    def test_a_row_a_receive_writes_moves_only_after_the_flag(self):
        """A receive writes its row by physical id; a move an allocation needs
        copies it only after the flag, without blocking the host."""
        pool, full, moved, page, schedule, copier = self._owner_end()
        epp = full.entry_bytes_per_page
        flags = CompletionFlags(num_slots=4)
        flag = flags.acquire()
        written = torch.cuda.Event()
        with torch.cuda.stream(copier):
            torch.cuda._sleep(500_000_000)
            pool.raw[page * epp : (page + 1) * epp].fill_(42)
            written.record()
        self._record(full, page, flag, writes=True)

        with torch.cuda.stream(schedule):
            self.assertEqual(full.flush_opportunistic(), 0, "an idle flush moved it")
            self.assertGreater(full.flush_for_allocation(), 0)
            copied = torch.cuda.Event()
            copied.record()
        self.assertFalse(written.query(), "the flush blocked the host")
        written.synchronize()
        time.sleep(0.05)
        self.assertFalse(copied.query(), "the copy ran before the flag")
        flag.publish()
        copied.synchronize()
        new_page = int(full.ids.virtual_to_physical[moved][0])
        self.assertNotEqual(new_page, page)
        self.assertTrue(
            bool((pool.raw[new_page * epp : (new_page + 1) * epp] == 42).all()),
            "the moved row lost the receive's write",
        )


if __name__ == "__main__":
    unittest.main()
