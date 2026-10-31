"""Physical Mamba relocation preserves payloads without a main-state gather."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_unified_mamba_views import _make_pool

from sglang.srt.mem_cache.kv.alloc.band import Band
from sglang.srt.mem_cache.kv.layout.state import StateView
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_cuda_ci(est_time=18, stage="extra-a", runner_config="1-gpu-small")


class TestMambaCompaction(CustomTestCase):
    def _fixture(self, device, direction="up", lazy=True, count=32):
        pool, spec = _make_pool(
            mamba_layer_num=2,
            conv_state_shapes=[(8, 3), (4, 2)],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(16, 64, 128) if device == "cuda" else (2, 4, 8),
            temporal_dtype=torch.float32,
            want_slots=count + 8,
            device=device,
            grow_direction=direction,
        )
        kv = StateView(
            arena=pool,
            spec=pool.spec("mamba"),
            mamba_layer_ids=[1, 3],
        )
        alloc = Band(
            arena=pool,
            sub_pool_name="mamba",
            device=device,
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        full = Band(
            arena=pool,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        alloc.bind_peer(full)
        full.bind_peer(alloc)
        full_ids = full.alloc(4).clone()
        full_pages = full.ids.virtual_to_physical[full_ids].cpu().tolist()
        ids = alloc.alloc(count).clone()
        ids = ids[torch.argsort(alloc.ids.virtual_to_physical[ids])]
        original_pages = alloc.ids.virtual_to_physical[ids].cpu().tolist()
        # Independent, byte-varying reference. Keep floating values finite for COW.
        gen = torch.Generator().manual_seed(42137)
        raw = torch.randint(0, 64, pool.raw.shape, dtype=torch.uint8, generator=gen)
        pool.raw.copy_(raw)
        # Unit moves of the state band, as (dst, src) per call.
        move = Mock()
        arena_move = pool.move_units

        def recorded(name, dst, src):
            if name == "mamba":
                move(dst, src)
            return arena_move(name, dst, src)

        pool.move_units = recorded
        return SimpleNamespace(
            pool=pool,
            spec=spec,
            kv=kv,
            alloc=alloc,
            full=full,
            ids=ids,
            original_pages=original_pages,
            full_pages=full_pages,
            raw=raw,
            move=move,
            count=count,
        )

    def _verify(self, f, freed):
        raw = f.pool.raw.cpu()
        current = f.alloc.ids.virtual_to_physical[f.ids].cpu().tolist()
        expected_pairs = set()
        for i, (v, old, new) in enumerate(
            zip(f.ids.cpu().tolist(), f.original_pages, current)
        ):
            if i in freed:
                self.assertEqual(new, -1)
                continue
            if old != new:
                expected_pairs.add((old, new))
            n = f.spec.entry_bytes()
            self.assertTrue(
                torch.equal(
                    raw[new * n : (new + 1) * n], f.raw[old * n : (old + 1) * n]
                )
            )
            self.assertEqual(int(f.alloc.ids.physical_to_virtual[new]), v)
        self.assertTrue(
            torch.equal(raw[: f.spec.entry_bytes()], f.raw[: f.spec.entry_bytes()])
        )
        for p in f.full_pages:
            n = f.full.entry_bytes
            self.assertTrue(
                torch.equal(raw[p * n : (p + 1) * n], f.raw[p * n : (p + 1) * n])
            )
        actual_pairs = set()
        for call in f.move.call_args_list:
            dst, src = call.args
            src, dst = src.cpu().tolist(), dst.cpu().tolist()
            self.assertFalse(set(src) & set(dst))
            self.assertEqual(len(dst), len(set(dst)))
            actual_pairs.update(zip(src, dst))
        self.assertEqual(actual_pairs, expected_pairs)
        self.assertTrue(expected_pairs)

    def _exercise(self, device, direction, mode, count=32):
        f = self._fixture(device, direction, mode != "eager", count=count)
        holes = f.count // 4
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
        f.alloc.free(f.ids[holes : 2 * holes])
        if mode == "urgent":
            self.assertEqual(f.alloc.flush_for_allocation(), holes)
        elif mode == "opportunistic":
            self.assertEqual(f.alloc.flush_opportunistic(), holes)
        if device == "cuda":
            torch.cuda.synchronize()
            extra_peak = torch.cuda.max_memory_allocated() - before
            # A unit move copies in place: no entry-sized scratch.
            self.assertLess(extra_peak, f.spec.entry_bytes())
        self._verify(f, set(range(holes, 2 * holes)))
        self.assertEqual(sum(c.args[0].numel() for c in f.move.call_args_list), holes)
        self.assertEqual(f.alloc.flush_for_allocation(), 0)
        # Ordinary COW copies whole entries after relocation.
        cow_id = f.alloc.alloc(1).clone()
        dst = f.alloc.ids.virtual_to_physical[cow_id]
        src = f.alloc.ids.virtual_to_physical[f.ids[-1:]]
        f.kv.copy_from(src, dst)
        state_tensors = [*f.kv.mamba_cache.conv, f.kv.mamba_cache.temporal]
        for tensor in state_tensors:
            self.assertTrue(torch.equal(tensor[:, src], tensor[:, dst]))
        sources = [t[:, src].clone() for t in state_tensors]
        f.kv.clear_slots(dst)
        for tensor, source in zip(state_tensors, sources):
            self.assertEqual(torch.count_nonzero(tensor[:, dst]).item(), 0)
            self.assertTrue(torch.equal(tensor[:, src], source))
        before_empty = f.pool.raw.clone()
        f.pool.move_units("mamba", f.ids[:0], f.ids[:0])
        self.assertTrue(torch.equal(f.pool.raw, before_empty))

    def test_side_states_beside_the_entry_are_refused(self):
        f = self._fixture("cpu", "down", False, count=4)
        with self.assertRaisesRegex(ValueError, "whole state in its entry"):
            f.kv.register_slot_state(object())

    def test_cpu_directions_and_move_paths(self):
        for direction in ("up", "down"):
            for mode in ("eager", "urgent", "opportunistic"):
                for count in (4, 32):
                    with self.subTest(direction=direction, mode=mode, count=count):
                        self._exercise("cpu", direction, mode, count=count)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_directions_and_move_paths(self):
        for direction in ("up", "down"):
            for mode in ("eager", "urgent", "opportunistic"):
                for count in (4, 32):
                    with self.subTest(direction=direction, mode=mode, count=count):
                        self._exercise("cuda", direction, mode, count=count)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_transfer_and_forward_writers_finish_before_move(self):
        for writer in ("forward", "load", "write"):
            with self.subTest(writer=writer):
                f = self._fixture("cuda")
                f.alloc.free(f.ids[8:16])
                n = f.spec.entry_bytes()
                # The slots the flush moves: a backup reads them, and must see
                # their bytes after the move.
                moved = [f.original_pages[i] for i in range(f.count - 8, f.count)]
                stream = torch.cuda.Stream()
                done = torch.cuda.Event()
                host = torch.empty(f.raw.shape, dtype=torch.uint8, pin_memory=True)
                host.copy_((f.raw + 1) % 64 if writer != "write" else f.raw)
                expected = host.clone()
                torch.cuda.synchronize()
                with torch.cuda.stream(stream):
                    # A bounded real delayed producer, not a fake event. Assert
                    # pending before entry; don't count already-fired coverage.
                    torch.cuda._sleep(1_000_000_000)
                    if writer == "write":
                        for page in moved:
                            host[page * n : (page + 1) * n].copy_(
                                f.pool.raw[page * n : (page + 1) * n],
                                non_blocking=True,
                            )
                    else:
                        f.pool.raw.copy_(host, non_blocking=True)
                    done.record()
                self.assertFalse(done.query())
                if writer == "forward":
                    # The forward rewrites every slot: a move waits for the
                    # writers its sources were recorded with.
                    f.alloc.set_latest_forward_done_event(done)
                    f.alloc.set_inflight_forward(done, f.ids)
                else:
                    # A load writes every slot; a backup reads the moved ones.
                    f.alloc.set_hicache_transfer_done_event(writer, done)
                    f.alloc.record_transfer(done, f.ids, writes=writer == "load")
                self.assertEqual(f.alloc.flush_for_allocation(), 8)
                if writer == "write":
                    # The vacated slots, retaken and overwritten after the
                    # backup's fence.
                    retaken = f.alloc.take_physical(8)
                    self.assertEqual(sorted(retaken.tolist()), sorted(moved))
                    f.pool.raw[min(moved) * n : (max(moved) + 1) * n].fill_(7)
                torch.cuda.synchronize()
                self.assertTrue(done.query())
                if writer == "write":
                    for page in moved:
                        self.assertTrue(
                            torch.equal(
                                host[page * n : (page + 1) * n],
                                expected[page * n : (page + 1) * n],
                            ),
                            f"the backup read slot {page} after it was reused",
                        )
                    f.pool.raw[min(moved) * n : (max(moved) + 1) * n].copy_(
                        f.raw[min(moved) * n : (max(moved) + 1) * n]
                    )
                else:
                    f.raw = expected
                self._verify(f, set(range(8, 16)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_every_unfinished_forward_guards_its_state_writes(self):
        # A pipeline stage keeps several microbatch forwards in flight, each
        # writing its own requests' state slots in place. A non-urgent flush must
        # not move a slot any of them writes; an urgent one orders its copies
        # after all of them on the GPU, and returns to the host while they run.
        f = self._fixture("cuda")
        f.alloc.free(f.ids[8:16])
        n = f.spec.entry_bytes()
        # The oldest forward writes the topmost survivor, the first one moved.
        writers = (f.count - 1, f.count - 2)
        expected = f.raw.clone()
        payloads = {}
        for k, i in enumerate(writers):
            page = f.original_pages[i]
            expected[page * n : (page + 1) * n] = 100 + k
            payloads[i] = torch.full((n,), 100 + k, dtype=torch.uint8, device="cuda")
        schedule = torch.cuda.Stream()
        forward = torch.cuda.Stream()
        with torch.cuda.stream(schedule):
            # Loading a kernel module and growing the caching allocator both wait
            # for every stream; warm both first, as a running scheduler has.
            f.alloc.prewarm_compaction()
            # Both pointer alignments the write-set translation is launched with.
            f.alloc.ids.translate(f.ids[:1])
            f.alloc.ids.translate(f.ids[1:2])
            f.alloc._to_device([0])
            torch.empty(1, device="cuda")
            torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        f.move.reset_mock()  # the warm-up copies the sink onto itself
        events = []
        for i in writers:
            page = f.original_pages[i]
            done = torch.cuda.Event()
            with torch.cuda.stream(forward):
                torch.cuda._sleep(500_000_000)
                f.pool.raw[page * n : (page + 1) * n].copy_(payloads[i])
                done.record()
            f.alloc.set_latest_forward_done_event(done)
            f.alloc.set_inflight_forward(done, f.ids[i : i + 1])
            events.append(done)
        with torch.cuda.stream(schedule):
            self.assertFalse(events[0].query())
            early = f.alloc.flush_opportunistic()
            self.assertFalse(events[0].query(), "the forwards ended before the flush")
            late = f.alloc.flush_for_allocation()
            self.assertFalse(
                events[0].query(), "the urgent relief blocked the host on the forward"
            )
        torch.cuda.synchronize()
        f.raw = expected
        self._verify(f, set(range(8, 16)))
        self.assertEqual((early, late), (0, 8))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_urgent_flush_waits_only_for_writers_of_its_sources(self):
        # A running forward writes one slot that stays put and reads the slots
        # the urgent flush moves. The copies must not wait for it; the slots the
        # frontier retreats past stay readable until it ends, so taking that
        # space back and overwriting it waits for the forward on the GPU.
        f = self._fixture("cuda")
        f.alloc.free(f.ids[8:16])
        n = f.spec.entry_bytes()
        moved = list(range(f.count - 8, f.count))
        vacated = [f.original_pages[i] for i in moved]
        kept = f.original_pages[0]
        expected = f.raw.clone()
        expected[kept * n : (kept + 1) * n] = 100
        payload = torch.full((n,), 100, dtype=torch.uint8, device="cuda")
        seen = torch.empty((len(vacated), n), dtype=torch.uint8, device="cuda")
        schedule = torch.cuda.Stream()
        forward = torch.cuda.Stream()
        with torch.cuda.stream(schedule):
            # Loading a kernel module and growing the caching allocator both wait
            # for every stream; warm both first, as a running scheduler has.
            f.alloc.prewarm_compaction()
            f.alloc.ids.translate(f.ids[:1])
            f.alloc.ids.translate(f.ids[1:2])
            f.alloc._to_device([0])
            torch.empty(1, device="cuda")
            torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        f.move.reset_mock()  # the warm-up copies the sink onto itself
        done = torch.cuda.Event()
        with torch.cuda.stream(forward):
            torch.cuda._sleep(500_000_000)
            f.pool.raw[kept * n : (kept + 1) * n].copy_(payload)
            for k, page in enumerate(vacated):
                seen[k].copy_(f.pool.raw[page * n : (page + 1) * n])
            done.record()
        f.alloc.set_latest_forward_done_event(done)
        f.alloc.set_inflight_forward(done, f.ids[:1])
        with torch.cuda.stream(schedule):
            self.assertEqual(f.alloc.flush_for_allocation(), 8)
            copied = torch.cuda.Event()
            copied.record()
            # The frontier retreated past the vacated slots; take them back.
            retaken = f.alloc.take_physical(8)
            f.pool.raw[min(vacated) * n : (max(vacated) + 1) * n].fill_(7)
        copied.synchronize()
        self.assertFalse(
            done.query(), "the copies waited for a forward that wrote none"
        )
        torch.cuda.synchronize()
        self.assertEqual(sorted(retaken.tolist()), sorted(vacated))
        self.assertEqual(max(vacated) - min(vacated) + 1, len(vacated))
        reads = torch.stack([f.raw[p * n : (p + 1) * n] for p in vacated])
        self.assertTrue(
            torch.equal(seen.cpu(), reads), "a vacated slot was overwritten"
        )
        raw = f.pool.raw.cpu()
        self.assertTrue(
            torch.equal(
                raw[kept * n : (kept + 1) * n], expected[kept * n : (kept + 1) * n]
            )
        )
        for i in moved:
            page = int(f.alloc.ids.virtual_to_physical[f.ids[i]])
            old = f.original_pages[i]
            self.assertTrue(
                torch.equal(
                    raw[page * n : (page + 1) * n], f.raw[old * n : (old + 1) * n]
                ),
                f"slot {i} lost its bytes",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_pending_read_sources_drain_with_closed_gate(self):
        f = self._fixture("cuda")
        f.alloc.free(f.ids[8:16])
        self.assertEqual(f.alloc.flush_opportunistic(), 8)
        # Mamba opportunistic moves occur at a quiescent write boundary. A
        # delayed reader is the separate source-reuse contract exercised here.
        old_sources = f.original_pages[-8:]
        n = f.spec.entry_bytes()
        observed = torch.empty((8, n), dtype=torch.uint8, device="cuda")
        expected = torch.stack([f.raw[p * n : (p + 1) * n] for p in old_sources])
        stream = torch.cuda.Stream()
        done = torch.cuda.Event()
        # Seed the pending state after a real copy. Some drivers serialize the
        # flush's D2H hole snapshot with other streams; that cannot exercise
        # pending routing end-to-end. CPU event tests cover that routing. Here
        # an actual delayed reader validates the CUDA completion/drain boundary.
        sources = f.alloc._free_phys_pages
        self.assertEqual(set(sources.cpu().tolist()), set(old_sources))
        f.alloc.set_latest_forward_done_event(done)
        f.alloc._free_phys_pages = sources[:0]
        f.alloc._pending_reuse[done] = (list(old_sources), [sources])
        f.alloc._pending_reuse_pages_cpu.update(old_sources)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            for i, p in enumerate(old_sources):
                observed[i].copy_(f.pool.raw[p * n : (p + 1) * n])
            done.record()
        self.assertFalse(done.query())
        f.alloc.pin_sources = (lambda: True,)
        calls = f.move.call_count
        self.assertEqual(f.alloc.flush_opportunistic(), 0)
        self.assertFalse(done.query(), "pending retention was not observed")
        self.assertEqual(f.alloc._pending_reuse_pages_cpu, set(old_sources))
        self.assertEqual(f.alloc._free_phys_pages.numel(), 0)
        done.synchronize()
        self.assertTrue(torch.equal(observed.cpu(), expected))
        # The reader's result is processed: its forward retires.
        f.alloc.retire_forward(done)
        self.assertEqual(f.alloc.flush_opportunistic(), 0)
        self.assertFalse(f.alloc._pending_reuse)
        self.assertFalse(f.alloc._pending_reuse_pages_cpu)
        self.assertEqual(set(f.alloc._free_phys_pages.cpu().tolist()), set(old_sources))
        self.assertEqual(f.alloc.flush_opportunistic(), 0)
        self.assertEqual(f.alloc._free_phys_pages.numel(), 8)
        self.assertEqual(f.move.call_count, calls)
        self._verify(f, set(range(8, 16)))
        reused = f.alloc.alloc(1).clone()
        physical = int(f.alloc.ids.virtual_to_physical[reused][0])
        self.assertIn(physical, old_sources)
        f.pool.raw[physical * n : (physical + 1) * n].fill_(77)
        # The recycled virtual ID can be one of the freed IDs; verify only the
        # original survivors and their relocated bytes after overwriting src.
        for i, old in enumerate(f.original_pages):
            if 8 <= i < 16:
                continue
            current = int(f.alloc.ids.virtual_to_physical[f.ids[i]])
            self.assertTrue(
                torch.equal(
                    f.pool.raw[current * n : (current + 1) * n].cpu(),
                    f.raw[old * n : (old + 1) * n],
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_graph_reads_relocated_state(self):
        f = self._fixture("cuda")
        # Graph reads the same stable views through the updated v2p map.
        observed = torch.empty_like(f.kv.mamba_cache.temporal[:, f.original_pages])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            observed.copy_(
                f.kv.mamba_cache.temporal[:, f.alloc.ids.virtual_to_physical[f.ids]]
            )
        expected = f.kv.mamba_cache.temporal[
            :, f.alloc.ids.virtual_to_physical[f.ids]
        ].clone()
        f.alloc.free(f.ids[8:16])
        self.assertEqual(f.alloc.flush_for_allocation(), 8)
        # Freed IDs are invalid: only replay live IDs, preserving captured shape.
        f.ids[8:16].copy_(f.ids[:8])
        expected[:, 8:16].copy_(expected[:, :8])
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(observed, expected))


class _Event:
    """A forward's done event, fired by the test."""

    def __init__(self):
        self.fired = False

    def query(self):
        return self.fired


class TestForwardWriteSets(CustomTestCase):
    def _allocator(self):
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(chain="mamba", full_layer_ids=(0,), state_layer_ids=(1,))
        )
        return bundle.token_to_kv_pool_allocator

    def test_state_band_records_every_slot_the_forward_writes(self):
        # State kernels write the working slots of the batch's rows in place,
        # plus its tracking and copy-on-write slots; compaction must see them.
        allocator = self._allocator()
        storage = allocator.storage
        storage.requests.req_index_to_mamba_index_mapping[torch.tensor([1, 2])] = (
            torch.tensor([5, 6], dtype=torch.int32)
        )
        tokens = torch.tensor([10, 11])
        storage.record_forward(
            _Event(),
            written_tokens=tokens,
            token_writes_unknown=False,
            req_pool_indices=torch.tensor([1, 2]),
            state_slots=(torch.tensor([7, -1]), None),
        )
        state = allocator.mamba_allocator._inflight_forwards
        self.assertEqual(state[-1][1].tolist(), [5, 6, 7, -1])
        owner = allocator.full_attn_allocator._inflight_forwards
        self.assertIs(owner[-1][1], tokens)

    def test_host_copies_are_recorded_on_the_members_they_touch(self):
        # A submitted host copy names its rows per sub-pool view; a move then
        # waits it, or releases what it reads, only for those rows.
        allocator = self._allocator()
        storage = allocator.storage
        owner_ids, state_slots = torch.tensor([3, 4]), torch.tensor([1])
        transfers = [
            SimpleNamespace(device_pool=storage.view(name), device_indices=ids)
            for name, ids in (
                (allocator.roles.owner, owner_ids),
                (allocator.roles.state, state_slots),
            )
        ]
        # A fused draft's view is not a sub-pool: its rows ride in the target's.
        transfers.append(
            SimpleNamespace(device_pool=object(), device_indices=torch.tensor([9]))
        )
        done = _Event()
        storage.record_host_transfer(done, transfers, writes=True)
        for band, ids in (
            (allocator.full_attn_allocator, owner_ids),
            (allocator.mamba_allocator, state_slots),
        ):
            ((signal, recorded, writes, release, unit_tokens),) = band._transfers
            self.assertIs(signal, done)
            self.assertIs(recorded, ids)
            self.assertTrue(writes)
            self.assertIsNone(release)
            self.assertIsNone(unit_tokens)  # controller ids, translated when read
        done.fired = True
        allocator.full_attn_allocator._prune_transfers()
        self.assertEqual(allocator.full_attn_allocator._transfers, [])

    def test_unknown_token_writes_hold_non_urgent_moves(self):
        # A pipeline stage's verify forward produces its own write slots, so
        # nothing may move until it ends, even with a newer forward recorded.
        allocator = self._allocator()
        owner = allocator.full_attn_allocator
        ids = owner.alloc(4)
        owner.free(ids[1:3])
        unknown, known = _Event(), _Event()
        for event, tokens in ((unknown, None), (known, ids[3:])):
            allocator.storage.record_forward(
                event,
                written_tokens=tokens,
                token_writes_unknown=tokens is None,
                req_pool_indices=torch.tensor([0]),
                state_slots=(),
            )
        self.assertEqual(owner.flush_opportunistic(), 0)
        # Retiring the newer forward retires the older one with it.
        unknown.fired = known.fired = True
        allocator.storage.retire_forward(known)
        self.assertGreater(owner.flush_opportunistic(), 0)


if __name__ == "__main__":
    unittest.main()
