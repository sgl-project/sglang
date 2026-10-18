"""Physical Mamba relocation preserves payloads without a main-state gather."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_unified_mamba_views import _make_pool

from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedMambaPool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-small")


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
        kv = UnifiedMambaPool(
            unified_buffer=pool,
            sub_pool_name="mamba",
            spec_state_size=0,
            mamba_layer_ids=[1, 3],
        )
        alloc = MultiEndedAllocator(
            unified_buffer=pool,
            sub_pool_name="mamba",
            kvcache=kv,
            device=device,
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        full = MultiEndedAllocator(
            unified_buffer=pool,
            sub_pool_name="full",
            kvcache=Mock(),
            device=device,
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        alloc.bind_peer(full)
        full.bind_peer(alloc)
        full_ids = full.alloc(4).clone()
        full_pages = full.virtual_to_physical[full_ids].cpu().tolist()
        ids = alloc.alloc(count).clone()
        ids = ids[torch.argsort(alloc.virtual_to_physical[ids])]
        original_pages = alloc.virtual_to_physical[ids].cpu().tolist()
        short = ShortConvPool(
            size=pool.max_slots("mamba") - 1,
            state_shape=(32, 3),
            layer_ids=[1, 3],
            dtype=torch.bfloat16,
            device=device,
        )
        ngram = NGramPool(
            size=pool.max_slots("mamba") - 1,
            context_len=8,
            eos_token_id=0,
            device=device,
        )
        kv.register_slot_state(short)
        kv.register_slot_state(ngram)
        # Independent, byte-varying reference. Keep floating values finite for COW.
        gen = torch.Generator().manual_seed(42137)
        raw = torch.randint(0, 64, pool._raw.shape, dtype=torch.uint8, generator=gen)
        pool._raw.copy_(raw)
        short.conv_state.copy_(
            torch.arange(short.conv_state.numel()).reshape(short.conv_state.shape) % 127
        )
        ngram.context.copy_(
            torch.arange(ngram.context.numel()).reshape(ngram.context.shape)
        )
        short_ref = short.conv_state.cpu().clone()
        ngram_ref = ngram.context.cpu().clone()
        move = Mock(wraps=kv.move_kv_cache)
        kv.move_kv_cache = move
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
            short=short,
            ngram=ngram,
            short_ref=short_ref,
            ngram_ref=ngram_ref,
            move=move,
            count=count,
        )

    def _verify(self, f, freed):
        raw = f.pool._raw.cpu()
        current = f.alloc.virtual_to_physical[f.ids].cpu().tolist()
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
            self.assertEqual(int(f.alloc.physical_to_virtual[new]), v)
            self.assertTrue(
                torch.equal(f.short.conv_state[:, new].cpu(), f.short_ref[:, old])
            )
            self.assertTrue(torch.equal(f.ngram.context[new].cpu(), f.ngram_ref[old]))
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
            # Bound this fixture, including its small real PLE siblings. Arbitrary
            # sibling geometries retain their own gather scratch requirements.
            self.assertLess(extra_peak, f.spec.entry_bytes())
        self._verify(f, set(range(holes, 2 * holes)))
        self.assertEqual(sum(c.args[0].numel() for c in f.move.call_args_list), holes)
        self.assertEqual(f.alloc.flush_for_allocation(), 0)
        # Ordinary COW still copies the real sibling pools after relocation.
        cow_id = f.alloc.alloc(1).clone()
        dst = f.alloc.virtual_to_physical[cow_id]
        src = f.alloc.virtual_to_physical[f.ids[-1:]]
        f.kv.copy_from(src, dst)
        for tensor in [
            *f.kv.mamba_cache.conv,
            f.kv.mamba_cache.temporal,
            f.short.conv_state,
        ]:
            self.assertTrue(torch.equal(tensor[:, src], tensor[:, dst]))
        self.assertTrue(torch.equal(f.ngram.context[src], f.ngram.context[dst]))
        before_empty = f.pool._raw.clone()
        f.kv.move_kv_cache(f.ids[:0], f.ids[:0])
        self.assertTrue(torch.equal(f.pool._raw, before_empty))

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
                        host.copy_(f.pool._raw, non_blocking=True)
                    else:
                        f.pool._raw.copy_(host, non_blocking=True)
                    done.record()
                self.assertFalse(done.query())
                if writer == "forward":
                    f.alloc.set_latest_forward_done_event(done)
                else:
                    f.alloc.set_hicache_transfer_done_event(writer, done)
                self.assertEqual(f.alloc.flush_for_allocation(), 8)
                torch.cuda.synchronize()
                self.assertTrue(done.query())
                if writer == "write":
                    self.assertTrue(torch.equal(host, expected))
                else:
                    f.raw = expected
                self._verify(f, set(range(8, 16)))

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
        f.alloc._free_phys_pages = sources[:0]
        f.alloc._pending_reuse[done] = (list(old_sources), [sources])
        f.alloc._pending_reuse_pages_cpu.update(old_sources)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            for i, p in enumerate(old_sources):
                observed[i].copy_(f.pool._raw[p * n : (p + 1) * n])
            done.record()
        self.assertFalse(done.query())
        f.alloc.host_transfer_move_gate = lambda: False
        calls = f.move.call_count
        self.assertEqual(f.alloc.flush_opportunistic(), 0)
        self.assertFalse(done.query(), "pending retention was not observed")
        self.assertEqual(f.alloc._pending_reuse_pages_cpu, set(old_sources))
        self.assertEqual(f.alloc._free_phys_pages.numel(), 0)
        done.synchronize()
        self.assertTrue(torch.equal(observed.cpu(), expected))
        self.assertEqual(f.alloc.flush_opportunistic(), 0)
        self.assertFalse(f.alloc._pending_reuse)
        self.assertFalse(f.alloc._pending_reuse_pages_cpu)
        self.assertEqual(set(f.alloc._free_phys_pages.cpu().tolist()), set(old_sources))
        self.assertEqual(f.alloc.flush_opportunistic(), 0)
        self.assertEqual(f.alloc._free_phys_pages.numel(), 8)
        self.assertEqual(f.move.call_count, calls)
        self._verify(f, set(range(8, 16)))
        reused = f.alloc.alloc(1).clone()
        physical = int(f.alloc.virtual_to_physical[reused][0])
        self.assertIn(physical, old_sources)
        f.pool._raw[physical * n : (physical + 1) * n].fill_(77)
        # The recycled virtual ID can be one of the freed IDs; verify only the
        # original survivors and their relocated bytes after overwriting src.
        for i, old in enumerate(f.original_pages):
            if 8 <= i < 16:
                continue
            current = int(f.alloc.virtual_to_physical[f.ids[i]])
            self.assertTrue(
                torch.equal(
                    f.pool._raw[current * n : (current + 1) * n].cpu(),
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
                f.kv.mamba_cache.temporal[:, f.alloc.virtual_to_physical[f.ids]]
            )
        expected = f.kv.mamba_cache.temporal[
            :, f.alloc.virtual_to_physical[f.ids]
        ].clone()
        f.alloc.free(f.ids[8:16])
        self.assertEqual(f.alloc.flush_for_allocation(), 8)
        # Freed IDs are invalid: only replay live IDs, preserving captured shape.
        f.ids[8:16].copy_(f.ids[:8])
        expected[:, 8:16].copy_(expected[:, :8])
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(observed, expected))


if __name__ == "__main__":
    unittest.main()
