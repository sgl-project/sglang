import random
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _dsv4_native_full_regions,
)
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.dual_ended import DualEndedHostArena
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Fence:
    def __init__(self):
        self.done = False
        self.waits = 0

    def query(self):
        return self.done

    def synchronize(self):
        self.waits += 1
        self.done = True


class TestDualEndedArena(TestCase):
    def pool(self):
        # Match the native width ratios. This makes all eight alignment phases
        # observable without relying on an NPU driver in CI.
        return DualEndedHostArena(
            {"kv": (8, 1), "idx": (2, 1), "kv2": (4, 3), "idx2": (1, 3)},
            (8, 43),
            128,
            4,
            lambda n: torch.empty(n, dtype=torch.uint8),
        )

    def test_group_native_index_pages_without_copy(self):
        names = (
            (PoolName.DEEPSEEK_V4_C1, PoolName.DEEPSEEK_V4_C1_INDEXER),
            (PoolName.DEEPSEEK_V4_C2, PoolName.DEEPSEEK_V4_C2_INDEXER),
        )
        model = SimpleNamespace(
            start_layer=0,
            kv_pools={1: object(), 2: object()},
            index_pools={1: object(), 2: object()},
        )
        infos = {}
        for ratio, pair in enumerate(names, 1):
            for name, slots in zip(pair, (128 // ratio, 32)):
                buf = torch.zeros((16, slots, 1, 8), dtype=torch.bfloat16)
                infos[name] = SimpleNamespace(
                    layer_ids=(3,),
                    buffer_info=SimpleNamespace(layout="pa_nd", buffers=[buf]),
                )
        regions = _dsv4_native_full_regions(model, infos, 128)
        self.assertEqual(len(regions), 4)
        for name, _, buffers, mapping in regions:
            source = infos[name].buffer_info.buffers[0]
            self.assertEqual(buffers[0].data_ptr(), source.data_ptr())
            self.assertEqual(buffers[0].numel(), source.numel())
            self.assertEqual(buffers[0].shape[1], 128 if name in names[0] else 64)
            self.assertEqual(mapping, {3: 0})
        infos[names[1][1]].buffer_info.buffers = [source.to(torch.uint8)]
        self.assertEqual(_dsv4_native_full_regions(model, infos, 128), [])

    def test_prepared_indices_and_native_host_view(self):
        arena = self.pool()
        f = arena.full.alloc(2, 4)
        pool = object.__new__(DeepSeekV4PagedHostPool)
        pool.slot_page_size = 4
        pool.host_region = arena.region("kv")
        pool.transfer_layout_domain = arena
        pool.device_buffers = [torch.zeros((2, 2, 1, 2), dtype=torch.bfloat16)]
        region = pool.host_region
        h, d = pool.prepare_transfer_indices(f, f, "kernel_ascend")
        self.assertIs(d, f)
        row = region.physical_rows(1)
        self.assertEqual(h[4:].tolist(), list(range(4 * row, 4 * row + 4)))
        pool._host_page_view(0)[row].fill_(1)
        self.assertTrue(
            torch.equal(region.layer_rows(0)[row].view(torch.bfloat16), torch.ones(4))
        )
        with self.assertRaises(ValueError):
            pool.prepare_transfer_indices(f[:1], f[:1], "kernel_ascend")

    def test_fractional_full_phases_survive_relocation(self):
        a = self.pool()
        f = a.full.alloc(a.full.available(), 1)
        for key in a._regions:
            if key is None:
                continue
            r = a.region(key)
            for i in range(r.layers):
                r.layer_rows(i)[r.physical_rows(f)] = (
                    torch.arange(len(f)) % 127 + i
                ).to(torch.uint8)[:, None]
        before = dict(a.sides["full"].logical_to_physical)
        a.full.free(f[1:20], 1)
        self.assertIsNotNone(a.swa.alloc(1, 1))
        self.assertGreater(a.moves, 0)
        moved = [
            p
            for p, pos in a.sides["full"].logical_to_physical.items()
            if before[p] != pos
        ]
        self.assertTrue(
            any(
                before[p] % 8 != a.sides["full"].logical_to_physical[p] % 8
                for p in moved
            )
        )
        for p in moved:
            for key, r in a._regions.items():
                if key is not None:
                    for i in range(r.layers):
                        self.assertTrue(
                            torch.all(
                                r.layer_rows(i)[r.physical_rows(p)] == p % 127 + i
                            )
                        )

    def test_fences_on_hole_and_delayed_release(self):
        a = self.pool()
        f = a.full.alloc(40, 4)
        s = a.swa.alloc(4, 4)
        a.swa.free(s[4:8], 4)
        logical = int(s[-4]) // 4
        old = a.sides["swa"].logical_to_physical[logical]
        fence = Fence()
        a.acquire_layout()
        self.assertEqual(a.region().physical_rows(logical), old * 43)
        a.release_layout(fence, ("load", 0))
        self.assertIsNotNone(a.full.alloc(a.full.available(), 4))
        self.assertGreater(fence.waits, 0)
        self.assertNotEqual(a.sides["swa"].logical_to_physical[logical], old)
        fence2 = Fence()
        a.acquire_layout()
        a.release_layout(fence2, ("write", 0))
        a.full.free(f[:4], 4)
        with self.assertRaises(ValueError):
            a.full.free(f[:4], 4)
        self.assertIn(int(f[0]) // 4, a.sides["full"].logical_to_physical)
        fence2.done = True
        a.full.available()
        self.assertNotIn(int(f[0]) // 4, a.sides["full"].logical_to_physical)

    def test_mixed_pressure_preserves_all_live_page_contents(self):
        arena = self.pool()
        rng = random.Random(20261123)
        live = {"full": {}, "swa": {}}
        for step in range(500):
            name = rng.choice(("full", "swa"))
            owner = getattr(arena, name)
            active = live[name]
            if active and rng.random() < 0.45:
                page = rng.choice(list(active))
                owner.free(torch.tensor([page]), 1)
                del active[page]
            else:
                slots = owner.alloc(rng.randrange(1, 6), 1)
                if slots is not None:
                    for page in slots.tolist():
                        self.assertNotIn(page, active)
                        active[page] = 1 + step % 191
                        for region in arena._regions.values():
                            if region.owner is owner:
                                row = region.physical_rows(page)
                                region.layer_rows(0)[row : row + region.layers].fill_(
                                    active[page]
                                )
            if step % 10 == 0 or step == 499:
                for region in arena._regions.values():
                    for page, value in live[region.owner.name].items():
                        row = region.physical_rows(page)
                        data = region.layer_rows(0)[row : row + region.layers]
                        self.assertTrue(torch.all(data == value))
                used = sum(
                    len(pages) * arena.sides[side].page_bytes
                    for side, pages in live.items()
                )
                self.assertEqual(arena.free_bytes(), arena._allocatable_bytes - used)
        self.assertGreater(arena.moves, 0)

    def test_metadata_allocation_failure_and_retry(self):
        a = self.pool()
        before = a.full.available()
        with patch.object(torch, "tensor", side_effect=MemoryError("test")):
            with self.assertRaises(MemoryError):
                a.full.alloc(3, 4)
        self.assertEqual(a.full.available(), before)
        self.assertEqual(a.full.alloc(3, 4).tolist(), list(range(12)))


if __name__ == "__main__":
    main()
