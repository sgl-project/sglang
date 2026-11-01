"""A float band's relocation waits, on the GPU, only for a running forward
that writes a moved page, and returns to the host while that forward still
runs; pages it vacates are fenced against later writes while a forward may
still read them."""

import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
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

register_cuda_ci(est_time=8, stage="extra-a", runner_config="1-gpu-small")

_ROLES = ChainRoles(owner="full", window="swa", state="mamba")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFloatRelocationAgainstARunningForward(CustomTestCase):
    def _float(self):
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
        # The stand-in views follow unit moves with their own id buffers.
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
        flt = alloc.swa_attn_allocator
        ids = alloc.alloc(12)
        self.assertIsNotNone(ids)
        alloc.free(ids[:4])
        live = ids[4:]
        epp = flt.entry_bytes_per_page
        gap_low, _ = flt._gap_pages()
        ask = (gap_low + 2) * epp

        schedule, forward = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(schedule):
            # A running scheduler has loaded these kernels and pinned buffers.
            flt.prewarm_compaction()
            flt._to_device([0])
            # Every specialization the write-set translation is launched with.
            flt.ids.translate(live[:1])
            flt.ids.translate(live[1:2])
            flt.ids.translate(live)
            torch.empty(1, device="cuda")
            torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        return pool, flt, live, epp, ask, schedule, forward

    def test_float_move_waits_on_the_gpu_not_the_host(self):
        """REGRESSION: a float move waited for the forward before reading its
        holes and built its page lists with pageable copies, so the scheduler
        thread stalled until the forward ended. Every live float page is
        rewritten by a delayed forward; the moved pages must carry those
        bytes, and the move must return before the forward does."""
        pool, flt, live, epp, ask, schedule, forward = self._float()
        pages = flt.ids.virtual_to_physical[live].tolist()
        done = torch.cuda.Event()
        with torch.cuda.stream(forward):
            torch.cuda._sleep(500_000_000)
            for k, page in enumerate(pages):
                pool.raw[page * epp : (page + 1) * epp].fill_(100 + k)
            done.record()
        flt.set_latest_forward_done_event(done)
        flt.set_inflight_forward(done, live)

        with torch.cuda.stream(schedule):
            opened = open_float_side(flt, "low", ask)
            self.assertFalse(done.query(), "the move blocked the host on the forward")
        torch.cuda.synchronize()
        self.assertGreaterEqual(opened, ask)
        moved = flt.ids.virtual_to_physical[live].tolist()
        self.assertNotEqual(moved, pages, "the geometry asked for no relocation")
        raw = pool.raw.cpu()
        for k, page in enumerate(moved):
            self.assertTrue(
                bool((raw[page * epp : (page + 1) * epp] == 100 + k).all()),
                f"live page {k} lost the forward's write",
            )

    def test_pages_a_forward_only_reads_move_at_once_and_stay_fenced(self):
        """A move waits only for a forward that writes a moved page. A delayed
        forward that only reads the live pages must not hold the move: the
        copies run while it still runs. The pages it vacates stay readable:
        a take that lands on them, and a write into them, wait that forward."""
        pool, flt, live, epp, ask, schedule, forward = self._float()
        pages = flt.ids.virtual_to_physical[live].tolist()
        original = pool.raw.clone()
        seen = torch.empty((len(pages), epp), dtype=torch.uint8, device="cuda")
        with torch.cuda.stream(forward):
            torch.cuda._sleep(500_000_000)
            for k, page in enumerate(pages):
                seen[k].copy_(pool.raw[page * epp : (page + 1) * epp])
            done = torch.cuda.Event()
            done.record()
        flt.set_latest_forward_done_event(done)
        flt.set_inflight_forward(done, None)

        with torch.cuda.stream(schedule):
            opened = open_float_side(flt, "low", ask)
            copied = torch.cuda.Event()
            copied.record()
        copied.synchronize()
        self.assertFalse(done.query(), "the move waited for a reading forward")
        moved = flt.ids.virtual_to_physical[live].tolist()
        vacated = sorted(set(pages) - set(moved))
        self.assertTrue(vacated, "the geometry vacated no page")
        with torch.cuda.stream(schedule):
            # The vacated pages now lie in the gap; overwrite them, as a later
            # hand-out of that space would, after waiting their fence.
            first, end = vacated[0], vacated[-1] + 1
            flt._wait_fences(first, end)
            pool.raw[first * epp : end * epp].fill_(7)
        self.assertFalse(done.query(), "the forward ended before the overwrite")
        torch.cuda.synchronize()
        self.assertGreaterEqual(opened, ask)
        expected = torch.stack(
            [original[p * epp : (p + 1) * epp] for p in pages]
        ).cuda()
        self.assertTrue(torch.equal(seen, expected), "a vacated page was overwritten")
        raw = pool.raw.cpu()
        for k, page in enumerate(moved):
            self.assertTrue(
                torch.equal(
                    raw[page * epp : (page + 1) * epp],
                    original[pages[k] * epp : (pages[k] + 1) * epp].cpu(),
                ),
                f"live page {k} lost its bytes",
            )


if __name__ == "__main__":
    unittest.main()
