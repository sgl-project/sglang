"""A float band's relocation is ordered after the running forward on the GPU,
and returns to the host while that forward still runs."""

import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
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

register_cuda_ci(est_time=8, stage="extra-a", runner_config="1-gpu-small")

_ROLES = ChainRoles(owner="full", window="swa", state="mamba")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFloatRelocationAgainstARunningForward(CustomTestCase):
    def test_make_room_waits_on_the_gpu_not_the_host(self):
        """REGRESSION: `make_room` waited for the forward before reading its
        holes and built its page lists with pageable copies, so the scheduler
        thread stalled until the forward ended. Every live float page is
        rewritten by a delayed forward; the moved pages must carry those
        bytes, and `make_room` must return before the forward does."""
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
            torch.empty(1, device="cuda")
            torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()

        pages = flt.ids.virtual_to_physical[live].tolist()
        done = torch.cuda.Event()
        with torch.cuda.stream(forward):
            torch.cuda._sleep(500_000_000)
            for k, page in enumerate(pages):
                pool.raw[page * epp : (page + 1) * epp].fill_(100 + k)
            done.record()
        flt.set_latest_forward_done_event(done)
        flt.set_inflight_forward(done, None)

        with torch.cuda.stream(schedule):
            opened = flt.make_room(side="low", min_bytes=ask)
            self.assertFalse(done.query(), "make_room blocked the host on the forward")
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


if __name__ == "__main__":
    unittest.main()
