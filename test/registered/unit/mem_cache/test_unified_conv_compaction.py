"""Conv-only compaction in the real full/SWA/state unified pool stack."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_swa_pools
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_cuda_ci(est_time=13, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=13, suite="stage-b-test-1-gpu-small-amd")


def make_conv_bundle(device, page_size=1, lazy=True, state_slots=40):
    # Six heterogeneous Conv streams and empty temporal state, like Inkling.
    shapes = [(64, 512), (64, 512), (32, 512), (32, 512), (16, 512), (16, 512)]
    if device == "cpu":
        shapes = [(2, 64), (2, 64), (1, 64), (1, 64), (1, 32), (1, 32)]
    params = SimpleNamespace(
        shape=SimpleNamespace(conv=shapes, temporal=(0, 0, 0)),
        dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.bfloat16),
        layers=[1, 3],
    )
    return init_unified_mamba_swa_pools(
        device=device,
        kv_cache_dtype=torch.bfloat16,
        head_num=1,
        head_dim=64,
        v_head_dim=64,
        swa_head_num=1,
        swa_head_dim=64,
        swa_v_head_dim=64,
        page_size=page_size,
        start_layer=0,
        end_layer=4,
        swa_attention_layer_ids=[2],
        full_attention_layer_ids=[0],
        mamba_layer_ids=[1, 3],
        mamba2_cache_params=params,
        full_max_total_num_tokens=64,
        swa_max_total_num_tokens=64,
        max_mamba_cache_size=state_slots,
        model_context_len=64,
        extra_max_context_len=0,
        max_num_reqs=4,
        enable_memory_saver=False,
        enable_mamba_extra_buffer=False,
        enable_mamba_extra_buffer_lazy=False,
        disable_overlap_schedule=False,
        need_sort=False,
        lazy_compaction=lazy,
        sliding_window_size=16,
    )


class TestConvCompaction(CustomTestCase):
    def _exercise(self, device, page_size, mode):
        bundle = make_conv_bundle(device, page_size, lazy=mode != "eager")
        pool = bundle.unified_memory_pool
        composite = bundle.token_to_kv_pool_allocator
        state = bundle.req_to_token_pool.mamba_pool
        alloc = composite.mamba_allocator
        self.assertEqual(state.mamba_cache.temporal.numel(), 0)
        self.assertFalse(state._should_fuse_slot_ops())
        self.assertIs(alloc.high_peer, composite.swa_attn_allocator)
        self.assertIs(
            composite.swa_attn_allocator.high_peer, composite.full_attn_allocator
        )
        ids = alloc.alloc(32).clone()
        tokens = composite.alloc(4 * page_size).clone()
        allocators = [
            alloc,
            composite.swa_attn_allocator,
            composite.full_attn_allocator,
        ]
        live_ids = [ids, tokens, tokens]
        before_maps = [
            a.translate_kv_loc(v).cpu().clone() for a, v in zip(allocators, live_ids)
        ]
        reference = torch.randint(
            0,
            64,
            pool._raw.shape,
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(781),
        )
        pool._raw.copy_(reference)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
        alloc.free(ids[8:16])
        if mode == "urgent":
            self.assertEqual(alloc.flush_for_allocation(), 8)
        elif mode == "opportunistic":
            self.assertEqual(alloc.flush_opportunistic(), 8)
        if device == "cuda":
            torch.cuda.synchronize()
            # The old gather of eight slots from the largest Conv exceeds this.
            self.assertLess(torch.cuda.max_memory_allocated() - before, 128 * 1024)
        # Consume space reclaimed by Conv compaction from the high end. This
        # must relocate the live SWA float toward the newly freed low band.
        full = composite.full_attn_allocator
        need = (full._current_gap_bytes() // full.entry_bytes_per_page + 1) * page_size
        self.assertIsNotNone(full.alloc(need))
        self.assertFalse(
            torch.equal(
                composite.swa_attn_allocator.translate_kv_loc(tokens).cpu(),
                before_maps[1],
            )
        )
        self.assertEqual(composite.verify_byte_accounting(), [])
        actual = pool._raw.cpu()
        moved = 0
        for group, (a, virtual, old) in enumerate(
            zip(allocators, live_ids, before_maps)
        ):
            current = a.translate_kv_loc(virtual).cpu()
            for i, (v, src, dst) in enumerate(
                zip(virtual.cpu().tolist(), old.tolist(), current.tolist())
            ):
                if group == 0 and 8 <= i < 16:
                    self.assertEqual(int(a.virtual_to_physical[v]), -1)
                    continue
                if group == 0:
                    moved += src != dst
                elif group == 2:
                    self.assertEqual(src, dst)
                self.assertEqual(
                    int(a.physical_to_virtual[dst // a.page_size]), v // a.page_size
                )
                n = a.entry_bytes
                self.assertTrue(
                    torch.equal(
                        actual[dst * n : (dst + 1) * n],
                        reference[src * n : (src + 1) * n],
                    )
                )
        self.assertEqual(moved, 8)
        self.assertTrue(
            torch.equal(actual[: alloc.entry_bytes], reference[: alloc.entry_bytes])
        )
        # Ordinary COW remains available for Conv-only states after compaction.
        new = alloc.alloc(1).clone()
        src = alloc.virtual_to_physical[ids[-1:]]
        dst = alloc.virtual_to_physical[new]
        state.copy_from(src, dst)
        for tensor in state.mamba_cache.conv:
            self.assertTrue(torch.equal(tensor[:, src], tensor[:, dst]))

    def test_cpu_conv_only_tri_pool(self):
        for ps in (1, 4):
            for mode in ("eager", "urgent", "opportunistic"):
                with self.subTest(page_size=ps, mode=mode):
                    self._exercise("cpu", ps, mode)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_conv_only_tri_pool(self):
        for ps in (1, 4):
            for mode in ("eager", "urgent", "opportunistic"):
                with self.subTest(page_size=ps, mode=mode):
                    self._exercise("cuda", ps, mode)


if __name__ == "__main__":
    unittest.main()
