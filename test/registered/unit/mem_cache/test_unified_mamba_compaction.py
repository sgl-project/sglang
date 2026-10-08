"""Mamba compaction preserves state without a batch-sized CUDA gather."""

import unittest
from unittest.mock import Mock

import torch
from test_unified_mamba_views import _make_pool

from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.mem_cache.unified_memory_pool import UnifiedMambaPool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-small")


class TestMambaCompaction(unittest.TestCase):
    def _compact(self, device, temporal_shape, count):
        pool, spec = _make_pool(
            mamba_layer_num=2,
            conv_state_shapes=[(8, 3), (4, 2)],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=temporal_shape,
            temporal_dtype=torch.float32,
            want_slots=count + 8,
            device=device,
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
            lazy_compaction=True,
        )
        ids = alloc.alloc(count).clone()
        ids = ids[torch.argsort(alloc.virtual_to_physical[ids], descending=True)]
        state = [*kv.mamba_cache.conv, kv.mamba_cache.temporal]
        for i, p in enumerate(alloc.virtual_to_physical[ids].tolist()):
            for tensor_idx, tensor in enumerate(state):
                for layer in range(2):
                    tensor[layer, p].fill_(i + 1 + tensor_idx * 64 + layer * 256)
        sibling = Mock()
        kv.register_slot_state(sibling)
        alloc.free(ids[: count // 2])
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
        moves = alloc.flush_for_allocation()
        if device == "cuda":
            torch.cuda.synchronize()
            extra_peak = torch.cuda.max_memory_allocated() - before
            # Old advanced indexing gathers count/2 full temporal states.
            self.assertLess(extra_peak, spec.entry_bytes())
        self.assertEqual(moves, count // 2)
        sibling.copy_slots.assert_called_once()
        self.assertEqual(sibling.copy_slots.call_args.args[0].numel(), moves)
        for i, v in enumerate(ids[count // 2 :].tolist(), count // 2):
            p = int(alloc.virtual_to_physical[v])
            self.assertEqual(int(alloc.physical_to_virtual[p]), v)
            for tensor_idx, tensor in enumerate(state):
                for layer in range(2):
                    self.assertTrue(
                        bool(
                            (
                                tensor[layer, p]
                                == i + 1 + tensor_idx * 64 + layer * 256
                            ).all()
                        )
                    )
        self.assertEqual(alloc.live_page_count, count // 2)
        self.assertEqual(alloc.flush_for_allocation(), 0)

    def test_cpu_state_and_siblings(self):
        self._compact("cpu", (2, 4, 8), 16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_peak_and_state(self):
        self._compact("cuda", (16, 64, 128), 32)


if __name__ == "__main__":
    unittest.main()
