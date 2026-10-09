"""Mamba compaction preserves state without a batch-sized CUDA gather."""

import unittest
from unittest.mock import Mock

import torch
from test_unified_mamba_views import _make_pool

from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.mem_cache.unified_memory_pool import UnifiedMambaPool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-small")


class TestMambaCompaction(CustomTestCase):
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
        full = MultiEndedAllocator(
            unified_buffer=pool,
            sub_pool_name="full",
            kvcache=Mock(),
            device=device,
            is_id_owner=True,
            lazy_compaction=True,
        )
        alloc.bind_peer(full)
        full.bind_peer(alloc)
        full_ids = full.alloc(4)
        full_pages = full.virtual_to_physical[full_ids]
        full_views = pool._raw[: full.num_pages * full.entry_bytes].view(
            full.num_pages, full.entry_bytes
        )
        full_views[full_pages] = 37
        pool._raw[: spec.entry_bytes()].fill_(91)
        ids = alloc.alloc(count).clone()
        ids = ids[torch.argsort(alloc.virtual_to_physical[ids], descending=True)]
        state = [*kv.mamba_cache.conv, kv.mamba_cache.temporal]
        original_pages = alloc.virtual_to_physical[ids].clone()
        sibling_values = torch.zeros(pool.max_slots("mamba"), device=device)
        for i, p in enumerate(original_pages.tolist()):
            sibling_values[p] = i + 100
            for tensor_idx, tensor in enumerate(state):
                for layer in range(2):
                    tensor[layer, p].fill_(i + 1 + tensor_idx * 64 + layer * 256)

        def copy_sibling(src, dst):
            sibling_values[dst] = sibling_values[src]

        sibling = Mock()
        sibling.copy_slots.side_effect = copy_sibling
        kv.register_slot_state(sibling)
        # Keep the high and low live ends, freeing interior slots only. This
        # checks both relocated survivors and live slots untouched by the move.
        holes = count // 4
        alloc.free(ids[holes : 2 * holes])
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
        moves = alloc.flush_for_allocation()
        if device == "cuda":
            torch.cuda.synchronize()
            extra_peak = torch.cuda.max_memory_allocated() - before
            # Old advanced indexing gathers every moved temporal state.
            self.assertLess(extra_peak, spec.entry_bytes())
        self.assertEqual(moves, holes)
        sibling.copy_slots.assert_called_once()
        self.assertEqual(sibling.copy_slots.call_args.args[0].numel(), moves)
        copied_src, copied_dst = sibling.copy_slots.call_args.args
        expected_pairs = set()
        for i, v in enumerate(ids.tolist()):
            if holes <= i < 2 * holes:
                continue
            p = int(alloc.virtual_to_physical[v])
            old_p = int(original_pages[i])
            if old_p != p:
                expected_pairs.add((old_p, p))
            self.assertEqual(float(sibling_values[p]), i + 100)
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
        self.assertEqual(
            set(zip(copied_src.tolist(), copied_dst.tolist())), expected_pairs
        )
        self.assertEqual(len(expected_pairs), holes)
        self.assertTrue(bool((pool._raw[: spec.entry_bytes()] == 91).all()))
        self.assertTrue(bool((full_views[full_pages] == 37).all()))
        self.assertEqual(alloc.live_page_count, count - holes)
        self.assertEqual(alloc.flush_for_allocation(), 0)

        # Ordinary COW still uses the inherited copy_from path and carries the
        # same sibling state after physical compaction.
        cow_id = alloc.alloc(1).clone()
        cow_dst = alloc.virtual_to_physical[cow_id]
        cow_src = alloc.virtual_to_physical[ids[-1:]]
        kv.copy_from(cow_src, cow_dst)
        for tensor in state:
            self.assertTrue(torch.equal(tensor[:, cow_src], tensor[:, cow_dst]))
        self.assertTrue(torch.equal(sibling_values[cow_src], sibling_values[cow_dst]))

    def test_cpu_state_and_siblings(self):
        self._compact("cpu", (2, 4, 8), 16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_peak_and_state(self):
        self._compact("cuda", (16, 64, 128), 32)


if __name__ == "__main__":
    unittest.main()
