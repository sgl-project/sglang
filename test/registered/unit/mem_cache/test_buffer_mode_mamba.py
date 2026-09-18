"""CPU regression coverage for buffer-mode Mamba transfer ownership."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    InitLoadBackParams,
    InsertResult,
)
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    _StagedPrefetch,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StagedPrefetchPlan
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBufferModeMamba(unittest.TestCase):
    def test_swa_repair_defers_redundant_mamba_free_until_ack(self):
        # FULL covers four tokens; an ancestor's SWA is gone, while the tail
        # still owns its checkpoint. H2D is queued and completes only below.
        cache = MagicMock()
        cache.page_size = 1
        cache.supports_swa.return_value = True
        cache.token_to_kv_pool_allocator.full_available_size.return_value = 4
        cache.token_to_kv_pool_allocator.full_to_swa_index_mapping = torch.tensor(
            [0, 0, 0, 23, 24]
        )
        cache.tree_core.empty_match_result.device_indices = torch.empty(
            0, dtype=torch.int64
        )
        cache.tree_core.swa_tombstone_ranges.return_value = [(0, 2)]
        cache.tree_core.attach_swa_window.return_value = []
        prefix = torch.arange(1, 5)
        cache.match_prefix.return_value = SimpleNamespace(
            device_indices=prefix, last_device_node=2
        )
        mamba = MagicMock()
        cache.components = {ComponentType.MAMBA: mamba}
        request_slot, canonical_slot, redundant_slot = 5, 7, 8
        state = torch.zeros(10, 2)
        expected = torch.tensor([11.0, 12.0])
        state[canonical_slot] = expected
        tail = SimpleNamespace(
            component_data={
                ComponentType.MAMBA: SimpleNamespace(
                    value=torch.tensor([canonical_slot])
                )
            }
        )
        # Use the real component's insertion decision for an existing state.
        component = SimpleNamespace(
            component_type=ComponentType.MAMBA, tree_core=MagicMock()
        )

        def insert(params):
            result = InsertResult(prefix_len=4)
            MambaComponent.commit_insert_component_data(
                component, tail, False, params, result, []
            )
            return result

        cache.insert.side_effect = insert
        cc = cache.cache_controller
        cc.prefetch_tokens_occupied = 4
        mamba_entry, swa_entry = MagicMock(), MagicMock()
        cc.mem_pool_host.entry_map = {
            PoolName.MAMBA: mamba_entry,
            PoolName.SWA: swa_entry,
        }
        node_copy = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([10]))
        swa_copy = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(20, 24))
        queued = []

        def load(*, host_indices, node_id, extra_pools):
            self.assertEqual(host_indices.numel(), 0)
            for transfer in extra_pools:
                if transfer.name == PoolName.SWA:
                    transfer.device_indices = torch.arange(30, 34)
                else:
                    if transfer.device_indices is None:
                        transfer.device_indices = torch.tensor([redundant_slot])
                    queued.append(transfer)
            return torch.empty(0, dtype=torch.int64)

        cc.load.side_effect = load
        request = CacheRequestHandle("swa-repair", 0)
        key = RadixKey(array("q", [1, 2, 3, 4]))
        req = SimpleNamespace(
            rid=request.rid,
            cache_request_handle=request,
            extra_key=None,
            cache_salt=None,
            prefix_indices=prefix,
            last_node=2,
            host_hit_length=0,
            swa_host_hit_length=4,
            mamba_host_hit_length=1,
            kv=SimpleNamespace(mamba_pool_idx=torch.tensor(request_slot)),
            staged_prefetch_plan=StagedPrefetchPlan(9, key, 4, 0, 4, 1),
        )
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline.reset()
        pipeline.staged_prefetches[request] = _StagedPrefetch(
            request=request,
            key_tokens=key.token_ids,
            extra_key=None,
            cache_salt=None,
            matched_len=0,
            num_tokens=4,
            occupied_tokens=4,
            host_indices=torch.arange(4),
            aux_xfers=[swa_copy, node_copy],
            hash_values=["a", "b", "c", "d"],
            operation_id=9,
        )
        loaded, last_node = pipeline.init_load_back(
            InitLoadBackParams(best_match_node=2, host_hit_length=0, req=req)
        )
        self.assertEqual(loaded.numel(), 0)
        self.assertEqual(last_node, 2)
        self.assertEqual(len(queued), 2)  # Both node and request H2D remain.
        mamba_entry.device_free_fn.assert_not_called()
        mamba_entry.host_pool.free.assert_not_called()
        self.assertEqual(
            tail.component_data[ComponentType.MAMBA].value.item(), canonical_slot
        )
        self.assertTrue(torch.equal(state[canonical_slot], expected))
        cache.tree_core.attach_swa_window.assert_called_once()

        # Simulate completion of the queued copies, then deliver their ack.
        for transfer in queued:
            state[transfer.device_indices] = expected
        self.assertTrue(torch.equal(state[request_slot], expected))
        self.assertTrue(pipeline.try_finish_load_back(-10))
        freed = mamba_entry.device_free_fn.call_args.args[0]
        self.assertEqual(freed.tolist(), [redundant_slot])
        mamba_entry.device_free_fn.assert_called_once()
        mamba_entry.host_pool.free.assert_called_once()
        self.assertTrue(torch.equal(state[canonical_slot], expected))
        self.assertEqual(cc.prefetch_tokens_occupied, 0)
        self.assertFalse(pipeline.try_finish_load_back(-10))
        mamba_entry.device_free_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
