"""Host reloads must preserve ID domains, allocation ownership, and stream order."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, MambaStateBufferInfo
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _split_hicache_size
from sglang.srt.mem_cache.kv.alloc.pricing import ends_compaction_allowed
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    chain_snapshot,
    pin_every_band,
    token_ends,
)

register_cuda_ci(est_time=13, stage="extra-a", runner_config="1-gpu-small")


class TestHiCacheIndexDomains(unittest.TestCase):
    def test_dcp_translation_uses_local_virtual_ids(self):
        # Two non-adjacent pages, relocated to different physical pages.
        page_size, dcp_size = 4, 2
        logical = torch.cat((torch.arange(8, 16), torch.arange(24, 32)))
        v2p = torch.tensor([0, 5, 4, 2])

        def translate(ids):
            return v2p[ids // page_size] * (page_size * 3) + ids % page_size

        transfer = L2Transfer(
            SimpleNamespace(dcp_size=dcp_size),
            SimpleNamespace(),
            logical.clone(),
            logical,
            device_translate=translate,
        )
        resolved = L2TransferEngine._resolve_device_indices(transfer)
        for rank in range(dcp_size):
            expected = translate(maybe_dcp_kernel_indices(logical, dcp_size, rank))
            torch.testing.assert_close(
                maybe_dcp_kernel_indices(resolved, dcp_size, rank), expected
            )

    def test_fixed_size_uses_host_capacity_for_shared_buffers(self):
        pools = [
            SimpleNamespace(host_capacity_bytes=n, get_kv_size_bytes=lambda: (0, 0))
            for n in (600, 300, 100)
        ]
        self.assertEqual(_split_hicache_size(10, tuple(pools)), (6, 3, 1))


class TestHostGateCapacity(unittest.TestCase):
    def test_schedulable_memo_tracks_gate_without_allocator_mutation(self):
        from test_unified_capacity_memo import _build

        from sglang.test.unified_allocator_fixtures import setup_allocator_context

        setup_allocator_context()
        inst, allocator, kvcache = _build(lazy=True)
        ids = inst._alloc(allocator, kvcache, 8)
        allocator.free_swa(ids[2:6])
        full = allocator.full_attn_allocator
        state = {"open": True}
        pin_every_band(allocator, lambda: not state["open"])
        epoch = full._chain_capacity_epoch()
        before = full.schedulable_available_size()
        state["open"] = False
        self.assertEqual(full._chain_capacity_epoch(), epoch)
        self.assertEqual(full.schedulable_available_size(), full.available_size())
        self.assertGreater(before, full.schedulable_available_size())
        self.assertFalse(
            ends_compaction_allowed(chain_snapshot(allocator), token_ends(allocator))
        )
        self.assertEqual(allocator.verify_byte_accounting(), [])
        state["open"] = True
        self.assertEqual(full.schedulable_available_size(), before)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTransferStreamOrdering(unittest.TestCase):
    def test_write_back_drain_waits_for_forward_before_copying_queued_nodes(self):
        from sglang.srt.managers.cache_controller import (
            CacheOperation,
            HiCacheController,
        )
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        values = torch.full((8,), -1, dtype=torch.int64, device="cuda")
        expected = torch.arange(8, dtype=torch.int64, device="cuda")
        copied = torch.empty(8, dtype=torch.int64, pin_memory=True)
        forward = torch.cuda.Stream()
        engine = L2TransferEngine("kernel")
        host = SimpleNamespace(
            prepare_transfer_indices=lambda h, d, backend: (h, d),
            backup_from_device_all_layer_physical=lambda pool, h, d, backend: (
                copied.copy_(values[d], non_blocking=True)
            ),
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_fence_stream = forward
        controller.l2_transfer_engine = engine
        controller.mem_pool_device_allocator = Mock()
        controller.ack_write_queue = []
        # A tick's queued backup and the eviction victim share one drain.
        controller.write_queue = [
            CacheOperation(torch.arange(4), torch.arange(4, device="cuda"), 1),
            CacheOperation(torch.arange(4, 8), torch.arange(4, 8, device="cuda"), 2),
        ]
        controller._move_write_operation = lambda op: (
            op.host_indices,
            op.device_indices,
            op.pool_transfers,
        )
        controller._l2_transfers = lambda h, d, extra: [
            L2Transfer(host, SimpleNamespace(), h, d)
        ]
        controller._num_tokens_by_pool = lambda op: {}
        controller._transfer_num_bytes = lambda op: 64
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = controller
        cache.ongoing_write_through = {1: None, 2: None}
        cache._finish_write_through_ack = cache.ongoing_write_through.pop
        cache.metrics_collector = None
        # Warm up temporary allocations so cudaMalloc cannot hide a missing fence.
        operations = list(controller.write_queue)
        cache.writing_check(write_back=True)
        controller.write_queue = operations
        cache.ongoing_write_through.update({1: None, 2: None})
        torch.cuda.synchronize()

        with torch.cuda.stream(forward):
            torch.cuda._sleep(100_000_000)
            values.copy_(expected)
        try:
            cache.writing_check(write_back=True)
            self.assertTrue(torch.equal(copied, torch.arange(8)))
            self.assertEqual(controller.write_queue, [])
            self.assertEqual(controller.ack_write_queue, [])
            self.assertEqual(cache.ongoing_write_through, {})
        finally:
            forward.synchronize()

    def test_load_translation_follows_supplied_start_event(self):
        # The start event precedes translation; transfer must wait for both.
        engine = L2TransferEngine("kernel")
        ids = torch.tensor([2, 4, 7], device="cuda")
        output = torch.full_like(ids, -1)
        resolved = torch.full_like(ids, -2)
        torch.cuda.synchronize()
        start = torch.cuda.Event()
        start.record()

        def translate(indices):
            # Translation runs on the index stream, never behind queued copies.
            self.assertEqual(torch.cuda.current_stream(), engine.index_stream)
            torch.cuda._sleep(20_000_000)
            resolved.copy_(indices + 100)
            return resolved

        host = SimpleNamespace(
            layer_num=1,
            prepare_transfer_indices=lambda host, device, backend: (host, device),
            load_to_device_per_layer_physical=lambda pool, h, d, layer, backend, **kw: (
                output.copy_(d)
            ),
        )
        transfer = L2Transfer(
            host, SimpleNamespace(), torch.arange(3), ids, device_translate=translate
        )
        completion = engine.submit_host_to_device(
            [transfer], transfer_layer_id_max=1, start_event=start
        )
        completion.finish_event.synchronize()
        torch.testing.assert_close(output.cpu(), torch.tensor([102, 104, 107]))


class TestSwaBackupAfterCompaction(unittest.TestCase):
    def test_backup_resolves_current_binding(self):
        from sglang.srt.mem_cache.unified_cache.components.base import (
            CacheTransferPhase,
        )
        from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent

        node = SimpleNamespace(
            component_data={
                ComponentType.FULL: SimpleNamespace(value=torch.tensor([3, 7])),
                ComponentType.SWA: SimpleNamespace(value=torch.tensor([103, 107])),
            },
            id=1,
        )
        component = SimpleNamespace(
            component_type=ComponentType.SWA,
            pool_name=PoolName.SWA,
            tree_core=SimpleNamespace(has_swa_host_pool=True),
            _collect_unbacked_swa_nodes=lambda n: [n],
            _unified_allocator=lambda: object(),
            _translate_full_to_swa=lambda x: x + 200,
        )
        transfers = SWAComponent.build_hicache_transfers(
            component, node, CacheTransferPhase.BACKUP_HOST
        )
        torch.testing.assert_close(
            transfers[0].device_indices, torch.tensor([203, 207])
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDirectBackendTranslation(unittest.TestCase):
    def test_cpu_indices_translate_on_device_and_return_to_cpu(self):
        mapping = torch.tensor([0, 9, 6], device="cuda")

        def translate(ids):
            self.assertTrue(ids.is_cuda)
            return mapping[ids]

        transfer = L2Transfer(
            SimpleNamespace(),
            SimpleNamespace(device="cuda"),
            torch.tensor([0, 1]),
            torch.tensor([2, 1]),
            device_translate=translate,
        )
        result = L2TransferEngine._resolve_device_indices(transfer)
        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, torch.tensor([6, 9]))


class TestTriPoolAssembly(CustomTestCase):
    def test_swa_allocation_and_rollback_match_pool_id_ownership(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
        from sglang.srt.mem_cache.kv.alloc.band import (
            Band,
        )

        for unified in (False, True):
            with self.subTest(unified=unified):
                # Unified memory always builds a Band SWA end.
                swa_allocator = (
                    object.__new__(Band)
                    if unified
                    else SimpleNamespace(alloc=Mock(), free=Mock())
                )
                composite = SimpleNamespace(
                    swa_attn_allocator=swa_allocator,
                    state_slots=SimpleNamespace(alloc=Mock(), free=Mock()),
                )
                params = SimpleNamespace(
                    mtp_draft_device_pools=(),
                    mtp_draft_mamba_pools=(),
                    token_to_kv_pool_allocator=composite,
                )
                memory = MagicMock(enable_unified_memory=unified, hicache_size=0)
                host = MagicMock()
                with (
                    patch.object(assembler, "get_memory", return_value=memory),
                    patch.object(
                        assembler, "_get_allocator_type", return_value="default"
                    ),
                    patch.object(assembler, "build_kv_host_pool", return_value=host),
                    patch.object(assembler, "build_mamba_host_pool", return_value=host),
                    patch.object(assembler, "HybridCacheController"),
                ):
                    group, _ = assembler.build_hybrid_mamba_swa_stack(
                        params=params,
                        full_kv_pool=object(),
                        swa_kv_pool=object(),
                        mamba_pool=SimpleNamespace(
                            get_device_pool_infos=lambda: (
                                DevicePoolInfo(
                                    pool_name=PoolName.MAMBA,
                                    indices_from_pool=PoolName.MAMBA,
                                    layer_ids=(2,),
                                    buffer_info=MambaStateBufferInfo(
                                        temporal=torch.zeros(1, 2, 1),
                                        conv=(torch.zeros(1, 2, 1),),
                                    ),
                                ),
                            )
                        ),
                        full_layer_mapping={0: 0},
                        swa_layer_mapping={1: 0},
                        start_layer=0,
                        page_size=1,
                        tp_group=None,
                        load_cache_event=None,
                        storage_backend=None,
                    )
                entry = group.entry_map[PoolName.SWA]
                if unified:
                    self.assertEqual(
                        entry.device_alloc_fn, swa_allocator.alloc_physical
                    )
                    self.assertEqual(
                        entry.device_free_fn, swa_allocator.cancel_physical_reservation
                    )
                else:
                    self.assertIs(entry.device_alloc_fn, swa_allocator.alloc)
                    self.assertIs(entry.device_free_fn, swa_allocator.free)


if __name__ == "__main__":
    unittest.main()
