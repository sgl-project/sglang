"""Host reloads must preserve ID domains, allocation ownership, and stream order."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _split_hicache_size
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.pool_host.group import PoolEntry
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-small")


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
            SimpleNamespace(host_transfer_translate=translate),
            logical.clone(),
            logical,
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

        inst, allocator, kvcache = _build(lazy=True)
        ids = inst._alloc(allocator, kvcache, 8)
        allocator.free_swa(ids[2:6])
        full = allocator.full_attn_allocator
        state = {"open": True}
        allocator.set_host_transfer_move_gate(lambda: state["open"])
        epoch = full._chain_capacity_epoch()
        before = full.schedulable_available_size()
        state["open"] = False
        self.assertEqual(full._chain_capacity_epoch(), epoch)
        self.assertEqual(full.schedulable_available_size(), full.available_size())
        self.assertGreater(before, full.schedulable_available_size())
        self.assertFalse(allocator._compaction_allowed())
        self.assertEqual(allocator.verify_byte_accounting(), [])
        state["open"] = True
        self.assertEqual(full.schedulable_available_size(), before)


class TestSwaLoadAllocation(unittest.TestCase):
    def _controller(self, bind, free=None, evict=None):
        controller = object.__new__(HybridCacheController)
        entry = PoolEntry(
            name=PoolName.SWA,
            host_pool=SimpleNamespace(),
            device_pool=SimpleNamespace(),
            layer_mapper=lambda i: i,
            device_indices_from_anchor_fn=bind,
            device_free_fn=free or Mock(),
            device_evict_fn=evict,
        )
        controller.mem_pool_host = SimpleNamespace(entry_map={PoolName.SWA: entry})
        return controller

    def _transfer(self, parts, count):
        transfer = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(count))
        transfer.anchor_index_parts = parts
        return transfer

    def test_swa_only_load_uses_resident_full_ids(self):
        bind = Mock(side_effect=lambda x: x + 100)
        controller = self._controller(bind)
        transfer = self._transfer([torch.tensor([13, 14])], 2)
        result = controller._resolve_device_transfers(
            [transfer], torch.empty(0, dtype=torch.int64)
        )
        self.assertIsNotNone(result)
        torch.testing.assert_close(transfer.device_indices, torch.tensor([113, 114]))

    def test_mixed_load_skips_resident_swa_nodes(self):
        bind = Mock(side_effect=lambda x: x + 100)
        controller = self._controller(bind)
        transfer = self._transfer([torch.tensor([13, 14]), slice(2, 4)], 4)
        controller._resolve_device_transfers(
            [transfer], torch.tensor([20, 21, 22, 23, 24, 25])
        )
        torch.testing.assert_close(
            transfer.device_indices, torch.tensor([113, 114, 122, 123])
        )

    def test_binding_retries_after_eviction(self):
        bind = Mock(side_effect=[None, torch.tensor([41, 42])])
        evict = Mock()
        controller = self._controller(bind, evict=evict)
        transfer = self._transfer([slice(0, 2)], 2)
        self.assertIsNotNone(
            controller._resolve_device_transfers([transfer], torch.tensor([11, 12]))
        )
        evict.assert_called_once_with(2)
        self.assertEqual(bind.call_count, 2)

    def test_rollback_releases_swa_binding_by_virtual_ids(self):
        free = Mock()
        controller = self._controller(lambda x: x + 100, free=free)
        transfer = self._transfer([torch.tensor([13, 14])], 2)
        # A missing sidecar source fails after SWA has bound its pages.
        sidecar = PoolTransfer(name=PoolName.MAMBA, indices_from_pool=PoolName.INDEXER)
        result = controller._resolve_device_transfers(
            [transfer, sidecar], torch.empty(0, dtype=torch.int64)
        )
        self.assertIsNone(result)
        torch.testing.assert_close(free.call_args.args[0], torch.tensor([13, 14]))
        self.assertIsNone(transfer.device_indices)

    def test_independent_allocations_precede_kernel_id_resolution(self):
        events = []
        controller = self._controller(lambda x: events.append("bind") or x + 100)
        controller.mem_pool_host.entry_map[PoolName.MAMBA] = PoolEntry(
            name=PoolName.MAMBA,
            host_pool=SimpleNamespace(),
            device_pool=SimpleNamespace(),
            layer_mapper=lambda i: i,
            device_alloc_fn=lambda n: events.append("mamba") or torch.arange(n),
            device_free_fn=Mock(),
        )
        swa = self._transfer([slice(0, 2)], 2)
        mamba = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.arange(1))
        self.assertIsNotNone(
            controller._resolve_device_transfers([swa, mamba], torch.tensor([10, 11]))
        )
        self.assertEqual(events, ["mamba", "bind"])

    def test_tree_spec_preserves_node_correspondence(self):
        kv = PoolTransfer(
            name=PoolName.KV, host_indices=torch.arange(4), nodes_to_load=[2, 3]
        )
        swa = PoolTransfer(
            name=PoolName.SWA, host_indices=torch.arange(4), nodes_to_load=[1, 3]
        )
        nodes = {
            i: SimpleNamespace(
                id=i,
                key=[i, i],
                load_back_pending_id=None,
                component_data={
                    ComponentType.FULL: SimpleNamespace(
                        value=torch.tensor([10, 11]) if i == 1 else None
                    )
                },
            )
            for i in (1, 2, 3)
        }
        full_component = SimpleNamespace(
            component_type=ComponentType.FULL,
            build_hicache_transfers=lambda *a, **k: [kv],
        )
        swa_component = SimpleNamespace(
            component_type=ComponentType.SWA,
            build_hicache_transfers=lambda *a, **k: [swa],
        )
        core = SimpleNamespace(
            node_by_id=nodes.__getitem__,
            components=[full_component, swa_component],
            components_by_type={
                ComponentType.FULL: full_component,
                ComponentType.SWA: swa_component,
            },
        )
        UnifiedTreeCore.build_load_back_spec(core, 3)
        parts = swa.anchor_index_parts
        torch.testing.assert_close(parts[0], torch.tensor([10, 11]))
        self.assertEqual(parts[1], slice(2, 4))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTransferStreamOrdering(unittest.TestCase):
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
            self.assertEqual(torch.cuda.current_stream(), engine.host_to_device_stream)
            torch.cuda._sleep(20_000_000)
            resolved.copy_(indices + 100)
            return resolved

        host = SimpleNamespace(
            layer_num=1,
            load_to_device_per_layer=lambda pool, h, d, layer, backend, **kw: (
                output.copy_(d)
            ),
        )
        transfer = L2Transfer(
            host,
            SimpleNamespace(host_transfer_translate=translate),
            torch.arange(3),
            ids,
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
            SimpleNamespace(device="cuda", host_transfer_translate=translate),
            torch.tensor([0, 1]),
            torch.tensor([2, 1]),
        )
        result = L2TransferEngine._resolve_device_indices(transfer)
        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, torch.tensor([6, 9]))


class TestTriPoolAssembly(unittest.TestCase):
    def test_swa_allocation_and_rollback_match_pool_id_ownership(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler

        for unified in (False, True):
            with self.subTest(unified=unified):
                swa_allocator = SimpleNamespace(alloc=Mock(), free=Mock())
                composite = SimpleNamespace(
                    swa_attn_allocator=swa_allocator,
                    bind_swa_for_loaded_rows=Mock(),
                    free_swa=Mock(),
                )
                params = SimpleNamespace(
                    token_to_kv_pool_allocator=composite,
                    req_to_token_pool=SimpleNamespace(
                        mamba_allocator=SimpleNamespace(alloc=Mock(), free=Mock())
                    ),
                )
                memory = MagicMock(enable_unified_memory=unified, hicache_size=0)
                host = MagicMock()
                with (
                    patch.object(assembler, "get_memory", return_value=memory),
                    patch.object(
                        assembler, "_get_allocator_type", return_value="default"
                    ),
                    patch.object(assembler, "build_kv_host_pool", return_value=host),
                    patch.object(assembler, "MambaPoolHost", return_value=host),
                    patch.object(assembler, "HybridCacheController"),
                ):
                    group, _ = assembler.build_hybrid_mamba_swa_stack(
                        params=params,
                        full_kv_pool=object(),
                        swa_kv_pool=object(),
                        mamba_pool=object(),
                        full_layer_mapping={0: 0},
                        swa_layer_mapping={1: 0},
                        mamba_layer_mapping={2: 0},
                        page_size=1,
                        tp_group=None,
                        load_cache_event=None,
                        storage_backend=None,
                    )
                entry = group.entry_map[PoolName.SWA]
                if unified:
                    self.assertIs(
                        entry.device_indices_from_anchor_fn,
                        composite.bind_swa_for_loaded_rows,
                    )
                    self.assertIs(entry.device_free_fn, composite.free_swa)
                    self.assertIsNone(entry.device_alloc_fn)
                else:
                    self.assertIs(entry.device_alloc_fn, swa_allocator.alloc)
                    self.assertIs(entry.device_free_fn, swa_allocator.free)
                    self.assertIsNone(entry.device_indices_from_anchor_fn)


if __name__ == "__main__":
    unittest.main()
