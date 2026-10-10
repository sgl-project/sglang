"""Host reloads must preserve ID domains, allocation ownership, and stream order."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _split_hicache_size
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
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
            prepare_transfer_indices=lambda host, device, backend: (host, device),
            load_to_device_per_layer_physical=lambda pool, h, d, layer, backend, **kw: (
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
        from sglang.srt.mem_cache.allocator.unified_sub_pool import (
            MultiEndedAllocator,
        )
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler

        for unified in (False, True):
            with self.subTest(unified=unified):
                # Unified memory always builds a MultiEndedAllocator SWA end.
                swa_allocator = (
                    object.__new__(MultiEndedAllocator)
                    if unified
                    else SimpleNamespace(alloc=Mock(), free=Mock())
                )
                composite = SimpleNamespace(swa_attn_allocator=swa_allocator)
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
