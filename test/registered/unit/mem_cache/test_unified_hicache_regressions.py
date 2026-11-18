"""Host reloads must preserve ID domains, allocation ownership, and stream order."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, MambaStateBufferInfo
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _split_hicache_size
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    chain_snapshot,
    hold_every_band,
    release_every_band,
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
    def test_capacity_read_tracks_holds(self):
        from test_unified_capacity_memo import _build

        from sglang.test.unified_allocator_fixtures import (
            largest_take,
            setup_allocator_context,
        )

        setup_allocator_context()
        inst, allocator, kvcache = _build(lazy=True)
        ids = inst._alloc(allocator, kvcache, 8)
        allocator.free_swa(ids[2:6])
        owner = allocator.full_attn_allocator.sub_pool_name
        before = largest_take(allocator, owner)
        hold_every_band(allocator)
        # A held band's holes cannot be compacted away: the take shrinks.
        self.assertGreater(before, largest_take(allocator, owner))
        self.assertTrue(any(band.blocked for band in chain_snapshot(allocator).bands))
        self.assertEqual(allocator.verify_byte_accounting(), [])
        release_every_band(allocator)
        self.assertEqual(largest_take(allocator, owner), before)


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
        controller.kv_storage = None
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

    def _write_back_rig(self, forward, delay_cycles: int):
        """A write-back cache whose victims hold the same rows of a target KV, a
        draft KV and a state pool, copied to host by the real engine; each copy
        spins ``delay_cycles`` on the device-to-host stream before it reads."""
        from sglang.srt.managers.cache_controller import HiCacheController
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        pools = {
            name: torch.zeros(128, 256, device="cuda")
            for name in ("target", "draft", "state")
        }
        hosts = {name: torch.zeros(64, 256, pin_memory=True) for name in pools}

        def host_pool(name):
            def backup(pool, h, d, backend):
                torch.cuda._sleep(delay_cycles)
                hosts[name].copy_(pools[name][d], non_blocking=True)

            return SimpleNamespace(
                prepare_transfer_indices=lambda h, d, backend: (h, d),
                backup_from_device_all_layer_physical=backup,
            )

        host_pools = [host_pool(name) for name in pools]
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_fence_stream = controller.load_fence_stream = forward
        controller.l2_transfer_engine = L2TransferEngine("kernel")
        controller.enable_storage = False
        controller.kv_storage = None
        controller.mem_pool_device_allocator = Mock()
        controller.ack_write_queue = []
        controller.write_queue = []
        controller._move_write_operation = lambda op: (
            op.host_indices,
            op.device_indices,
            op.pool_transfers,
        )
        controller._l2_transfers = lambda h, d, extra: [
            L2Transfer(host, SimpleNamespace(), h, d) for host in host_pools
        ]
        controller._num_tokens_by_pool = lambda op: {}
        controller._transfer_num_bytes = lambda op: 0
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = controller
        cache.ongoing_write_through = {}
        cache._finish_write_through_ack = cache.ongoing_write_through.pop
        cache.metrics_collector = None
        return cache, pools, hosts

    def _write_back_and_reuse(
        self, cache, forward, pools, hosts, node_id, overlap, land
    ):
        """One eviction: a forward writes the victim's rows, the victim's copy is
        queued and landed by ``land``, and the freed rows are reused at once by
        target and draft extends (forward stream), a state reset (current stream)
        and a load of the host copy (host-to-device stream). With ``overlap``, a
        forward queued before the eviction still writes the victim's last row and
        another request's row. Returns the victim's rows and the load's result."""
        from sglang.srt.managers.cache_controller import CacheOperation

        controller = cache.cache_controller
        engine = controller.l2_transfer_engine
        order = torch.randperm(128)
        rows = order[:64].sort().values.cuda()
        live_row = order[64:65].cuda()
        with torch.cuda.stream(forward):
            victim = {name: torch.randn(64, 256, device="cuda") for name in pools}
            for name, values in victim.items():
                pools[name][rows] = values
            if overlap:
                # Overlap scheduling: the previous forward is still running.
                torch.cuda._sleep(20_000_000)
                victim["target"][-1] = node_id
                pools["target"][rows[-1:]] = victim["target"][-1:]
                pools["draft"][live_row] = float(node_id)
        controller.write_queue = [CacheOperation(torch.arange(64), rows, node_id)]
        cache.ongoing_write_through[node_id] = None

        land()
        in_flight = torch.cuda.Event()
        in_flight.record(engine.device_to_host_stream)
        # No host wait: the delayed copy is still running.
        self.assertFalse(in_flight.query())
        self.assertEqual(controller.ack_write_queue, [])
        self.assertEqual(cache.ongoing_write_through, {})

        with torch.cuda.stream(forward):
            pools["target"][rows] = -1.0
            pools["draft"][rows] = -2.0
        pools["state"][rows] = -3.0
        with torch.cuda.stream(engine.host_to_device_stream):
            loaded = {
                name: host.to("cuda", non_blocking=True) for name, host in hosts.items()
            }
        torch.cuda.synchronize()
        for name, value in (("target", -1.0), ("draft", -2.0), ("state", -3.0)):
            self.assertTrue(bool((pools[name][rows] == value).all()))
        if overlap:
            self.assertTrue(bool((pools["draft"][live_row] == node_id).all()))
        return victim, loaded

    def test_ordered_write_back_demotion_keeps_host_copies_under_immediate_reuse(self):
        """Fault injection: every copy starts ~20 ms late. Without the stream waits
        the reuse lands first and corrupts every host copy and the load; with them
        the copies and the load are bitwise intact, with and without a forward
        queued ahead (overlap scheduling)."""
        from sglang.srt.environ import envs

        forward = torch.cuda.Stream()
        cache, pools, hosts = self._write_back_rig(forward, delay_cycles=50_000_000)
        with envs.SGLANG_OPT_HICACHE_WRITE_BACK_ORDERED_DEMOTE.override(True):
            self.assertTrue(cache._write_back_demote_ordered())
        self.assertFalse(cache._write_back_demote_ordered())

        def land_without_waits():
            cache.cache_controller.start_writing()
            cache.cache_controller.ack_write_queue.clear()
            cache.ongoing_write_through.clear()

        # Warm up kernels and allocations: a first launch must not outlast the delay.
        self._write_back_and_reuse(
            cache, forward, pools, hosts, 1, overlap=True, land=land_without_waits
        )
        # Positive control: the same eviction with the acks dropped, no waits.
        victim, loaded = self._write_back_and_reuse(
            cache, forward, pools, hosts, 2, overlap=False, land=land_without_waits
        )
        for name in pools:
            self.assertFalse(torch.equal(hosts[name], victim[name].cpu()), name)
            self.assertFalse(torch.equal(loaded[name], victim[name]), name)

        for cycle in range(32):
            overlap = cycle % 2 == 1
            with self.subTest(cycle=cycle, overlap=overlap):
                victim, loaded = self._write_back_and_reuse(
                    cache,
                    forward,
                    pools,
                    hosts,
                    cycle + 3,
                    overlap=overlap,
                    land=cache._order_write_backs_before_reuse,
                )
                for name in pools:
                    self.assertTrue(torch.equal(hosts[name], victim[name].cpu()), name)
                    self.assertTrue(torch.equal(loaded[name], victim[name]), name)


class TestSwaBackupAfterCompaction(unittest.TestCase):
    def test_backup_names_window_rows_by_owner_ids(self):
        # The copy translates them at submission, so a page move between the
        # backup's queueing and its submission cannot misdirect it.
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
        )
        transfers = SWAComponent.build_hicache_transfers(
            component, node, CacheTransferPhase.BACKUP_HOST
        )
        torch.testing.assert_close(transfers[0].device_indices, torch.tensor([3, 7]))


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
        """A static SWA stack loads and rolls back through the same allocator;
        a unified pool loads through its storage (`KVStorage.host_load_alloc`)."""
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler

        swa_allocator = SimpleNamespace(alloc=Mock(), free=Mock())
        composite = SimpleNamespace(
            swa_attn_allocator=swa_allocator,
            state_slots=SimpleNamespace(alloc=Mock(), free=Mock()),
        )
        params = SimpleNamespace(
            page_size=1,
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
            token_to_kv_pool_allocator=composite,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        memory = MagicMock(enable_unified_memory=False, hicache_size=0)
        host = MagicMock()
        with (
            patch.object(assembler, "get_memory", return_value=memory),
            patch.object(assembler, "_get_allocator_type", return_value="default"),
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
                load_cache_event=None,
                storage_backend=None,
            )
        entry = group.entry_map[PoolName.SWA]
        self.assertIs(entry.device_alloc_fn, swa_allocator.alloc)
        self.assertIs(entry.device_free_fn, swa_allocator.free)


if __name__ == "__main__":
    unittest.main()
