"""Unit tests for HiCache staged write-back host-pool dispatch."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import cache_controller as manager_cache_controller
from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.managers.cache_controller import CacheOperation as ManagerCacheOperation
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import l2_transfer as transfer_module
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    DSAIndexerPoolHost,
    HostPoolGroup,
    LogicalHostPool,
    PoolEntry,
)
from sglang.srt.mem_cache.mla_host_dedup import MLAHostDedupBroadcaster
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MEMORY_POOL_HOST_MODULE = "sglang.srt.mem_cache.memory_pool_host"
MHA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mha"
MLA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mla"


def _indices(start: int, end: int) -> torch.Tensor:
    return torch.arange(start, end, dtype=torch.int64)


def _ptr_key_from_layers(src_layers) -> tuple[int, ...]:
    return tuple(int(src_layers[i].data_ptr()) for i in range(len(src_layers)))


def _ptr_key_from_tensor(ptrs: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(ptr) for ptr in ptrs.cpu().tolist())


def _device_pool_stub(*, layer_num: int, **fields) -> SimpleNamespace:
    """Minimal device-pool stand-in with layer-split fields real pools expose."""
    return SimpleNamespace(
        layer_num=layer_num,
        layer_shard_enabled=False,
        **fields,
    )


def _dedup_context(broadcaster, *, producer_stream=None) -> SimpleNamespace:
    return SimpleNamespace(
        broadcaster=broadcaster,
        is_dummy_rank=not broadcaster.is_src,
        producer_stream=producer_stream,
        last_write_finish_event=None,
        prefetch_sync_groups=None,
    )


def _host_group_stub(captured, *, can_use_write_back_jit: bool) -> SimpleNamespace:
    class FakeHostPool:
        size_per_token = 2

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ):
            captured.append(host_indices)

    entries = [
        PoolEntry(
            name=name,
            host_pool=FakeHostPool(),
            device_pool=None,
            layer_mapper=lambda layer_id: layer_id,
            is_primary_index_anchor=name == PoolName.KV,
        )
        for name in (PoolName.KV, PoolName.SWA, PoolName.DEEPSEEK_V4_C4)
    ]
    return SimpleNamespace(
        layout="page_first",
        can_use_write_back_jit=can_use_write_back_jit,
        anchor_entry=entries[0],
        entry_map={entry.name: entry for entry in entries},
    )


def _cpu_staged_lf_pf_copy(
    src_registry,
    *,
    ptr_src,
    src_indices,
    dst_indices,
    dst,
    **_,
):
    src_layers = src_registry[_ptr_key_from_tensor(ptr_src)]
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    for layer_id, src in enumerate(src_layers):
        dst[dst_indices, layer_id] = src[src_indices]


def _cpu_staged_mha_lf_pf_copy(
    src_registry,
    *,
    k_ptr_src,
    v_ptr_src,
    src_indices,
    dst_indices,
    dst_k,
    dst_v,
    **_,
):
    k_src_layers = src_registry[_ptr_key_from_tensor(k_ptr_src)]
    v_src_layers = src_registry[_ptr_key_from_tensor(v_ptr_src)]
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    for layer_id, (k_src, v_src) in enumerate(zip(k_src_layers, v_src_layers)):
        dst_k[dst_indices, layer_id] = k_src[src_indices]
        dst_v[dst_indices, layer_id] = v_src[src_indices]


def _cpu_jit_one_layer_mha_copy(
    *,
    k_cache_dst,
    v_cache_dst,
    k_cache_src,
    v_cache_src,
    indices_dst,
    indices_src,
    **_,
):
    indices_dst = indices_dst.to(dtype=torch.int64, device="cpu")
    indices_src = indices_src.to(dtype=torch.int64, device="cpu")
    k_cache_dst[indices_dst] = k_cache_src[indices_src]
    v_cache_dst[indices_dst] = v_cache_src[indices_src]


def _cpu_jit_one_layer_mla_copy(
    *,
    cache_dst,
    cache_src,
    indices_dst,
    indices_src,
    **_,
):
    indices_dst = indices_dst.to(dtype=torch.int64, device="cpu")
    indices_src = indices_src.to(dtype=torch.int64, device="cpu")
    cache_dst[indices_dst] = cache_src[indices_src]


def _cpu_per_layer_pf_lf_copy(
    *,
    src,
    dst,
    src_indices,
    dst_indices,
    layer_id,
    **_,
):
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    dst[dst_indices] = src[src_indices, layer_id]


class _FakeEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self):
        pass

    def wait(self, stream):
        pass


class _FakeStream:
    def __init__(self, operations=None):
        self.operations = operations
        self.synchronize_count = 0
        self.waited_streams = []
        self.waited_events = []

    def wait_stream(self, stream):
        self.waited_streams.append(stream)

    def wait_event(self, event):
        self.waited_events.append(event)

    def synchronize(self):
        self.synchronize_count += 1
        if self.operations is not None:
            self.operations.append(("synchronize", None))


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    def Stream():
        return _FakeStream()

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


class TestHiCacheStagedWriteBackDispatch(CustomTestCase):
    def setUp(self):
        transfer_module._timing_events_supported.cache_clear()
        self.addCleanup(transfer_module._timing_events_supported.cache_clear)

    @staticmethod
    def _start_writing(controller):
        if not hasattr(controller, "mla_dedup"):
            controller.mla_dedup = None
        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            controller.l2_transfer_engine = L2TransferEngine("kernel")
            controller.start_writing()

    def test_hybrid_load_forwards_merged_pool_transfers(self):
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            keys=["page-key"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        op = CacheOperation(_indices(0, 4), _indices(4, 8), 7)
        op.pool_transfers = [transfer]
        controller = mock.Mock(spec=HybridCacheController)
        controller.load_queue = [op, op]
        controller.layer_done_counter = mock.MagicMock()
        controller.layer_done_counter.update_producer.return_value = 0
        controller._move_op_indices.side_effect = lambda op: (
            op.host_indices,
            op.device_indices,
            op.pool_transfers,
        )
        controller.mem_pool_host = _host_group_stub([], can_use_write_back_jit=False)
        controller.has_draft = False
        controller.has_mtp_draft = False
        controller.mla_dedup = None
        controller.mla_dedup_enabled = False
        controller._mla_skip_host_io = False
        controller._l2_transfers.side_effect = lambda *args: (
            HybridCacheController._l2_transfers(controller, *args)
        )
        controller._l2_load_transfers.side_effect = lambda *args: (
            HybridCacheController._l2_load_transfers(controller, *args)
        )
        controller._num_tokens_by_pool.return_value = {}
        controller._transfer_num_bytes.return_value = 0
        controller.l2_transfer_engine = mock.Mock()
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller.l2_transfer_engine.submit_host_to_device.return_value = completion
        controller.layer_num = 2
        controller.ack_load_queue = []

        self.assertEqual(HybridCacheController.start_loading(controller), 0)

        merged_op = controller._move_op_indices.call_args.args[0]
        merged_transfer = merged_op.pool_transfers[0]
        self.assertEqual(merged_transfer.host_indices.tolist(), [0, 1, 0, 1])
        self.assertEqual(merged_transfer.keys, ["page-key", "page-key"])
        self.assertEqual(merged_transfer.hit_policy, PoolHitPolicy.TRAILING_PAGES)
        controller._l2_load_transfers.assert_called_once()
        l2_transfers = (
            controller.l2_transfer_engine.submit_host_to_device.call_args.args[0]
        )
        self.assertEqual(len(l2_transfers), 2)
        self.assertEqual(l2_transfers[1].host_indices.tolist(), [0, 1, 0, 1])
        self.assertEqual(
            len(
                HybridCacheController._l2_transfers(
                    controller, _indices(0, 0), _indices(0, 0), [merged_transfer]
                )
            ),
            1,
        )
        controller._num_tokens_by_pool.assert_called_once_with(merged_op)
        self.assertEqual(controller.ack_load_queue[0].node_ids, [7, 7])

    def test_l2_transfer_maps_global_layers(self):
        host_pool = mock.Mock()
        transfer = L2Transfer(
            host_pool=host_pool,
            device_pool=mock.sentinel.device_pool,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            layer_mapper={1: 0, 3: 1}.get,
        )
        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            L2TransferEngine("kernel").submit_host_to_device([transfer], layer_num=4)

        self.assertEqual(
            [
                call.args[3]
                for call in host_pool.load_to_device_per_layer.call_args_list
            ],
            [0, 1],
        )

    def test_packed_draft_load_is_flattened_into_l2_transfers(self):
        host_pool = mock.Mock()
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=PoolEntry(
                name=PoolName.KV,
                host_pool=host_pool,
                device_pool=mock.sentinel.target_device_pool,
                layer_mapper={0: 0, 1: 1, 2: 2}.get,
                is_primary_index_anchor=True,
            ),
            entry_map={},
        )
        controller.layer_num = 2
        controller.has_mtp_draft = True
        controller.mtp_draft_device_pools = (mock.sentinel.draft_device_pool,)
        controller.has_draft = False

        self.assertEqual(
            len(controller._l2_transfers(_indices(0, 2), _indices(2, 4))), 1
        )
        transfers = controller._l2_load_transfers(_indices(0, 2), _indices(2, 4))

        self.assertEqual(len(transfers), 2)
        self.assertFalse(transfers[0].is_draft)
        self.assertTrue(transfers[1].is_draft)
        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            L2TransferEngine("kernel").submit_host_to_device(transfers, layer_num=2)
        self.assertEqual(
            [
                call.args[3]
                for call in host_pool.load_to_device_per_layer.call_args_list
            ],
            [0, 2, 1],
        )
        self.assertIs(
            host_pool.load_to_device_per_layer.call_args_list[1].args[0],
            mock.sentinel.draft_device_pool,
        )
        self.assertTrue(
            host_pool.load_to_device_per_layer.call_args_list[1].kwargs["is_draft"]
        )

    def test_mixed_staged_write_resolves_indices_per_pool(self):
        anchor_host_pool = SimpleNamespace(can_use_write_back_jit=True)
        extra_host_pool = SimpleNamespace(can_use_write_back_jit=False)
        anchor_entry = PoolEntry(
            name=PoolName.KV,
            host_pool=anchor_host_pool,
            device_pool=None,
            layer_mapper=lambda layer_id: layer_id,
            is_primary_index_anchor=True,
        )
        extra_entry = PoolEntry(
            name=PoolName.SWA,
            host_pool=extra_host_pool,
            device_pool=None,
            layer_mapper=lambda layer_id: layer_id,
        )
        host_group = SimpleNamespace(
            layout="page_first",
            can_use_write_back_jit=False,
            supports_per_pool_backup_indices=True,
            anchor_entry=anchor_entry,
            entry_map={PoolName.KV: anchor_entry, PoolName.SWA: extra_entry},
        )
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=_indices(4, 6),
            device_indices=_indices(6, 8),
        )
        op = CacheOperation(
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            node_id=1,
            pool_transfers=[transfer],
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.io_backend = "kernel"
        controller.mem_pool_host = host_group
        controller.move_indices = mock.Mock(
            return_value=(mock.sentinel.host_indices, mock.sentinel.device_indices)
        )

        host_indices, device_indices, pool_transfers = controller._move_write_operation(
            op
        )

        self.assertIs(host_indices, op.host_indices)
        self.assertIs(device_indices, op.device_indices)
        controller.move_indices.assert_called_once_with(
            transfer.host_indices, transfer.device_indices
        )
        self.assertIs(pool_transfers[0].host_indices, mock.sentinel.host_indices)
        self.assertIs(pool_transfers[0].device_indices, mock.sentinel.device_indices)

    def _patched_transfers(self, src_registry=None, module=MEMORY_POOL_HOST_MODULE):
        staged_side_effect = None
        if src_registry is not None:
            staged_side_effect = lambda **kwargs: _cpu_staged_lf_pf_copy(
                src_registry, **kwargs
            )
        return (
            mock.patch(
                f"{module}.jit_transfer_hicache_all_layer_mla_staged_lf_pf",
                side_effect=staged_side_effect,
            ),
            mock.patch(
                f"{module}.transfer_kv_all_layer_mla_lf_pf",
                create=True,
            ),
            mock.patch(
                f"{module}.transfer_kv_per_layer_mla_pf_lf",
                side_effect=_cpu_per_layer_pf_lf_copy,
                create=True,
            ),
        )

    def test_mla_dedup_dummy_host_pools_are_allocator_only(self):
        mla_device_pool = _device_pool_stub(
            layer_num=2,
            store_dtype=torch.float16,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            size=8,
            start_layer=0,
            end_layer=2,
        )
        mla_host = MLATokenToKVPoolHost(
            mla_device_pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=2,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )

        self.assertTrue(mla_host._is_dummy)
        self.assertIsNone(mla_host.kv_buffer)
        self.assertIsNone(mla_host.data_ptrs)
        self.assertEqual(mla_host.get_contiguous_buf_infos(), ([], [], []))
        slots = mla_host.alloc(2)
        self.assertIsNotNone(slots)
        self.assertEqual(slots.tolist(), [0, 1])
        with self.assertRaisesRegex(AssertionError, "load on a dummy"):
            mla_host.load_to_device_per_layer(
                mla_device_pool, slots, slots, layer_id=0, io_backend="kernel"
            )

        dsa_device_pool = _device_pool_stub(
            layer_num=2,
            store_dtype=torch.float16,
            size=8,
            start_layer=0,
            end_layer=2,
            index_head_dim=8,
            quant_block_size=4,
        )
        indexer_host = DSAIndexerPoolHost(
            dsa_device_pool,
            mla_host,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )

        self.assertTrue(indexer_host._is_dummy)
        self.assertIsNone(indexer_host.index_k_with_scale_buffer)
        self.assertIsNone(indexer_host.index_k_device_ptrs)
        self.assertEqual(indexer_host.size, mla_host.size)
        with self.assertRaisesRegex(AssertionError, "load on a dummy"):
            indexer_host.load_to_device_per_layer(
                dsa_device_pool, slots, slots, layer_id=0, io_backend="kernel"
            )

    def test_mha_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        head_num = 1
        head_dim = 4
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        k_layers = [
            (torch.arange(8 * head_num * head_dim, dtype=torch.uint8) + layer_id * 40)
            .reshape(8, head_num, head_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        v_layers = [
            (
                torch.arange(8 * head_num * head_dim, dtype=torch.uint8)
                + 100
                + layer_id * 40
            )
            .reshape(8, head_num, head_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        expected_k = [layer[device_indices].clone() for layer in k_layers]
        expected_v = [layer[device_indices].clone() for layer in v_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            k_buffer=k_layers,
            v_buffer=v_layers,
            k_data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in k_layers], dtype=torch.uint64
            ),
            v_data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in v_layers], dtype=torch.uint64
            ),
        )

        host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host.layout = "page_first"
        host.page_size = 1
        host.layer_num = layer_num
        host.head_num = head_num
        host.head_dim = head_dim
        host.element_dim = head_num * head_dim
        host.token_stride_size = host.element_dim
        host.layout_dim = host.token_stride_size * layer_num
        host.dtype = torch.uint8
        host.can_use_jit = True
        host.can_use_write_back_jit = True
        host.kv_buffer = torch.zeros(
            2, 8, layer_num, head_num, head_dim, dtype=torch.uint8
        )
        host.k_data_refs = [host.k_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.v_data_refs = [host.v_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.staging_k_buffer = torch.empty(
            4, layer_num, head_num, head_dim, dtype=torch.uint8
        )
        host.staging_v_buffer = torch.empty_like(host.staging_k_buffer)
        src_registry = {
            _ptr_key_from_layers(k_layers): k_layers,
            _ptr_key_from_layers(v_layers): v_layers,
        }

        with (
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.jit_transfer_hicache_all_layer_staged_lf_pf",
                side_effect=lambda **kwargs: _cpu_staged_mha_lf_pf_copy(
                    src_registry, **kwargs
                ),
            ) as staged,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.transfer_kv_all_layer_lf_pf",
                create=True,
            ) as fallback,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.jit_transfer_hicache_one_layer",
                side_effect=_cpu_jit_one_layer_mha_copy,
            ) as load,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.can_use_write_back_jit_kernel",
                return_value=True,
            ) as can_use_write_back_jit_kernel,
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            for layer in k_layers + v_layers:
                layer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        can_use_write_back_jit_kernel.assert_not_called()
        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(k_layers[layer_id][device_indices], expected_k[layer_id])
            )
            self.assertTrue(
                torch.equal(v_layers[layer_id][device_indices], expected_v[layer_id])
            )
            self.assertTrue(
                torch.equal(host.k_buffer[host_indices, layer_id], expected_k[layer_id])
            )
            self.assertTrue(
                torch.equal(host.v_buffer[host_indices, layer_id], expected_v[layer_id])
            )

    def test_mla_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        kv_cache_dim = 5
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        device_layers = [
            (torch.arange(8 * kv_cache_dim, dtype=torch.uint8) + layer_id * 50)
            .reshape(8, 1, kv_cache_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        expected = [layer[device_indices].clone() for layer in device_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            kv_buffer=device_layers,
            data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in device_layers], dtype=torch.uint64
            ),
        )

        host = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host.device_pool = device_pool
        host.layout = "page_first"
        host.page_size = 1
        host.layer_num = layer_num
        host.kv_cache_dim = kv_cache_dim
        host.token_stride_size = kv_cache_dim
        host.layout_dim = host.token_stride_size * layer_num
        host.dtype = torch.uint8
        host._is_dummy = False
        host.can_use_jit = True
        host.can_use_write_back_jit = True
        host.kv_buffer = torch.zeros(8, layer_num, 1, kv_cache_dim, dtype=torch.uint8)
        host.data_refs = [host.kv_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.staging_buffer = torch.empty(
            4, layer_num, 1, kv_cache_dim, dtype=torch.uint8
        )
        src_registry = {_ptr_key_from_layers(device_layers): device_layers}

        staged_patch, fallback_patch, _ = self._patched_transfers(
            src_registry, module=MLA_POOL_HOST_MODULE
        )
        with (
            staged_patch as staged,
            fallback_patch as fallback,
            mock.patch(
                f"{MLA_POOL_HOST_MODULE}.jit_transfer_hicache_one_layer_mla",
                side_effect=_cpu_jit_one_layer_mla_copy,
            ) as load,
            mock.patch(
                f"{MLA_POOL_HOST_MODULE}.can_use_write_back_jit_kernel",
                return_value=True,
            ) as can_use_write_back_jit_kernel,
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            for layer in device_layers:
                layer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        can_use_write_back_jit_kernel.assert_not_called()
        for layer_id, layer in enumerate(device_layers):
            self.assertTrue(torch.equal(layer[device_indices], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_indices, layer_id], expected[layer_id])
            )

    @unittest.skip(
        "TODO: Mamba pool is currently incompatible with write-back staging "
        "kernel; re-enable once the staging bug is fixed."
    )
    def test_mamba_backup_then_load_roundtrip_uses_staged(self):
        num_layers = 2
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        temporal = torch.arange(num_layers * 8 * 3, dtype=torch.uint8).reshape(
            num_layers, 8, 1, 3
        )
        conv = (torch.arange(num_layers * 8 * 2, dtype=torch.uint8) + 97).reshape(
            num_layers, 8, 1, 2
        )
        device_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=temporal.clone(), conv=[conv.clone()])
        )
        expected_temporal = device_pool.mamba_cache.temporal[:, device_indices].clone()
        expected_conv = device_pool.mamba_cache.conv[0][:, device_indices].clone()

        host = MambaPoolHost.__new__(MambaPoolHost)
        host.layout = "page_first"
        host.num_mamba_layers = num_layers
        host.device_pool = SimpleNamespace(device="cpu")
        host.temporal_buffer = torch.zeros(8, num_layers, 1, 3, dtype=torch.uint8)
        host.conv_buffer = [
            torch.zeros(8, num_layers, 1, 2, dtype=torch.uint8),
        ]
        host.conv_state_shapes = [(2,)]
        host.temporal_staging_buffer = torch.empty(
            4, num_layers, 1, 3, dtype=torch.uint8
        )
        host.conv_staging_buffers = [
            torch.empty(4, num_layers, 1, 2, dtype=torch.uint8),
        ]
        host._temporal_can_use_jit = True
        host._conv_can_use_jit = [True]
        host.can_use_write_back_jit = True
        host.temporal_device_ptrs = torch.tensor(
            [layer.data_ptr() for layer in device_pool.mamba_cache.temporal],
            dtype=torch.uint64,
        )
        host.conv_device_ptrs = [
            torch.tensor(
                [layer.data_ptr() for layer in device_pool.mamba_cache.conv[0]],
                dtype=torch.uint64,
            )
        ]

        src_registry = {
            _ptr_key_from_layers(device_pool.mamba_cache.temporal): list(
                device_pool.mamba_cache.temporal
            ),
            _ptr_key_from_layers(device_pool.mamba_cache.conv[0]): list(
                device_pool.mamba_cache.conv[0]
            ),
        }

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            device_pool.mamba_cache.temporal.zero_()
            device_pool.mamba_cache.conv[0].zero_()
            for layer_id in range(num_layers):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 2)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, 4)
        self.assertTrue(
            torch.equal(
                device_pool.mamba_cache.temporal[:, device_indices], expected_temporal
            )
        )
        self.assertTrue(
            torch.equal(
                device_pool.mamba_cache.conv[0][:, device_indices], expected_conv
            )
        )

    def test_deepseek_v4_paged_pool_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        slot_page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_rows = torch.tensor([0, 2], dtype=torch.int64)
        device_rows = torch.tensor([1, 3], dtype=torch.int64)
        device_buffers = [
            (torch.arange(5 * 4, dtype=torch.uint8) + layer_id * 50).reshape(5, 4)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_rows].clone() for buffer in device_buffers]

        host = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        host.pool_name = "c4"
        host.layout = "page_first"
        host.slot_page_size = slot_page_size
        host.layer_num = layer_num
        host.item_bytes = 4
        host.dtype = torch.uint8
        host.device_buffers = device_buffers
        host.device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_buffers], dtype=torch.uint64
        )
        host.kv_buffer = torch.zeros(
            4, host.layer_num, host.item_bytes, dtype=torch.uint8
        )
        host.staging_buffer = torch.empty(
            4, host.layer_num, host.item_bytes, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_buffers): device_buffers}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_buffers:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=None,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_buffers):
            self.assertTrue(torch.equal(buffer[device_rows], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_rows, layer_id], expected[layer_id])
            )

    def test_deepseek_v4_state_pool_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        swa_page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_rows = torch.tensor([0, 2], dtype=torch.int64)
        device_rows = torch.tensor([1, 3], dtype=torch.int64)
        device_page_views = [
            (torch.arange(5 * 5, dtype=torch.uint8) + layer_id * 60).reshape(5, 5)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_rows].clone() for buffer in device_page_views]

        host = DeepSeekV4StateHostPool.__new__(DeepSeekV4StateHostPool)
        host.pool_name = "c4_state"
        host.layout = "page_first"
        host.swa_page_size = swa_page_size
        host.layer_num = layer_num
        host.state_page_bytes = 5
        host.dtype = torch.uint8
        host.device_page_views = device_page_views
        host.device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_page_views], dtype=torch.uint64
        )
        host.kv_buffer = torch.zeros(
            4, host.layer_num, host.state_page_bytes, dtype=torch.uint8
        )
        host.staging_buffer = torch.empty(
            4, host.layer_num, host.state_page_bytes, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_page_views): device_page_views}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_page_views:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=None,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_page_views):
            self.assertTrue(torch.equal(buffer[device_rows], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_rows, layer_id], expected[layer_id])
            )

    def test_dsa_indexer_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_page_indices = torch.tensor([0, 2], dtype=torch.int64)
        device_page_indices = torch.tensor([1, 3], dtype=torch.int64)
        indexer_page_stride_size = 8
        device_layers = [
            (
                torch.arange(5 * indexer_page_stride_size, dtype=torch.uint8)
                + layer_id * 70
            ).reshape(5, 1, indexer_page_stride_size)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_page_indices].clone() for buffer in device_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            index_k_with_scale_buffer=device_layers,
        )

        host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        host.device_pool = device_pool
        host.layout = "page_first"
        host.page_size = page_size
        host.layer_num = layer_num
        host.indexer_page_stride_size = indexer_page_stride_size
        host.indexer_layout_dim = host.layer_num * host.indexer_page_stride_size
        host.index_k_device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_layers], dtype=torch.uint64
        )
        host.index_k_with_scale_buffer = torch.zeros(
            4, host.layer_num, 1, host.indexer_page_stride_size, dtype=torch.uint8
        )
        host._is_dummy = False
        host.staging_buffer = torch.empty(
            4, host.layer_num, 1, host.indexer_page_stride_size, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_layers): device_layers}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=device_pool,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_layers:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=device_pool,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_layers):
            self.assertTrue(
                torch.equal(buffer[device_page_indices], expected[layer_id])
            )
            self.assertTrue(
                torch.equal(
                    host.index_k_with_scale_buffer[host_page_indices, layer_id],
                    expected[layer_id],
                )
            )

    def test_logical_host_pool_preserves_page_first_group_layout(self):
        logical_host_pool = LogicalHostPool(8, 2, layout="page_first")
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=logical_host_pool,
                    device_pool=None,
                    layer_mapper=lambda _: 0,
                    is_primary_index_anchor=True,
                )
            ]
        )

        self.assertEqual(group.layout, "page_first")
        self.assertTrue(group.can_use_write_back_jit)

    def test_host_pool_group_destroys_logical_anchor(self):
        logical_host_pool = LogicalHostPool(8, 2, layout="page_first")
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=logical_host_pool,
                    device_pool=None,
                    layer_mapper=lambda _: 0,
                    is_primary_index_anchor=True,
                )
            ]
        )

        self.assertIsNone(group.destroy())

    def test_write_back_jit_hybrid_write_keeps_extra_host_indices_on_cpu(self):
        captured = []

        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [
            CacheOperation(
                host_indices=_indices(0, 4),
                device_indices=_indices(4, 8),
                node_id=1,
                pool_transfers=[
                    PoolTransfer(
                        name=PoolName.DEEPSEEK_V4_C4,
                        host_indices=_indices(0, 4),
                        device_indices=_indices(4, 8),
                    )
                ],
            )
        ]
        controller.io_backend = "kernel"
        controller.mem_pool_host = _host_group_stub(
            captured, can_use_write_back_jit=True
        )
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.ack_write_queue = []
        controller.move_hybrid_indices = mock.Mock(
            side_effect=AssertionError(
                "write-back JIT kernel write should not move indices"
            )
        )

        self._start_writing(controller)

        controller.move_hybrid_indices.assert_not_called()
        self.assertEqual([indices.device.type for indices in captured], ["cpu", "cpu"])

    def test_hybrid_write_moves_indices_without_write_back_jit(self):
        captured = []

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C4,
                    host_indices=_indices(0, 4),
                    device_indices=_indices(4, 8),
                )
            ],
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = _host_group_stub(
            captured, can_use_write_back_jit=False
        )
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.ack_write_queue = []
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, op.pool_transfers)
        )

        self._start_writing(controller)

        controller.move_hybrid_indices.assert_called_once()
        self.assertEqual([indices.device.type for indices in captured], ["cpu", "cpu"])

    def test_write_back_jit_cache_controller_keeps_host_indices_on_cpu(self):
        captured = {}

        class FakeHostPool:
            layout = "page_first"
            can_use_write_back_jit = True
            size_per_token = 2

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [
            CacheOperation(
                host_indices=_indices(0, 4),
                device_indices=_indices(4, 8),
                node_id=1,
            )
        ]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostPool()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.device = "cuda"
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            side_effect=AssertionError(
                "write-back JIT kernel write should not move indices"
            )
        )

        self._start_writing(controller)

        controller.move_indices.assert_not_called()
        self.assertEqual(captured["host_indices"].device.type, "cpu")

    def test_cache_controller_moves_indices_without_write_back_jit(self):
        captured = {}

        class FakeHostPool:
            layout = "page_first"
            can_use_write_back_jit = False
            size_per_token = 2

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostPool()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.device = "cuda"
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        self._start_writing(controller)

        controller.move_indices.assert_called_once()
        self.assertEqual(captured["host_indices"].device.type, "cpu")

    def test_mla_dedup_peer_still_writes_local_draft_pool(self):
        target_writes = []
        draft_writes = []

        class FakeTargetHostPool:
            layout = "page_first"
            can_use_write_back_jit = False
            size_per_token = 1

            def backup_from_device_all_layer(self, *args):
                target_writes.append(args)

        class FakeDraftHostPool:
            layout = "page_first"
            can_use_write_back_jit = True
            size_per_token = 1

            def backup_from_device_all_layer(self, *args):
                draft_writes.append(args)

        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.mem_pool_host_draft = FakeDraftHostPool()
        controller.mem_pool_device_draft = object()
        controller.has_draft = True
        controller.mla_dedup = _dedup_context(SimpleNamespace(is_src=False))
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        self._start_writing(controller)

        self.assertEqual(target_writes, [])
        self.assertEqual(len(draft_writes), 1)
        self.assertEqual(draft_writes[0][1].device.type, "cpu")

    def test_mla_dedup_load_restores_draft_on_every_rank(self):
        draft_loads = []
        broadcasts = []

        class FakeDraftHostPool:
            layer_num = 2

            def load_to_device_per_layer(self, *args):
                draft_loads.append(args)

        class FakeProducerEvent:
            start_event = _FakeEvent()
            finish_event = _FakeEvent()

            def __init__(self):
                self.completed_layers = []

            def complete(self, layer_index):
                self.completed_layers.append(layer_index)

        producer_event = FakeProducerEvent()
        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host_draft = FakeDraftHostPool()
        controller.mem_pool_device_draft = object()
        controller.has_draft = True
        controller.layer_num = 3
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[producer_event]
        )
        controller.mla_dedup = _dedup_context(
            SimpleNamespace(
                is_src=False,
                prepare_broadcast=lambda device_indices, stream: (
                    device_indices,
                    None,
                ),
                broadcast_loaded_layer=lambda layer_id, prepared: broadcasts.append(
                    (layer_id, prepared)
                ),
            )
        )
        controller.load_stream = _FakeStream()
        controller.l2_transfer_engine = SimpleNamespace(
            host_to_device_stream=controller.load_stream
        )
        controller.ack_load_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_loading()

        self.assertEqual([layer_id for layer_id, _ in broadcasts], [0, 1, 2])
        self.assertEqual(len(draft_loads), 2)
        self.assertEqual(producer_event.completed_layers, [0, 1, 2])
        self.assertEqual(controller.load_stream.synchronize_count, 1)
        self.assertEqual(len(controller.ack_load_queue), 1)
        ack = controller.ack_load_queue[0]
        self.assertEqual(ack.num_tokens, 4)
        self.assertIsNot(ack.start_event, producer_event.start_event)
        self.assertIsNot(ack.finish_event, producer_event.finish_event)

    def test_mla_dedup_source_load_and_broadcast_are_layerwise(self):
        operations = []

        class FakeTargetHostPool:
            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                operations.append(("target", layer_id))

        class FakeDraftHostPool:
            layer_num = 2

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                operations.append(("draft", layer_id))

        class FakeProducerEvent:
            start_event = _FakeEvent()
            finish_event = _FakeEvent()

            def complete(self, layer_index):
                operations.append(("complete", layer_index))

        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.mem_pool_host_draft = FakeDraftHostPool()
        controller.mem_pool_device_draft = object()
        controller.has_draft = True
        controller.layer_num = 3
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[FakeProducerEvent()]
        )
        controller.mla_dedup = _dedup_context(
            SimpleNamespace(
                is_src=True,
                prepare_broadcast=lambda device_indices, stream: operations.append(
                    ("prepare", None)
                )
                or (device_indices, None),
                broadcast_loaded_layer=lambda layer_id, prepared: operations.append(
                    ("broadcast", layer_id)
                ),
            )
        )
        controller.load_stream = _FakeStream(operations)
        controller.l2_transfer_engine = SimpleNamespace(
            host_to_device_stream=controller.load_stream
        )
        controller.ack_load_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_loading()

        self.assertEqual(
            operations,
            [
                ("prepare", None),
                ("target", 0),
                ("broadcast", 0),
                ("draft", 0),
                ("complete", 0),
                ("target", 1),
                ("broadcast", 1),
                ("draft", 1),
                ("complete", 1),
                ("target", 2),
                ("broadcast", 2),
                ("complete", 2),
                ("synchronize", None),
            ],
        )

    def test_mla_layer_broadcast_reuses_full_staging_capacity(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.is_src = True
        broadcaster.src_global_rank = 0
        broadcaster.group = object()

        layer_buffers = [
            torch.arange(24, dtype=torch.float32).reshape(6, 1, 4),
            torch.arange(24, 48, dtype=torch.float32).reshape(6, 1, 4),
        ]
        target = torch.tensor([0, 2, 5], dtype=torch.int64)
        # Capacity is six rows, modeling the retained 2 layers * 3-row chunk.
        staging = torch.empty(2 * 3 * 4, dtype=torch.float32)

        with mock.patch.object(torch.distributed, "broadcast") as broadcast:
            broadcaster._bcast_layer(layer_buffers, staging, target, 4, layer_id=1)

        broadcast.assert_called_once()
        expected = layer_buffers[1].index_select(0, target)
        torch.testing.assert_close(
            staging[: expected.numel()].view_as(expected), expected
        )

        broadcaster.is_src = False
        received = [torch.zeros_like(layer) for layer in layer_buffers]
        with mock.patch.object(torch.distributed, "broadcast"):
            broadcaster._bcast_layer(received, staging, target, 4, layer_id=1)
        torch.testing.assert_close(received[1].index_select(0, target), expected)

    def test_mla_dedup_chunk_tokens_uses_environment(self):
        device_pool = _device_pool_stub(
            layer_num=2,
            device=torch.device("cpu"),
            kv_cache_dim=4,
            kv_buffer=[torch.empty(3, 1, 4), torch.empty(3, 1, 4)],
        )

        with (
            envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(7),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 2),
            ),
        ):
            broadcaster = MLAHostDedupBroadcaster(
                device_pool, group=object(), src_global_rank=0
            )

        self.assertEqual(broadcaster.chunk_tokens, 7)
        self.assertEqual(broadcaster.kv_staging.numel(), 2 * 7 * 4)

    def test_mla_dedup_chunk_tokens_must_be_positive(self):
        device_pool = _device_pool_stub(
            layer_num=2,
            device=torch.device("cpu"),
            kv_cache_dim=4,
            kv_buffer=[torch.empty(3, 1, 4), torch.empty(3, 1, 4)],
        )

        with (
            envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(0),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 2),
            ),
            self.assertRaisesRegex(ValueError, "must be positive"),
        ):
            MLAHostDedupBroadcaster(device_pool, group=object(), src_global_rank=0)

    def test_mla_dedup_peer_prefetch_uses_synchronized_increment(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mla_dedup = _dedup_context(SimpleNamespace(is_src=False))
        controller.page_size = 4
        operation = PrefetchOperation(
            request_id="request",
            token_ids=[],
            pool_transfers=[object()],
        )
        operation.hash_value = ["page-0", "page-1"]
        operation.increment = mock.Mock(return_value=True)

        controller._page_transfer(operation)

        operation.increment.assert_called_once_with(8)
        self.assertTrue(operation.pool_transfers_done)

    def test_mla_dedup_build_eagerly_warms_dedicated_nccl_group(self):
        tp_group = object()
        dedicated_group = object()
        device_pool = _device_pool_stub(
            layer_num=2,
            device=torch.device("cpu"),
            kv_cache_dim=4,
            kv_buffer=[torch.empty(3, 1, 4)],
        )

        with (
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 2),
            ),
            mock.patch.object(
                torch.distributed,
                "get_process_group_ranks",
                return_value=[4, 5],
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
                return_value=dedicated_group,
            ) as create_group,
            mock.patch.object(torch.distributed, "broadcast") as broadcast,
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            broadcaster = MLAHostDedupBroadcaster.build(
                device_pool, tp_group, attn_tp_group=None
            )

        create_group.assert_called_once_with(group_ranks=[4, 5], backend="nccl")
        broadcast.assert_called_once()
        warmup = broadcast.call_args.args[0]
        self.assertEqual(warmup.numel(), 1)
        self.assertIs(broadcast.call_args.kwargs["group"], dedicated_group)
        self.assertEqual(broadcast.call_args.kwargs["src"], 4)
        synchronize.assert_called_once_with(device_pool.device)
        self.assertIs(broadcaster.group, dedicated_group)

    def test_mla_dedup_indexer_pages_preserve_logical_order(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.device = torch.device("cpu")
        broadcaster.device_pool = SimpleNamespace(page_size=4)
        broadcaster.idx_bufs = [object()]

        # Logical page 0 is backed by physical page 2, while logical page 1 is
        # backed by physical page 0. unique() would sort this to [0, 2] and
        # swap the payload when another TP rank uses different physical pages.
        device_indices = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3], dtype=torch.int64)
        prepared_indices, page_indices = broadcaster.prepare_broadcast(
            device_indices, _FakeStream()
        )

        self.assertIs(prepared_indices, device_indices)
        torch.testing.assert_close(page_indices, torch.tensor([2, 0]))

    def test_mla_dedup_indexer_rejects_partial_pages(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.device = torch.device("cpu")
        broadcaster.device_pool = SimpleNamespace(page_size=4)
        broadcaster.idx_bufs = [object()]

        with self.assertRaisesRegex(ValueError, "page-aligned device indices"):
            broadcaster.prepare_broadcast(torch.arange(7), _FakeStream())

    def test_mla_dedup_write_fences_forward_stream_both_directions(self):
        class FakeHostPool:
            layout = "layer_first"
            can_use_write_back_jit = False
            size_per_token = 1

            def backup_from_device_all_layer(self, *args):
                pass

        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        producer_stream = object()
        write_stream = _FakeStream()
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [op]
        controller.io_backend = "direct"
        controller.mem_pool_host = FakeHostPool()
        controller.mem_pool_device = object()
        controller.has_draft = False
        controller.mla_dedup = _dedup_context(
            SimpleNamespace(is_src=True), producer_stream=producer_stream
        )
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(transfer_module, "device_module", _FakeDeviceModule):
            controller.l2_transfer_engine = L2TransferEngine("direct")
            controller.l2_transfer_engine.device_to_host_stream = write_stream
            controller.start_writing()

        self.assertEqual(write_stream.waited_streams, [producer_stream])
        self.assertIs(
            controller.mla_dedup.last_write_finish_event,
            controller.ack_write_queue[0].finish_event,
        )
        forward_stream = _FakeStream()
        controller.wait_for_last_write(forward_stream)
        self.assertEqual(
            forward_stream.waited_events,
            [controller.ack_write_queue[0].finish_event],
        )

    def test_hybrid_mla_dedup_peer_still_writes_local_draft_pool(self):
        target_writes = []
        draft_writes = []

        class FakeTargetHostPool:
            layout = "page_first"
            can_use_write_back_jit = False

            def backup_from_device_all_layer(self, *args, **kwargs):
                target_writes.append((args, kwargs))

        class FakeDraftHostPool:
            layout = "page_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(self, *args):
                draft_writes.append(args)

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        target_host_pool = FakeTargetHostPool()
        target_host_pool._is_dummy = True
        controller.mem_pool_device = object()
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(
                host_pool=target_host_pool,
                device_pool=controller.mem_pool_device,
                layer_mapper=None,
            ),
            entry_map={},
        )
        controller.mem_pool_host_draft = FakeDraftHostPool()
        controller.mem_pool_device_draft = object()
        controller.has_draft = True
        controller.mla_dedup = _dedup_context(SimpleNamespace(is_src=False))
        controller.ack_write_queue = []
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._record_transfer_indices_on_stream = lambda *args: None
        controller._num_tokens_by_pool = lambda op: {"kv": len(op.device_indices)}
        controller._mla_transfer_num_bytes = lambda op: 0

        self._start_writing(controller)

        self.assertEqual(target_writes, [])
        self.assertEqual(len(draft_writes), 1)
        self.assertEqual(draft_writes[0][1].device.type, "cpu")

    def test_hybrid_mla_dedup_loads_extra_pools_layerwise(self):
        operations = []
        pool_transfers = [mock.sentinel.pool_transfer]

        class FakeTargetHostPool:
            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                pool_transfers=None,
            ):
                operations.append(("load", layer_id, pool_transfers))

        class FakeProducerEvent:
            start_event = _FakeEvent()
            finish_event = _FakeEvent()

            def complete(self, layer_index):
                operations.append(("complete", layer_index))

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.has_draft = False
        controller.layer_num = 2
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[FakeProducerEvent()]
        )
        controller.mla_dedup = _dedup_context(
            SimpleNamespace(
                is_src=True,
                prepare_broadcast=lambda device_indices, stream: (
                    device_indices,
                    None,
                ),
                broadcast_loaded_layer=lambda layer_id, prepared: operations.append(
                    ("broadcast", layer_id)
                ),
            )
        )
        controller.load_stream = _FakeStream(operations)
        controller.l2_transfer_engine = SimpleNamespace(
            host_to_device_stream=controller.load_stream
        )
        controller.ack_load_queue = []
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, pool_transfers)
        )
        controller._record_transfer_indices_on_stream = mock.Mock()

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_loading()

        self.assertEqual(
            operations,
            [
                ("load", 0, pool_transfers),
                ("broadcast", 0),
                ("complete", 0),
                ("load", 1, pool_transfers),
                ("broadcast", 1),
                ("complete", 1),
                ("synchronize", None),
            ],
        )
        controller._record_transfer_indices_on_stream.assert_called_once_with(
            controller.l2_transfer_engine.host_to_device_stream,
            op.host_indices,
            op.device_indices,
            pool_transfers,
        )


if __name__ == "__main__":
    unittest.main()
