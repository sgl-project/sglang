"""Unit tests for HiCache staged write-back host-pool dispatch."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import cache_controller as manager_cache_controller
from sglang.srt.managers.cache_controller import CacheOperation as ManagerCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheController,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    DSAIndexerPoolHost,
    HostPoolGroup,
    LogicalHostPool,
    MambaPoolHost,
    MLATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MEMORY_POOL_HOST_MODULE = "sglang.srt.mem_cache.memory_pool_host"
MHA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mha"


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
    def record(self):
        pass

    def wait(self, stream):
        pass


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


class TestHiCacheStagedWriteBackDispatch(unittest.TestCase):
    def _patched_transfers(self, src_registry=None):
        staged_side_effect = None
        if src_registry is not None:
            staged_side_effect = lambda **kwargs: _cpu_staged_lf_pf_copy(
                src_registry, **kwargs
            )
        return (
            mock.patch(
                f"{MEMORY_POOL_HOST_MODULE}.jit_transfer_hicache_all_layer_mla_staged_lf_pf",
                side_effect=staged_side_effect,
            ),
            mock.patch(
                f"{MEMORY_POOL_HOST_MODULE}.transfer_kv_all_layer_mla_lf_pf",
                create=True,
            ),
            mock.patch(
                f"{MEMORY_POOL_HOST_MODULE}.transfer_kv_per_layer_mla_pf_lf",
                side_effect=_cpu_per_layer_pf_lf_copy,
                create=True,
            ),
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
        host.can_use_jit = True
        host.can_use_write_back_jit = True
        host.kv_buffer = torch.zeros(8, layer_num, 1, kv_cache_dim, dtype=torch.uint8)
        host.data_refs = [host.kv_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.staging_buffer = torch.empty(
            4, layer_num, 1, kv_cache_dim, dtype=torch.uint8
        )
        src_registry = {_ptr_key_from_layers(device_layers): device_layers}

        staged_patch, fallback_patch, _ = self._patched_transfers(src_registry)
        with (
            staged_patch as staged,
            fallback_patch as fallback,
            mock.patch(
                f"{MEMORY_POOL_HOST_MODULE}.jit_transfer_hicache_one_layer_mla",
                side_effect=_cpu_jit_one_layer_mla_copy,
            ) as load,
            mock.patch(
                f"{MEMORY_POOL_HOST_MODULE}.can_use_write_back_jit_kernel",
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

    def test_write_back_jit_hybrid_write_keeps_extra_host_indices_on_cpu(self):
        captured = {}

        class FakeHostGroup:
            layout = "page_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                io_backend,
                pool_transfers=None,
            ):
                captured["host_indices"] = host_indices
                captured["pool_transfers"] = pool_transfers

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
        controller.mem_pool_host = FakeHostGroup()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller._record_transfer_indices_on_stream = lambda *args: None
        controller.move_hybrid_indices = mock.Mock(
            side_effect=AssertionError(
                "write-back JIT kernel write should not move indices"
            )
        )

        with mock.patch.object(
            hybrid_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_hybrid_indices.assert_not_called()
        self.assertEqual(captured["host_indices"].device.type, "cpu")
        self.assertEqual(captured["pool_transfers"][0].host_indices.device.type, "cpu")

    def test_hybrid_write_moves_indices_without_write_back_jit(self):
        captured = {}

        class FakeHostGroup:
            layout = "page_first"
            can_use_write_back_jit = False

            def backup_from_device_all_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                io_backend,
                pool_transfers=None,
            ):
                captured["host_indices"] = host_indices
                captured["pool_transfers"] = pool_transfers

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
        controller.mem_pool_host = FakeHostGroup()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller._record_transfer_indices_on_stream = lambda *args: None
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, op.pool_transfers)
        )

        with mock.patch.object(
            hybrid_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_hybrid_indices.assert_called_once()
        self.assertEqual(captured["host_indices"].device.type, "cpu")
        self.assertEqual(captured["pool_transfers"][0].host_indices.device.type, "cpu")

    def test_hybrid_write_moves_indices_without_page_first_layout(self):
        captured = {}

        class FakeHostGroup:
            layout = "layer_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                io_backend,
                pool_transfers=None,
            ):
                captured["host_indices"] = host_indices
                captured["pool_transfers"] = pool_transfers

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
        controller.mem_pool_host = FakeHostGroup()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller._record_transfer_indices_on_stream = lambda *args: None
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, op.pool_transfers)
        )

        with mock.patch.object(
            hybrid_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_hybrid_indices.assert_called_once()
        self.assertEqual(captured["host_indices"].device.type, "cpu")
        self.assertEqual(captured["pool_transfers"][0].host_indices.device.type, "cpu")

    def test_write_back_jit_cache_controller_keeps_host_indices_on_cpu(self):
        captured = {}

        class FakeHostPool:
            layout = "page_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [
            ManagerCacheOperation(
                host_indices=_indices(0, 4),
                device_indices=_indices(4, 8),
                node_id=1,
            )
        ]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostPool()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            side_effect=AssertionError(
                "write-back JIT kernel write should not move indices"
            )
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_indices.assert_not_called()
        self.assertEqual(captured["host_indices"].device.type, "cpu")

    def test_cache_controller_moves_indices_without_write_back_jit(self):
        captured = {}

        class FakeHostPool:
            layout = "page_first"
            can_use_write_back_jit = False

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        op = ManagerCacheOperation(
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
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_indices.assert_called_once()
        self.assertEqual(captured["host_indices"].device.type, "cpu")

    def test_cache_controller_moves_indices_without_page_first_layout(self):
        captured = {}

        class FakeHostPool:
            layout = "layer_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        op = ManagerCacheOperation(
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
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_indices.assert_called_once()
        self.assertEqual(captured["host_indices"].device.type, "cpu")

    def test_cache_controller_copies_canary_with_kv_indices(self):
        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        bridge = mock.Mock()
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [op]
        controller.io_backend = "direct"
        controller.mem_pool_host = SimpleNamespace(
            layout="layer_first",
            can_use_write_back_jit=False,
            backup_from_device_all_layer=mock.Mock(),
        )
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )
        controller.kv_canary_hicache_bridge = bridge

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        bridge.backup.assert_called_once_with(
            host_indices=op.host_indices,
            device_indices=op.device_indices,
            io_backend="direct",
        )

    def test_cache_controller_restores_canary_before_layer_completion(self):
        call_order = []
        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        producer_event = SimpleNamespace(
            start_event=_FakeEvent(),
            finish_event=_FakeEvent(),
            complete=lambda layer_id: call_order.append(("complete", layer_id)),
        )
        bridge = mock.Mock()
        bridge.restore.side_effect = lambda **_: call_order.append(("canary", None))
        controller = HiCacheController.__new__(HiCacheController)
        controller.load_queue = [op]
        controller.io_backend = "direct"
        controller.mem_pool_host = SimpleNamespace(
            layout="layer_first",
            load_to_device_per_layer=lambda *args: call_order.append(("kv", args[3])),
        )
        controller.mem_pool_device = None
        controller.mem_pool_device_allocator = None
        controller.has_draft = False
        controller.layer_num = 2
        controller.load_stream = object()
        controller.ack_load_queue = []
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0,
            events=[producer_event],
        )
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )
        controller.kv_canary_hicache_bridge = bridge

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_loading()

        bridge.restore.assert_called_once_with(
            host_indices=op.host_indices,
            device_indices=op.device_indices,
            io_backend="direct",
        )
        self.assertEqual(call_order[0], ("canary", None))
        self.assertEqual(
            call_order[1:], [("kv", 0), ("complete", 0), ("kv", 1), ("complete", 1)]
        )

    def test_hybrid_controller_passes_swa_transfer_to_canary(self):
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=_indices(1, 3),
            device_indices=_indices(5, 7),
        )
        op = CacheOperation(
            host_indices=_indices(0, 2),
            device_indices=_indices(4, 6),
            node_id=1,
            pool_transfers=[swa_transfer],
        )
        bridge = mock.Mock()
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [op]
        controller.io_backend = "direct"
        controller.mem_pool_host = SimpleNamespace(
            layout="layer_first",
            can_use_write_back_jit=False,
            backup_from_device_all_layer=mock.Mock(),
        )
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller._record_transfer_indices_on_stream = mock.Mock()
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, [swa_transfer])
        )
        controller.kv_canary_hicache_bridge = bridge

        with mock.patch.object(
            hybrid_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        bridge.backup.assert_called_once_with(
            host_indices=op.host_indices,
            device_indices=op.device_indices,
            pool_transfers=[swa_transfer],
            io_backend="direct",
        )


if __name__ == "__main__":
    unittest.main()
