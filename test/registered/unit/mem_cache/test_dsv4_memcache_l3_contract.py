import ctypes
import threading
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, InsertResult
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store import (
    NpuMemcacheStore,
)
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    PrepareLoadBackResult,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeObjectStore:
    def __init__(self, existing=()):
        self.existing = set(existing)

    def batch_is_exist(self, keys):
        return [1 if key in self.existing else 0 for key in keys]


class _LifecycleObjectStore:
    instances = []
    remove_all_result = 0

    def __init__(self):
        self.setup_calls = 0
        self.init_calls = []
        self.registered_buffers = []
        self.put_calls = []
        self.objects = {}
        self.remove_all_calls = 0
        self.closed = False
        self.__class__.instances.append(self)

    def setup(self, _config):
        self.setup_calls += 1
        return 0

    def init(self, device_id, init_bm):
        self.init_calls.append((device_id, init_bm))
        return 0

    def register_buffer(self, ptr, size):
        self.registered_buffers.append((ptr, size))
        return 0

    def batch_put_from(self, keys, ptrs, sizes, direct=None):
        self.put_calls.append((keys, ptrs, sizes, direct))
        for key, ptr, size in zip(keys, ptrs, sizes):
            self.objects[key] = ctypes.string_at(ptr, size)
        return [0] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes, direct=None):
        self.get_call = (keys, ptrs, sizes, direct)
        results = []
        for key, ptr, size in zip(keys, ptrs, sizes):
            value = self.objects.get(key)
            if value is None or len(value) != size:
                results.append(-1)
                continue
            ctypes.memmove(ptr, value, size)
            results.append(0)
        return results

    def remove_all(self):
        self.remove_all_calls += 1
        return self.remove_all_result

    def close(self):
        self.closed = True
        return None


def _make_memcache(existing=()):
    backend = NpuMemcacheStore.__new__(NpuMemcacheStore)
    backend.store = _FakeObjectStore(existing)
    backend._store_initialized = True
    backend.mem_pool_host = SimpleNamespace(kv_buffer=None)
    backend.registered_pools = {}
    backend.mla_suffix = ""
    backend.mha_suffix = "0"
    backend.extra_backend_tag = None
    backend.is_mla_backend = True
    backend.storage_config = None
    return backend


def _make_lazy_memcache(protocol="device_sdma"):
    _LifecycleObjectStore.instances.clear()
    _LifecycleObjectStore.remove_all_result = 0
    backend = NpuMemcacheStore.__new__(NpuMemcacheStore)
    backend.store = None
    backend.storage_config = SimpleNamespace(tp_rank=3)
    backend._store_initialized = False
    backend._store_init_lock = threading.Lock()
    backend._pending_buffers = []
    backend._store_factory = _LifecycleObjectStore
    backend._local_cfg = object()
    backend._device_id = 3
    backend._init_bm = True
    backend._protocol = protocol
    backend._defer_runtime_init = True
    return backend


def test_lazy_clear_uses_metadata_only_client_without_initializing_runtime_store():
    backend = _make_lazy_memcache()

    backend.clear()

    assert not backend._store_initialized
    assert backend.store is None
    assert len(_LifecycleObjectStore.instances) == 1
    clear_client = _LifecycleObjectStore.instances[0]
    assert clear_client.setup_calls == 1
    assert clear_client.init_calls == [(3, False)]
    assert clear_client.remove_all_calls == 1
    assert clear_client.closed


def test_lazy_clear_propagates_memcache_remove_all_failure():
    backend = _make_lazy_memcache()
    _LifecycleObjectStore.remove_all_result = -7

    try:
        backend.clear()
    except RuntimeError as exc:
        assert "remove_all failed with code -7" in str(exc)
    else:
        raise AssertionError("Memcache clear failure must not be reported as success")


def test_device_transports_lazy_init_is_limited_to_logical_anchor():
    dsv4_group = LogicalHostPool(4096, 128)
    ordinary_group = SimpleNamespace(entries=[SimpleNamespace(name=PoolName.KV)])

    assert NpuMemcacheStore._should_lazy_init(dsv4_group, "device_sdma", True)
    assert NpuMemcacheStore._should_lazy_init(dsv4_group, "device_rdma", True)
    assert not NpuMemcacheStore._should_lazy_init(ordinary_group, "device_sdma", True)
    assert not NpuMemcacheStore._should_lazy_init(ordinary_group, "device_rdma", True)
    assert not NpuMemcacheStore._should_lazy_init(dsv4_group, "host_shm", True)
    assert not NpuMemcacheStore._should_lazy_init(dsv4_group, "device_rdma", False)


def test_lazy_store_reports_miss_and_defers_host_registration():
    backend = _make_lazy_memcache()
    tensor = torch.empty(16, dtype=torch.uint8)

    backend.register_buffer(tensor)

    assert _LifecycleObjectStore.instances == []
    assert backend._batch_exist(["k0", "k1"]) == [0, 0]
    assert backend._get_batch_zero_copy_impl(["k0"], [123], [16]) == [-1]

    backend.prepare_for_backup()

    store = _LifecycleObjectStore.instances[0]
    assert store.setup_calls == 1
    assert store.init_calls == [(3, True)]
    assert store.registered_buffers == [(tensor.data_ptr(), 16)]


def test_lazy_store_first_put_initializes_only_once():
    backend = _make_lazy_memcache()
    source = ctypes.create_string_buffer(b"0123456789abcdef")

    assert backend._put_batch_zero_copy_impl(
        ["k0"], [ctypes.addressof(source)], [16]
    ) == [0]
    backend.prepare_for_backup()

    assert len(_LifecycleObjectStore.instances) == 1
    store = _LifecycleObjectStore.instances[0]
    assert len(store.put_calls) == 1
    assert store.put_calls[0][0] == ["k0"]
    assert store.put_calls[0][2:] == ([16], None)
    assert store.put_calls[0][1] == [ctypes.addressof(source)]
    assert store.objects["k0"] == b"0123456789abcdef"


def test_lazy_store_keeps_original_addresses_for_io():
    backend = _make_lazy_memcache()
    source = ctypes.create_string_buffer(b"0123456789abcdef")
    destination = ctypes.create_string_buffer(16)

    assert backend._put_batch_zero_copy_impl(
        ["k0"], [ctypes.addressof(source)], [16]
    ) == [0]
    assert backend._get_batch_zero_copy_impl(
        ["k0"], [ctypes.addressof(destination)], [16]
    ) == [16]

    store = _LifecycleObjectStore.instances[0]
    assert store.put_calls[0][1] == [ctypes.addressof(source)]
    assert store.put_calls[0][2:] == ([16], None)
    assert store.get_call[1] == [ctypes.addressof(destination)]
    assert store.get_call[2:] == ([16], None)
    assert destination.raw == b"0123456789abcdef"


def test_device_rdma_lazy_init_registers_buffers_and_keeps_zero_copy_io():
    backend = _make_lazy_memcache(protocol="device_rdma")
    tensor = torch.empty(16, dtype=torch.uint8)
    source = ctypes.create_string_buffer(b"0123456789abcdef")
    destination = ctypes.create_string_buffer(16)

    backend.register_buffer(tensor)
    assert _LifecycleObjectStore.instances == []

    backend.prepare_for_backup()

    store = _LifecycleObjectStore.instances[0]
    assert store.registered_buffers == [(tensor.data_ptr(), 16)]
    assert backend._put_batch_zero_copy_impl(
        ["k0"], [ctypes.addressof(source)], [16]
    ) == [0]
    assert backend._get_batch_zero_copy_impl(
        ["k0"], [ctypes.addressof(destination)], [16]
    ) == [16]
    assert store.put_calls[0][1] == [ctypes.addressof(source)]
    assert store.get_call[1] == [ctypes.addressof(destination)]
    assert destination.raw == b"0123456789abcdef"


def test_logical_anchor_is_a_successful_noop():
    backend = _make_memcache()

    assert backend.batch_exists(["h0", "h1"]) == 2
    assert backend.batch_get_v1(["h0", "h1"], torch.arange(256)) == [True, True]
    assert backend.batch_set_v1(["h0", "h1"], torch.arange(256)) == [True, True]


def test_side_pool_registration_does_not_precheck_layout():
    backend = _make_memcache()
    backend.register_buffer = Mock()
    buffer = object()
    host_pool = SimpleNamespace(
        layout="layer_first",
        get_hybrid_pool_buffer=lambda: [buffer],
    )

    backend.register_mem_host_pool_v2(host_pool, PoolName.DEEPSEEK_V4_C4)
    backend.register_buffer.assert_called_once_with(buffer)


def test_c128_exists_derives_group_endpoint_from_hash_chain():
    keys = [f"h{i}" for i in range(16)]
    object_key = f"h15__{PoolName.DEEPSEEK_V4_C128}"
    backend = _make_memcache([object_key])
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        keys=["__placeholder__"],
        logical_pages_per_object=16,
    )

    result = backend.batch_exists_v2(keys, [transfer])

    assert result.kv_hit_pages == 16
    assert result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C128] == 1


def test_logical_anchor_is_all_or_nothing_when_c128_object_is_missing():
    keys = [f"h{i}" for i in range(16)]
    backend = _make_memcache()
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        keys=["h15"],
        logical_pages_per_object=16,
    )

    result = backend.batch_exists_v2(keys, [transfer])

    assert result.kv_hit_pages == 0
    assert PoolName.DEEPSEEK_V4_C128 not in result.extra_pool_hit_pages


def test_indexer_missing_scale_makes_logical_anchor_miss():
    keys = ["h0", "h1", "h2"]
    existing = {
        "h0__deepseek_v4_c4_indexer",
        "h0__deepseek_v4_c4_indexer_scale",
        "h1__deepseek_v4_c4_indexer",
        # h1 scale is deliberately absent.
        "h2__deepseek_v4_c4_indexer",
        "h2__deepseek_v4_c4_indexer_scale",
    }
    backend = _make_memcache(existing)
    transfer = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4_INDEXER)
    scale_transfer = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE)

    result = backend.batch_exists_v2(keys, [transfer, scale_transfer])

    assert result.kv_hit_pages == 1
    assert result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C4_INDEXER] == 3
    assert result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE] == 1


def test_logical_anchor_returns_common_partial_prefix_for_coarse_c128():
    keys = [f"h{i}" for i in range(32)]
    existing = {
        *[f"h{i}__{PoolName.DEEPSEEK_V4_C4}" for i in range(16)],
        f"h15__{PoolName.DEEPSEEK_V4_C128}",
    }
    backend = _make_memcache(existing)
    transfers = [
        PoolTransfer(name=PoolName.DEEPSEEK_V4_C4),
        PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C128,
            keys=["h15", "h31"],
            logical_pages_per_object=16,
        ),
    ]

    result = backend.batch_exists_v2(keys, transfers)

    assert result.kv_hit_pages == 16
    assert result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C4] == 16
    assert result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C128] == 1


def test_partial_prefix_finds_available_trailing_window():
    keys = [f"h{i}" for i in range(32)]
    existing = {
        *[f"h{i}__{PoolName.DEEPSEEK_V4_C4}" for i in range(16)],
        f"h14__{PoolName.SWA}",
        f"h15__{PoolName.SWA}",
    }
    backend = _make_memcache(existing)
    trailing = PoolTransfer(
        name=PoolName.SWA,
        keys=["h30", "h31"],
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )

    result = backend.batch_exists_v2(
        keys,
        [PoolTransfer(name=PoolName.DEEPSEEK_V4_C4), trailing],
    )

    assert result.kv_hit_pages == 16
    assert result.extra_pool_hit_pages[PoolName.SWA] == 16


def test_physical_pools_round_trip_independently():
    backend = _make_memcache()
    backend.store = _LifecycleObjectStore()
    backend._store_initialized = True
    backend._batch_exist = lambda keys: [int(k in backend.store.objects) for k in keys]
    buffers = {
        PoolName.DEEPSEEK_V4_C4: ctypes.create_string_buffer(b"compressed-kv"),
        PoolName.DEEPSEEK_V4_C128: ctypes.create_string_buffer(b"c128-kv"),
        PoolName.DEEPSEEK_V4_C4_INDEXER: ctypes.create_string_buffer(b"index-key"),
        PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE: ctypes.create_string_buffer(b"scale"),
    }
    expected = {name: bytes(buf) for name, buf in buffers.items()}
    transfers = []
    for name, buf in buffers.items():
        page_size = 16 if name == PoolName.DEEPSEEK_V4_C128 else 128
        backend.registered_pools[name] = SimpleNamespace(
            page_size=page_size,
            get_page_buffer_meta=lambda indices, buf=buf: (
                [ctypes.addressof(buf)],
                [ctypes.sizeof(buf)],
            ),
        )
        transfers.append(
            PoolTransfer(name=name, host_indices=torch.arange(page_size), keys=["h0"])
        )

    assert backend.batch_set_v2(transfers) == {name: [True] for name in buffers}
    assert set(backend.store.objects) == {f"h0__{name}" for name in buffers}
    for buf in buffers.values():
        ctypes.memset(ctypes.addressof(buf), 0, ctypes.sizeof(buf))
    assert backend.batch_get_v2(transfers) == {name: [True] for name in buffers}
    assert {name: bytes(buf) for name, buf in buffers.items()} == expected


def _make_host_group():
    pool = LogicalHostPool(4096, 128)
    return HostPoolGroup(
        [
            PoolEntry(
                PoolName.KV,
                pool,
                None,
                lambda layer: layer,
                is_primary_index_anchor=True,
            )
        ]
    )


def test_c128_prefetch_transfer_uses_runtime_coverage():
    component = C128SidecarComponent.__new__(C128SidecarComponent)
    component.cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            c128_attn_allocator=SimpleNamespace(page_size=16)
        )
    )
    component.tree_core = SimpleNamespace(page_size=128)

    transfer = component.build_hicache_transfers(
        SimpleNamespace(),
        phase=CacheTransferPhase.PREFETCH,
        staging_tokens=16,
        prefetch_tokens=16 * 128,
    )[0]

    assert transfer.name == PoolName.DEEPSEEK_V4_C128
    assert transfer.keys == ["__placeholder__"]
    assert transfer.host_indices is None
    assert transfer.logical_pages_per_object == 16


def test_c128_prefetch_transfer_supports_page_size_thirty_two():
    component = C128SidecarComponent.__new__(C128SidecarComponent)
    component.cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            c128_attn_allocator=SimpleNamespace(page_size=32)
        )
    )
    component.tree_core = SimpleNamespace(page_size=128)

    transfer = component.build_hicache_transfers(
        SimpleNamespace(),
        phase=CacheTransferPhase.PREFETCH,
        staging_tokens=32,
        prefetch_tokens=32 * 128,
    )[0]

    assert transfer.keys == ["__placeholder__"]
    assert transfer.host_indices is None
    assert transfer.logical_pages_per_object == 32


def test_c128_exists_accepts_two_explicit_group_keys():
    keys = [f"h{i}" for i in range(32)]
    backend = _make_memcache(
        [
            f"h15__{PoolName.DEEPSEEK_V4_C128}",
            f"h31__{PoolName.DEEPSEEK_V4_C128}",
        ]
    )
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        keys=["h15", "h31"],
        logical_pages_per_object=16,
    )

    result = backend.batch_exists_v2(keys, [transfer])

    assert result.kv_hit_pages == 32
    assert result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C128] == 2


class _FakeLRU:
    def __init__(self):
        self.nodes = set()

    def in_list(self, node):
        return node in self.nodes

    def insert_mru(self, node):
        self.nodes.add(node)


class _FakeNode:
    def __init__(self, key, parent=None, hashes=None, host_value=None):
        self.id = id(self)
        self.key = key
        self.parent = parent
        self.hash_value = hashes
        self.component_data = {ComponentType.C128: ComponentData(host_value=host_value)}


def _make_c128_component_and_path(page_size=16):
    component = C128SidecarComponent.__new__(C128SidecarComponent)
    component.cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            c128_attn_allocator=SimpleNamespace(page_size=page_size)
        )
    )
    root = _FakeNode([])
    hashes = [f"h{i}" for i in range(page_size)]
    tail = _FakeNode(list(range(128 * page_size)), parent=root, hashes=hashes)
    lru = _FakeLRU()
    component.tree_core = SimpleNamespace(
        page_size=128,
        root_node=root,
        host_lru_lists={ComponentType.C128: lru},
        node_by_id=lambda node_id: tail if node_id == tail.id else root,
        _update_evictable_leaf_sets=lambda node: None,
    )
    return component, root, tail, lru


def test_c128_storage_prefetch_alignment_uses_absolute_anchor_depth():
    component, root, aligned_tail, _ = _make_c128_component_and_path()
    partial_anchor = _FakeNode(list(range(128)), parent=root, hashes=["h0"])

    assert component.align_storage_prefetch_length(root, 4095) == 2048
    assert component.align_storage_prefetch_length(aligned_tail, 4096) == 4096
    assert component.align_storage_prefetch_length(partial_anchor, 4096) == 0


def test_c128_backup_uses_group_endpoint_hash():
    component, _, tail, _ = _make_c128_component_and_path()
    tail.component_data[ComponentType.C128].host_value = torch.arange(16)

    transfer = component.build_hicache_transfers(
        tail, CacheTransferPhase.BACKUP_STORAGE
    )[0]

    assert transfer.keys == ["h15"]
    assert torch.equal(transfer.host_indices, torch.arange(16))


def test_c128_prefetch_commit_publishes_only_complete_group():
    component, root, tail, lru = _make_c128_component_and_path()
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        host_indices=torch.arange(16),
        keys=["h15"],
    )
    result = PoolTransferResult(
        kv_hit_pages=16,
        extra_pool_hit_pages={PoolName.DEEPSEEK_V4_C128: 1},
    )

    component.commit_hicache_transfer(
        root,
        CacheTransferPhase.PREFETCH,
        [transfer],
        cache_actions=[],
        insert_result=InsertResult(
            prefix_len=0,
            total_len=2048,
            inserted_host_node=tail.id,
        ),
        pool_storage_result=result,
    )

    assert torch.equal(
        tail.component_data[ComponentType.C128].host_value, torch.arange(16)
    )
    assert lru.in_list(tail)


def test_c128_successful_load_back_rebinds_pages_to_request():
    component, _, tail, _ = _make_c128_component_and_path()
    tail.component_data[ComponentType.C128].value = torch.tensor([7])
    bound = []
    component.cache.req_to_token_pool = SimpleNamespace(
        set_c128_prefix_pages=lambda req, pages: bound.append((req, pages.clone()))
    )
    req = SimpleNamespace(best_match_node=tail.id)

    prep = component.prepare_load_back(tail.id, req=req)
    component.finalize_load_back(req, prep, success=True)

    assert prep == PrepareLoadBackResult()
    assert len(bound) == 1
    assert bound[0][0] is req
    assert torch.equal(bound[0][1], torch.tensor([7]))


def test_c128_failed_load_back_keeps_provisional_request_mapping():
    component, _, tail, _ = _make_c128_component_and_path()
    component.cache.req_to_token_pool = SimpleNamespace(
        set_c128_prefix_pages=lambda req, pages: (_ for _ in ()).throw(
            AssertionError("failed load-back must not rebind")
        )
    )
    req = SimpleNamespace(best_match_node=tail.id)

    component.finalize_load_back(
        req,
        component.prepare_load_back(tail.id, req=req),
        success=False,
    )


def test_c128_prefetch_sizes_without_allocating_until_hit():
    component, root, _, _ = _make_c128_component_and_path()
    component._c128_kv_pool_host = SimpleNamespace(
        alloc=Mock(return_value=torch.arange(16))
    )
    prep = component.prepare_prefetch(root.id, prefetch_tokens=4096)
    assert prep.staging_tokens == 32
    component._c128_kv_pool_host.alloc.assert_not_called()

    # Only the first of the two candidate objects exists: allocate one page.
    staging = component.alloc_prefetch_staging(16)
    component._c128_kv_pool_host.alloc.assert_called_once_with(16)
    assert staging.numel() == 16


def test_hit_time_staging_counts_c128_objects_not_full_pages():
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        keys=["h15", "h31"],
        logical_pages_per_object=16,
    )
    alloc = Mock(return_value=torch.arange(16))
    cache = SimpleNamespace(
        page_size=128,
        cache_controller=SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                entry_map={
                    PoolName.DEEPSEEK_V4_C128: SimpleNamespace(
                        host_pool=SimpleNamespace(page_size=16)
                    )
                }
            )
        ),
        components={ComponentType.C128: SimpleNamespace(alloc_prefetch_staging=alloc)},
    )
    info = SimpleNamespace(comp_xfers={ComponentType.C128: [transfer]})
    assert UnifiedRadixCache._alloc_prefetch_aux_staging(cache, info, 2048)
    alloc.assert_called_once_with(16)
    assert transfer.host_indices.numel() == 16


def test_upstream_hybrid_pool_keys_remain_supported():
    backend = _make_memcache()
    backend.registered_pools[PoolName.MAMBA] = SimpleNamespace(
        temporal_state_elem_size=0, conv_buffer=[object(), object()]
    )
    assert backend._get_hybrid_page_component_keys(
        ["h0"], PoolTransfer(name=PoolName.MAMBA)
    ) == (["h0_0_conv_0", "h0_0_conv_1"], 2)
    assert backend._get_hybrid_page_component_keys(
        ["h0"], PoolTransfer(name=PoolName.DRAFT)
    ) == (["h0_0_k", "h0_0_v"], 2)
    with pytest.raises(ValueError, match="Unsupported hybrid pool"):
        backend._get_hybrid_page_component_keys(
            ["h0"], PoolTransfer(name="unsupported")
        )


@pytest.mark.parametrize("fp8_packed", [False, True])
def test_upstream_mla_indexer_scale_and_packed_kv_keys(fp8_packed):
    backend = _make_memcache()
    sizes = 3 if fp8_packed else 4
    backend.mem_pool_host = SimpleNamespace(
        layout="page_first_kv_split",
        dsa_kv_cache_store_fp8=fp8_packed,
        index_k_buffer=object(),
        index_k_scale_buffer=object(),
        get_page_buffer_meta=lambda indices: (list(range(sizes)), [16] * sizes),
    )
    keys, _, _ = backend._get_mla_buffer_meta(["h0"], torch.arange(128))
    assert keys == ["h0__k"] + ([] if fp8_packed else ["h0__v"]) + [
        "h0__index_k",
        "h0__scale",
    ]
    assert backend._get_key_multiplier() == sizes


@pytest.mark.parametrize("backup_skip", [False, True])
def test_backup_initializes_runtime_on_every_rank(backup_skip):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = 128
    controller.backup_skip = backup_skip
    controller.storage_backend_type = "npu_memcache"
    controller.mem_pool_host = _make_host_group()
    backend = controller.storage_backend = _make_memcache()
    backend.prepare_for_backup = Mock()
    backend.batch_set_v2 = Mock(return_value={PoolName.DEEPSEEK_V4_C128: [True]})
    controller.page_set_func = controller._page_set_zero_copy
    operation = PrefetchOperation(
        CacheRequestHandle("req", 0),
        list(range(2048)),
        pool_transfers=[PoolTransfer(name=PoolName.DEEPSEEK_V4_C128, keys=["h15"])],
    )
    operation.hash_value = [f"h{i}" for i in range(16)]
    operation.host_indices = torch.arange(2048)

    controller._page_backup(operation)

    backend.prepare_for_backup.assert_called_once()
    assert backend.batch_set_v2.call_count == int(not backup_skip)


@pytest.mark.parametrize(
    "failed_pool",
    [
        None,
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
        PoolName.DEEPSEEK_V4_C128,
        PoolName.SWA,
    ],
)
def test_logical_anchor_prefetch_requires_all_physical_pools(failed_pool):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = 128
    controller.mem_pool_host = _make_host_group()
    controller.prefetch_sync_queue = Queue()
    controller.storage_backend = _make_memcache()
    controller.page_get_func = controller._page_get_zero_copy
    controller.storage_backend.batch_get_v2 = lambda transfers, **kwargs: {
        t.name: [t.name != failed_pool] * len(t.keys) for t in transfers
    }
    transfers = [
        PoolTransfer(name=name, indices_from_pool=PoolName.KV)
        for name in (
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
        )
    ] + [
        PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C128,
            keys=["__placeholder__"],
            host_indices=torch.arange(16),
            logical_pages_per_object=16,
        ),
        PoolTransfer(
            name=PoolName.SWA,
            keys=["__placeholder__"],
            host_indices=torch.arange(128),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        ),
    ]
    request = CacheRequestHandle("req", 0)
    operation = PrefetchOperation(request, list(range(2048)), pool_transfers=transfers)
    operation.hash_value = [f"h{i}" for i in range(16)]
    operation.host_indices = torch.arange(2048)

    controller._page_transfer(operation)
    for ack in controller.prefetch_sync_queue.queue:
        if ack.completed_tokens is not None:
            operation.completed_tokens = ack.completed_tokens
        if ack.pool_hits is not None:
            operation.pool_storage_result.update_extra_pool_hit_pages(ack.pool_hits)
    operation.pool_transfers_done = True
    controller.append_host_mem_release = Mock()
    controller.prefetch_tokens_occupied = 2048
    cache = SimpleNamespace(
        page_size=128,
        cache_controller=controller,
        storage_existence_cache=SimpleNamespace(invalidate_beyond=Mock()),
        _finish_storage_prefetch=Mock(),
        buffer_pipeline=None,
        ongoing_prefetch={request: operation},
        _prefetch_occupied_span=lambda *_: 2048,
        prefetch_loaded_tokens_by_reqid={},
        prefetch_loaded_storage_start_by_reqid={},
    )
    accepted = UnifiedRadixCache._check_hybrid_prefetch_result(
        cache,
        request,
        operation,
        operation.completed_tokens,
        operation.hash_value,
        operation.host_indices,
        None,
        None,
        list(range(2048)),
    )
    assert accepted == (failed_pool is None)
    assert controller.append_host_mem_release.call_count == int(failed_pool is not None)


@pytest.mark.parametrize("assume_stored", [False, True])
@pytest.mark.parametrize("hit_tokens", [2048, 4096])
def test_c128_read_uses_namespaced_keys_after_rank_hit_reduction(
    assume_stored, hit_tokens
):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = 128
    controller.mem_pool_host = _make_host_group()
    controller.prefetch_sync_queue = Queue()
    controller.page_get_func = controller._page_get_zero_copy
    key = RadixKey(list(range(4096)), extra_key="tenant", cache_salt="salt")
    hashes = get_storage_hash_str(key, page_size=128)
    backend = controller.storage_backend = _make_memcache(
        [f"{h}__{PoolName.DEEPSEEK_V4_C128}" for h in hashes[15::16]]
    )
    backend.batch_get_v2 = Mock(
        return_value={PoolName.DEEPSEEK_V4_C128: [True] * (hit_tokens // 2048)}
    )
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        keys=["__placeholder__"] * 2,
        logical_pages_per_object=16,
    )
    operation = PrefetchOperation(
        CacheRequestHandle("req", 0),
        key,
        pool_transfers=[transfer],
        assume_stored=assume_stored,
    )
    hit_hashes, tokens = controller._storage_hit_query(operation)
    assert tokens == 4096
    assert transfer.host_indices is None

    # The scheduler allocates staging after reducing the hit length across ranks.
    operation.hash_value = hit_hashes[: hit_tokens // 128]
    operation.host_indices = torch.arange(hit_tokens)
    transfer.host_indices = torch.arange(hit_tokens // 128)
    controller._page_transfer(operation)

    assert transfer.keys == hashes[15 : hit_tokens // 128 : 16]
    backend.batch_get_v2.assert_called_once()


def test_trailing_pools_find_a_common_shorter_prefix():
    backend = _make_memcache(
        {
            "h1__deepseek_v4_c4_state",
            "h3__deepseek_v4_c4_state",
            "h1__deepseek_v4_c4_indexer_state",
            "h2__deepseek_v4_c4_indexer_state",
        }
    )
    result = backend.batch_exists_v2(
        [f"h{i}" for i in range(4)],
        [
            PoolTransfer(
                name=name,
                keys=["__placeholder__"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
            for name in (
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            )
        ],
    )
    assert result.kv_hit_pages == 2
    assert result.restorable_prefix_pages == [2]


@pytest.mark.parametrize("coarse", [False, True])
def test_memory_pressure_preserves_complete_c128_groups(coarse):
    request = CacheRequestHandle("req", 0)
    operation = PrefetchOperation(
        request,
        list(range(2048)),
        pool_transfers=[
            PoolTransfer(
                name=PoolName.DEEPSEEK_V4_C128 if coarse else PoolName.SWA,
                keys=["__placeholder__"],
                logical_pages_per_object=16 if coarse else 1,
                hit_policy=PoolHitPolicy.ALL_PAGES
                if coarse
                else PoolHitPolicy.TRAILING_PAGES,
            )
        ],
    )
    operation.storage_hit_count = 2048
    operation.hash_value = [f"h{i}" for i in range(16)]
    host_pool = SimpleNamespace(
        alloc=Mock(side_effect=[None, None, torch.arange(1024)]),
        available_size=lambda: 1024,
    )
    controller = SimpleNamespace(
        mem_pool_host=host_pool,
        prefetch_hit_queue=Queue(),
        prefetch_buffer=Queue(),
        ack_prefetch_queue=Queue(),
        ack_backup_queue=Queue(),
        host_mem_release_queue=Queue(),
    )
    controller.prefetch_hit_queue.put(operation)
    cache = SimpleNamespace(
        cache_controller=controller,
        host_memory_mode="cache",
        page_size=128,
        prefetch_threshold=128,
        ongoing_prefetch={request: Mock()},
        storage_prefetch_retries=Mock(),
        evict_host=Mock(),
        _record_storage_prefetch_hit=Mock(),
        _invalidate_absent_from_hit_query=Mock(),
        _account_prefetch_outcome=Mock(),
        _alloc_prefetch_aux_staging=Mock(return_value=True),
        _resolve_storage_prefetch_tokens=Mock(),
        _finish_storage_prefetch=Mock(),
        revoke_pending_prefetch=Mock(),
        _log_storage_prefetch_deferred=Mock(),
    )

    UnifiedRadixCache._drain_storage_control_queues_impl(cache, 1, 0, 0, 0, {}, False)

    if coarse:
        assert host_pool.alloc.call_count == 2
        assert controller.prefetch_buffer.empty()
        cache.revoke_pending_prefetch.assert_called_once_with(request)
    else:
        assert controller.prefetch_buffer.get_nowait() is operation
        assert operation.storage_hit_count == 1024
        cache.revoke_pending_prefetch.assert_not_called()
