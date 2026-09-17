import ctypes
import threading
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.managers.cache_controller import HiCacheController
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
    backend._use_dram_staging = protocol == "device_sdma"
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


def test_device_transports_lazy_init_is_limited_to_dsv4_pool_groups():
    dsv4_group = SimpleNamespace(
        entries=[SimpleNamespace(name=PoolName.DEEPSEEK_V4_C4)]
    )
    ordinary_group = SimpleNamespace(entries=[SimpleNamespace(name=PoolName.KV)])

    assert NpuMemcacheStore._should_lazy_init(dsv4_group, "device_sdma", True)
    assert NpuMemcacheStore._should_lazy_init(dsv4_group, "device_rdma", True)
    assert not NpuMemcacheStore._should_lazy_init(ordinary_group, "device_sdma", True)
    assert not NpuMemcacheStore._should_lazy_init(ordinary_group, "device_rdma", True)
    assert not NpuMemcacheStore._should_lazy_init(dsv4_group, "host_shm", True)
    assert not NpuMemcacheStore._should_lazy_init(dsv4_group, "device_rdma", False)


def test_logical_anchor_uses_controller_pool_names_for_lazy_init():
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mem_pool_device = object()
    controller.mem_pool_host = SimpleNamespace(
        layout="page_first_direct",
        entries=[
            SimpleNamespace(name=PoolName.KV),
            SimpleNamespace(name=PoolName.DEEPSEEK_V4_C4),
        ],
    )
    controller.enable_storage_metrics = False
    controller.get_attn_cp_rank_and_size = lambda: (0, 1)

    parallel = SimpleNamespace(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
    with (
        patch(
            "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
            return_value=False,
        ),
        patch(
            "sglang.srt.managers.cache_controller.get_parallel",
            return_value=parallel,
        ),
    ):
        storage_config = controller._generate_storage_config("dsv4", {})

    logical_anchor = SimpleNamespace(kv_buffer=None)
    assert storage_config.host_pool_names == (
        str(PoolName.KV),
        str(PoolName.DEEPSEEK_V4_C4),
    )

    assert NpuMemcacheStore._should_lazy_init(
        logical_anchor,
        "device_sdma",
        True,
        storage_config.host_pool_names,
    )
    assert NpuMemcacheStore._should_lazy_init(
        logical_anchor,
        "device_rdma",
        True,
        storage_config.host_pool_names,
    )
    assert not NpuMemcacheStore._should_lazy_init(
        logical_anchor,
        "host_shm",
        True,
        storage_config.host_pool_names,
    )


def test_lazy_store_reports_miss_and_skips_misclassified_host_registration():
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
    assert store.registered_buffers == []


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
    assert store.put_calls[0][1] != [ctypes.addressof(source)]
    assert store.objects["k0"] == b"0123456789abcdef"


def test_lazy_store_stages_npu_pinned_host_addresses_through_process_dram():
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
    assert store.put_calls[0][1] != [ctypes.addressof(source)]
    assert store.put_calls[0][2:] == ([16], None)
    assert store.get_call[1] != [ctypes.addressof(destination)]
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


def test_side_pool_registration_rejects_layer_first_layout():
    backend = _make_memcache()
    backend.register_buffer = lambda _buffer: None
    host_pool = SimpleNamespace(
        layout="layer_first",
        get_hybrid_pool_buffer=lambda: [object()],
    )

    try:
        backend.register_mem_host_pool_v2(host_pool, PoolName.DEEPSEEK_V4_C4)
    except ValueError as exc:
        assert "page-first" in str(exc)
    else:
        raise AssertionError("layer_first side pool registration must fail")


def test_c128_exists_uses_explicit_terminal_key_for_aligned_candidate():
    keys = [f"h{i}" for i in range(16)]
    object_key = f"h15__{PoolName.DEEPSEEK_V4_C128}"
    backend = _make_memcache([object_key])
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        keys=["h15"],
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


def test_partial_prefix_realigns_trailing_state_keys():
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
    assert trailing.keys == ["h14", "h15"]
    assert result.extra_pool_hit_pages[PoolName.SWA] == 2


def test_partial_prefix_trims_coarse_buffer_and_releases_tail():
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mem_pool_host = SimpleNamespace(
        entry_map={
            PoolName.DEEPSEEK_V4_C128: SimpleNamespace(
                host_pool=SimpleNamespace(page_size=16)
            )
        }
    )
    released = []
    controller.append_host_mem_release = lambda **kwargs: released.extend(
        kwargs["extra_pools"]
    )
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        host_indices=torch.arange(32),
        keys=["h15", "h31"],
        logical_pages_per_object=16,
    )

    controller._trim_prefetch_transfers([transfer], [f"h{i}" for i in range(32)], 16)

    assert transfer.keys == ["h15"]
    assert torch.equal(transfer.host_indices, torch.arange(16))
    assert len(released) == 1
    assert torch.equal(released[0].host_indices, torch.arange(16, 32))


@pytest.mark.parametrize(
    "name",
    [
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
    ],
)
def test_storage_derived_pools_reuse_logical_anchor_indices(name):
    anchor = torch.cat((torch.arange(128), torch.arange(256, 384)))
    transfer = PoolTransfer(name=name, indices_from_pool=PoolName.KV)
    operation = SimpleNamespace(
        host_indices=anchor, hash_value=["h0", "h1"], pool_transfers=[transfer]
    )
    controller = HybridCacheController.__new__(HybridCacheController)
    controller._resolve_sidecar_nonkv_derived_pool_transfers(operation)
    assert transfer.host_indices is anchor
    assert transfer.keys == operation.hash_value


def test_refactored_indexer_pools_round_trip_independently():
    backend = _make_memcache()
    backend.store = _LifecycleObjectStore()
    backend._store_initialized = True
    backend._use_dram_staging = True
    backend._batch_exist = lambda keys: [int(k in backend.store.objects) for k in keys]
    buffers = {
        PoolName.DEEPSEEK_V4_C4_INDEXER: ctypes.create_string_buffer(b"index-key"),
        PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE: ctypes.create_string_buffer(b"scale"),
    }
    expected = {name: bytes(buf) for name, buf in buffers.items()}
    transfers = []
    for name, buf in buffers.items():
        backend.registered_pools[name] = SimpleNamespace(
            page_size=128,
            get_page_buffer_meta=lambda indices, buf=buf: (
                [ctypes.addressof(buf)],
                [ctypes.sizeof(buf)],
            ),
        )
        transfers.append(
            PoolTransfer(name=name, host_indices=torch.arange(128), keys=["h0"])
        )

    assert backend.batch_set_v2(transfers) == {name: [True] for name in buffers}
    assert set(backend.store.objects) == {f"h0__{name}" for name in buffers}
    for buf in buffers.values():
        ctypes.memset(ctypes.addressof(buf), 0, ctypes.sizeof(buf))
    assert backend.batch_get_v2(transfers) == {name: [True] for name in buffers}
    assert {name: bytes(buf) for name, buf in buffers.items()} == expected


def _make_host_group(physical=False):
    pool = LogicalHostPool(4096, 128)
    if physical:
        pool.kv_buffer = torch.empty(1)
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


@pytest.mark.parametrize("physical", [False, True])
def test_host_group_dispatches_primary_io_by_anchor_pool(physical):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mem_pool_host = _make_host_group(physical)
    controller.storage_host_pool = controller.mem_pool_host.anchor_entry.host_pool
    controller.page_size = 128
    controller.storage_backend_type = "npu_memcache"
    controller.prefetch_sync_queue = Queue()
    controller._page_transfer_sidecar = Mock()
    controller.backup_skip = False
    controller.storage_backend = SimpleNamespace(prepare_for_backup=Mock())
    operation = PrefetchOperation(
        CacheRequestHandle("req", 0), list(range(128)), pool_transfers=[]
    )
    operation.hash_value = ["h0"]
    assert controller.mem_pool_host.kv_buffer is controller.storage_host_pool.kv_buffer
    with patch.object(HiCacheController, "_page_transfer", return_value=1) as read:
        controller._page_transfer(operation)
        assert read.call_count == int(physical)
    with patch.object(HiCacheController, "_page_backup") as write:
        controller._page_backup(operation)
        assert write.call_count == int(physical)


@pytest.mark.parametrize("stored", [True, False])
def test_logical_anchor_backup_requires_successful_physical_pool_write(stored):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = 128
    controller.backup_skip = False
    controller.mem_pool_host = _make_host_group()
    controller.storage_host_pool = controller.mem_pool_host.anchor_entry.host_pool
    controller.storage_backend = SimpleNamespace(
        prepare_for_backup=Mock(),
        batch_set_v2=lambda transfers, **kwargs: {PoolName.DEEPSEEK_V4_C128: [stored]},
    )
    operation = SimpleNamespace(
        pool_transfers=[PoolTransfer(name=PoolName.DEEPSEEK_V4_C128, keys=["h15"])],
        pool_storage_result=PoolTransferResult(kv_hit_pages=0, extra_pool_hit_pages={}),
        prefix_keys=None,
        hash_value=[f"h{i}" for i in range(16)],
        completed_tokens=0,
    )

    controller._page_backup(operation)

    assert operation.completed_tokens == (2048 if stored else 0)


def test_virtual_anchor_prefetch_skips_primary_io_and_loads_real_pool():
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = 128
    controller.storage_backend_type = "npu_memcache"
    controller.mem_pool_host = _make_host_group()
    controller.storage_host_pool = controller.mem_pool_host.anchor_entry.host_pool
    controller.prefetch_sync_queue = Queue()
    calls = []
    controller.storage_backend = SimpleNamespace(
        batch_get_v2=lambda transfers, **kwargs: (
            calls.append(transfers) or {PoolName.DEEPSEEK_V4_C128: [True]}
        )
    )
    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        host_indices=torch.arange(16),
        keys=["h15"],
    )
    operation = PrefetchOperation(
        CacheRequestHandle("req", 0), list(range(2048)), pool_transfers=[transfer]
    )
    operation.hash_value = [f"h{i}" for i in range(16)]
    operation.host_indices = torch.arange(2048)

    controller._page_transfer(operation)

    acks = []
    while not controller.prefetch_sync_queue.empty():
        acks.append(controller.prefetch_sync_queue.get())

    assert max(ack.completed_tokens or 0 for ack in acks) == 2048
    assert len(calls) == 1
    assert any(
        ack.pool_hits and ack.pool_hits.get(PoolName.DEEPSEEK_V4_C128.value, 0) == 1
        for ack in acks
    )


@pytest.mark.parametrize(
    "failed_pool",
    [
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
    ],
)
def test_logical_anchor_rejects_failed_derived_read(failed_pool):
    request = CacheRequestHandle("req", 0)
    transfers = [
        PoolTransfer(name=failed_pool, keys=["h0"], indices_from_pool=PoolName.KV),
        PoolTransfer(name=PoolName.SWA, keys=["h0"]),
    ]
    operation = PrefetchOperation(
        CacheRequestHandle("req", 0), list(range(128)), pool_transfers=transfers
    )
    operation.completed_tokens = 128
    operation.pool_transfers_done = True
    operation.pool_storage_result.update_extra_pool_hit_pages(
        {failed_pool: 0, PoolName.SWA: 1}
    )
    release = Mock()
    host_group = _make_host_group()
    cache = SimpleNamespace(
        page_size=128,
        cache_controller=SimpleNamespace(
            mem_pool_host=host_group,
            storage_host_pool=host_group.anchor_entry.host_pool,
            append_host_mem_release=release,
            prefetch_tokens_occupied=128,
        ),
        storage_existence_cache=SimpleNamespace(invalidate_beyond=Mock()),
        _finish_storage_prefetch=Mock(),
        buffer_pipeline=None,
        ongoing_prefetch={request: operation},
        _prefetch_occupied_span=lambda *_: 128,
        prefetch_loaded_tokens_by_reqid={},
        prefetch_loaded_storage_start_by_reqid={},
    )
    assert not UnifiedRadixCache._check_hybrid_prefetch_result(
        cache,
        request,
        operation,
        128,
        ["h0"],
        torch.arange(128),
        None,
        None,
        list(range(128)),
    )
    release.assert_called_once()
    assert cache.prefetch_loaded_tokens_by_reqid[request] == 0


def test_sidecar_exception_preserves_ack_sequence():
    def run(fail):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 128
        controller.storage_backend_type = "npu_memcache"
        controller.mem_pool_host = _make_host_group()
        controller.storage_host_pool = controller.mem_pool_host.anchor_entry.host_pool
        controller.prefetch_sync_queue = Queue()
        controller.prefetch_buffer = Queue()
        controller.storage_stop_event = _OneIterationStopEvent()
        controller.storage_backend = SimpleNamespace(
            batch_get_v2=Mock(
                side_effect=RuntimeError("injected read failure") if fail else None,
                return_value={PoolName.SWA: [True]},
            )
        )
        operation = PrefetchOperation(
            CacheRequestHandle("req", 0),
            list(range(128)),
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.SWA, keys=["h0"], host_indices=torch.arange(128)
                )
            ],
        )
        operation.hash_value = ["h0"]
        operation.host_indices = torch.arange(128)
        controller.prefetch_buffer.put(operation)
        controller.prefetch_io_aux_func()
        return list(controller.prefetch_sync_queue.queue)

    success, failure = run(False), run(True)

    def signature(acks):
        return [
            (
                a.completed_tokens is not None,
                a.pool_hits is not None,
                bool(a.completed_req),
            )
            for a in acks
        ]

    assert (
        signature(success)
        == signature(failure)
        == [(True, False, False), (False, True, False), (False, False, True)]
    )
    assert failure[1].pool_hits == {}


class _OneIterationStopEvent:
    def __init__(self):
        self._checks = 0

    def is_set(self):
        self._checks += 1
        return self._checks > 1


def _run_one_hybrid_prefetch_worker(page_transfer):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.storage_stop_event = _OneIterationStopEvent()
    controller.prefetch_buffer = Queue()
    controller.prefetch_sync_queue = Queue()
    controller._page_transfer = page_transfer
    controller.append_host_mem_release = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("the scheduler, not the IO worker, owns prefetch release")
    )

    operation = PrefetchOperation(
        CacheRequestHandle("req-terminal", 0), list(range(128))
    )
    operation.host_indices = torch.arange(128)
    controller.prefetch_buffer.put(operation)
    controller.prefetch_io_aux_func()

    acks = []
    while not controller.prefetch_sync_queue.empty():
        acks.append(controller.prefetch_sync_queue.get())
    return operation, acks


def test_hybrid_prefetch_worker_emits_terminal_ack_on_success():
    operation, acks = _run_one_hybrid_prefetch_worker(lambda _operation: None)

    assert len(acks) == 1
    assert acks[0].operation is operation
    assert acks[0].rid == operation.request_id
    assert acks[0].completed_req is True
    assert not operation.is_terminated()


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


@pytest.mark.parametrize("assume_stored", [False, True])
def test_c128_query_resolves_namespaced_endpoint_keys(assume_stored):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = 128
    controller.storage_backend_type = "npu_memcache"
    controller.mem_pool_host = SimpleNamespace(entry_map={})
    key = RadixKey(list(range(4096)), extra_key="tenant", cache_salt="salt")
    hashes = get_storage_hash_str(key, page_size=128)
    controller.storage_backend = _make_memcache(
        [f"{h}__{PoolName.DEEPSEEK_V4_C128}" for h in hashes[15::16]]
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
    _, tokens = controller._storage_hit_query(operation)
    assert tokens == 4096
    assert transfer.keys == hashes[15::16]
    assert transfer.host_indices is None


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
