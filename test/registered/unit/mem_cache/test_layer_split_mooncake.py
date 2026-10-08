"""Real pools/adapters; only native SDK byte I/O is replaced with host memcpy."""

import ctypes
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.layers.cp.utils import get_layer_shard_range
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    LayerShardStorageSpec,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.layer_split.layer_split_host_view import LayerSplitHostView
from sglang.srt.mem_cache.layer_split.layer_split_staging_pool import StagingComponent
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ByteStore:
    def __init__(self):
        self.objects = {}
        self.registered = []
        self.short_get = False
        self.short_response = False

    def register_buffer(self, pointer, size):
        self.registered.append((pointer, size))
        return 0

    def batch_is_exist(self, keys):
        return [int(key in self.objects) for key in keys]

    def batch_put_from_multi_buffers(self, keys, pointers, sizes, config=None):
        for key, ptrs, lengths in zip(keys, pointers, sizes):
            self.objects.setdefault(
                key, b"".join(ctypes.string_at(p, n) for p, n in zip(ptrs, lengths))
            )
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, pointers, sizes):
        results = []
        for key, ptrs, lengths in zip(keys, pointers, sizes):
            if key not in self.objects:
                results.append(-1)
                continue
            value = self.objects[key]
            if self.short_get:
                value = value[:-1]
            offset = 0
            for ptr, length in zip(ptrs, lengths):
                part = value[offset : offset + length]
                if part:
                    ctypes.memmove(ptr, part, len(part))
                offset += length
            results.append(len(value))
        return results[:-1] if self.short_response else results

    def batch_put_from(self, keys, pointers, sizes, config=None):
        return self.batch_put_from_multi_buffers(
            keys, [[p] for p in pointers], [[n] for n in sizes], config
        )

    def batch_get_into(self, keys, pointers, sizes):
        return self.batch_get_into_multi_buffers(
            keys, [[p] for p in pointers], [[n] for n in sizes]
        )


def pools(rank, shards, layers):
    capacity = -(-layers // shards)
    device = SimpleNamespace(
        layer_num=layers,
        layer_shard_enabled=True,
        layer_shard_rank=rank,
        layer_shard_size=shards,
        _owned_local_layer_range=lambda: get_layer_shard_range(rank, shards, layers),
    )
    kv = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
    idx = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
    for pool in (kv, idx):
        pool.device_pool = device
        pool.layout = "layer_first"
        pool.page_size = 2
        pool.size = 8
        pool.dcp_size = 1
        pool.layer_num = pool.target_layer_num = capacity
        pool.mtp_draft_device_pools = ()
        pool.device = "cpu"
        pool.can_use_write_back_jit = False
    device.host_pool_decls = lambda: (
        SimpleNamespace(pool_name=PoolName.KV, owned_device_layers=None),
        SimpleNamespace(pool_name=PoolName.INDEXER, owned_device_layers=None),
    )
    lo, hi = get_layer_shard_range(rank, shards, layers)
    idx.layer_num = idx.target_layer_num = hi - lo
    kv.dtype = torch.float16
    kv.kv_cache_dim = 2
    kv.token_stride_size = 4
    kv.kv_buffer = torch.zeros((capacity, 8, 1, 2), dtype=torch.float16)
    idx.indexer_dtype = torch.uint8
    idx.indexer_size_per_token = 3
    idx.indexer_page_stride_size = 6
    idx.index_k_with_scale_buffer = torch.zeros(
        (idx.layer_num, 4, 6), dtype=torch.uint8
    )
    return HostPoolGroup(
        [
            PoolEntry(
                PoolName.KV, kv, device, lambda i: i, is_primary_index_anchor=True
            ),
            PoolEntry(PoolName.INDEXER, idx, device, lambda i: i),
        ]
    )


def adapter(wire, host, shard=None, staged=False):
    result = MooncakeStore.__new__(MooncakeStore)
    result.layer_shard = shard
    result.mem_pool_host = host.get_pool(PoolName.KV)
    result.registered_pools = {PoolName.INDEXER: host.get_pool(PoolName.INDEXER)}
    result.external_buffer_pools = (
        frozenset((PoolName.KV, PoolName.INDEXER)) if staged else frozenset()
    )
    result.store = wire
    result.config_prefix = "test-model"
    result.is_mla_backend = True
    result.mla_suffix = "" if shard is None else shard.key_suffix
    result._replicate_config_cls = SimpleNamespace
    result._use_group_semantics = False
    result.enable_storage_metrics = False
    return result


@pytest.mark.parametrize("staged", [False, True])
def test_backends_share_native_and_external_buffer_resolution(staged):
    host = pools(0, 2, 5)
    store = adapter(ByteStore(), host, staged=staged)
    assert type(store)._transfer_buffer_pool is HiCacheStorage._transfer_buffer_pool
    for name in (PoolName.KV, PoolName.INDEXER):
        transfer = PoolTransfer(name)
        if staged:
            with pytest.raises(ValueError, match="explicit physical buffer binding"):
                store._transfer_buffer_pool(transfer)
        else:
            expected = host.get_pool(name)
            assert store._transfer_buffer_pool(transfer) is expected
        alias = f"staging-{name}"
        physical_pool = object()
        store.registered_pools[alias] = physical_pool
        transfer.buffer_pool_name = alias
        assert store._transfer_buffer_pool(transfer) is physical_pool
        transfer.buffer_pool_name = "unregistered-alias"
        with pytest.raises(KeyError):
            store._transfer_buffer_pool(transfer)


@pytest.mark.parametrize("layers,shards", [(5, 2), (78, 8), (2, 4)])
def test_direct_rank_keys_owned_spans_roundtrip_and_empty_rank(layers, shards):
    wire = ByteStore()
    keys = ["page-a", "page-b"]
    indices = torch.tensor([0, 1, 4, 5])
    cases = []
    total_bytes = 0
    for rank in range(shards):
        host = pools(rank, shards, layers)
        view = LayerSplitHostView(host)
        start, end = get_layer_shard_range(rank, shards, layers)
        shard = LayerShardStorageSpec(rank, shards, layers, start, end)
        store = adapter(wire, host, shard=shard)
        expected = {}
        for component, shard_view in view.shard_views.items():
            for page, token_base in enumerate((0, 4)):
                size = shard_view.owned_layers * 2 * shard_view.row_bytes
                value = (torch.arange(size) + rank * 29 + page * 17).to(torch.uint8)
                expected[component, token_base] = value
                shard_view.write_page(token_base, value)
                total_bytes += size
        transfers = [
            PoolTransfer(name, keys=keys, host_indices=indices)
            for name in (PoolName.KV, PoolName.INDEXER)
        ]
        result = store.batch_set_v2(transfers)
        assert all(value == [True, True] for value in result.values())
        assert store.batch_exists_v2(keys, transfers).kv_hit_pages == 2
        cases.append((store, host, view, transfers, expected))
    assert len(wire.objects) == 2 * 2 * min(layers, shards)
    assert (
        sum(map(len, wire.objects.values())) == total_bytes == layers * 2 * 2 * (4 + 3)
    )
    for store, host, view, transfers, expected in cases:
        host.get_pool(PoolName.KV).kv_buffer.fill_(0)
        host.get_pool(PoolName.INDEXER).index_k_with_scale_buffer.fill_(0)
        result = store.batch_get_v2(transfers)
        assert all(value == [True, True] for value in result.values())
        for (component, token_base), want in expected.items():
            got = torch.empty_like(want)
            view.shard_views[component].read_page(token_base, got)
            assert torch.equal(got, want)
        for shard_view in view.shard_views.values():
            pool = shard_view.pool
            buffer = (
                pool.kv_buffer
                if shard_view.component == "target"
                else pool.index_k_with_scale_buffer
            )
            assert not torch.count_nonzero(buffer[shard_view.owned_layers :]).item()


def staged_case():
    wire = ByteStore()
    store = adapter(wire, pools(0, 2, 5), staged=True)
    transfers, slabs = [], []
    for name, width in ((PoolName.KV, 4), (PoolName.INDEXER, 3)):
        slab = StagingComponent(
            name=str(name),
            num_pages=2,
            layer_count=5,
            page_size=2,
            row_bytes=width,
            pin_memory=False,
        )
        slab.buffer.view(-1).copy_(torch.arange(slab.buffer.numel()).to(torch.uint8))
        alias = f"physical-{name}"
        store.register_mem_host_pool_v2(slab, alias)
        transfers.append(
            PoolTransfer(
                name,
                keys=["p0", "p1"],
                host_indices=torch.arange(2),
                buffer_pool_name=alias,
                indices_from_pool=PoolName.KV if name == PoolName.INDEXER else None,
            )
        )
        slabs.append(slab)
    return store, wire, transfers, slabs


def test_staged_disjoint_slabs_use_independent_objects_and_query_keys():
    store, wire, transfers, slabs = staged_case()
    expected = [slab.buffer.clone() for slab in slabs]
    result = store.batch_set_v2(transfers)
    assert result == {PoolName.KV: [True, True], PoolName.INDEXER: [True, True]}
    assert len(wire.objects) == 4
    assert sum(map(len, wire.objects.values())) == 2 * 5 * 2 * (4 + 3)
    assert (
        store.batch_exists_v2(
            ["p0", "p1", "absent"],
            [PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)],
        ).kv_hit_pages
        == 2
    )
    for slab in slabs:
        slab.buffer.zero_()
    assert store.batch_get_v2(transfers) == result
    assert all(torch.equal(slab.buffer, value) for slab, value in zip(slabs, expected))


@pytest.mark.parametrize("fault", ["short_get", "short_response"])
def test_mooncake_staged_short_get_fails_both_components_closed(fault):
    store, wire, transfers, _ = staged_case()
    store.batch_set_v2(transfers)
    setattr(wire, fault, True)
    assert store.batch_get_v2(transfers) == {
        PoolName.KV: [False, False],
        PoolName.INDEXER: [False, False],
    }


def test_mooncake_staged_rejects_implicit_l2_binding():
    store, _, transfers, _ = staged_case()
    transfers[0].buffer_pool_name = None
    with pytest.raises(ValueError, match="physical buffer binding"):
        store.batch_get_v2(transfers)


def test_mooncake_external_pool_registration_does_not_register_l2():
    store, wire, _, _ = staged_case()
    prior = list(wire.registered)
    store.register_mem_pool_host(store.mem_pool_host)
    store.register_mem_host_pool_v2(
        store.registered_pools[PoolName.INDEXER], PoolName.INDEXER
    )
    assert wire.registered == prior


def test_mooncake_direct_long_indexer_get_uses_native_storage_batch_bound():
    pages = 2 * STORAGE_BATCH_SIZE + 3
    host = pools(0, 2, 5)
    indexer = host.get_pool(PoolName.INDEXER)
    indexer.size = pages * indexer.page_size
    anchor = host.get_pool(PoolName.KV)
    anchor.size = indexer.size
    anchor.kv_buffer = torch.zeros(
        (anchor.layer_num, anchor.size, 1, anchor.kv_cache_dim), dtype=anchor.dtype
    )
    indexer.index_k_with_scale_buffer = (
        torch.arange(indexer.layer_num * pages * indexer.indexer_page_stride_size)
        .to(torch.uint8)
        .reshape(indexer.layer_num, pages, indexer.indexer_page_stride_size)
    )
    wire = ByteStore()
    store = adapter(
        wire,
        host,
        shard=LayerShardStorageSpec(0, 2, 5, 0, 3),
    )
    transfer = PoolTransfer(
        PoolName.INDEXER,
        keys=[f"page-{i}" for i in range(pages)],
        host_indices=torch.arange(pages * indexer.page_size),
    )
    expected = indexer.index_k_with_scale_buffer.clone()
    assert store.batch_set_v2([transfer])[PoolName.INDEXER] == [True] * pages
    indexer.index_k_with_scale_buffer.zero_()
    with mock.patch.object(
        wire, "batch_get_into_multi_buffers", wraps=wire.batch_get_into_multi_buffers
    ) as get:
        assert store.batch_get_v2([transfer])[PoolName.INDEXER] == [True] * pages
    assert [len(call.args[0]) for call in get.call_args_list] == [128, 128, 3]
    assert torch.equal(indexer.index_k_with_scale_buffer, expected)


@pytest.mark.parametrize(
    "lengths,expected_batches",
    [
        ([4, 6, 1, 9, 11, 1], [2, 2, 1, 1]),
        ([10, 10, 10], [1, 1, 1]),
    ],
)
def test_mooncake_layer_split_get_byte_bound_preserves_objects_and_order(
    lengths, expected_batches
):
    wire = ByteStore()
    store = adapter(wire, pools(0, 2, 5), staged=True)
    keys = [f"object-{i}" for i in range(len(lengths))]
    buffers = [ctypes.create_string_buffer(n) for n in lengths]
    for i, (key, n) in enumerate(zip(keys, lengths)):
        wire.objects[key] = bytes([i + 1]) * n
    pointers = [[ctypes.addressof(b)] for b in buffers]
    sizes = [[n] for n in lengths]
    module = sys.modules[MooncakeStore.__module__]
    with (
        mock.patch.object(module, "_LAYER_SPLIT_MAX_GET_BYTES", 10),
        mock.patch.object(
            wire,
            "batch_get_into_multi_buffers",
            wraps=wire.batch_get_into_multi_buffers,
        ) as get,
    ):
        assert store._layer_split_io(keys, pointers, sizes, False) == [True] * len(keys)
    assert [len(call.args[0]) for call in get.call_args_list] == expected_batches
    assert [key for call in get.call_args_list for key in call.args[0]] == keys
    assert [b.raw for b in buffers] == [wire.objects[key] for key in keys]


def test_mooncake_late_chunk_response_length_mismatch_fails_closed():
    wire = ByteStore()
    store = adapter(wire, pools(0, 2, 5), staged=True)
    count = STORAGE_BATCH_SIZE + 1
    keys = [str(i) for i in range(count)]
    with mock.patch.object(
        store, "_get_batch_zero_copy_impl", side_effect=[[1] * STORAGE_BATCH_SIZE, []]
    ) as get:
        assert (
            store._layer_split_io(keys, [[0]] * count, [[1]] * count, False)
            == [False] * count
        )
    assert get.call_count == 2


def test_direct_primary_v1_and_v2_use_identical_keys():
    wire = ByteStore()
    host = pools(7, 8, 78)
    start, end = get_layer_shard_range(7, 8, 78)
    shard = LayerShardStorageSpec(7, 8, 78, start, end)
    store = adapter(wire, host, shard=shard)
    keys, indices = ["p0", "p2"], torch.tensor([0, 1, 4, 5])
    host.get_pool(PoolName.KV).kv_buffer.view(torch.uint8).fill_(51)
    assert store.batch_set_v1(keys, indices) == [True, True]
    assert len(wire.objects) == 2
    assert store.batch_exists(keys) == 2
    host.get_pool(PoolName.KV).kv_buffer.zero_()
    assert store.batch_get_v1(keys, indices) == [True, True]
    saved = dict(wire.objects)
    assert store.batch_set_v2(
        [PoolTransfer(PoolName.KV, keys=keys, host_indices=indices)]
    )[PoolName.KV] == [True, True]
    assert wire.objects == saved


def test_components_do_not_require_a_packing_group():
    store, wire, transfers, _ = staged_case()
    transfers[1].indices_from_pool = None
    assert store.batch_set_v2(transfers) == {
        PoolName.KV: [True, True],
        PoolName.INDEXER: [True, True],
    }
    assert len(wire.objects) == 4


def test_direct_rank_without_indexer_layers_does_not_require_an_indexer_object():
    host = pools(0, 2, 5)
    indexer = host.get_pool(PoolName.INDEXER)
    indexer.layer_num = indexer.target_layer_num = 0
    indexer.index_k_with_scale_buffer = torch.empty((0, 4, 6), dtype=torch.uint8)
    wire = ByteStore()
    store = adapter(wire, host, shard=LayerShardStorageSpec(0, 2, 5, 0, 3))
    indices = torch.arange(2)
    assert store.batch_set_v1(["page"], indices) == [True]
    transfer = PoolTransfer(PoolName.INDEXER, keys=["page"], host_indices=indices)
    assert store.batch_set_v2([transfer]) == {PoolName.INDEXER: [True]}
    assert store.batch_get_v2([transfer]) == {PoolName.INDEXER: [True]}
    result = store.batch_exists_v2(["page"], [transfer])
    assert result.kv_hit_pages == 1
    assert result.extra_pool_hit_pages[PoolName.INDEXER] == 1
    assert len(wire.objects) == 1


@pytest.mark.parametrize("missing_component", [PoolName.KV, PoolName.INDEXER])
def test_staged_query_and_get_do_not_treat_a_half_page_as_complete(missing_component):
    store, wire, transfers, _ = staged_case()
    store.batch_set_v2(transfers)
    transfer = next(t for t in transfers if t.name == missing_component)
    object_keys, _ = store._get_hybrid_page_component_keys(transfer.keys, transfer)
    object_keys = store._tag_keys(object_keys)
    saved = wire.objects.pop(object_keys[1])
    sidecar = PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)
    assert store.batch_exists_v2(["p0", "p1"], [sidecar]).kv_hit_pages == 1
    result = store.batch_get_v2(transfers)
    assert result[missing_component] == [True, False]
    other = PoolName.INDEXER if missing_component == PoolName.KV else PoolName.KV
    assert result[other] == [True, True]
    wire.objects[object_keys[1]] = saved
    assert store.batch_exists_v2(["p0", "p1"], [sidecar]).kv_hit_pages == 2


def test_staged_partial_write_is_a_cache_miss_until_other_component_arrives():
    store, wire, transfers, _ = staged_case()
    assert store.batch_set_v2(transfers[:1]) == {PoolName.KV: [True, True]}
    assert len(wire.objects) == 2
    sidecar = PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)
    assert store.batch_exists_v2(["p0", "p1"], [sidecar]).kv_hit_pages == 0
    assert store.batch_set_v2(transfers[1:]) == {PoolName.INDEXER: [True, True]}
    assert len(wire.objects) == 4
    assert store.batch_exists_v2(["p0", "p1"], [sidecar]).kv_hit_pages == 2


def test_staged_set_reports_component_failure_independently():
    from sglang.srt.mem_cache.layer_split.layer_split_utils import complete_page_mask

    store, wire, transfers, _ = staged_case()
    indexer_keys, _ = store._get_hybrid_page_component_keys(
        transfers[1].keys, transfers[1]
    )
    indexer_keys = store._tag_keys(indexer_keys)
    native_put = wire.batch_put_from

    def put(keys, pointers, sizes, config=None):
        if keys[0] in indexer_keys:
            return [-1] * len(keys)
        return native_put(keys, pointers, sizes, config)

    with mock.patch.object(wire, "batch_put_from", side_effect=put):
        result = store.batch_set_v2(transfers)
    assert result == {PoolName.KV: [True, True], PoolName.INDEXER: [False, False]}
    assert complete_page_mask(result, 2) == [False, False]
    assert len(wire.objects) == 2
