import threading
from array import array
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache_linker import (
    DevicePoolEntry,
    DevicePoolGroup,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, slots=None):
        self.slots = slots
        self.freed = []
        self.mapping = []

    def available_size(self):
        return 100

    def alloc(self, size):
        if self.slots is None:
            return torch.arange(1, size + 1, dtype=torch.int64)
        value = self.slots[:size].clone()
        self.slots = self.slots[size:]
        return value

    def free(self, value):
        self.freed.append(value.clone())

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def test_sparse_multi_component_layer_ranges():
    k0 = torch.zeros((8, 3), dtype=torch.uint8)
    k2 = torch.zeros((8, 5), dtype=torch.uint8)
    v0 = torch.zeros((8, 7), dtype=torch.uint8)
    v2 = torch.zeros((8, 11), dtype=torch.uint8)
    pool = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=[[k0, k2], [v0, v2]],
        layer_mapping={0: 0, 2: 1},
        page_size=2,
        rows_are_pages=False,
        packed=False,
    )

    indices = torch.tensor([0, 1, 4, 5])
    locations = pool.prepare_locations(indices)
    assert locations == [0, 4]
    pointers, sizes = pool.get_page_buffer_meta(indices)
    assert pointers == [
        buffer[row].data_ptr() for row in locations for buffer in (k0, k2, v0, v2)
    ]
    assert sizes == [6, 10, 14, 22] * 2
    assert pool.get_prepared_layer_range_meta(locations, 1) is None

    pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
    assert pointers == [
        [k2[0].data_ptr()],
        [v2[0].data_ptr()],
        [k2[4].data_ptr()],
        [v2[4].data_ptr()],
    ]
    assert sizes == [[10], [22], [10], [22]]
    assert offsets == [[6], [14], [6], [14]]


def test_lookup_returns_sparse_mamba_boundaries():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    pools = [
        SimpleNamespace(
            name=name,
            indices_from_pool=name,
            translate_indices=lambda indices: indices,
        )
        for name in (PoolName.KV, PoolName.MAMBA)
    ]
    linker.pool_group = DevicePoolGroup(pools, num_layers=4, page_size=1)
    linker.pools = linker.pool_group.entry_map
    linker.stats = {"lookup": 0}
    linker.storage = MooncakeStore.__new__(MooncakeStore)
    linker.storage.mem_pool_host = SimpleNamespace(kv_buffer=None)
    linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
        [f"{key}_{transfer.name}" for key in keys],
        1,
    )
    linker.storage._tag_keys = lambda keys: keys
    linker.storage._batch_exist = lambda keys: [
        int(key.endswith("_kv") or key[0] in ("b", "d")) for key in keys
    ]
    valid = linker.lookup(
        "rid",
        [
            PoolTransfer(name=PoolName.KV, keys=["a", "b", "c", "d"]),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["d"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ],
    )
    assert valid == [2, 4]


def test_tail_hashes_honor_radix_key_limit():
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(page_size=256)
    key = RadixKey(array("q", range(256)), limit=255)
    result = SimpleNamespace(last_device_node=None)

    assert wrapper._tail_hashes(key, result, device_hit_len=0) == []


def test_load_and_offload_share_gc_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(mooncake_direct_linker, "freeze_gc", calls.append)
    monkeypatch.setattr(
        mooncake_direct_linker.device_module,
        "Event",
        lambda: SimpleNamespace(record=lambda: None),
    )

    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker.page_size = 2
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=2)
    linker.gc_frozen = False
    linker.offload_queue = Queue()
    linker.pending_loads = {"first": [object()]}
    linker.layer_done_counter = SimpleNamespace(update_producer=lambda: 3)
    linker.load_queue = Queue()
    linker.stats = {"load": 0}

    assert linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert linker.start_layer_wise_loading() == 3
    linker.pending_loads = {"second": [object()]}
    assert linker.start_layer_wise_loading() == 3

    assert calls == ["Mooncake direct linker"]
    assert linker.stats["load"] == 2


def test_offload_runs_on_background_thread(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()
    worker_threads = []
    event_calls = []

    monkeypatch.setattr(mooncake_direct_linker, "freeze_gc", lambda _: None)

    class _Event:
        def record(self):
            event_calls.append("record")

        def synchronize(self):
            event_calls.append("synchronize")

    monkeypatch.setattr(mooncake_direct_linker.device_module, "Event", _Event)

    class _Storage:
        def batch_set_v2(self, transfers):
            assert event_calls == ["record", "synchronize"]
            worker_threads.append(threading.get_ident())
            started.set()
            assert release.wait(timeout=5)
            return {
                transfer.name: [True] * len(transfer.keys) for transfer in transfers
            }

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
        get_hybrid_pool_buffer=lambda: [],
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.page_size = 2
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=2)
    linker.pools = linker.pool_group.entry_map
    linker.storage = _Storage()
    linker.gc_frozen = False
    linker.stats = {"lookup": 0, "load": 0, "offload": 0}
    linker.offload_queue = Queue()
    linker.offload_results = Queue()
    linker.offload_thread = threading.Thread(
        target=linker.offload_thread_func, daemon=True
    )
    linker.offload_thread.start()

    assert linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert started.wait(timeout=5)
    assert linker.num_completed_offloads() == 0
    assert worker_threads == [linker.offload_thread.ident]
    assert worker_threads[0] != caller_thread

    release.set()
    linker.offload_queue.join()
    assert linker.num_completed_offloads() == 1
    assert linker.pop_completed_offload()
    linker.offload_queue.put(None)
    linker.offload_thread.join(timeout=5)


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.OFFLOAD
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    results = []
    linker = SimpleNamespace(
        offload=lambda transfers: True,
        num_completed_offloads=lambda: len(results),
        pop_completed_offload=lambda: results.pop(0),
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache_linker = linker
    wrapper.pending_offloads = []
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    node = SimpleNamespace(id=node_id, external_cache_stored=False)
    wrapper.cache = SimpleNamespace(
        _components_tuple=(_Component(),),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
        resolve_node_handle=lambda value: node if value == node_id else None,
    )

    wrapper.offload_nodes([node_id])
    assert locks == [node_id]
    assert node.external_cache_stored
    assert not unlocks

    results.append(False)
    assert wrapper.num_completed_offloads() == 1
    wrapper.drain_offloads(finish_count=1)
    assert not node.external_cache_stored
    assert unlocks == [(node_id, lock_params)]


def test_check_hicache_events_drains_common_tp_offloads():
    drained = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        num_completed_offloads=lambda: 3,
        drain_offloads=drained.append,
    )

    def reduce_to_common_count(count, op):
        assert op == torch.distributed.ReduceOp.MIN
        count.fill_(1)

    cache._all_reduce_attn_groups = reduce_to_common_count
    cache.check_hicache_events()

    assert drained == [1]


def test_deepseek_v4_device_pool_group_maps_sparse_sidecars():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4LayerItem,
        DeepSeekV4TokenToKVPool,
    )

    def state_pool():
        return SimpleNamespace(
            ring_size=2,
            kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((8, 3))),
        )

    kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    kvcache._unified_kv = False
    kvcache.start_layer = 0
    kvcache.end_layer = 3
    kvcache.swa_page_size = 2
    kvcache.swa_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)]
    )
    kvcache.c4_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)]
    )
    kvcache.c4_indexer_kv_pool = SimpleNamespace(
        index_k_with_scale_buffer=[
            torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
        ]
    )
    kvcache.c128_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)]
    )
    kvcache.layer_mapping = [
        DeepSeekV4LayerItem(4, 0),
        DeepSeekV4LayerItem(128, 0),
        DeepSeekV4LayerItem(4, 1),
    ]
    kvcache.compress_state_pools = [state_pool(), None, state_pool()]
    kvcache.indexer_compress_state_pools = [state_pool(), None, state_pool()]

    group = resolve_hybrid_device_pool_group(
        kvcache=kvcache,
        page_size=2,
        params=SimpleNamespace(req_to_token_pool=None),
        components={ComponentType.FULL, ComponentType.SWA},
    )
    assert group.num_layers == 3
    assert set(group.entry_map) == {
        PoolName.SWA,
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    }
    assert group.sources[PoolName.DEEPSEEK_V4_C4] == PoolName.KV
    assert group.sources[PoolName.DEEPSEEK_V4_C4_STATE] == PoolName.SWA
    c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
    pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 2
    assert sizes == [5, 5]
    _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
    assert sizes == [[5]]
    assert offsets == [[5]]
    assert c4_pool.get_prepared_layer_range_meta([0], 1) is None


def test_mamba_strategy_rejects_direct_linker():
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
    kvcache.use_mla = False
    kvcache.full_attention_layer_id_mapping = {0: 0, 2: 1}
    kvcache.full_kv_pool = SimpleNamespace(
        size=6,
        k_scale_buffer=None,
        k_buffer=[torch.zeros((8, 3)), torch.zeros((8, 5))],
        v_buffer=[torch.zeros((8, 7)), torch.zeros((8, 11))],
    )
    req_pool = SimpleNamespace(
        mamba_ckpt_pool=None,
        mamba_map={1: 0, 3: 1},
        mamba_pool=SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.zeros((2, 5, 2, 3)),
                conv=[torch.zeros((2, 5, 4))],
            )
        ),
        translate_mamba_indices=lambda indices: indices,
    )

    with pytest.raises(ValueError, match="does not support the direct external linker"):
        resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(req_to_token_pool=req_pool),
            components={ComponentType.FULL, ComponentType.MAMBA},
        )


def test_dsa_device_pool_group_uses_assembler_strategy():
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    kvcache = DSATokenToKVPool.__new__(DSATokenToKVPool)
    kvcache.page_size = 2
    kvcache.layer_num = 2
    kvcache.kv_buffer = [
        torch.zeros((8, 3), dtype=torch.uint8),
        torch.zeros((8, 5), dtype=torch.uint8),
    ]
    kvcache.index_key_cache = SimpleNamespace(
        buffer=[
            torch.zeros((4, 7), dtype=torch.uint8),
            torch.zeros((4, 11), dtype=torch.uint8),
        ]
    )

    group = resolve_hybrid_device_pool_group(
        kvcache=kvcache,
        page_size=2,
        params=SimpleNamespace(req_to_token_pool=None),
        components={ComponentType.FULL},
    )

    assert group.num_layers == 2
    assert set(group.entry_map) == {PoolName.KV, PoolName.INDEXER}
    assert group.sources == {
        PoolName.KV: PoolName.KV,
        PoolName.INDEXER: PoolName.KV,
    }


def test_device_pool_group_allows_partial_side_pool_load():
    swa_pool = SimpleNamespace(
        name=PoolName.SWA,
        indices_from_pool=PoolName.SWA,
        translate_indices=lambda indices: indices + 100,
    )
    group = DevicePoolGroup([swa_pool], num_layers=1, page_size=2)
    transfer = PoolTransfer(
        name=PoolName.SWA,
        keys=["b", "d"],
        device_indices=torch.tensor([20, 21, 24, 25]),
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )

    assert group.resolve_transfers([transfer]) == []
    [resolved] = group.resolve_transfers(
        [transfer], allow_partial=True, allow_missing_kv=True
    )
    assert resolved.name == PoolName.SWA
    assert resolved.keys == ["b", "d"]
    assert resolved.host_indices.tolist() == [120, 121, 124, 125]


def test_swa_linker_finish_maps_or_releases_slots():
    swa_allocator = _Allocator()
    allocator = SimpleNamespace(
        swa_attn_allocator=swa_allocator,
        set_full_to_swa_mapping=swa_allocator.set_full_to_swa_mapping,
    )
    component = SWAComponent.__new__(SWAComponent)
    component.cache = SimpleNamespace(
        page_size=64,
        token_to_kv_pool_allocator=allocator,
    )
    component.sliding_window_size = 128
    req = SimpleNamespace(kv=SimpleNamespace(swa_evicted_seqlen=0))
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1, 2, 3, 4]))
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([20, 21]))

    component.finish_external_linker_load(req, full, swa, prefix_len=256, success=True)
    mapped_full, mapped_swa = swa_allocator.mapping[0]
    assert mapped_full.tolist() == [3, 4]
    assert mapped_swa.tolist() == [20, 21]
    assert req.kv.swa_evicted_seqlen == 128

    component.finish_external_linker_load(req, full, swa, prefix_len=256, success=False)
    assert swa_allocator.freed[0].tolist() == [20, 21]


def test_mamba_component_rejects_external_linker():
    component = MambaComponent.__new__(MambaComponent)
    with pytest.raises(AssertionError, match="does not support external linker mode"):
        component.build_external_linker_transfer(
            LinkerTransferPhase.LOAD, None, ["a", "b"]
        )


def test_insert_filters_overlapping_full_and_swa_load_pages():
    canonical_swa = torch.tensor([300, 301, 202, 203, 304, 305, 206, 207])
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(
        page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(
            translate_loc_from_full_to_swa=lambda indices: canonical_swa
        ),
    )
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
    )
    canonical_tail = torch.tensor([10, 11, 102, 103, 14, 15, 106, 107])
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([200, 201, 202, 203, 204, 205, 206, 207]),
    )
    filtered = wrapper._filter_load_pages_after_insert(
        [(None, full), (None, swa)],
        canonical_tail,
    )
    assert filtered == [full, swa]
    assert full.keys == ["b", "d"]
    assert full.device_indices.tolist() == [102, 103, 106, 107]
    assert swa.keys == ["b", "d"]
    assert swa.device_indices.tolist() == [202, 203, 206, 207]
