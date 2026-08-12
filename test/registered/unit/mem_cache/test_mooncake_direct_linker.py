import threading
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_mappings import (
    DevicePoolEntry,
    DevicePoolGroup,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
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


def test_prefetch_loads_before_returning():
    worker_threads = []

    class _Storage:
        def batch_get_v2(self, transfers):
            worker_threads.append(threading.get_ident())
            return {
                transfer.name: [True] * len(transfer.keys) for transfer in transfers
            }

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.page_size = 2
    linker.pool_group = DevicePoolGroup([pool], num_layers=3, page_size=2)
    linker.storage = _Storage()
    linker.load_strategy = "prefetch"
    linker.gc_frozen = True
    linker.pending_loads = {}
    linker.stats = {"load": 0}
    linker.load_queue = Queue()
    linker.load_thread = threading.Thread(
        target=linker.load_thread_func, daemon=True
    )
    linker.load_thread.start()

    assert linker.load(
        "rid",
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["a"],
                device_indices=torch.tensor([0, 1]),
            )
        ],
    )
    assert worker_threads == [linker.load_thread.ident]
    assert linker.pending_loads == {}
    assert linker.stats["load"] == 1

    linker.load_queue.put(None)
    linker.load_thread.join(timeout=5)


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
    wrapper.drain_offloads()
    assert not node.external_cache_stored
    assert unlocks == [(node_id, lock_params)]


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

    group = resolve_hybrid_device_pool_group(kvcache, 2, None)
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


def test_qwen35_device_pool_group_maps_full_and_mamba_layers():
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

    group = resolve_hybrid_device_pool_group(kvcache, 2, req_pool)
    pools = group.entry_map
    assert group.num_layers == 4
    assert set(pools) == {PoolName.KV, PoolName.MAMBA}
    assert group.sources == {
        PoolName.KV: PoolName.KV,
        PoolName.MAMBA: PoolName.MAMBA,
    }
    assert pools[PoolName.MAMBA].translate_indices(torch.tensor([1])).tolist() == [1]
    assert pools[PoolName.KV].get_prepared_layer_range_meta([0], 1) is None
    assert pools[PoolName.MAMBA].get_prepared_layer_range_meta([0], 0) is None
    pointers, sizes = pools[PoolName.KV].get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 4
    assert sizes == [24, 40, 56, 88]


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


def test_mamba_linker_load_allocates_cache_and_request_slots():
    allocator = _Allocator(slots=torch.tensor([7, 8]))
    req_pool = SimpleNamespace(mamba_allocator=allocator, mamba_ckpt_pool=None)
    component = MambaComponent.__new__(MambaComponent)
    component.cache = SimpleNamespace(
        req_to_token_pool=req_pool,
        evict=lambda params: None,
    )

    transfer = component.build_external_linker_transfer(
        LinkerTransferPhase.LOAD, None, ["a", "b"]
    )
    assert transfer.keys == ["b", "b"]
    assert transfer.device_indices.tolist() == [7, 8]

    req = SimpleNamespace(
        mamba_pool_idx=None,
        mamba_cow_src_index=torch.tensor([99]),
        mamba_needs_clear=True,
    )
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1]))
    component.finish_external_linker_load(req, full, transfer, prefix_len=2, success=True)
    assert req.mamba_pool_idx.item() == 8
    assert req.mamba_cow_src_index is None
    assert not req.mamba_needs_clear

    failed = PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([9, 10]))
    component.finish_external_linker_load(req, full, failed, prefix_len=2, success=False)
    assert allocator.freed[-1].tolist() == [9, 10]


def test_overlapping_load_retargets_freed_slots_to_tree_values():
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            translate_loc_from_full_to_swa=lambda indices: indices + 1000
        ),
        tree_core=SimpleNamespace(
            get_component_device_value=lambda node_id, component_type: torch.tensor(
                [30]
            )
        ),
    )

    full = PoolTransfer(
        name=PoolName.KV, device_indices=torch.tensor([100, 101, 102, 103])
    )
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([200, 201]))
    mamba = PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([300, 301]))
    canonical_tail = torch.tensor([10, 11, 12, 13])
    rematched = SimpleNamespace(
        device_indices=torch.cat([torch.tensor([1, 2]), canonical_tail]),
        last_device_node=30,
    )

    load_transfers = wrapper._point_at_canonical(
        [(None, full), (None, swa), (None, mamba)],
        rematched,
        canonical_tail,
    )

    assert load_transfers == [full, swa, mamba]
    assert full.device_indices.tolist() == canonical_tail.tolist()
    assert swa.device_indices.tolist() == [1012, 1013]
    assert mamba.device_indices.tolist() == [30, 301]
