import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
    _storage_suffix,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _SingleBufferPool:
    page_size = 1

    def get_page_buffer_meta(self, indices):
        return [3000 + i for i in range(len(indices))], [8] * len(indices)


class _MultiBufferPool:
    page_size = 1

    def get_page_buffer_meta(self, indices):
        ptrs = []
        sizes = []
        for i in range(len(indices)):
            ptrs.extend([4000 + i * 10, 4001 + i * 10])
            sizes.extend([8, 16])
        return ptrs, sizes


class _RecordingCounter:
    def __init__(self):
        self.completed = []
        self.failures = []

    def complete(self, index, layer):
        self.completed.append((index, layer))

    def fail(self, index, error):
        self.failures.append((index, error))


_MISSING = object()


class _PatchManager:
    def __init__(self, test_case):
        self.test_case = test_case

    def setattr(self, target, name, value=_MISSING):
        if value is _MISSING:
            patcher = patch(target, new=name)
        else:
            patcher = patch.object(target, name, value)
        patched = patcher.start()
        self.test_case.addCleanup(patcher.stop)
        return patched


def _make_v2_store():
    store = MooncakeStore.__new__(MooncakeStore)
    store.registered_pools = {}
    store.mem_pool_host = SimpleNamespace(kv_buffer=None)
    store.is_mla_backend = True
    store.mla_suffix = ""
    store.mha_suffix = ""
    store.config_prefix = None
    return store


class TestMooncakeDirectLinker(unittest.TestCase):
    def test_storage_suffix_matches_replication_topology(self):
        assert (
            _storage_suffix(
                rank_replicated=True,
                tp_rank=3,
                attn_cp_rank=1,
                pp_rank=2,
            )
            == "cp1_pp2"
        )
        assert (
            _storage_suffix(
                rank_replicated=False,
                tp_rank=3,
                attn_cp_rank=1,
                pp_rank=2,
            )
            == "tp3_cp1_pp2"
        )

    def test_init_wires_topology_with_injected_storage(self):
        self._check_init_wires_topology(
            distributed=False,
            inject_storage=True,
            include_mamba=False,
            expected_tp_rank=0,
        )

    def test_init_wires_distributed_topology_with_created_storage(self):
        self._check_init_wires_topology(
            distributed=True,
            inject_storage=False,
            include_mamba=True,
            expected_tp_rank=2,
        )

    def _check_init_wires_topology(
        self, distributed, inject_storage, include_mamba, expected_tp_rank
    ):
        patches = _PatchManager(self)
        buffer = torch.zeros(8, dtype=torch.uint8)
        pool = SimpleNamespace(get_hybrid_pool_buffer=lambda: [buffer, buffer.view(-1)])
        pools = {PoolName.KV: pool}
        if include_mamba:
            pools[PoolName.MAMBA] = pool
        pool_group = SimpleNamespace(
            entry_map=pools,
            num_layers=2,
            rank_replicated=False,
        )
        patches.setattr(
            mooncake_direct_linker,
            "resolve_hybrid_device_pool_group",
            lambda **kwargs: pool_group,
        )
        patches.setattr(
            mooncake_direct_linker,
            "get_memory",
            lambda: SimpleNamespace(hicache_storage_backend_extra_config="config"),
        )
        patches.setattr(
            mooncake_direct_linker,
            "get_model",
            lambda: SimpleNamespace(model_path="model"),
        )
        patches.setattr(
            mooncake_direct_linker.HybridCacheController,
            "parse_storage_backend_extra_config",
            staticmethod(lambda config: ({"backend": "value"},)),
        )
        patches.setattr(torch.distributed, "is_available", lambda: distributed)
        patches.setattr(torch.distributed, "is_initialized", lambda: distributed)
        patches.setattr(torch.distributed, "get_rank", lambda group: 2)
        patches.setattr(torch.distributed, "get_world_size", lambda group: 8)
        config_kwargs = {}
        patches.setattr(
            mooncake_direct_linker,
            "HiCacheStorageConfig",
            lambda **kwargs: config_kwargs.update(kwargs) or SimpleNamespace(**kwargs),
        )

        registered = []
        closed = []
        storage = SimpleNamespace(
            store=SimpleNamespace(
                register_buffer=lambda ptr, size: registered.append((ptr, size)) or 0
            ),
            close=lambda: closed.append(True),
        )
        patches.setattr(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.MooncakeStore",
            lambda config, mem_pool: storage,
        )
        registered_counters = []
        params = SimpleNamespace(
            page_size=2,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: object()),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            pp_rank=1,
            pp_size=2,
            attn_cp_rank=3,
            attn_cp_size=4,
            req_to_token_pool=SimpleNamespace(
                register_layer_transfer_counter=registered_counters.append
            ),
        )

        linker = MooncakeDirectLinker(
            SimpleNamespace(tp_size=8),
            params,
            components={object()},
            storage=storage if inject_storage else None,
        )

        assert len(registered) == 1
        assert linker.pools == pools
        assert linker.num_layers == 2
        assert linker.offload_owner
        assert registered_counters == (
            [linker.layer_done_counter] if include_mamba else []
        )
        assert storage.mem_pool_host is pool_group
        assert storage.registered_pools == linker.pools
        expected_suffix = f"tp{expected_tp_rank}_cp3_pp1"
        assert storage.mla_suffix == expected_suffix
        assert storage.mha_suffix == expected_suffix
        assert config_kwargs == {
            "tp_rank": expected_tp_rank,
            "tp_size": 8,
            "pp_rank": 1,
            "pp_size": 2,
            "attn_cp_rank": 3,
            "attn_cp_size": 4,
            "is_mla_model": False,
            "enable_storage_metrics": False,
            "is_page_first_layout": False,
            "model_name": "model",
            "extra_config": {"backend": "value"},
        }
        assert linker.load_thread.is_alive()
        assert linker.offload_thread.is_alive()

        linker.close()
        assert not linker.load_thread.is_alive()
        assert not linker.offload_thread.is_alive()
        assert closed == [True]

    def test_register_buffers_rejects_mooncake_registration_failure(self):
        buffer = torch.zeros(8, dtype=torch.uint8)
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pools = {
            PoolName.KV: SimpleNamespace(get_hybrid_pool_buffer=lambda: [buffer])
        }
        linker.storage = SimpleNamespace(
            store=SimpleNamespace(register_buffer=lambda ptr, size: -1)
        )

        with self.assertRaisesRegex(RuntimeError, "error code: -1"):
            linker.register_buffers()

    def test_lookup_returns_every_sparse_restorable_boundary(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers: transfers
        )
        linker.storage = SimpleNamespace(
            batch_exists_v2=lambda keys, transfers: SimpleNamespace(
                restorable_prefix_pages=[2, 4]
            )
        )
        linker.stats = {"lookup": 0}

        result = linker.lookup(
            "rid",
            [PoolTransfer(name=PoolName.KV, keys=["a", "b", "c", "d"])],
        )

        assert result == [2, 4]
        assert linker.stats["lookup"] == 1

    def test_lookup_reports_a_storage_miss(self):
        transfer = PoolTransfer(name=PoolName.KV, keys=["page"])
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers: transfers
        )
        linker.storage = SimpleNamespace(
            batch_exists_v2=lambda keys, transfers: SimpleNamespace(
                restorable_prefix_pages=[]
            )
        )
        linker.stats = {"lookup": 0}

        assert linker.lookup("rid", [transfer]) == []
        assert linker.stats["lookup"] == 1

    def test_lookup_short_circuits_when_transfer_cannot_be_resolved(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pool_group = SimpleNamespace(resolve_transfers=lambda transfers: [])
        linker.storage = SimpleNamespace(
            batch_exists_v2=lambda keys, transfers: self.fail("storage was queried")
        )
        linker.stats = {"lookup": 0}

        assert (
            linker.lookup("rid", [PoolTransfer(name=PoolName.KV, keys=["page"])]) == []
        )
        assert linker.stats["lookup"] == 0

    def test_lookup_short_circuits_for_empty_kv_keys(self):
        transfer = PoolTransfer(name=PoolName.KV, keys=[])
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers: transfers
        )
        linker.storage = SimpleNamespace(
            batch_exists_v2=lambda keys, transfers: self.fail("storage was queried")
        )
        linker.stats = {"lookup": 0}

        assert linker.lookup("rid", [transfer]) == []
        assert linker.stats["lookup"] == 0

    def test_load_queues_partial_transfers_rejects_duplicates_and_cancels(self):
        expanded = [PoolTransfer(name=PoolName.SWA, keys=["page"])]
        resolve_calls = []

        def resolve(transfers, **kwargs):
            resolve_calls.append(kwargs)
            return expanded

        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pool_group = SimpleNamespace(resolve_transfers=resolve)
        linker.pending_loads = {}

        assert linker.load("rid", [object()])
        assert linker.pending_loads == {"rid": expanded}
        assert resolve_calls == [{"allow_partial": True, "allow_missing_kv": True}]
        with self.assertRaisesRegex(RuntimeError, "already queued"):
            linker.load("rid", [object()])
        assert linker.cancel_queued_load("rid")
        assert not linker.cancel_queued_load("rid")

        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, **kwargs: []
        )
        assert not linker.load("miss", [object()])

    def test_start_layer_wise_loading_returns_sentinel_for_empty_queue(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pending_loads = {}

        assert linker.start_layer_wise_loading() == -1

    def test_freeze_gc_once_is_idempotent(self):
        patches = _PatchManager(self)
        calls = []
        patches.setattr(mooncake_direct_linker, "freeze_gc", calls.append)
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.gc_frozen = False

        linker.freeze_gc_once()
        linker.freeze_gc_once()

        assert calls == ["Mooncake direct linker"]
        assert linker.gc_frozen

    def test_layer_counter_failure_unblocks_waiters_and_cleans_final_future(self):
        counter = LayerWiseLoadCounter(num_layers=2)
        counter.set_consumer(99)
        counter.wait_until(0)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, ValueError("injected load failure"))
        counter.fail(index, ValueError("a completed future must not be overwritten"))

        for layer in range(2):
            with self.assertRaisesRegex(
                RuntimeError, "layer-wise KV load failed"
            ) as exc:
                counter.wait_until(layer)
            assert isinstance(exc.exception.__cause__, ValueError)

        assert index not in counter.futures

    def test_load_worker_reports_scheduler_event_failure_and_stays_alive(self):
        patches = _PatchManager(self)

        class _Event:
            instance_count = 0

            def __init__(self):
                type(self).instance_count += 1
                self.should_fail = type(self).instance_count == 1

            def record(self):
                pass

            def synchronize(self):
                if self.should_fail:
                    raise RuntimeError("injected event failure")

        patches.setattr(mooncake_direct_linker.device_module, "Event", _Event)

        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.gc_frozen = True
        linker.pending_loads = {"rid": [object()]}
        linker.completed_loads = Queue()
        linker.layer_done_counter = LayerWiseLoadCounter(num_layers=1)
        linker.load_queue = Queue()
        linker.stats = {"load": 0}
        thread = threading.Thread(target=linker.load_thread_func, daemon=True)
        thread.start()

        index = linker.start_layer_wise_loading()
        linker.layer_done_counter.set_consumer(index)
        with self.assertRaisesRegex(RuntimeError, "layer-wise KV load failed") as exc:
            linker.layer_done_counter.wait_until(0)
        assert "injected event failure" in str(exc.exception.__cause__)

        linker.load_queue.join()
        assert linker.num_completed_loads() == 1
        assert linker.pop_completed_load() == ["rid"]

        completed_batches = []

        def complete_load(counter_index, request_transfers):
            completed_batches.append(request_transfers)
            linker.layer_done_counter.complete(counter_index, 0)

        linker.load_layer_wise = complete_load
        second_transfer = object()
        linker.pending_loads = {"rid2": [second_transfer]}
        index = linker.start_layer_wise_loading()
        linker.layer_done_counter.set_consumer(index)
        linker.layer_done_counter.wait_until(0)
        linker.load_queue.join()

        assert completed_batches == [[[second_transfer]]]
        assert linker.pop_completed_load() == ["rid2"]
        assert thread.is_alive()
        linker.load_queue.put(None)
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_layer_wise_load_uses_sparse_ranges_and_completes_every_layer(self):
        calls = []

        class _Store:
            def batch_get_session_start(self, keys):
                calls.append(("start", list(keys)))
                return [0] * len(keys)

            def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
                calls.append(("get", list(keys), ptrs, sizes, offsets))
                return [sum(item) for item in sizes]

            def batch_get_session_end(self, keys):
                calls.append(("end", list(keys)))

        def layer_meta(locations, layer):
            if layer == 1:
                return None
            return (
                [[100 + layer], [200 + layer]],
                [[4], [8]],
                [[layer * 12], [layer * 12]],
            )

        counter = LayerWiseLoadCounter(num_layers=3)
        index = counter.update_producer()
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.storage = SimpleNamespace(
            store=_Store(),
            _get_hybrid_page_component_keys=lambda keys, transfer: (
                [f"{key}_kv" for key in keys],
                1,
            ),
            _tag_keys=lambda keys: keys,
        )
        linker.pools = {
            PoolName.KV: SimpleNamespace(
                prepare_locations=lambda indices: [0, 2],
                get_prepared_layer_range_meta=layer_meta,
            )
        }
        linker.num_layers = 3
        linker.layer_done_counter = counter

        linker.load_layer_wise(
            index,
            [
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["a", "b"],
                        host_indices=torch.tensor([0, 1]),
                    )
                ]
            ],
        )

        counter.set_consumer(index)
        for layer in range(3):
            counter.wait_until(layer)
        assert [call[0] for call in calls] == ["start", "get", "get", "end"]
        assert calls[1][1] == ["a_kv", "b_kv"]
        assert calls[1][3] == [[4], [8]]
        assert calls[2][4] == [[24], [24]]

    def test_layer_wise_load_reports_session_start_failure(self):
        counter = _RecordingCounter()
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.storage = SimpleNamespace(
            store=SimpleNamespace(
                batch_get_session_start=lambda keys: [-1],
                batch_get_session_end=lambda keys: self.fail(
                    "an unstarted session was closed"
                ),
            ),
            _get_hybrid_page_component_keys=lambda keys, transfer: (list(keys), 1),
            _tag_keys=lambda keys: keys,
        )
        linker.pools = {
            PoolName.KV: SimpleNamespace(prepare_locations=lambda indices: [0])
        }
        linker.num_layers = 1
        linker.layer_done_counter = counter

        linker.load_layer_wise(
            7,
            [
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["page"],
                        host_indices=torch.tensor([0]),
                    )
                ]
            ],
        )

        assert counter.completed == []
        assert len(counter.failures) == 1
        assert "session start failed" in str(counter.failures[0][1])

    def test_layer_wise_load_reports_range_failure_and_closes_session(self):
        session_ends = []
        counter = _RecordingCounter()
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.storage = SimpleNamespace(
            store=SimpleNamespace(
                batch_get_session_start=lambda keys: [0],
                batch_get_into_multi_buffer_ranges=lambda keys, ptrs, sizes, offsets: (
                    None
                ),
                batch_get_session_end=lambda keys: session_ends.append(list(keys)),
            ),
            _get_hybrid_page_component_keys=lambda keys, transfer: (list(keys), 1),
            _tag_keys=lambda keys: keys,
        )
        linker.pools = {
            PoolName.KV: SimpleNamespace(
                prepare_locations=lambda indices: [0],
                get_prepared_layer_range_meta=lambda locations, layer: (
                    [[100]],
                    [[4]],
                    [[0]],
                ),
            )
        }
        linker.num_layers = 1
        linker.layer_done_counter = counter

        linker.load_layer_wise(
            8,
            [
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["page"],
                        host_indices=torch.tensor([0]),
                    )
                ]
            ],
        )

        assert counter.completed == []
        assert len(counter.failures) == 1
        assert "range get failed" in str(counter.failures[0][1])
        assert session_ends == [["page"]]

    def test_layer_wise_load_reports_session_cleanup_failure_after_completion(self):
        counter = _RecordingCounter()

        def fail_cleanup(keys):
            raise RuntimeError("injected cleanup failure")

        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.storage = SimpleNamespace(
            store=SimpleNamespace(
                batch_get_session_start=lambda keys: [0],
                batch_get_into_multi_buffer_ranges=lambda keys, ptrs, sizes, offsets: [
                    4
                ],
                batch_get_session_end=fail_cleanup,
            ),
            _get_hybrid_page_component_keys=lambda keys, transfer: (list(keys), 1),
            _tag_keys=lambda keys: keys,
        )
        linker.pools = {
            PoolName.KV: SimpleNamespace(
                prepare_locations=lambda indices: [0],
                get_prepared_layer_range_meta=lambda locations, layer: (
                    [[100]],
                    [[4]],
                    [[0]],
                ),
            )
        }
        linker.num_layers = 1
        linker.layer_done_counter = counter

        linker.load_layer_wise(
            9,
            [
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["page"],
                        host_indices=torch.tensor([0]),
                    )
                ]
            ],
        )

        assert counter.completed == [(9, 0)]
        assert len(counter.failures) == 1
        assert "injected cleanup failure" in str(counter.failures[0][1])

    def test_replicated_non_owner_offload_completes_without_storage_io(self):
        expanded = [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                host_indices=torch.tensor([0, 1]),
            )
        ]
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, allow_partial: expanded
        )
        linker.gc_frozen = True
        linker.offload_owner = False
        linker.offload_queue = Queue()
        linker.offload_results = Queue()

        assert linker.offload(expanded)
        assert linker.offload_queue.empty()
        assert linker.num_completed_offloads() == 1
        assert linker.pop_completed_offload()

    def test_owner_offload_worker_reports_success_and_failures(self):
        patches = _PatchManager(self)
        event_calls = []

        class _Event:
            def record(self):
                event_calls.append("record")

            def synchronize(self):
                event_calls.append("synchronize")

        patches.setattr(mooncake_direct_linker.device_module, "Event", _Event)

        storage_calls = []

        def batch_set_v2(transfers):
            storage_calls.append(transfers)
            if len(storage_calls) <= 2:
                return {PoolName.KV: [True, True]}
            if len(storage_calls) == 3:
                return {PoolName.KV: [True, False]}
            raise RuntimeError("injected offload failure")

        expanded = [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page0", "page1"],
                host_indices=torch.tensor([0, 1]),
            )
        ]
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.page_size = 2
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, allow_partial: expanded
        )
        linker.storage = SimpleNamespace(batch_set_v2=batch_set_v2)
        linker.gc_frozen = True
        linker.offload_owner = True
        linker.offload_queue = Queue()
        linker.offload_results = Queue()
        linker.stats = {"offload": 0}
        thread = threading.Thread(target=linker.offload_thread_func, daemon=True)
        thread.start()

        for _ in range(4):
            assert linker.offload(expanded)
        linker.offload_queue.join()

        assert [linker.pop_completed_offload() for _ in range(4)] == [
            True,
            True,
            False,
            False,
        ]
        assert linker.num_completed_offloads() == 0
        assert linker.stats["offload"] == 2
        assert event_calls.count("record") == 4
        assert event_calls.count("synchronize") == 4
        assert thread.is_alive()

        linker.offload_queue.put(None)
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_offload_short_circuits_when_transfer_cannot_be_resolved(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, allow_partial: []
        )

        assert not linker.offload([object()])

    def test_reset_discards_pending_and_completed_state(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pending_loads = {"rid": [object()]}
        linker.load_queue = Queue()
        linker.completed_loads = Queue()
        linker.completed_loads.put(["rid"])
        linker.offload_queue = Queue()
        linker.offload_results = Queue()
        linker.offload_results.put(True)
        linker.layer_done_counter = LayerWiseLoadCounter(num_layers=2)
        linker.layer_done_counter.update_producer()
        linker.layer_done_counter.set_consumer(0)

        linker.reset()

        assert linker.pending_loads == {}
        assert linker.completed_loads.empty()
        assert linker.offload_results.empty()
        assert linker.layer_done_counter.producer_index == -1
        assert linker.layer_done_counter.consumer_index == -1
        assert linker.layer_done_counter.futures == {}

    def test_v2_physical_kv_gets_multi_layer_buffers(self):
        store = _make_v2_store()
        store.registered_pools[PoolName.KV] = _MultiBufferPool()
        get_call = {}

        def get(keys, ptrs, sizes):
            get_call.update(keys=keys, ptrs=ptrs, sizes=sizes)
            return [sum(item) for item in sizes]

        store._get_batch_zero_copy_impl = get
        result = store.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=["page0", "page1"],
                    host_indices=torch.tensor([0, 1]),
                )
            ]
        )

        assert result[PoolName.KV] == [True, True]
        assert get_call == {
            "keys": ["page0__k", "page1__k"],
            "ptrs": [[4000, 4001], [4010, 4011]],
            "sizes": [[8, 16], [8, 16]],
        }

    def test_v2_lookup_returns_sparse_trailing_boundaries(self):
        store = _make_v2_store()
        store.registered_pools[PoolName.SWA] = _SingleBufferPool()
        existing = {"page1__swa", "page3__swa"}
        store._batch_exist = lambda keys: [int(key in existing) for key in keys]
        page_keys = [f"page{i}" for i in range(4)]

        result = store.batch_exists_v2(
            page_keys,
            [
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=["one-page-window"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ],
        )

        assert result.kv_hit_pages == 4
        assert result.restorable_prefix_pages == [2, 4]
        assert result.extra_pool_hit_pages[PoolName.SWA] == 4

    def test_v2_lookup_intersects_all_pages_with_sparse_trailing_boundaries(self):
        store = _make_v2_store()
        store.registered_pools[PoolName.INDEXER] = _SingleBufferPool()
        store.registered_pools[PoolName.SWA] = _SingleBufferPool()
        existing = {
            "page0__indexer",
            "page1__indexer",
            "page2__indexer",
            "page1__swa",
            "page2__swa",
        }
        store._batch_exist = lambda keys: [int(key in existing) for key in keys]
        page_keys = [f"page{i}" for i in range(4)]

        result = store.batch_exists_v2(
            page_keys,
            [
                PoolTransfer(name=PoolName.INDEXER),
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=["window0", "window1"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
            ],
        )

        assert result.kv_hit_pages == 3
        assert result.restorable_prefix_pages == [3]
        assert result.extra_pool_hit_pages == {
            PoolName.KV: 4,
            PoolName.INDEXER: 3,
            PoolName.SWA: 3,
        }

    def test_dsa_index_buffers_elided_without_external_linker(self):
        self._check_external_linker_keeps_dsa_index_buffers(False, True)

    def test_dsa_index_buffers_kept_with_external_linker(self):
        self._check_external_linker_keeps_dsa_index_buffers(True, False)

    def _check_external_linker_keeps_dsa_index_buffers(
        self, external_linker_enabled, expected
    ):
        patches = _PatchManager(self)
        from sglang.srt.mem_cache import kv_cache_configurator

        patches.setattr(
            kv_cache_configurator,
            "get_memory",
            lambda: SimpleNamespace(
                enable_hisparse=False,
                enable_hierarchical_cache=False,
                enable_unified_cache_external_linker=external_linker_enabled,
            ),
        )
        patches.setattr(
            kv_cache_configurator,
            "get_disagg",
            lambda: SimpleNamespace(disaggregation_mode="null"),
        )

        assert (
            kv_cache_configurator._should_elide_dsa_index_k(is_draft_worker=False)
            is expected
        )


if __name__ == "__main__":
    unittest.main()
