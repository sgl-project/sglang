import contextlib
import json
import os
import threading
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.umbp import umbp_direct_linker
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    LayerWiseLoadCounter,
    UMBPDirectLinker,
    _object_sizes_per_page,
    _ordered_layers,
    _PoolRangePlan,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache_linker import (
    DevicePoolEntry,
    DevicePoolGroup,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DeploymentMode(Enum):
    Local = 0
    StandaloneProcess = 1
    Distributed = 2


def _assert_page_pointers_match_views(test, entry, indices):
    """Verify precomputed row pointers against tensor-view data pointers."""
    pointers, _ = entry.get_page_buffer_meta(indices)
    test.assertEqual(
        pointers,
        [
            buffer[row].data_ptr()
            for row in entry.prepare_locations(indices)
            for component in entry.components
            for buffer in component
        ],
    )


class TestUMBPDirectLinker(unittest.TestCase):
    page_size = 2
    num_layers = 3

    def setUp(self):
        self.kv_buffers = [
            torch.zeros((32, 1, 4), dtype=torch.uint8) for _ in range(self.num_layers)
        ]
        self.indexer_buffers = [
            torch.zeros((16, 6), dtype=torch.uint8) for _ in range(self.num_layers)
        ]
        identity = {layer: layer for layer in range(self.num_layers)}
        self.pool_group = DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=PoolName.KV,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[self.kv_buffers],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=False,
                ),
                DevicePoolEntry(
                    name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[self.indexer_buffers],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=True,
                ),
            ],
            self.num_layers,
            self.page_size,
        )
        self.pools = self.pool_group.entry_map

        self.client = MagicMock()
        self.client.is_distributed.return_value = True
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess
        self.client.get_backend_mode.return_value = _DeploymentMode.Local
        self.client.supports_ranged_io.return_value = True
        self.client.register_memory.return_value = True
        self.client.batch_exists.side_effect = lambda keys: [True] * len(keys)
        self.client.batch_put_ranges_from_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        self.client.batch_get_ranges_into_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        self.client.report_external_kv_blocks.return_value = True
        self.client.revoke_external_kv_blocks.return_value = True
        self.client.revoke_all_external_kv_blocks_at_tier.return_value = True

        self.storage = MagicMock()
        self.storage.client = self.client
        self.storage._disable_zero_copy_register = False
        self.storage._get_hybrid_page_component_keys.side_effect = (
            lambda keys, transfer, rank_suffix=None: (
                [f"{key}_{rank_suffix or 'rank'}_{transfer.name}" for key in keys],
                1,
            )
        )

        self.freeze_gc_patcher = patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.freeze_gc"
        )
        self.freeze_gc_mock = self.freeze_gc_patcher.start()
        self.addCleanup(self.freeze_gc_patcher.stop)
        self.event_patcher = patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.device_module.Event",
            side_effect=lambda: SimpleNamespace(
                record=lambda: None, synchronize=lambda: None
            ),
        )
        self.event_patcher.start()
        self.addCleanup(self.event_patcher.stop)

        self.server_args = SimpleNamespace(
            hicache_storage_backend_extra_config=None,
            tp_size=1,
            pp_size=1,
            attn_cp_size=1,
            enable_dp_attention=False,
            model_path="test-model",
        )
        self.params = SimpleNamespace(
            page_size=self.page_size,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
        )
        self.connectors = []

    def tearDown(self):
        for connector in self.connectors:
            connector.close()

    def make_connector(
        self,
        extra_config=None,
        pool_group=None,
    ):
        pool_group = pool_group or self.pool_group
        self.server_args.hicache_storage_backend_extra_config = (
            json.dumps(extra_config) if extra_config is not None else None
        )
        with (
            # A sibling test leaves gloo initialized without sglang parallel groups.
            patch("torch.distributed.is_initialized", return_value=False),
            patch.object(
                umbp_direct_linker,
                "resolve_hybrid_device_pool_group",
                return_value=pool_group,
            ),
            patch(
                "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker._parse_storage_extra_config",
                return_value=dict(extra_config or {}),
            ),
        ):
            connector = UMBPDirectLinker(
                self.server_args,
                self.params,
                components={ComponentType.FULL},
                _storage=self.storage,
            )
        self.connectors.append(connector)
        return connector

    def transfer(self, pages=2):
        starts = torch.arange(pages, dtype=torch.int64) * self.page_size
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        indices = (starts[:, None] + offsets).flatten()
        return PoolTransfer(
            name=PoolName.KV,
            device_indices=indices,
            keys=[f"page-{index}" for index in range(pages)],
        )

    @staticmethod
    def wait_for_offloads(connector):
        connector._offload_queue.join()

    def test_object_key_and_pointer_order_are_page_major(self):
        connector = self.make_connector()
        transfer = connector.pool_group.resolve_transfers([self.transfer(pages=2)])[0]

        keys = connector._object_keys(transfer)
        entry = connector.pools[transfer.name]
        locations = entry.prepare_locations(transfer.host_indices)

        # One object per page now; the layer lives inside it as a byte range.
        self.assertEqual(keys, [f"page-{page}_rank_kv" for page in range(2)])

        # `batch_*_ranges` pairs keys with range entries positionally, so the
        # two must stay the same length for every layer. A mismatch here is the
        # shape that silently shifts every range by one object.
        for layer in range(self.num_layers):
            ptrs, sizes, offsets = entry.get_prepared_layer_range_meta(locations, layer)
            self.assertEqual(len(ptrs), len(keys))
            self.assertEqual(len(sizes), len(keys))
            self.assertEqual(len(offsets), len(keys))

        # Offsets must tile the object exactly, in layer order, and the object
        # size the connector declares must match that tiling -- deriving it
        # from the emitted ranges instead would hide a dropped trailing layer.
        per_page = _object_sizes_per_page(entry)
        self.assertEqual(len(per_page), 1)
        cursor = 0
        for layer in range(self.num_layers):
            _, sizes, offsets = entry.get_prepared_layer_range_meta(locations, layer)
            self.assertEqual(offsets[0], [cursor])
            cursor += sizes[0][0]
        self.assertEqual(cursor, per_page[0])

    def test_dsa_transfer_resolution_matches_legacy_expansion(self):
        connector = self.make_connector()
        source = self.transfer(pages=2)

        resolved = connector.pool_group.resolve_transfers([source])

        self.assertEqual([transfer.name for transfer in resolved], list(self.pools))
        for transfer in resolved:
            self.assertEqual(transfer.keys, source.keys)
            self.assertTrue(torch.equal(transfer.host_indices, source.device_indices))
            self.assertIsNone(transfer.indices_from_pool)
            _assert_page_pointers_match_views(
                self, connector.pools[transfer.name], transfer.host_indices
            )

    def test_load_plan_materializes_a_shared_source_only_once(self):
        connector = self.make_connector()
        resolved = connector.pool_group.resolve_transfers([self.transfer(pages=2)])
        self.assertIs(resolved[0].host_indices, resolved[1].host_indices)

        with patch.object(
            umbp_direct_linker,
            "_materialize_cpu_indices",
            wraps=umbp_direct_linker._materialize_cpu_indices,
        ) as materialize:
            plans = connector._build_load_plans([resolved])

        self.assertEqual(materialize.call_count, 1)
        by_name = {plan.name: plan for plan in plans}
        # KV buffers are token-row based; INDEXER buffers are page-row based.
        # Only the immutable source snapshot is shared, never final row geometry.
        self.assertEqual(by_name[PoolName.KV].locations, [0, 2])
        self.assertEqual(by_name[PoolName.INDEXER].locations, [0, 1])

    def test_async_offload_snapshot_uses_distinct_reusable_pinned_slots(self):
        connector = UMBPDirectLinker.__new__(UMBPDirectLinker)
        connector._async_offload_index_snapshot = True
        connector._offload_index_fallback_warned = False
        connector._offload_index_stream = None
        connector._offload_index_done = None
        connector._offload_index_device = None
        connector._offload_index_buffers = []

        device = torch.device("cuda:3")

        class FakeCudaTensor(torch.Tensor):
            @property
            def device(self):
                return device

            def record_stream(self, stream):
                pass

        def indices(values):
            return torch.tensor(values, dtype=torch.int64).as_subclass(FakeCudaTensor)

        allocation_sizes = []

        def allocate(count):
            allocation_sizes.append(count)
            return torch.empty(count, dtype=torch.int64)

        stream = MagicMock()
        done = MagicMock()
        fake_device_module = SimpleNamespace(
            device=MagicMock(side_effect=lambda _: contextlib.nullcontext()),
            Stream=MagicMock(return_value=stream),
            Event=MagicMock(return_value=done),
            stream=MagicMock(side_effect=lambda _: contextlib.nullcontext()),
        )

        with (
            patch.object(umbp_direct_linker, "device_module", fake_device_module),
            patch.object(
                connector,
                "_allocate_offload_index_buffer",
                side_effect=allocate,
            ),
        ):
            first = connector._materialize_offload_indices(indices([11, 12, 13, 14]), 0)
            first_before_reuse = first.clone()
            second = connector._materialize_offload_indices(indices([21, 22]), 1)
            slot_zero_ptr = first.data_ptr()
            slot_one_before_reuse = second.clone()
            reused = connector._materialize_offload_indices(indices([31, 32]), 0)

        self.assertEqual(first_before_reuse.tolist(), [11, 12, 13, 14])
        self.assertEqual(second.tolist(), [21, 22])
        self.assertEqual(reused.tolist(), [31, 32])
        self.assertEqual(reused.data_ptr(), slot_zero_ptr)
        self.assertNotEqual(second.data_ptr(), slot_zero_ptr)
        self.assertTrue(torch.equal(second, slot_one_before_reuse))
        self.assertNotIn(0, allocation_sizes)

    def test_async_offload_snapshot_records_before_copy_failure(self):
        connector = UMBPDirectLinker.__new__(UMBPDirectLinker)
        connector._async_offload_index_snapshot = True
        connector._offload_index_fallback_warned = False
        connector._offload_index_stream = None
        connector._offload_index_done = None
        connector._offload_index_device = None
        connector._offload_index_buffers = []

        indices = MagicMock()
        indices.device = torch.device("cuda:2")
        indices.dtype = torch.int64
        indices.ndim = 1
        indices.is_contiguous.return_value = True
        indices.numel.return_value = 4
        indices.detach.return_value = indices

        buffer = MagicMock()
        buffer.numel.return_value = 4
        buffer.__getitem__.return_value = buffer
        timeline = []
        indices.record_stream.side_effect = lambda _: timeline.append("record")

        def fail_copy(*args, **kwargs):
            timeline.append("copy")
            raise RuntimeError("copy failed")

        buffer.copy_.side_effect = fail_copy
        stream = MagicMock()
        done = MagicMock()
        fake_device_module = SimpleNamespace(
            device=lambda _: contextlib.nullcontext(),
            Stream=lambda **_: stream,
            Event=lambda: done,
            stream=lambda _: contextlib.nullcontext(),
        )
        fallback = object()

        with (
            patch.object(umbp_direct_linker, "device_module", fake_device_module),
            patch.object(
                connector, "_allocate_offload_index_buffer", return_value=buffer
            ),
            patch.object(
                umbp_direct_linker,
                "_materialize_cpu_indices",
                return_value=fallback,
            ) as sync_copy,
            self.assertLogs(umbp_direct_linker.logger, level="ERROR"),
        ):
            result = connector._materialize_offload_indices(indices, 0)

        self.assertIs(result, fallback)
        self.assertEqual(timeline, ["record", "copy"])
        stream.synchronize.assert_not_called()
        done.record.assert_not_called()
        sync_copy.assert_called_once_with(indices)
        self.assertFalse(connector._async_offload_index_snapshot)
        self.assertIs(connector._offload_index_buffers[0], buffer)

    def test_offload_snapshot_shape_fallback_warns_once(self):
        connector = UMBPDirectLinker.__new__(UMBPDirectLinker)
        connector._async_offload_index_snapshot = True
        connector._offload_index_fallback_warned = False
        connector._offload_index_device = None
        device = torch.device("cuda:1")

        class FakeCudaTensor(torch.Tensor):
            @property
            def device(self):
                return device

        indices = torch.tensor([3, 5], dtype=torch.int32).as_subclass(FakeCudaTensor)
        with patch.object(umbp_direct_linker.logger, "warning") as warning:
            first = connector._materialize_offload_indices(indices, 0)
            second = connector._materialize_offload_indices(indices, 0)

        self.assertEqual(first.tolist(), [3, 5])
        self.assertEqual(first.dtype, torch.int64)
        self.assertEqual(second.tolist(), [3, 5])
        warning.assert_called_once()

        with patch.object(umbp_direct_linker.logger, "warning") as warning:
            connector._materialize_offload_indices(
                torch.tensor([7, 9], dtype=torch.int32), 0
            )
        warning.assert_not_called()

    def test_resolver_accepts_deepseek_v4_pool(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def state_pool():
            return SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(
                    kv_score=torch.zeros((8, 3), dtype=torch.uint8)
                ),
            )

        kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        kvcache._unified_kv = False
        kvcache.start_layer = 0
        kvcache.end_layer = 3
        kvcache.swa_page_size = self.page_size
        kvcache.swa_kv_pool = SimpleNamespace(
            kv_buffer=[
                torch.zeros((8, 3), dtype=torch.uint8) for _ in range(kvcache.end_layer)
            ]
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
            DeepSeekV4LayerItem(4, 1),
            DeepSeekV4LayerItem(128, 0),
            DeepSeekV4LayerItem(4, 0),
        ]
        kvcache.compress_state_pools = [state_pool(), None, state_pool()]
        kvcache.indexer_compress_state_pools = [state_pool(), None, state_pool()]

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=self.params,
            components={ComponentType.FULL, ComponentType.SWA},
        )

        self.assertEqual(group.num_layers, 3)
        self.assertEqual(len(group.entry_map), 6)
        self.assertEqual(
            _ordered_layers(group.entry_map[PoolName.DEEPSEEK_V4_C4]), [2, 0]
        )
        probe = torch.arange(self.page_size, dtype=torch.int64)
        for entry in group.entry_map.values():
            _assert_page_pointers_match_views(self, entry, probe)

    def test_pool_layers_follow_buffer_indices_not_logical_layer_order(self):
        buffers = [
            torch.zeros((4, 3), dtype=torch.uint8),
            torch.zeros((4, 5), dtype=torch.uint8),
        ]
        entry = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[buffers],
            layer_mapping={0: 1, 2: 0},
            page_size=2,
            rows_are_pages=False,
        )

        layers = _ordered_layers(entry)
        ptrs, _ = entry.get_page_buffer_meta(torch.tensor([0, 1]))

        self.assertEqual(layers, [2, 0])
        self.assertEqual(ptrs, [buffers[0][0].data_ptr(), buffers[1][0].data_ptr()])

    def test_load_plans_allow_different_page_counts_per_pool(self):
        connector = self.make_connector()
        kv_transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([0, 1, 2, 3]),
            keys=["kv-0", "kv-1"],
        )
        indexer_transfer = PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=torch.tensor([0, 1]),
            keys=["indexer-0"],
        )

        plans = connector._build_load_plans([[kv_transfer, indexer_transfer]])

        # One object per page per pool, and the rows the ranges are built from
        # must stay in step with the keys.
        self.assertEqual(
            {plan.name: len(plan.keys) for plan in plans},
            {PoolName.KV: 2, PoolName.INDEXER: 1},
        )
        for plan in plans:
            self.assertEqual(plan.entries_per_page, 1)
            self.assertEqual(len(plan.locations), len(plan.keys))

    def test_layerwise_load_completes_logical_layers_without_objects(self):
        connector = UMBPDirectLinker.__new__(UMBPDirectLinker)
        connector.num_layers = 3
        connector.layer_group = 1
        connector.pool_layers = {PoolName.KV: [0, 2]}
        connector.storage = self.storage
        connector._trace_perf = False
        connector._traced = 0
        connector._trace_budget = 0
        connector._lookup_traced = 0
        connector._start_traced = 0
        connector._exists_build_ms = 0.0
        connector._exists_rpc_ms = 0.0
        connector._exists_keys = 0
        connector.layer_done_counter = LayerWiseLoadCounter(connector.num_layers)
        connector.pools = {PoolName.KV: self.pools[PoolName.KV]}
        plan = _PoolRangePlan(
            name=PoolName.KV,
            keys=["page"],
            locations=[0],
            entries_per_page=1,
        )
        counter = connector.layer_done_counter.update_producer()
        connector.layer_done_counter.set_consumer(counter)

        connector._run_layer_wise_batch(counter, [plan])

        connector.layer_done_counter.wait_until(2)
        # Layers 0 and 2 belong to the pool; layer 1 has no objects and must
        # still complete so the forward thread is released.
        self.assertEqual(self.client.batch_get_ranges_into_ptr.call_count, 2)

    def test_lookup_stops_at_first_partial_page_across_chunks(self):
        connector = self.make_connector()
        # One object per page now, so a chunk holds far more pages than before
        # and all four fit in one probe per pool: pages 1-3 present, page 4 not.
        self.client.batch_exists.side_effect = [
            [True, True, True, False],
            [True, True, True, False],
        ]
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.CHUNK_PAGES",
            2,
        ):
            hit = connector.lookup("rid", [self.transfer(pages=4)])

        self.assertEqual(hit, [1, 2, 3])
        self.assertEqual(self.client.batch_exists.call_count, 2)

    def test_trailing_hit_policy_returns_sparse_valid_prefixes(self):
        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=["tail"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        valid = UMBPDirectLinker._apply_hit_policy(
            [1, 2, 3, 4], [True, True, False, True], transfer
        )

        self.assertEqual(valid, [1, 2, 4])

    def test_lookup_uses_full_kv_key_domain_for_trailing_pool(self):
        pool_group = self._hybrid_pool_group()
        connector = self.make_connector(pool_group=pool_group)
        kv = PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(8),
            keys=["p0", "p1", "p2", "p3"],
        )
        swa = PoolTransfer(
            name=PoolName.SWA,
            device_indices=torch.tensor([6, 7]),
            keys=["p3"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        valid = connector.lookup("rid", [kv, swa])

        self.assertEqual(valid, [1, 2, 3, 4])
        queried = [
            key
            for call in self.client.batch_exists.call_args_list
            for key in call.args[0]
        ]
        self.assertIn("p0_rank_swa", queried)
        self.assertIn("p3_rank_swa", queried)

    def test_load_accepts_swa_without_kv_after_tree_deduplication(self):
        connector = self.make_connector(pool_group=self._hybrid_pool_group())
        swa = PoolTransfer(
            name=PoolName.SWA,
            device_indices=torch.tensor([0, 1]),
            keys=["tail"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        self.assertTrue(connector.load("rid", [swa]))

        self.assertEqual(len(connector._pending), 1)
        queued = connector._pending["rid"]
        self.assertEqual([transfer.name for transfer in queued], [PoolName.SWA])
        self.assertEqual(queued[0].keys, ["tail"])

    def test_offload_allows_partial_hybrid_sources(self):
        connector = self.make_connector(pool_group=self._hybrid_pool_group())
        kv = PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(8),
            keys=["p0", "p1", "p2", "p3"],
        )

        self.assertTrue(connector.offload([kv]))
        self.wait_for_offloads(connector)

        # Count first: that call is where ranks agree on what may be consumed.
        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertTrue(connector.pop_completed_offload())
        sent_keys = [
            key
            for call in self.client.batch_put_ranges_from_ptr.call_args_list
            for key in call.args[0]
        ]
        self.assertTrue(sent_keys)
        self.assertTrue(all("deepseek_v4_c4" in key for key in sent_keys))
        self.assertTrue(all("_swa_" not in key for key in sent_keys))

    def _hybrid_pool_group(self):
        identity = {0: 0}
        return DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=PoolName.DEEPSEEK_V4_C4,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[[torch.zeros((4, 3), dtype=torch.uint8)]],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=True,
                ),
                DevicePoolEntry(
                    name=PoolName.SWA,
                    indices_from_pool=PoolName.SWA,
                    device_pool=None,
                    components=[[torch.zeros((4, 5), dtype=torch.uint8)]],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=True,
                ),
            ],
            num_layers=1,
            page_size=self.page_size,
        )

    def test_offload_is_chunked_on_logical_page_boundaries(self):
        connector = self.make_connector()
        # Offload puts one range per layer on an object, so a budget of
        # 2 * num_layers ranges is a budget of 2 objects.
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.RANGES_PER_CALL",
            2 * self.num_layers,
        ):
            self.assertTrue(connector.offload([self.transfer(pages=5)]))
            self.wait_for_offloads(connector)

        # Chunk boundaries may fall between objects, never inside one.
        calls = self.client.batch_put_ranges_from_ptr.call_args_list
        self.assertEqual(len(calls), 6)
        self.assertEqual([len(call.args[0]) for call in calls], [2, 2, 1, 2, 2, 1])
        for call in calls:
            for object_size, sizes in zip(call.args[1], call.args[3]):
                self.assertEqual(sum(sizes), object_size)

    def test_offload_keeps_key_range_and_object_size_pairing(self):
        """Keep keys, ranges, and declared object sizes aligned while chunking."""
        connector = self.make_connector()

        expected = {}
        for transfer in connector.pool_group.resolve_transfers(
            [self.transfer(pages=3)]
        ):
            plan = connector._build_load_plans([[transfer]])[0]
            ptrs, sizes, offsets = connector._all_layer_ranges(plan)
            for index, key in enumerate(plan.keys):
                expected[key] = (ptrs[index], sizes[index], offsets[index])

        self.client.batch_put_ranges_from_ptr.reset_mock()
        self.assertTrue(connector.offload([self.transfer(pages=3)]))
        self.wait_for_offloads(connector)

        seen = {}
        for call in self.client.batch_put_ranges_from_ptr.call_args_list:
            keys, object_sizes = call.args[0], call.args[1]
            ptrs, sizes, offsets = call.args[2], call.args[3], call.args[4]
            self.assertEqual(len(keys), len(object_sizes))
            self.assertEqual(len(keys), len(ptrs))
            self.assertEqual(len(keys), len(sizes))
            self.assertEqual(len(keys), len(offsets))
            for index, key in enumerate(keys):
                seen[key] = (ptrs[index], sizes[index], offsets[index])
                # The declared object size must match the tiling, or the tier's
                # exact-tiling check -- the only write-time guard against a
                # dropped trailing layer -- has nothing to compare against.
                self.assertEqual(
                    object_sizes[index],
                    max(o + z for o, z in zip(offsets[index], sizes[index])),
                )
                self.assertEqual(sum(sizes[index]), object_sizes[index])

        self.assertEqual(seen, expected)

    def test_offload_success_produces_exactly_one_result(self):
        event_calls = []

        class _Event:
            def record(self):
                event_calls.append("record")

            def synchronize(self):
                event_calls.append("synchronize")

        def put_after_event(keys, *args):
            self.assertEqual(event_calls, ["record", "synchronize"])
            return [True] * len(keys)

        self.client.batch_put_ranges_from_ptr.side_effect = put_after_event
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.device_module.Event",
            _Event,
        ):
            connector = self.make_connector()

            self.assertTrue(connector.offload([self.transfer(pages=1)]))
            self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertTrue(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_offload_failure_produces_exactly_one_false_result(self):
        self.client.batch_put_ranges_from_ptr.return_value = [False]
        self.client.batch_put_ranges_from_ptr.side_effect = None
        connector = self.make_connector()

        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_offload_exceptions_produce_exactly_one_false_result(self):
        self.client.batch_put_ranges_from_ptr.side_effect = RuntimeError("put failed")
        connector = self.make_connector()

        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.assertEqual(connector.num_completed_offloads(), 0)

        class _FailingEvent:
            def record(self):
                pass

            def synchronize(self):
                raise RuntimeError("event failed")

        self.client.batch_put_ranges_from_ptr.reset_mock()
        self.client.batch_put_ranges_from_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.device_module.Event",
            _FailingEvent,
        ):
            connector = self.make_connector()
            self.assertTrue(connector.offload([self.transfer(pages=1)]))
            self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.client.batch_put_ranges_from_ptr.assert_not_called()

    def test_offload_results_are_fifo(self):
        def result_for_key(keys, *args):
            return [not keys[0].startswith("fail-")] * len(keys)

        self.client.batch_put_ranges_from_ptr.side_effect = result_for_key
        connector = self.make_connector()
        success = self.transfer(pages=1)
        success.keys = ["success-page"]
        failure = self.transfer(pages=1)
        failure.keys = ["fail-page"]

        self.assertTrue(connector.offload([success]))
        self.assertTrue(connector.offload([failure]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 2)
        self.assertTrue(connector.pop_completed_offload())
        self.assertFalse(connector.pop_completed_offload())

    def test_offload_materializes_a_shared_source_only_once(self):
        connector = self.make_connector()

        with patch.object(
            connector,
            "_materialize_offload_indices",
            wraps=connector._materialize_offload_indices,
        ) as materialize:
            self.assertTrue(connector.offload([self.transfer(pages=2)]))
            self.wait_for_offloads(connector)

        self.assertEqual(materialize.call_count, 1)
        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertTrue(connector.pop_completed_offload())
        self.assertEqual(self.client.batch_put_ranges_from_ptr.call_count, 2)

    @contextlib.contextmanager
    def peer_reductions(self, connector, peer_values):
        """Inject peer values into successive MIN reductions."""
        pending = [list(values) for values in peer_values]

        def fake_all_reduce(tensor, op=None, group=None):
            self.assertTrue(pending, "connector issued more reductions than scripted")
            peer = torch.tensor(pending.pop(0), dtype=tensor.dtype)
            self.assertEqual(peer.numel(), tensor.numel())
            tensor.copy_(torch.minimum(tensor, peer))

        connector._offload_sync_groups = (object(),)
        with patch("torch.distributed.all_reduce", fake_all_reduce):
            yield
        self.assertFalse(pending, f"scripted reductions never happened: {pending}")

    def offload_pages(self, connector, count):
        for _ in range(count):
            self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

    def test_completed_offload_outcomes_agree_across_ranks(self):
        connector = self.make_connector()
        self.offload_pages(connector, 2)

        # Counting is local; each pop reduces exactly one outcome.
        with self.peer_reductions(connector, [[1], [0]]):
            self.assertEqual(connector.num_completed_offloads(), 2)
            self.assertTrue(connector.pop_completed_offload())
            self.assertFalse(connector.pop_completed_offload())

    def test_idle_steps_issue_no_collective(self):
        """Counting completed offloads never issues a collective."""
        connector = self.make_connector()
        connector._offload_sync_groups = (object(),)

        with patch("torch.distributed.all_reduce") as all_reduce:
            self.assertEqual(connector.num_completed_offloads(), 0)

        all_reduce.assert_not_called()

    def test_wrapper_drain_releases_nodes_with_agreed_outcomes(self):
        connector = self.make_connector()
        self.offload_pages(connector, 3)
        nodes = [SimpleNamespace(id=i, external_cache_stored=False) for i in range(3)]
        cache = SimpleNamespace(
            released=[],
        )
        cache.resolve_node_handle = lambda node_id: nodes[node_id]
        cache.dec_lock_ref = lambda node_id, params: cache.released.append(node_id)
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache
        wrapper.cache_linker = connector
        wrapper.pending_offloads = [(node.id, object()) for node in nodes]

        with self.peer_reductions(connector, [[0]]):
            wrapper.drain_offloads(1)

        self.assertEqual(cache.released, [0])
        self.assertEqual(len(wrapper.pending_offloads), 2)
        self.assertFalse(nodes[0].external_cache_stored)

    def test_reset_waits_for_offload_and_discards_stale_result(self):
        worker_started = threading.Event()
        release_worker = threading.Event()
        reset_started = threading.Event()
        reset_done = threading.Event()

        def blocked_put(keys, *args):
            worker_started.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("test did not release offload worker")
            return [True] * len(keys)

        self.client.batch_put_ranges_from_ptr.side_effect = blocked_put
        connector = self.make_connector()
        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.assertTrue(worker_started.wait(timeout=5))

        def run_reset():
            reset_started.set()
            connector.reset()
            reset_done.set()

        reset_thread = threading.Thread(target=run_reset)
        reset_thread.start()
        self.assertTrue(reset_started.wait(timeout=5))
        try:
            self.assertTrue(reset_thread.is_alive())
            self.assertFalse(reset_done.is_set())
        finally:
            release_worker.set()
            reset_thread.join(timeout=5)

        self.assertFalse(reset_thread.is_alive())
        self.assertTrue(reset_done.is_set())
        self.assertEqual(connector.num_completed_offloads(), 0)

    def test_background_load_completes_each_layer(self):
        connector = self.make_connector()
        # The finest granularity, one call per layer. Grouping is covered by
        # test_layer_group_folds_layers_into_one_call_per_object.
        connector.layer_group = 1
        self.assertTrue(connector.load("rid", [self.transfer(pages=3)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        connector.layer_done_counter.wait_until(self.num_layers - 1)

        # One get per layer and pool at this size.
        self.assertEqual(
            self.client.batch_get_ranges_into_ptr.call_count,
            self.num_layers * len(self.pools),
        )

    def test_load_and_offload_share_one_gc_freeze(self):
        self.freeze_gc_mock.reset_mock()
        connector = self.make_connector()

        self.assertTrue(connector.load("rid", [self.transfer(pages=1)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        connector.layer_done_counter.wait_until(self.num_layers - 1)
        self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

        self.freeze_gc_mock.assert_called_once_with("UMBP direct linker")

    def test_background_load_uses_full_object_budget_per_call(self):
        connector = self.make_connector()
        connector.layer_group = 1
        self.assertTrue(connector.load("rid", [self.transfer(pages=7)]))
        # Layer-wise puts a single range on an object, so a budget of 2 ranges
        # is a budget of 2 objects.
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.RANGES_PER_CALL",
            2,
        ):
            counter = connector.start_layer_wise_loading()
            connector.layer_done_counter.set_consumer(counter)
            connector.layer_done_counter.wait_until(self.num_layers - 1)

        # 7 pages go out as [2, 2, 2, 1] per layer and pool.
        calls = self.client.batch_get_ranges_into_ptr.call_args_list
        self.assertEqual(len(calls), self.num_layers * len(self.pools) * 4)
        self.assertEqual(
            [len(call.args[0]) for call in calls],
            [2, 2, 2, 1] * (self.num_layers * len(self.pools)),
        )
        # Every key carries exactly one range per layer for these pools.
        for call in calls:
            for ranges in call.args[1]:
                self.assertEqual(len(ranges), 1)

    def test_background_load_does_not_split_a_load_that_fits_the_budget(self):
        """Do not split a layer-wise load that fits the range budget."""
        connector = self.make_connector()
        connector.layer_group = 1  # the budget, not the group, is under test
        pages = 7
        self.assertTrue(connector.load("rid", [self.transfer(pages=pages)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        connector.layer_done_counter.wait_until(self.num_layers - 1)

        calls = self.client.batch_get_ranges_into_ptr.call_args_list
        self.assertEqual(len(calls), self.num_layers * len(self.pools))
        for call in calls:
            self.assertEqual(len(call.args[0]), pages)

    def test_layer_group_folds_layers_into_one_call_per_object(self):
        """Fold a layer group into one call while completing every layer."""
        connector = self.make_connector()
        group = 2
        connector.layer_group = group
        pages = 3
        self.assertTrue(connector.load("rid", [self.transfer(pages=pages)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        # Completing every layer is the property the forward thread depends on;
        # waiting on the last one would hang if any were skipped.
        for layer in range(self.num_layers):
            connector.layer_done_counter.wait_until(layer)

        # 3 layers in groups of 2 is a ragged split, [0,1] then [2], which is
        # the normal case: no model's layer count divides by the group size.
        group_sizes = [
            min(group, self.num_layers - start)
            for start in range(0, self.num_layers, group)
        ]
        self.assertEqual(group_sizes, [2, 1])
        calls = self.client.batch_get_ranges_into_ptr.call_args_list
        self.assertEqual(len(calls), len(group_sizes) * len(self.pools))
        for call, expected_ranges in zip(
            calls, [size for size in group_sizes for _ in self.pools]
        ):
            self.assertEqual(len(call.args[0]), pages)
            for ranges in call.args[1]:
                self.assertEqual(len(ranges), expected_ranges)

    def test_layer_group_reads_the_same_bytes_as_ungrouped(self):
        """Grouping must not disturb which range lands at which pointer."""
        connector = self.make_connector()
        transfer = self.transfer(pages=3)

        def ranges_for(layer_group):
            self.client.batch_get_ranges_into_ptr.reset_mock()
            connector.layer_group = layer_group
            self.assertTrue(connector.load(f"rid{layer_group}", [transfer]))
            counter = connector.start_layer_wise_loading()
            connector.layer_done_counter.set_consumer(counter)
            connector.layer_done_counter.wait_until(self.num_layers - 1)
            triples = set()
            for call in self.client.batch_get_ranges_into_ptr.call_args_list:
                for key, ptrs, sizes, offsets in zip(*call.args[:4]):
                    triples.update(zip([key] * len(ptrs), ptrs, sizes, offsets))
            return triples

        self.assertEqual(ranges_for(2), ranges_for(1))

    def test_background_load_failure_reaches_consumer(self):
        connector = self.make_connector()
        # One call per layer, so the failure names a single layer.
        connector.layer_group = 1
        call_index = 0

        def fail_second_layer(keys, *args):
            nonlocal call_index
            call_index += 1
            if call_index == 3:
                return [False] * len(keys)
            return [True] * len(keys)

        self.client.batch_get_ranges_into_ptr.side_effect = fail_second_layer
        self.assertTrue(connector.load("rid", [self.transfer(pages=2)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        with self.assertRaisesRegex(
            RuntimeError, "UMBP layer-wise KV load failed"
        ) as raised:
            connector.layer_done_counter.wait_until(1)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(
            str(raised.exception.__cause__),
            "UMBP get failed for pool=kv, layer=1: success=0/2.",
        )

    def test_rejects_unsafe_storage_options(self):
        with self.assertRaisesRegex(ValueError, "ssd_enabled=false"):
            self.make_connector({"ssd_enabled": True})
        with self.assertRaisesRegex(ValueError, "cache_remote_fetches=false"):
            self.make_connector({"cache_remote_fetches": True})
        with self.assertRaisesRegex(ValueError, "smallest per-layer object"):
            self.make_connector({"dram_page_size": 1024})
        with self.assertRaisesRegex(ValueError, "smallest per-layer object"):
            self.make_connector({"dram_page_size": 0})

        connector = self.make_connector({"dram_page_size": 6})
        self.assertFalse(connector._closed)

    def test_rejects_removed_split_load(self):
        with patch.dict(os.environ, {"UMBP_LOAD_SPLIT": "1"}):
            with self.assertRaisesRegex(ValueError, "UMBP_LOAD_SPLIT"):
                self.make_connector()

    def test_standalone_process_skips_distributed_page_config(self):
        self.client.is_distributed.return_value = False
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess

        connector = self.make_connector(
            {
                "standalone_address": "unix:///tmp/umbp-test.sock",
                "dram_page_size": 1024,
            }
        )

        self.assertEqual(connector.deployment_mode, _DeploymentMode.StandaloneProcess)

    def test_rejects_deployment_modes_without_ranged_io(self):
        """Require a StandaloneProcess client for ranged I/O."""
        for mode in (_DeploymentMode.Local, _DeploymentMode.Distributed):
            with self.subTest(mode=mode):
                self.client.get_deployment_mode.return_value = mode
                with self.assertRaisesRegex(ValueError, "StandaloneProcess"):
                    self.make_connector()

    def test_rejects_standalone_server_without_ranged_capability(self):
        self.client.supports_ranged_io.return_value = False

        with self.assertRaisesRegex(
            ValueError, "UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES"
        ):
            self.make_connector()

        self.storage.close.assert_called_once()

    def test_external_kv_reconcile_checks_every_rank_and_pool(self):
        self.client.get_backend_mode.return_value = _DeploymentMode.Distributed
        self.server_args.tp_size = 2
        with patch.dict(
            os.environ,
            {
                "UMBP_EXTKV_REPORT": "1",
                "UMBP_EXTKV_FLUSH_MS": "3600000",
                "UMBP_EXTKV_RECONCILE_SECONDS": "3600",
            },
        ):
            connector = self.make_connector()

        connector._queue_extkv_pages(["0123456789abcdef0123456789abcdef"])
        connector._flush_extkv()

        self.client.report_external_kv_blocks.assert_called_once()
        reported_hash = self.client.report_external_kv_blocks.call_args.args[0][0]
        required = connector._extkv_required_keys[reported_hash]
        self.assertEqual(len(required), 4)  # 2 TP ranks x (KV + INDEXER)
        self.assertTrue(any("tp1_cp0_pp0_kv" in key for key in required))
        self.assertTrue(any("tp0_cp0_pp0_indexer" in key for key in required))

        # Losing a non-anchor object must revoke the logical block even while
        # every KV anchor remains present.
        self.client.batch_exists.side_effect = lambda keys: [
            "indexer" not in key for key in keys
        ]
        connector._reconcile_extkv()

        self.client.revoke_external_kv_blocks.assert_called_once()
        self.assertNotIn(reported_hash, connector._extkv_reported)

        # Re-report the same block, then lose only a non-tp0 KV key. The full
        # required-key AND must revoke it again.
        self.client.batch_exists.side_effect = lambda keys: [True] * len(keys)
        connector._queue_extkv_pages(["0123456789abcdef0123456789abcdef"])
        connector._flush_extkv()
        self.client.batch_exists.side_effect = lambda keys: [
            "tp1_cp0_pp0_kv" not in key for key in keys
        ]
        connector._reconcile_extkv()
        self.assertEqual(self.client.revoke_external_kv_blocks.call_count, 2)
        self.assertNotIn(reported_hash, connector._extkv_reported)

    def test_external_kv_reset_revokes_all_and_clears_state(self):
        self.client.get_backend_mode.return_value = _DeploymentMode.Distributed
        with patch.dict(
            os.environ,
            {
                "UMBP_EXTKV_REPORT": "1",
                "UMBP_EXTKV_FLUSH_MS": "3600000",
                "UMBP_EXTKV_RECONCILE_SECONDS": "3600",
            },
        ):
            connector = self.make_connector()
        connector._queue_extkv_pages(["fedcba9876543210fedcba9876543210"])
        connector._flush_extkv()
        self.assertEqual(len(connector._extkv_reported), 1)
        connector._queue_extkv_pages(["00112233445566778899aabbccddeeff"])
        self.assertEqual(len(connector._extkv_pending), 1)

        connector.reset()

        # Publishing a pending tail immediately before revoke-all only wastes a
        # report RPC. reset clears both reported and pending state directly.
        self.client.report_external_kv_blocks.assert_called_once()
        self.client.revoke_all_external_kv_blocks_at_tier.assert_called_once()
        self.assertFalse(connector._extkv_pending)
        self.assertFalse(connector._extkv_reported)
        self.assertFalse(connector._extkv_required_keys)

    def test_external_kv_repeated_report_revoke_keeps_live_state_bounded(self):
        self.client.get_backend_mode.return_value = _DeploymentMode.Distributed
        live_hashes = set()

        def report(hashes, _tier):
            live_hashes.update(hashes)
            return True

        def revoke(hashes, _tier):
            live_hashes.difference_update(hashes)
            return True

        self.client.report_external_kv_blocks.side_effect = report
        self.client.revoke_external_kv_blocks.side_effect = revoke
        self.client.batch_exists.side_effect = lambda keys: [False] * len(keys)
        with patch.dict(
            os.environ,
            {
                "UMBP_EXTKV_REPORT": "1",
                "UMBP_EXTKV_FLUSH_MS": "3600000",
                "UMBP_EXTKV_RECONCILE_SECONDS": "3600",
            },
        ):
            connector = self.make_connector()

        for index in range(5):
            page_hash = f"{index:032x}"
            connector._queue_extkv_pages([page_hash])
            connector._flush_extkv()
            self.assertEqual(len(live_hashes), 1)

            connector._reconcile_extkv()

            self.assertFalse(live_hashes)
            self.assertFalse(connector._extkv_reported)
            self.assertFalse(connector._extkv_required_keys)

    def test_external_kv_is_disabled_by_default(self):
        self.client.get_backend_mode.return_value = _DeploymentMode.Distributed
        connector = self.make_connector()

        connector._queue_extkv_pages(["page-extkv-disabled"])

        self.client.report_external_kv_blocks.assert_not_called()
        self.assertIsNone(connector._extkv_thread)

    def test_standalone_close_deregisters_before_storage_close(self):
        self.client.is_distributed.return_value = False
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess
        events = []
        self.client.deregister_memory.side_effect = lambda ptr: events.append(
            ("deregister", ptr)
        )
        self.storage.close.side_effect = lambda: events.append(("storage_close", None))
        connector = self.make_connector(
            {"standalone_address": "unix:///tmp/umbp-test.sock"}
        )

        connector.close()

        self.assertGreater(len(connector._registered), 0)
        self.assertEqual(events[0][0], "deregister")
        self.assertEqual(events[1][0], "storage_close")
        self.client.deregister_memory.assert_called_once_with(
            connector._registered[0][0]
        )

    def test_standalone_close_can_retry_deregistration(self):
        self.client.is_distributed.return_value = False
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess
        self.client.deregister_memory.side_effect = [RuntimeError("rpc failed"), None]
        connector = self.make_connector(
            {"standalone_address": "unix:///tmp/umbp-test.sock"}
        )

        with self.assertRaisesRegex(RuntimeError, "rpc failed"):
            connector.close()
        self.assertFalse(connector._closed)
        self.assertFalse(connector._offload_thread.is_alive())
        self.assertFalse(connector._load_thread.is_alive())

        connector.close()
        self.assertTrue(connector._closed)
        self.assertEqual(self.client.deregister_memory.call_count, 2)

    def test_does_not_retain_second_client_reference(self):
        connector = self.make_connector()
        self.assertNotIn("client", vars(connector))

    def test_close_is_idempotent_and_does_not_poison_queue(self):
        connector = self.make_connector()

        connector.close()
        connector.close()
        connector.reset()

        self.storage.close.assert_called_once()

    def test_store_component_keys_match_connector_pool_names(self):
        from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

        store = UMBPStore.__new__(UMBPStore)
        store.registered_pools = self.pools
        store.is_mla_backend = True
        store.mla_suffix = "tp0_cp0_pp0"
        store.config_prefix = None
        transfer = PoolTransfer(name=PoolName.INDEXER)

        keys, multiplier = store._get_hybrid_page_component_keys(
            ["page-0", "page-1"], transfer
        )

        self.assertEqual(multiplier, 1)
        self.assertEqual(
            keys,
            [
                "page-0_tp0_cp0_pp0_indexer",
                "page-1_tp0_cp0_pp0_indexer",
            ],
        )

        store.config_prefix = "model-a"
        tagged_keys, _ = store._get_hybrid_page_component_keys(["page-0"], transfer)
        self.assertEqual(tagged_keys, ["model-a_page-0_tp0_cp0_pp0_indexer"])

    def test_extra_backend_tag_is_a_known_store_option(self):
        from sglang.srt.mem_cache.storage.umbp.umbp_store import _COMMON_EXTRA_KEYS

        self.assertIn("extra_backend_tag", _COMMON_EXTRA_KEYS)

    def test_linker_backend_dispatches_to_mori(self):
        cache = SimpleNamespace(
            components={ComponentType.FULL: object()},
            tree_core=SimpleNamespace(enable_external_cache_linker=False),
            write_through_threshold=0,
        )
        args = SimpleNamespace(unified_cache_external_linker_backend="mori")
        params = MagicMock()
        expected = object()
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.UMBPDirectLinker",
            return_value=expected,
        ) as linker_cls:
            wrapper = UnifiedCacheLinkerWrapper(cache, args, params)

        linker_cls.assert_called_once_with(
            args, params, components={ComponentType.FULL}
        )
        self.assertIs(wrapper.cache_linker, expected)
        self.assertTrue(cache.tree_core.enable_external_cache_linker)
        self.assertEqual(cache.write_through_threshold, 1)


if __name__ == "__main__":
    unittest.main()
