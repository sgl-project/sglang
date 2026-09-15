"""Unit tests for external-linker device pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    _build_deepseek_v4_device_pool_group,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDevicePoolEntry(CustomTestCase):
    def test_sparse_multi_component_layer_ranges(self):
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
        self.assertEqual(locations, [0, 4])
        pointers, sizes = pool.get_page_buffer_meta(indices)
        self.assertEqual(
            pointers,
            [
                buffer[row].data_ptr()
                for row in locations
                for buffer in (k0, k2, v0, v2)
            ],
        )
        self.assertEqual(sizes, [6, 10, 14, 22] * 2)
        self.assertIsNone(pool.get_prepared_layer_range_meta(locations, 1))

        pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
        self.assertEqual(
            pointers,
            [
                [k2[0].data_ptr()],
                [v2[0].data_ptr()],
                [k2[4].data_ptr()],
                [v2[4].data_ptr()],
            ],
        )
        self.assertEqual(sizes, [[10], [22], [10], [22]])
        self.assertEqual(offsets, [[6], [14], [6], [14]])

    def test_rejects_invalid_pages_and_empty_buffers(self):
        with self.assertRaisesRegex(ValueError, "has no storage buffers"):
            DevicePoolEntry(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[],
                layer_mapping={},
                page_size=2,
                rows_are_pages=False,
            )

        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[torch.zeros((8, 3), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=2,
            rows_are_pages=False,
        )
        for indices, error in (
            (torch.tensor([0]), "multiple of page_size"),
            (torch.tensor([1, 2]), "aligned contiguous pages"),
            (torch.tensor([0, 2]), "aligned contiguous pages"),
            (torch.tensor([8, 9]), "exceeds buffer shapes"),
        ):
            with self.subTest(indices=indices.tolist()):
                with self.assertRaisesRegex(ValueError, error):
                    pool.prepare_locations(indices)


class TestDevicePoolGroup(CustomTestCase):
    def test_resolve_transfers_expands_physical_pools(self):
        entries = [
            SimpleNamespace(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices,
            ),
            SimpleNamespace(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices + 100,
            ),
        ]
        group = DevicePoolGroup(entries, num_layers=2, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["a", "b"],
            device_indices=torch.tensor([0, 1, 4, 5]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        resolved = group.resolve_transfers([transfer])

        self.assertEqual(
            [item.name for item in resolved], [PoolName.KV, PoolName.INDEXER]
        )
        self.assertEqual(resolved[0].host_indices.tolist(), [0, 1, 4, 5])
        self.assertEqual(resolved[1].host_indices.tolist(), [100, 101, 104, 105])
        self.assertTrue(
            all(item.hit_policy == PoolHitPolicy.ALL_PAGES for item in resolved)
        )

    def test_partial_side_pool_requires_explicit_opt_in(self):
        entry = SimpleNamespace(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            translate_indices=lambda indices: indices + 100,
        )
        group = DevicePoolGroup([entry], num_layers=1, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=["b", "d"],
            device_indices=torch.tensor([20, 21, 24, 25]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        self.assertEqual(group.resolve_transfers([transfer]), [])
        resolved = group.resolve_transfers(
            [transfer], allow_partial=True, allow_missing_kv=True
        )

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].name, PoolName.SWA)
        self.assertEqual(resolved[0].keys, ["b", "d"])
        self.assertEqual(resolved[0].host_indices.tolist(), [120, 121, 124, 125])
        self.assertEqual(resolved[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)


class TestHybridDevicePoolAssembler(CustomTestCase):
    def test_deepseek_v4_maps_sparse_sidecars(self):
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
        kvcache.start_layer = 1
        kvcache.end_layer = 4
        kvcache.swa_page_size = 2
        kvcache.swa_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)]
        )
        kvcache.c4_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)],
            bytes_per_page_padded=5,
        )
        kvcache.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[
                torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
            ]
        )
        kvcache.c128_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)],
            bytes_per_page_padded=11,
        )
        kvcache.layer_mapping = [
            DeepSeekV4LayerItem(0, -1),
            DeepSeekV4LayerItem(4, 0),
            DeepSeekV4LayerItem(128, 0),
            DeepSeekV4LayerItem(4, 1),
        ]
        kvcache.compress_state_pools = [None, state_pool(), None, state_pool()]
        kvcache.indexer_compress_state_pools = [
            None,
            state_pool(),
            None,
            state_pool(),
        ]
        draft_swa_buffers = [
            torch.zeros((8, 13), dtype=torch.uint8),
            torch.zeros((8, 17), dtype=torch.uint8),
        ]

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(
                mtp_draft_device_pools=(
                    SimpleNamespace(
                        swa_kv_pool=SimpleNamespace(kv_buffer=draft_swa_buffers)
                    ),
                )
            ),
            components={ComponentType.FULL, ComponentType.SWA},
        )

        self.assertEqual(group.num_layers, 3)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(
            set(group.entry_map),
            {
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            },
        )
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4], PoolName.KV)
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4_STATE], PoolName.SWA)

        c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
        pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
        self.assertEqual(len(pointers), 2)
        self.assertEqual(sizes, [5, 5])
        _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
        self.assertEqual(sizes, [[5]])
        self.assertEqual(offsets, [[5]])
        self.assertIsNone(c4_pool.get_prepared_layer_range_meta([0], 1))

        swa_pool = group.entry_map[PoolName.SWA]
        _, sizes, offsets = swa_pool.get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[3, 13]])
        self.assertEqual(offsets, [[0, 9]])
        _, sizes, offsets = swa_pool.get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[3, 17]])
        self.assertEqual(offsets, [[3, 22]])

    def test_unified_deepseek_v4_uses_only_compressed_pools(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

        for split_indexer in (False, True):
            with self.subTest(split_indexer=split_indexer):
                c4 = [torch.zeros((4, width), dtype=torch.uint8) for width in (5, 7)]
                c128 = [torch.zeros((4, 11), dtype=torch.uint8)]
                expected = {
                    PoolName.DEEPSEEK_V4_C4: c4,
                    PoolName.DEEPSEEK_V4_C128: c128,
                }
                if split_indexer:
                    payload = [
                        torch.zeros((4, 1, 4, 1, 16), dtype=torch.uint8)
                        for _ in range(2)
                    ]
                    scale = [
                        torch.zeros((4, 1, 4, 1), dtype=torch.uint8) for _ in range(2)
                    ]
                    indexer = SimpleNamespace(
                        index_k_with_scale_buffer=None,
                        index_k_payload_buffer=payload,
                        index_k_scale_buffer=scale,
                    )
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = [
                        b.flatten(1) for b in payload
                    ]
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE] = [
                        b.flatten(1) for b in scale
                    ]
                else:
                    buffers = [
                        torch.zeros((4, width), dtype=torch.uint8) for width in (13, 17)
                    ]
                    indexer = SimpleNamespace(index_k_with_scale_buffer=buffers)
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = buffers
                regions = {4: (c4, 7), 128: (c128, 11)}
                kvcache = SimpleNamespace(
                    _unified_kv=True,
                    start_layer=0,
                    end_layer=3,
                    layer_mapping=[
                        DeepSeekV4LayerItem(4, 1),
                        DeepSeekV4LayerItem(128, 0),
                        DeepSeekV4LayerItem(4, 0),
                    ],
                    # Unified KV has neither paged KV nor an index-addressed SWA pool.
                    swa_kv_pool=None,
                    c4_kv_pool=None,
                    c128_kv_pool=None,
                    swa_page_size=3,
                    c4_indexer_kv_pool=indexer,
                    unified_region_buffers=Mock(side_effect=regions.__getitem__),
                )
                group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

                self.assertEqual(set(group.entry_map), set(expected))
                self.assertEqual(set(group.sources.values()), {PoolName.KV})
                self.assertTrue(group.rank_replicated)
                for name, buffers in expected.items():
                    entry = group.entry_map[name]
                    actual = entry.components[0]
                    self.assertEqual(len(actual), len(buffers))
                    for got, want in zip(actual, buffers):
                        self.assertEqual(got.data_ptr(), want.data_ptr())
                        self.assertEqual(got.shape, want.shape)
                self.assertEqual(
                    kvcache.unified_region_buffers.call_args_list, [call(4), call(128)]
                )
                resolved = group.resolve_transfers(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=["page-0"],
                            device_indices=torch.tensor([0, 1]),
                        )
                    ]
                )
                self.assertEqual({t.name for t in resolved}, set(expected))

    def test_deepseek_v4_still_rejects_hisparse(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool

        kvcache = SimpleNamespace(
            c4_kv_pool=HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
        )
        with self.assertRaisesRegex(ValueError, "does not support HiSparse"):
            _build_deepseek_v4_device_pool_group(kvcache, 2)

    def test_dsa_uses_hybrid_assembler_strategy(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        def dsa_pool(kv_width, index_width):
            pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
            pool.page_size = 2
            pool.layer_num = 1
            pool.kv_buffer = [torch.zeros((8, kv_width), dtype=torch.uint8)]
            pool.index_key_cache = SimpleNamespace(
                buffer=[torch.zeros((4, index_width), dtype=torch.uint8)]
            )
            return pool

        kvcache = dsa_pool(3, 7)
        kvcache.layer_num = 2
        kvcache.kv_buffer.append(torch.zeros((8, 5), dtype=torch.uint8))
        kvcache.index_key_cache.buffer.append(torch.zeros((4, 11), dtype=torch.uint8))
        draft_pools = (dsa_pool(13, 17), dsa_pool(19, 23))

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(mtp_draft_device_pools=draft_pools),
            components={ComponentType.FULL},
        )

        self.assertEqual(group.num_layers, 2)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.INDEXER})
        self.assertEqual(
            group.sources,
            {
                PoolName.KV: PoolName.KV,
                PoolName.INDEXER: PoolName.KV,
            },
        )
        _, sizes, offsets = group.entry_map[PoolName.KV].get_prepared_layer_range_meta(
            [0], 0
        )
        self.assertEqual(sizes, [[6, 26]])
        self.assertEqual(offsets, [[0, 16]])
        _, sizes, offsets = group.entry_map[
            PoolName.INDEXER
        ].get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[7, 17]])
        self.assertEqual(offsets, [[0, 18]])
        _, sizes, offsets = group.entry_map[
            PoolName.INDEXER
        ].get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[11, 23]])
        self.assertEqual(offsets, [[7, 35]])

    def test_linker_requires_packed_draft(self):
        """Do not accept draft state that the linker would omit from storage."""
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        draft = SimpleNamespace(
            token_to_kv_pool=object(),
            model_config=SimpleNamespace(
                num_nextn_predict_layers=0,
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            ),
        )
        target = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.EAGLE)
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target),
            _draft_model_runners=lambda: (draft,),
        )
        for linker_enabled, nextn_layers in (
            (False, 0),
            (True, 0),
            (False, 1),
            (True, 1),
        ):
            draft.model_config.num_nextn_predict_layers = nextn_layers
            with (
                self.subTest(linker=linker_enabled, nextn=nextn_layers),
                patch.object(
                    spec,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=not linker_enabled,
                        enable_unified_cache_external_linker=linker_enabled,
                    ),
                ),
            ):
                if linker_enabled and not nextn_layers:
                    with self.assertRaisesRegex(
                        NotImplementedError, "only supports packed"
                    ):
                        spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(target.mtp_draft_device_pools, ())
                else:
                    plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(
                        plan.mode,
                        spec.HiCacheDraftMode.PACKED
                        if nextn_layers
                        else spec.HiCacheDraftMode.SIDECAR,
                    )
                    self.assertEqual(plan.device_pools, (draft.token_to_kv_pool,))
                    self.assertEqual(
                        target.mtp_draft_device_pools,
                        plan.device_pools if nextn_layers else (),
                    )

    def test_unsupported_strategy_fails_with_context(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        with self.assertRaisesRegex(
            ValueError,
            "does not support the direct external linker: _MambaStrategy",
        ):
            resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=SimpleNamespace(),
                components={ComponentType.FULL, ComponentType.MAMBA},
            )


class TestMooncakeLinkerPPLookup(CustomTestCase):
    def setUp(self):
        self.keys = [f"page{i}" for i in range(6)]
        self.existing = set()
        self.queried = []

    def make_linker(self, *, side_pool=None, pp_size=3):
        names = [PoolName.DEEPSEEK_V4_C4]
        if side_pool is not None:
            names.append(side_pool)
        entries = [
            SimpleNamespace(
                name=name,
                indices_from_pool=(
                    PoolName.SWA if name == PoolName.SWA else PoolName.KV
                ),
                components=[[], []],
            )
            for name in names
        ]
        group = DevicePoolGroup(entries, num_layers=1, page_size=1)
        storage = MooncakeStore.__new__(MooncakeStore)
        storage.mem_pool_host = group
        storage.registered_pools = group.entry_map
        storage.pp_rank, storage.pp_size = 0, pp_size
        storage.mla_suffix, storage.mha_suffix = "cp1_pp0", "tp2_cp1_pp0"
        storage.config_prefix = "model_pp0_tag"
        storage.is_mla_backend = True

        def exists(keys):
            self.queried.extend(keys)
            return [int(key in self.existing) for key in keys]

        storage._batch_exist = exists
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pool_group, linker.storage = group, storage
        linker.stats = {"lookup": 0}
        return linker

    def add_pages(self, pp_rank, pages, pool=PoolName.DEEPSEEK_V4_C4):
        self.existing.update(
            f"model_pp0_tag_{self.keys[page]}_cp1_pp{pp_rank}_{pool}" for page in pages
        )

    def test_pp0_uses_shortest_stage_prefix(self):
        for stage_lengths in ((6, 4, 2), (6, 4, 0)):
            with self.subTest(stage_lengths=stage_lengths):
                self.existing.clear()
                self.queried.clear()
                linker = self.make_linker()
                for pp_rank, pages in enumerate(stage_lengths):
                    self.add_pages(pp_rank, range(pages))
                result = linker.lookup(
                    "req", [PoolTransfer(PoolName.KV, keys=self.keys)]
                )
                self.assertEqual(result, list(range(1, min(stage_lengths) + 1)))
                self.assertEqual(len(self.queried), len(self.keys) * 3)
                self.assertEqual(linker.stats["lookup"], 1)
                # Query must not mutate suffixes used concurrently by load/offload.
                self.assertEqual(linker.storage.mla_suffix, "cp1_pp0")
                self.assertEqual(linker.storage.mha_suffix, "tp2_cp1_pp0")

    def test_pp_swa_requires_a_common_restorable_boundary(self):
        linker = self.make_linker(side_pool=PoolName.SWA)
        for pp_rank, swa_pages in enumerate((range(6), (0, 1, 4, 5), range(4))):
            self.add_pages(pp_rank, range(6))
            self.add_pages(pp_rank, swa_pages, PoolName.SWA)
        result = linker.lookup(
            "req",
            [
                PoolTransfer(PoolName.KV, keys=self.keys),
                PoolTransfer(
                    PoolName.SWA,
                    keys=self.keys[-2:],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
            ],
        )
        # Per-stage maxima are 6, 6, 4, but PP1 cannot restore at 4.
        self.assertEqual(result, [1, 2])

    def test_default_storage_query_and_single_pp_stay_local(self):
        self.add_pages(0, range(6))
        transfers = [PoolTransfer(PoolName.KV, keys=self.keys)]
        linker = self.make_linker()
        result = linker.storage.batch_exists_v2(
            self.keys, linker.pool_group.resolve_transfers(transfers)
        )
        self.assertEqual(result.restorable_prefix_pages, list(range(1, 7)))
        self.assertEqual(len(self.queried), 6)
        self.queried.clear()
        self.assertEqual(
            self.make_linker(pp_size=1).lookup("req", transfers), list(range(1, 7))
        )
        self.assertEqual(len(self.queried), 6)


if __name__ == "__main__":
    unittest.main()
