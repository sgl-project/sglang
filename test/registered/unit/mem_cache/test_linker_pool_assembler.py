"""Unit tests for external-linker device pool assembly."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
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


def _dsv4_pool(pool_type, **fields):
    pool = pool_type.__new__(pool_type)
    pool.__dict__.update(fields)
    return pool


class TestHybridDevicePoolAssembler(CustomTestCase):
    def test_deepseek_v4_maps_sparse_sidecars(self):
        from sglang.srt.mem_cache.deepseek_v4_compress_state import (
            CompressStatePool,
            KVAndScore,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        def state_pool():
            return _dsv4_pool(
                CompressStatePool,
                request_scoped=False,
                ring_size=2,
                kv_score_buffer=KVAndScore(torch.zeros((9, 4))),
            )

        kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        kvcache._unified_kv = False
        kvcache.start_layer = 1
        kvcache.end_layer = 4
        kvcache._stage_start = 1
        kvcache._stage_end = 4
        kvcache.stage_layer_ids = (1, 2, 3)
        kvcache.swa_page_size = 2
        kvcache.swa_kv_pool = _dsv4_pool(
            DeepSeekV4SingleKVPool,
            page_size=2,
            kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)],
        )
        kvcache.c4_kv_pool = _dsv4_pool(
            DeepSeekV4SingleKVPool,
            page_size=1,
            kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)],
            bytes_per_page_padded=5,
        )
        kvcache.c4_indexer_kv_pool = _dsv4_pool(
            DeepSeekV4IndexerPool,
            index_page_size=1,
            index_k_with_scale_buffer=[
                torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
            ],
        )
        kvcache.c128_kv_pool = _dsv4_pool(
            DeepSeekV4SingleKVPool,
            page_size=1,
            kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)],
            bytes_per_page_padded=11,
        )
        kvcache.compress_state_pools = [None, state_pool(), None, state_pool()]
        kvcache.indexer_compress_state_pools = [
            None,
            state_pool(),
            None,
            state_pool(),
        ]
        kvcache.sources_by_ratio = {4: [1, 3], 128: [2]}
        kvcache.index_pools = {4: kvcache.c4_indexer_kv_pool}
        draft_swa_buffers = [
            torch.zeros((8, 13), dtype=torch.uint8),
            torch.zeros((8, 17), dtype=torch.uint8),
        ]

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(
                mtp_draft_device_pools=(
                    _dsv4_pool(
                        DeepSeekV4TokenToKVPool,
                        start_layer=0,
                        end_layer=2,
                        _stage_start=0,
                        _stage_end=2,
                        stage_layer_ids=(0, 1),
                        sources_by_ratio={},
                        swa_kv_pool=_dsv4_pool(
                            DeepSeekV4SingleKVPool,
                            page_size=2,
                            kv_buffer=draft_swa_buffers,
                        ),
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

        for name, pools in (
            (PoolName.DEEPSEEK_V4_C4_STATE, kvcache.compress_state_pools),
            (
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                kvcache.indexer_compress_state_pools,
            ),
        ):
            pointers, sizes = group.entry_map[name].get_page_buffer_meta(
                torch.tensor([2, 3])
            )
            self.assertEqual(sizes, [32, 32])
            self.assertEqual(
                pointers,
                [
                    pools[layer].kv_score_buffer.kv_score.data_ptr() + 32
                    for layer in (1, 3)
                ],
            )
            with self.assertRaisesRegex(ValueError, "exceeds buffer shapes"):
                group.entry_map[name].get_page_buffer_meta(torch.tensor([8, 9]))

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

    def test_deepseek_v41_shared_kv_and_split_indexer_pages(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def index_pool(layers, rows, width):
            # Native FP4 split layout: two slots per page, ``width`` payload bytes
            # and two scale bytes per page row.
            return _dsv4_pool(
                DeepSeekV4IndexerPool,
                index_page_size=2,
                index_k_with_scale_buffer=None,
                index_k_payload_buffer=[
                    torch.arange(rows * width, dtype=torch.uint8).reshape(
                        rows, 1, 1, 2, width // 2
                    )
                    for _ in range(layers)
                ],
                index_k_scale_buffer=[
                    torch.zeros((rows, 1, 1, 2), dtype=torch.uint8)
                    for _ in range(layers)
                ],
            )

        def kv_pool(page_size, buffers):
            return _dsv4_pool(
                DeepSeekV4SingleKVPool, page_size=page_size, kv_buffer=buffers
            )

        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.start_layer, pool.end_layer = 1, 7
        pool.stage_layer_ids = tuple(range(1, 7))
        pool.swa_page_size = 8
        pool.swa_kv_pool = kv_pool(
            8, [torch.zeros((5, 3), dtype=torch.uint8) for _ in range(6)]
        )
        pool.c4_kv_pool = pool.c128_kv_pool = pool.c4_indexer_kv_pool = None
        pool.sources_by_ratio = {1: [4], 2: [2, 6]}
        pool.kv_pools = {
            1: kv_pool(1, [torch.zeros((5, 7), dtype=torch.uint8)]),
            2: kv_pool(1, [torch.zeros((5, 11), dtype=torch.uint8) for _ in range(2)]),
        }
        pool.index_pools = {1: index_pool(1, 23, 4), 2: index_pool(2, 13, 6)}
        pool.layer_mapping = [
            DeepSeekV4LayerItem(ratio, layer)
            for ratio, layer in [
                (0, -1),
                (0, -1),
                (2, 0),
                (2, 0),
                (1, 0),
                (1, 0),
                (2, 1),
            ]
        ]
        pool.compress_state_pools = [None] * 7
        pool.indexer_compress_state_pools = [None] * 7
        draft = _dsv4_pool(
            DeepSeekV4TokenToKVPool,
            _unified_kv=False,
            stage_layer_ids=(0, 1),
            sources_by_ratio={},
            swa_kv_pool=kv_pool(
                8, [torch.zeros((5, 13), dtype=torch.uint8) for _ in range(2)]
            ),
        )
        group = resolve_hybrid_device_pool_group(
            kvcache=pool,
            page_size=8,
            params=SimpleNamespace(mtp_draft_device_pools=(draft,)),
            components={ComponentType.FULL, ComponentType.SWA},
        )
        self.assertEqual(
            set(group.entry_map),
            {
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C1,
                PoolName.DEEPSEEK_V4_C1_INDEXER,
                PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE,
                PoolName.DEEPSEEK_V4_C2,
                PoolName.DEEPSEEK_V4_C2_INDEXER,
                PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE,
            },
        )
        self.assertEqual(group.entry_map[PoolName.DEEPSEEK_V4_C1].layer_mapping, {3: 0})
        c2 = group.entry_map[PoolName.DEEPSEEK_V4_C2]
        self.assertEqual(c2.layer_mapping, {1: 0, 5: 1})
        self.assertIsNone(c2.get_prepared_layer_range_meta([1], 2))
        self.assertEqual(
            c2.get_prepared_layer_range_meta([1], 5),
            ([[pool.kv_pools[2].kv_buffer[1][1].data_ptr()]], [[11]], [[11]]),
        )
        full = PoolTransfer(
            name=PoolName.KV, keys=["page"], device_indices=torch.arange(8, 16)
        )
        swa = PoolTransfer(
            name=PoolName.SWA,
            keys=["tail"],
            device_indices=torch.arange(8, 16),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        transfers = {t.name: t for t in group.resolve_transfers([full, swa])}
        self.assertEqual(transfers[PoolName.DEEPSEEK_V4_C2].keys, ["page"])
        self.assertEqual(
            transfers[PoolName.DEEPSEEK_V4_C2].hit_policy, PoolHitPolicy.ALL_PAGES
        )
        self.assertEqual(transfers[PoolName.SWA].keys, ["tail"])
        for ratio, name, width, raw_row in [
            (1, PoolName.DEEPSEEK_V4_C1_INDEXER, 16, 4),
            (2, PoolName.DEEPSEEK_V4_C2_INDEXER, 12, 2),
        ]:
            entry = group.entry_map[name]
            pointers, sizes = entry.get_page_buffer_meta(torch.arange(8, 16))
            original = pool.index_pools[ratio].index_k_payload_buffer
            self.assertEqual(pointers, [b[raw_row].data_ptr() for b in original])
            self.assertEqual(sizes, [width] * len(original))
            self.assertTrue(
                torch.equal(
                    entry.components[0][0][1],
                    original[0][raw_row : raw_row * 2].flatten(),
                )
            )
        self.assertEqual(
            group.entry_map[PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE].get_page_buffer_meta(
                torch.arange(8, 16)
            )[1],
            [8],
        )
        self.assertEqual(
            group.entry_map[PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE].get_page_buffer_meta(
                torch.arange(8, 16)
            )[1],
            [4, 4],
        )
        _, sizes, offsets = group.entry_map[PoolName.SWA].get_prepared_layer_range_meta(
            [1], 0
        )
        self.assertEqual(sizes, [[3, 13]])
        self.assertEqual(offsets, [[0, 18]])

    def test_deepseek_v41_request_window_drops_swa(self):
        """Encoder replay leaves no target SWA pool: skip SWA and the draft SWA
        rather than dereference them, and refuse pools that take SWA indices."""
        from sglang.srt.mem_cache.deepseek_v4_compress_state import (
            CompressStatePool,
            KVAndScore,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def state_pool():
            return _dsv4_pool(
                CompressStatePool,
                request_scoped=False,
                ring_size=2,
                kv_score_buffer=KVAndScore(torch.zeros((4, 3))),
            )

        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.start_layer, pool.end_layer = 0, 2
        pool.stage_layer_ids = (0, 1)
        pool.swa_page_size = 8
        pool.swa_kv_pool = None
        pool.c4_kv_pool = pool.c128_kv_pool = pool.c4_indexer_kv_pool = None
        pool.sources_by_ratio = {1: [1]}
        pool.kv_pools = {
            1: _dsv4_pool(
                DeepSeekV4SingleKVPool,
                page_size=1,
                kv_buffer=[torch.zeros((3, 7), dtype=torch.uint8)],
            )
        }
        pool.index_pools = {
            1: _dsv4_pool(
                DeepSeekV4IndexerPool,
                index_page_size=2,
                index_k_with_scale_buffer=[torch.zeros((9, 4), dtype=torch.uint8)],
            )
        }
        pool.layer_mapping = [DeepSeekV4LayerItem(0, -1), DeepSeekV4LayerItem(1, 0)]
        pool.compress_state_pools = [None, None]
        pool.indexer_compress_state_pools = [None, None]
        draft = _dsv4_pool(
            DeepSeekV4TokenToKVPool,
            _unified_kv=False,
            stage_layer_ids=(0,),
            sources_by_ratio={},
            swa_kv_pool=_dsv4_pool(
                DeepSeekV4SingleKVPool,
                page_size=8,
                kv_buffer=[torch.zeros((3, 5), dtype=torch.uint8)],
            ),
        )
        params = SimpleNamespace(mtp_draft_device_pools=(draft,))
        group = resolve_hybrid_device_pool_group(
            kvcache=pool,
            page_size=8,
            params=params,
            components={ComponentType.FULL, ComponentType.SWA},
        )
        self.assertEqual(
            set(group.entry_map),
            {PoolName.DEEPSEEK_V4_C1, PoolName.DEEPSEEK_V4_C1_INDEXER},
        )
        indexer = group.entry_map[PoolName.DEEPSEEK_V4_C1_INDEXER]
        _, sizes = indexer.get_page_buffer_meta(torch.arange(8, 16))
        self.assertEqual(sizes, [16])

        pool.layer_mapping[0] = DeepSeekV4LayerItem(4, 0)
        pool.sources_by_ratio[4] = [0]
        pool.c4_kv_pool = _dsv4_pool(
            DeepSeekV4SingleKVPool,
            page_size=1,
            kv_buffer=[torch.zeros((3, 5), dtype=torch.uint8)],
        )
        pool.c4_indexer_kv_pool = _dsv4_pool(
            DeepSeekV4IndexerPool,
            index_page_size=1,
            index_k_with_scale_buffer=[torch.zeros((3, 6), dtype=torch.uint8)],
        )
        pool.index_pools[4] = pool.c4_indexer_kv_pool
        pool.compress_state_pools[0] = state_pool()
        pool.indexer_compress_state_pools[0] = state_pool()
        with self.assertRaisesRegex(ValueError, "SWA page indices"):
            resolve_hybrid_device_pool_group(
                kvcache=pool,
                page_size=8,
                params=params,
                components={ComponentType.FULL, ComponentType.SWA},
            )

    def test_unified_deepseek_v4_uses_only_compressed_pools(self):

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
                    indexer = _dsv4_pool(
                        DeepSeekV4IndexerPool,
                        index_page_size=1,
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
                    indexer = _dsv4_pool(
                        DeepSeekV4IndexerPool,
                        index_page_size=1,
                        index_k_with_scale_buffer=buffers,
                    )
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = buffers
                regions = {4: (c4, 7), 128: (c128, 11)}
                kvcache = _dsv4_pool(
                    DeepSeekV4TokenToKVPool,
                    page_size=256,
                    _unified_kv=True,
                    start_layer=0,
                    end_layer=3,
                    _stage_start=0,
                    _stage_end=3,
                    stage_layer_ids=(0, 1, 2),
                    sources_by_ratio={4: [0, 2], 128: [1]},
                    index_pools={4: indexer},
                    # Unified KV has neither paged KV nor an index-addressed SWA pool.
                    swa_kv_pool=None,
                    c4_kv_pool=None,
                    c128_kv_pool=None,
                    swa_page_size=3,
                    c4_indexer_kv_pool=indexer,
                    unified_region_buffers=Mock(side_effect=regions.__getitem__),
                    # bf16 unified rows are a single pool: no rope sibling.
                    unified_rope_region_buffers=Mock(return_value=None),
                )
                group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

                # expected carries no *_ROPE entry, so this also pins that the
                # bf16 layout ships exactly what it did before.
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

    def test_unified_deepseek_v4_fp8_ships_the_rope_half(self):
        """fp8 rows are two pools; shipping nope alone leaves stale rope."""

        c4 = [torch.zeros((4, width), dtype=torch.uint8) for width in (5, 7)]
        c128 = [torch.zeros((4, 11), dtype=torch.uint8)]
        c4_rope = [torch.zeros((4, 3), dtype=torch.uint8) for _ in range(2)]
        c128_rope = [torch.zeros((4, 3), dtype=torch.uint8)]
        indexer = [torch.zeros((4, width), dtype=torch.uint8) for width in (13, 17)]
        regions = {4: (c4, 7), 128: (c128, 11)}
        rope_regions = {4: (c4_rope, 3), 128: (c128_rope, 3)}
        kvcache = _dsv4_pool(
            DeepSeekV4TokenToKVPool,
            page_size=256,
            _unified_kv=True,
            start_layer=0,
            end_layer=3,
            _stage_start=0,
            _stage_end=3,
            stage_layer_ids=(0, 1, 2),
            sources_by_ratio={4: [0, 2], 128: [1]},
            swa_kv_pool=None,
            c4_kv_pool=None,
            c128_kv_pool=None,
            swa_page_size=3,
            c4_indexer_kv_pool=_dsv4_pool(
                DeepSeekV4IndexerPool,
                index_page_size=1,
                index_k_with_scale_buffer=indexer,
            ),
            unified_region_buffers=Mock(side_effect=regions.__getitem__),
            unified_rope_region_buffers=Mock(side_effect=rope_regions.__getitem__),
        )
        kvcache.index_pools = {4: kvcache.c4_indexer_kv_pool}
        group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

        expected_rope = {
            PoolName.DEEPSEEK_V4_C4_ROPE: c4_rope,
            PoolName.DEEPSEEK_V4_C128_ROPE: c128_rope,
        }
        self.assertTrue(set(expected_rope).issubset(group.entry_map))
        for name, buffers in expected_rope.items():
            actual = group.entry_map[name].components[0]
            self.assertEqual(len(actual), len(buffers))
            for got, want in zip(actual, buffers):
                self.assertEqual(got.data_ptr(), want.data_ptr())
                self.assertEqual(got.shape, want.shape)
        self.assertEqual(
            kvcache.unified_rope_region_buffers.call_args_list, [call(4), call(128)]
        )

        # One row index addresses both halves: same layer mapping, same source.
        for rope_name, nope_name in (
            (PoolName.DEEPSEEK_V4_C4_ROPE, PoolName.DEEPSEEK_V4_C4),
            (PoolName.DEEPSEEK_V4_C128_ROPE, PoolName.DEEPSEEK_V4_C128),
        ):
            self.assertEqual(
                group.entry_map[rope_name].layer_mapping,
                group.entry_map[nope_name].layer_mapping,
            )
            self.assertEqual(group.sources[rope_name], group.sources[nope_name])

        # One KV transfer must expand to both halves.
        resolved = group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=["page-0"],
                    device_indices=torch.tensor([0, 1]),
                )
            ]
        )
        self.assertTrue(set(expected_rope).issubset({t.name for t in resolved}))

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
            pool.start_layer = 0
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
        with self.assertRaisesRegex(ValueError, "draft KV"):
            resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=SimpleNamespace(
                    mtp_draft_device_pools=(), sidecar_draft_device_pools=(object(),)
                ),
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

    def test_draft_plan_packs_only_nextn_mtp(self):
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        no_state = SimpleNamespace(
            layer_router=SimpleNamespace(part_view=lambda part: None)
        )
        draft = SimpleNamespace(
            token_to_kv_pool=no_state,
            runner_ids=SimpleNamespace(is_fused_draft=False),
            model_config=SimpleNamespace(
                num_nextn_predict_layers=0,
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            ),
        )
        target = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            token_to_kv_pool=no_state,
        )
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

    def test_mla_mamba_maps_kv_pages_and_state_slots(self):
        from sglang.srt.mem_cache.hybrid_cache import linker_pool_assembler
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MHATokenToKVPool,
            MLATokenToKVPool,
        )

        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        kvcache.start_layer = 0
        kvcache.full_attention_layer_id_mapping = {1: 0, 3: 1}
        kvcache.full_kv_pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        kvcache.full_kv_pool.kv_buffer = [
            torch.zeros((8, 1, 5), dtype=torch.uint8) for _ in range(2)
        ]
        mamba_cache = SimpleNamespace(
            temporal=torch.zeros((2, 4, 3), dtype=torch.uint8),
            conv=[torch.zeros((2, 4, 2), dtype=torch.uint8)],
        )
        draft = MHATokenToKVPool.__new__(MHATokenToKVPool)
        draft.page_size, draft.layer_num = 2, 1
        draft.k_scale_buffer = draft.native_k_scale_buffer = None
        draft.k_buffer = [torch.zeros((8, 3), dtype=torch.uint8)]
        draft.v_buffer = [torch.zeros((8, 3), dtype=torch.uint8)]
        params = SimpleNamespace(
            mtp_draft_device_pools=(),
            sidecar_draft_device_pools=(draft,),
            token_to_kv_pool_allocator=SimpleNamespace(
                state=SimpleNamespace(
                    pool=SimpleNamespace(mamba_cache=mamba_cache, num_mamba_layers=2),
                    layer_map={0: 0, 2: 1},
                    ckpt_pool=None,
                    short_conv_pool=SimpleNamespace(enabled=False),
                    ngram_pool=SimpleNamespace(enabled=False),
                )
            ),
        )

        def resolve(dcp_size=1):
            with (
                patch.object(
                    linker_pool_assembler,
                    "get_memory",
                    return_value=SimpleNamespace(enable_unified_memory=False),
                ),
                patch.object(
                    linker_pool_assembler,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_dcp_size=dcp_size),
                ),
            ):
                return resolve_hybrid_device_pool_group(
                    kvcache=kvcache,
                    page_size=2,
                    params=params,
                    components={ComponentType.FULL, ComponentType.MAMBA},
                )

        group = resolve()
        self.assertEqual(group.num_layers, 4)
        self.assertFalse(group.rank_replicated)
        kv, mamba = group.entry_map[PoolName.KV], group.entry_map[PoolName.MAMBA]
        self.assertEqual(kv.layer_mapping, {1: 0, 3: 1})
        self.assertEqual(mamba.layer_mapping, {0: 0, 2: 1})
        self.assertEqual(kv.prepare_locations(torch.tensor([2, 3])), [2])
        self.assertEqual(mamba.prepare_locations(torch.tensor([3])), [3])
        self.assertEqual(group.sources[PoolName.DRAFT], PoolName.KV)
        self.assertEqual(group.entry_map[PoolName.DRAFT].layer_mapping, {0: (0, 1)})

        with self.assertRaisesRegex(ValueError, "plain MLA"):
            resolve(dcp_size=2)
        params.mtp_draft_device_pools = (object(),)
        with self.assertRaisesRegex(ValueError, "plain MLA"):
            resolve()
        params.mtp_draft_device_pools = ()
        kvcache.full_kv_pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        with self.assertRaisesRegex(ValueError, "plain MLA"):
            resolve()

    def test_minimax_m31_nvfp4_maps_page_rows_and_draft(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import (
            MiniMaxM3Nvfp4IndexKPool,
            MiniMaxM3Nvfp4KVPool,
            MiniMaxM3Nvfp4MainKVPool,
        )

        def page_rows(pool_cls, row_bytes):
            pool = pool_cls.__new__(pool_cls)
            pool.page_size, pool.layer_num = 2, 2
            pool.rows = [torch.zeros((4, row_bytes), dtype=torch.uint8)] * 2
            return pool

        kvcache = MiniMaxM3Nvfp4KVPool.__new__(MiniMaxM3Nvfp4KVPool)
        kvcache.start_layer, kvcache.layer_num = 4, 2
        kvcache.main_pool = page_rows(MiniMaxM3Nvfp4MainKVPool, 5)
        kvcache.index_k_pool = page_rows(MiniMaxM3Nvfp4IndexKPool, 3)
        kvcache.index_k_layer_id_mapping = {4: 0, 5: 1}
        draft = MHATokenToKVPool.__new__(MHATokenToKVPool)
        draft.page_size, draft.layer_num = 2, 1
        draft.k_scale_buffer = draft.native_k_scale_buffer = None
        draft.k_buffer = [torch.zeros((8, 7), dtype=torch.uint8)]
        draft.v_buffer = [torch.zeros((8, 7), dtype=torch.uint8)]
        params = SimpleNamespace(
            mtp_draft_device_pools=(), sidecar_draft_device_pools=(draft,)
        )

        def resolve():
            return resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=params,
                components={ComponentType.FULL},
            )

        group = resolve()
        self.assertEqual(group.num_layers, 2)
        self.assertFalse(group.rank_replicated)
        self.assertEqual(
            group.sources,
            {
                PoolName.KV: PoolName.KV,
                PoolName.INDEXER: PoolName.KV,
                PoolName.DRAFT: PoolName.KV,
            },
        )
        indices = torch.tensor([2, 3, 6, 7])
        for name, row_bytes in ((PoolName.KV, 5), (PoolName.INDEXER, 3)):
            entry = group.entry_map[name]
            self.assertEqual(entry.layer_mapping, {0: 0, 1: 1})
            locations = entry.prepare_locations(indices)
            self.assertEqual(locations, [1, 3])
            _, sizes, offsets = entry.get_prepared_layer_range_meta(locations, 1)
            self.assertEqual(sizes, [[row_bytes]] * 2)
            self.assertEqual(offsets, [[row_bytes]] * 2)
        draft_entry = group.entry_map[PoolName.DRAFT]
        self.assertEqual(draft_entry.layer_mapping, {0: (0, 1)})
        self.assertEqual(draft_entry.prepare_locations(indices), [2, 6])

        params.mtp_draft_device_pools = (object(),)
        with self.assertRaisesRegex(ValueError, "M3.1 NVFP4"):
            resolve()
        params.mtp_draft_device_pools = ()
        kvcache.main_pool = page_rows(MHATokenToKVPool, 5)
        with self.assertRaisesRegex(ValueError, "M3.1 NVFP4"):
            resolve()

    @staticmethod
    def _standard_minimax_pool(start_layer=4, layer_num=4, index_layers=(5, 7)):
        from sglang.srt.mem_cache.memory_pool import (
            MHATokenToKOnlyPool,
            MHATokenToKVPool,
            MiniMaxSparseKVPool,
        )

        def buffers(count, width, dtype, base):
            return [
                (torch.arange(10 * width).reshape(10, 1, width) + base + layer * 100)
                .to(dtype)
                .contiguous()
                for layer in range(count)
            ]

        main = MHATokenToKVPool.__new__(MHATokenToKVPool)
        main.size, main.page_size = 8, 2
        main.start_layer, main.layer_num = start_layer, layer_num
        main.end_layer = start_layer + layer_num
        main.k_scale_buffer = main.native_k_scale_buffer = None
        main.v_scale_buffer = main.native_v_scale_buffer = None
        main.k_buffer = buffers(layer_num, 3, torch.bfloat16, 1)
        main.v_buffer = buffers(layer_num, 3, torch.bfloat16, 21)
        index = MHATokenToKOnlyPool.__new__(MHATokenToKOnlyPool)
        index.size, index.page_size, index.layer_num = 8, 2, len(index_layers)
        index.model_layer_ids = tuple(index_layers)
        index.k_buffer = buffers(len(index_layers), 2, torch.float32, 41)
        pool = MiniMaxSparseKVPool.__new__(MiniMaxSparseKVPool)
        pool.size, pool.page_size = 8, 2
        pool.start_layer, pool.end_layer = start_layer, start_layer + layer_num
        pool.layer_num = layer_num
        pool._enable_hisparse = False
        pool.dense_pool = pool.index_kv_pool = None
        pool.main_pool, pool.index_k_pool = main, index
        pool.index_k_layer_id_mapping = {
            layer: local for local, layer in enumerate(index_layers)
        }
        return pool

    @staticmethod
    def _resolve_standard_minimax(pool, **overrides):
        params = SimpleNamespace(
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
            sidecar_draft_device_pools=(),
            pp_size=1,
            attn_cp_size=1,
        )
        params.__dict__.update(overrides)
        return resolve_hybrid_device_pool_group(
            kvcache=pool, page_size=2, params=params, components={ComponentType.FULL}
        )

    def test_minimax_standard_all_layers_byte_round_trip(self):
        # Cover the exact M3 layer policy and a nonzero stage offset with holes.
        for start, layers, index_layers in (
            (0, 60, tuple(range(3, 60))),
            (4, 4, (5, 7)),
        ):
            with self.subTest(start=start, layers=layers):
                source = self._standard_minimax_pool(start, layers, index_layers)
                target = self._standard_minimax_pool(start, layers, index_layers)
                src_group = self._resolve_standard_minimax(source)
                dst_group = self._resolve_standard_minimax(target)
                self.assertEqual(
                    set(src_group.entry_map), {PoolName.KV, PoolName.INDEXER}
                )
                self.assertFalse(src_group.rank_replicated)
                self.assertEqual(
                    src_group.entry_map[PoolName.INDEXER].layer_mapping,
                    {layer - start: local for local, layer in enumerate(index_layers)},
                )
                src_indices = torch.tensor([2, 3, 6, 7])
                dst_indices = torch.tensor([0, 1, 4, 5])
                transfers = src_group.resolve_transfers(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=["page0", "page1"],
                            device_indices=src_indices,
                        )
                    ]
                )
                self.assertEqual(
                    {t.name for t in transfers}, {PoolName.KV, PoolName.INDEXER}
                )
                for name, src_entry in src_group.entry_map.items():
                    dst_entry = dst_group.entry_map[name]
                    originals = [b.clone() for b in src_entry.kv_buffer]
                    for buffer in dst_entry.kv_buffer:
                        buffer.fill_(-128)
                    ptrs, sizes = src_entry.get_page_buffer_meta(src_indices)
                    buffers_per_page = len(src_entry.kv_buffer)
                    objects = [
                        b"".join(
                            ctypes.string_at(ptr, size)
                            for ptr, size in zip(
                                ptrs[i : i + buffers_per_page],
                                sizes[i : i + buffers_per_page],
                            )
                        )
                        for i in range(0, len(ptrs), buffers_per_page)
                    ]
                    expected_bytes = (
                        layers * 2 * 2 * 3 * 2
                        if name == PoolName.KV
                        else len(index_layers) * 2 * 2 * 4
                    )
                    self.assertEqual(
                        [len(obj) for obj in objects], [expected_bytes] * 2
                    )
                    src_locations = src_entry.prepare_locations(src_indices)
                    dst_locations = dst_entry.prepare_locations(dst_indices)
                    for layer in range(layers):
                        src_meta = src_entry.get_prepared_layer_range_meta(
                            src_locations, layer
                        )
                        dst_meta = dst_entry.get_prepared_layer_range_meta(
                            dst_locations, layer
                        )
                        self.assertEqual(src_meta is None, dst_meta is None)
                        if dst_meta is None:
                            self.assertNotIn(layer, src_entry.layer_mapping)
                            continue
                        dst_ptrs, dst_sizes, offsets = dst_meta
                        for obj, row_ptrs, row_sizes, row_offsets in zip(
                            objects, dst_ptrs, dst_sizes, offsets
                        ):
                            for ptr, size, offset in zip(
                                row_ptrs, row_sizes, row_offsets
                            ):
                                self.assertGreaterEqual(offset, 0)
                                self.assertLessEqual(offset + size, len(obj))
                                ctypes.memmove(ptr, obj[offset : offset + size], size)
                    for old, src_buffer, dst_buffer in zip(
                        originals, src_entry.kv_buffer, dst_entry.kv_buffer
                    ):
                        torch.testing.assert_close(src_buffer, old, rtol=0, atol=0)
                        torch.testing.assert_close(
                            dst_buffer[dst_indices], old[src_indices], rtol=0, atol=0
                        )
                        self.assertTrue(
                            torch.all(dst_buffer[[2, 3, 6, 7, 8, 9]] == -128)
                        )

    def test_minimax_standard_rejects_unsupported_before_buffer_access(self):
        for field, value in (
            ("_enable_hisparse", True),
            ("dense_pool", object()),
            ("index_kv_pool", object()),
        ):
            with self.subTest(field=field):
                pool = self._standard_minimax_pool()
                setattr(pool, field, value)
                pool.main_pool = pool.index_k_pool = object()
                with self.assertRaisesRegex(ValueError, "index-K-only"):
                    self._resolve_standard_minimax(pool)
        for options in (
            dict(mtp_draft_device_pools=(object(),)),
            dict(mtp_draft_mamba_pools=(object(),)),
            dict(sidecar_draft_device_pools=(object(),)),
            dict(pp_size=2),
            dict(attn_cp_size=2),
        ):
            with self.subTest(options=options):
                pool = self._standard_minimax_pool()
                pool.main_pool = pool.index_k_pool = object()
                with self.assertRaisesRegex(ValueError, "index-K-only"):
                    self._resolve_standard_minimax(pool, **options)

    def test_minimax_standard_rejects_geometry_and_mapping_before_pointers(self):
        changes = (
            lambda p: setattr(p.main_pool, "page_size", 4),
            lambda p: setattr(p.index_k_pool, "page_size", 4),
            lambda p: setattr(p.index_k_pool, "size", 16),
            lambda p: setattr(p.main_pool, "size", 16),
            lambda p: setattr(p.main_pool, "start_layer", 0),
            lambda p: setattr(p.main_pool, "end_layer", 9),
            lambda p: setattr(p.main_pool, "layer_num", 3),
            lambda p: setattr(p, "end_layer", 9),
            lambda p: setattr(p, "index_k_layer_id_mapping", {5: 0, 7: 0}),
            lambda p: setattr(p, "index_k_layer_id_mapping", {5: 0, 8: 1}),
            lambda p: setattr(p.index_k_pool, "model_layer_ids", (7, 5)),
        )
        for change in changes:
            pool = self._standard_minimax_pool()
            change(pool)
            with patch(
                "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.DevicePoolEntry",
                side_effect=AssertionError("pointer access"),
            ):
                with self.assertRaisesRegex(ValueError, "geometry|mapping"):
                    self._resolve_standard_minimax(pool)

    def test_minimax_standard_rejects_unknown_pool_and_missing_layer_buffers(self):
        for field in ("main_pool", "index_k_pool"):
            pool = self._standard_minimax_pool()
            setattr(pool, field, object())
            with self.assertRaisesRegex(ValueError, "pool types"):
                self._resolve_standard_minimax(pool)
        for name in ("k_buffer", "v_buffer"):
            pool = self._standard_minimax_pool()
            setattr(pool.main_pool, name, [])
            with patch(
                "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.DevicePoolEntry",
                side_effect=AssertionError("pointer access"),
            ):
                with self.assertRaisesRegex(ValueError, "buffer layer counts"):
                    self._resolve_standard_minimax(pool)
        pool = self._standard_minimax_pool()
        pool.index_k_pool.k_buffer = []
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.DevicePoolEntry",
            side_effect=AssertionError("pointer access"),
        ):
            with self.assertRaisesRegex(ValueError, "buffer layer counts"):
                self._resolve_standard_minimax(pool)

    def test_minimax_standard_requires_index_pool_for_index_mapping(self):
        pool = self._standard_minimax_pool()
        pool.index_k_pool = None
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.DevicePoolEntry",
            side_effect=AssertionError("pointer access"),
        ):
            with self.assertRaisesRegex(ValueError, "requires an index pool"):
                self._resolve_standard_minimax(pool)
        pool.index_k_layer_id_mapping = {}
        group = self._resolve_standard_minimax(pool)
        self.assertEqual(set(group.entry_map), {PoolName.KV})
        self.assertEqual(group.num_layers, 4)

    def test_minimax_standard_never_drops_scale_buffers(self):
        for name in (
            "k_scale_buffer",
            "v_scale_buffer",
            "native_k_scale_buffer",
            "native_v_scale_buffer",
        ):
            with self.subTest(name=name):
                pool = self._standard_minimax_pool()
                setattr(pool.main_pool, name, [torch.ones(1)])
                with patch(
                    "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.DevicePoolEntry",
                    side_effect=AssertionError("pointer access"),
                ):
                    with self.assertRaisesRegex(ValueError, "scaled KV"):
                        self._resolve_standard_minimax(pool)

    def test_unsupported_strategy_fails_with_context(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = SWAKVPool.__new__(SWAKVPool)
        kvcache.full_kv_pool, kvcache.swa_kv_pool = object(), object()
        kvcache.layers_mapping = {}
        with self.assertRaisesRegex(
            ValueError,
            "does not support the direct external linker: _SwaStrategy",
        ):
            resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=SimpleNamespace(),
                components={ComponentType.FULL, ComponentType.SWA},
            )


if __name__ == "__main__":
    unittest.main()
