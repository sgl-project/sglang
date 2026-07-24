"""Unit tests for DeepSeek V4 CP Cache LayerSplit layouts and descriptors."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.cp_cache_layer_split import (
    build_cp_cache_layer_split_deepseek_v4_pool_layout,
    build_cp_cache_layer_split_deepseek_v4_worst_case_pool_layout,
    staging,
)
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_TRANSFER_ATTENTION_STATE,
    DSV4_TRANSFER_C4_INDEXER_KV,
    DSV4_TRANSFER_C4_KV,
    DSV4_TRANSFER_C128_KV,
    DSV4_TRANSFER_C128_STATE,
    DSV4_TRANSFER_INDEXER_STATE,
    DSV4_TRANSFER_SWA_KV,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCpCacheLayerSplitDeepSeekV4Layout(CustomTestCase):
    def test_pool_num_pages_includes_dummy_page(self):
        self.assertEqual(
            CpCacheLayerSplitDeepSeekV4TokenToKVPool._pool_num_pages(
                SimpleNamespace(size=256, page_size=128)
            ),
            3,
        )
        self.assertEqual(
            CpCacheLayerSplitDeepSeekV4TokenToKVPool._pool_num_pages(
                SimpleNamespace(size=257, page_size=128)
            ),
            3,
        )

    def test_layout_counts_owned_layers_per_family(self):
        ratios = [0, 4, 4, 128] + [0] * 56
        layout = build_cp_cache_layer_split_deepseek_v4_pool_layout(0, 4, 0, 60, ratios)

        self.assertEqual(layout.swa_layer_num, 15)
        self.assertEqual(layout.c4_layer_num, 2)
        self.assertEqual(layout.c128_layer_num, 1)
        self.assertEqual(layout.c4_indexer_layer_num, 2)
        self.assertEqual(layout.c4_state_layer_num, 2)
        self.assertEqual(layout.c128_state_layer_num, 1)
        self.assertEqual(layout.c4_indexer_state_layer_num, 2)

    def test_worst_case_layout_is_max_across_cp_ranks(self):
        ratios = [0, 4, 128, 4] * 15 + [4]
        layout = build_cp_cache_layer_split_deepseek_v4_worst_case_pool_layout(
            4, 0, 61, ratios
        )
        rank_layouts = [
            build_cp_cache_layer_split_deepseek_v4_pool_layout(rank, 4, 0, 61, ratios)
            for rank in range(4)
        ]

        self.assertEqual(
            layout.swa_layer_num, max(x.swa_layer_num for x in rank_layouts)
        )
        self.assertEqual(layout.c4_layer_num, max(x.c4_layer_num for x in rank_layouts))
        self.assertEqual(
            layout.c128_layer_num, max(x.c128_layer_num for x in rank_layouts)
        )


class TestCpCacheLayerSplitDeepSeekV4PoolInternals(CustomTestCase):
    def test_non_owned_read_requires_prefetch_only_during_cp_prefill(self):
        fake = object.__new__(CpCacheLayerSplitDeepSeekV4TokenToKVPool)
        fake._staging = staging.StagingBufferManager()
        expected_staging = fake._staging.allocate("swa", 1, lambda _: torch.empty(1))
        fake._swa_remapped_layer_id = None
        fake._owns_swa_layer_id = lambda _: False
        fake.cp_rank = 1

        fake._require_prefetched_reads = True
        with self.assertRaisesRegex(RuntimeError, "without prefetch"):
            fake.get_swa_key_buffer_radix(layer_id=3)

        fake._require_prefetched_reads = False
        self.assertIs(fake.get_swa_key_buffer_radix(layer_id=3), expected_staging)

    def test_state_pools_use_factories_only_for_owned_layers(self):
        fake = SimpleNamespace(
            compression_ratios=[0, 4, 128, 4],
            _stage_start=0,
            _stage_end=4,
            _owns_attention_state_layer_id=lambda layer_id: layer_id in (1, 2),
            _owns_indexer_state_layer_id=lambda layer_id: layer_id == 3,
            _make_attn_state_pool=lambda ratio, enabled: ("attention", ratio, enabled),
            _make_indexer_state_pool=lambda ratio, enabled: (
                "indexer",
                ratio,
                enabled,
            ),
        )

        CpCacheLayerSplitDeepSeekV4TokenToKVPool._init_paged_compress_states(fake, True)

        self.assertEqual(
            fake.compress_state_pools,
            [None, ("attention", 4, True), ("attention", 128, True), None],
        )
        self.assertEqual(
            fake.indexer_compress_state_pools,
            [None, None, None, ("indexer", 4, True)],
        )

    def test_owner_local_layer_map_tracks_each_owner_independently(self):
        fake = SimpleNamespace(
            cp_size=2,
            _stage_start=0,
            _stage_end=6,
            compression_ratios=[4, 128, 4, 4, 128, 4],
            _get_layer_owner_rank=lambda layer_id: 0 if layer_id < 3 else 1,
        )
        build_map = (
            CpCacheLayerSplitDeepSeekV4TokenToKVPool._build_owner_local_layer_map
        )

        self.assertEqual(
            build_map(fake, 4),
            {0: 0, 2: 1, 3: 0, 5: 1},
        )


class _FakeAllReduceComm:
    def __init__(self, remote_pages):
        self.remote_pages = remote_pages

    @contextmanager
    def change_state(self, enable):
        yield

    def all_reduce(self, mask):
        mask[self.remote_pages] = 1


class TestCpCacheLayerSplitStaging(CustomTestCase):
    def test_active_pages_include_remote_cp_ranks(self):
        indices = torch.tensor([0, 7, -1], dtype=torch.int32)
        eager_build_mask = torch.compiler.disable(staging.build_active_pages_mask)

        with patch.object(staging, "build_active_pages_mask", eager_build_mask):
            selected_pages = staging.active_pages_for_indices(
                indices,
                page_size=4,
                max_pages=4,
                pynccl_comm=_FakeAllReduceComm(remote_pages=[3]),
            )

        self.assertEqual(selected_pages.tolist(), [0, 1, 3])

    def test_indices_and_page_table_remap_preserve_padding(self):
        selected_pages = torch.tensor([1, 3], dtype=torch.int64)
        remap_indices = torch.compiler.disable(staging.remap_indices_to_staging)
        remap_page_table = torch.compiler.disable(staging.remap_page_table_to_staging)

        remapped_indices = remap_indices(
            torch.tensor([4, 5, 12, 15, -1], dtype=torch.int32),
            selected_pages,
            page_size=4,
            max_pages=4,
        )
        remapped_page_table = remap_page_table(
            torch.tensor([3, 1, -1], dtype=torch.int32),
            selected_pages,
            max_pages=4,
        )

        self.assertEqual(remapped_indices.tolist(), [0, 1, 4, 7, -1])
        self.assertEqual(remapped_page_table.tolist(), [1, 0, -1])


def _make_fake_hicache_pool(
    *,
    compression_ratios,
    owned_global_ids,
    stage_start=0,
    stage_end=None,
):
    if stage_end is None:
        stage_end = len(compression_ratios)

    layer_mapping = [None] * len(compression_ratios)
    c4_local = 0
    c128_local = 0
    for gid in range(stage_start, stage_end):
        ratio = compression_ratios[gid]
        if ratio == 4 and gid in owned_global_ids:
            layer_mapping[gid] = SimpleNamespace(
                compress_ratio=4, compress_layer_id=c4_local
            )
            c4_local += 1
        elif ratio == 128 and gid in owned_global_ids:
            layer_mapping[gid] = SimpleNamespace(
                compress_ratio=128, compress_layer_id=c128_local
            )
            c128_local += 1
        else:
            layer_mapping[gid] = SimpleNamespace(
                compress_ratio=ratio, compress_layer_id=None
            )

    fake = object.__new__(CpCacheLayerSplitDeepSeekV4TokenToKVPool)
    fake._swa_global_to_local = {
        gid: i for i, gid in enumerate(sorted(owned_global_ids))
    }
    fake.swa_kv_pool = SimpleNamespace(kv_buffer=[None] * len(owned_global_ids))
    fake.layer_mapping = layer_mapping
    fake.compression_ratios = compression_ratios
    fake._stage_start = stage_start
    fake._stage_end = stage_end
    fake._is_layer_owned = lambda gid: gid in owned_global_ids
    return fake


class TestCpCacheLayerSplitDeepSeekV4HiCacheMapping(CustomTestCase):
    def test_mapping_keys_match_owned_sets_and_values_are_contiguous(self):
        fake = _make_fake_hicache_pool(
            compression_ratios=[0, 4, 128, 4],
            owned_global_ids={2, 3},
        )
        mapping = fake.get_hicache_host_layer_mapping()

        self.assertEqual(
            mapping,
            {
                "swa": {2: 0, 3: 1},
                "c4_kv": {3: 0},
                "c128_kv": {2: 0},
                "c4_indexer": {3: 0},
                "c4_state": {3: 0},
                "c128_state": {2: 0},
                "c4_indexer_state": {3: 0},
            },
        )


def _make_fake_transfer_pool(*, cp_rank):
    cp_size = 2
    ratios = [0, 4, 4, 128, 4, 128, 0, 4]
    owned_swa_layers = range(0, 4) if cp_rank == 0 else range(4, 8)
    fake = object.__new__(CpCacheLayerSplitDeepSeekV4TokenToKVPool)
    fake.cp_rank = cp_rank
    fake.cp_size = cp_size
    fake._layer_shard_start_layer = 0
    fake._layer_shard_layer_num = len(ratios)
    fake._stage_start = 0
    fake._stage_end = len(ratios)
    fake.compression_ratios = ratios
    fake._swa_global_to_local = {
        layer_id: local_id for local_id, layer_id in enumerate(owned_swa_layers)
    }
    fake.swa_kv_pool = SimpleNamespace(kv_buffer=[None] * 4)
    return fake


class TestCpCacheLayerSplitDeepSeekV4TransferLayout(CustomTestCase):
    def test_sharded_layout_describes_only_owned_buffers(self):
        fake = _make_fake_transfer_pool(cp_rank=1)

        self.assertEqual(
            fake.get_kv_transfer_layout(),
            [
                (DSV4_TRANSFER_C4_KV, 4),
                (DSV4_TRANSFER_C4_KV, 7),
                (DSV4_TRANSFER_C4_INDEXER_KV, 4),
                (DSV4_TRANSFER_C4_INDEXER_KV, 7),
                (DSV4_TRANSFER_C128_KV, 5),
            ],
        )
        self.assertEqual(
            fake.get_state_transfer_layout(),
            [
                (DSV4_TRANSFER_SWA_KV, 4),
                (DSV4_TRANSFER_SWA_KV, 5),
                (DSV4_TRANSFER_SWA_KV, 6),
                (DSV4_TRANSFER_SWA_KV, 7),
                (DSV4_TRANSFER_ATTENTION_STATE, 4),
                (DSV4_TRANSFER_ATTENTION_STATE, 7),
                (DSV4_TRANSFER_INDEXER_STATE, 4),
                (DSV4_TRANSFER_INDEXER_STATE, 7),
            ],
        )
        self.assertEqual(
            fake.get_c128_state_transfer_layout(),
            [(DSV4_TRANSFER_C128_STATE, 5)],
        )


if __name__ == "__main__":
    unittest.main()
