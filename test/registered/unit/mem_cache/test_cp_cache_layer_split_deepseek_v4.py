"""Unit tests for DeepSeek V4 CP Cache LayerSplit layouts and descriptors."""

from __future__ import annotations

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
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


@contextmanager
def _override_sharding_envs(
    *,
    disable_swa=False,
    disable_c4=False,
    disable_c128=False,
    disable_c4_indexer=False,
):
    with ExitStack() as stack:
        stack.enter_context(
            envs.SGLANG_CP_CACHE_LAYER_SPLIT_DSV4_DISABLE_SWA_SHARDING.override(
                disable_swa
            )
        )
        stack.enter_context(
            envs.SGLANG_CP_CACHE_LAYER_SPLIT_DSV4_DISABLE_C4_SHARDING.override(
                disable_c4
            )
        )
        stack.enter_context(
            envs.SGLANG_CP_CACHE_LAYER_SPLIT_DSV4_DISABLE_C128_SHARDING.override(
                disable_c128
            )
        )
        stack.enter_context(
            envs.SGLANG_CP_CACHE_LAYER_SPLIT_DSV4_DISABLE_C4_INDEXER_SHARDING.override(
                disable_c4_indexer
            )
        )
        yield


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
        with _override_sharding_envs():
            layout = build_cp_cache_layer_split_deepseek_v4_pool_layout(
                0, 4, 0, 60, ratios
            )

        self.assertEqual(layout.swa_layer_num, 15)
        self.assertEqual(layout.c4_layer_num, 2)
        self.assertEqual(layout.c128_layer_num, 1)
        self.assertEqual(layout.c4_indexer_layer_num, 2)
        self.assertEqual(layout.c4_state_layer_num, 2)
        self.assertEqual(layout.c128_state_layer_num, 1)
        self.assertEqual(layout.c4_indexer_state_layer_num, 2)

    def test_worst_case_layout_is_max_across_cp_ranks(self):
        ratios = [0, 4, 128, 4] * 15 + [4]
        with _override_sharding_envs():
            layout = build_cp_cache_layer_split_deepseek_v4_worst_case_pool_layout(
                4, 0, 61, ratios
            )
            rank_layouts = [
                build_cp_cache_layer_split_deepseek_v4_pool_layout(
                    rank, 4, 0, 61, ratios
                )
                for rank in range(4)
            ]

        self.assertEqual(
            layout.swa_layer_num, max(x.swa_layer_num for x in rank_layouts)
        )
        self.assertEqual(layout.c4_layer_num, max(x.c4_layer_num for x in rank_layouts))
        self.assertEqual(
            layout.c128_layer_num, max(x.c128_layer_num for x in rank_layouts)
        )

    def test_replicated_c4_family_uses_stage_count(self):
        ratios = [0, 4, 128, 4, 4, 128, 0, 4]
        with _override_sharding_envs(disable_c4=True):
            layout = build_cp_cache_layer_split_deepseek_v4_pool_layout(
                1, 2, 0, 8, ratios
            )

        self.assertEqual(layout.c4_layer_num, 4)
        self.assertEqual(layout.c4_state_layer_num, 4)
        self.assertEqual(layout.c128_layer_num, 1)


class TestCpCacheLayerSplitDeepSeekV4PoolInternals(CustomTestCase):
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
            build_map(fake, 4, True),
            {0: 0, 2: 1, 3: 0, 5: 1},
        )
        self.assertEqual(
            build_map(fake, 4, False),
            {0: 0, 2: 1, 3: 2, 5: 3},
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
    swa_owned_global_ids,
    swa_buffer_count,
    c4_owned,
    c128_owned,
    c4_indexer_owned,
    c4_attn_state_owned,
    c128_attn_state_owned,
    c4_indexer_state_owned,
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
        if ratio == 4 and gid in c4_owned:
            layer_mapping[gid] = SimpleNamespace(
                compress_ratio=4, compress_layer_id=c4_local
            )
            c4_local += 1
        elif ratio == 128 and gid in c128_owned:
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
        gid: i for i, gid in enumerate(sorted(swa_owned_global_ids))
    }
    fake.swa_kv_pool = SimpleNamespace(kv_buffer=[None] * swa_buffer_count)
    fake.layer_mapping = layer_mapping
    fake.compression_ratios = compression_ratios
    fake._stage_start = stage_start
    fake._stage_end = stage_end
    fake._owns_c4_kv_layer_id = lambda gid: gid in c4_owned
    fake._owns_c128_kv_layer_id = lambda gid: gid in c128_owned
    fake._owns_indexer_kv_layer_id = lambda gid: gid in c4_indexer_owned
    fake._owns_attention_state_layer_id = lambda gid: (
        gid in c4_attn_state_owned or gid in c128_attn_state_owned
    )
    fake._owns_indexer_state_layer_id = lambda gid: gid in c4_indexer_state_owned
    return fake


class TestCpCacheLayerSplitDeepSeekV4HiCacheMapping(CustomTestCase):
    def test_mapping_keys_match_owned_sets_and_values_are_contiguous(self):
        fake = _make_fake_hicache_pool(
            compression_ratios=[0, 4, 128, 4],
            swa_owned_global_ids={0, 1, 2, 3},
            swa_buffer_count=2,
            c4_owned={1, 3},
            c128_owned=set(),
            c4_indexer_owned={1, 3},
            c4_attn_state_owned={1, 3},
            c128_attn_state_owned=set(),
            c4_indexer_state_owned={1, 3},
        )
        mapping = fake.get_hicache_host_layer_mapping()

        self.assertEqual(
            mapping,
            {
                "swa": {0: 0, 1: 1},
                "c4_kv": {1: 0, 3: 1},
                "c128_kv": {},
                "c4_indexer": {1: 0, 3: 1},
                "c4_state": {1: 0, 3: 1},
                "c128_state": {},
                "c4_indexer_state": {1: 0, 3: 1},
            },
        )

    def test_c4_indexer_mapping_can_diverge_from_c4_kv(self):
        fake = _make_fake_hicache_pool(
            compression_ratios=[4, 4, 4, 4],
            swa_owned_global_ids=set(),
            swa_buffer_count=0,
            c4_owned={0, 2},
            c128_owned=set(),
            c4_indexer_owned={0, 1, 2, 3},
            c4_attn_state_owned={0, 2},
            c128_attn_state_owned=set(),
            c4_indexer_state_owned={0, 2},
        )
        mapping = fake.get_hicache_host_layer_mapping()

        self.assertEqual(mapping["c4_kv"], {0: 0, 2: 1})
        self.assertEqual(mapping["c4_indexer"], {0: 0, 1: 1, 2: 2, 3: 3})


def _make_fake_transfer_pool(
    *,
    cp_rank,
    shard_swa=True,
    shard_c4=True,
    shard_c128=True,
    shard_c4_indexer=True,
):
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
    fake._shard_swa = shard_swa
    fake._shard_c4 = shard_c4
    fake._shard_c128 = shard_c128
    fake._shard_c4_indexer = shard_c4_indexer
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

    def test_replicated_non_owner_layout_keeps_none_placeholders(self):
        fake = _make_fake_transfer_pool(cp_rank=1, shard_c4=False)

        self.assertEqual(
            fake.get_kv_transfer_layout(),
            [
                None,
                None,
                None,
                None,
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
                None,
                None,
                None,
                None,
                (DSV4_TRANSFER_INDEXER_STATE, 4),
                (DSV4_TRANSFER_INDEXER_STATE, 7),
            ],
        )

    def test_replicated_c128_state_has_single_sender(self):
        rank0 = _make_fake_transfer_pool(cp_rank=0, shard_c128=False)
        rank1 = _make_fake_transfer_pool(cp_rank=1, shard_c128=False)

        self.assertEqual(
            rank0.get_c128_state_transfer_layout(),
            [
                (DSV4_TRANSFER_C128_STATE, 3),
                (DSV4_TRANSFER_C128_STATE, 5),
            ],
        )
        self.assertEqual(rank1.get_c128_state_transfer_layout(), [None, None])


if __name__ == "__main__":
    unittest.main()
