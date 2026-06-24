"""Unit tests for DeepSeek V4 CP KV LayerSplit layouts and descriptors."""

from __future__ import annotations

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache.cp_kv_layer_split import (
    build_cp_kv_layer_split_deepseek_v4_pool_layout,
    build_cp_kv_layer_split_deepseek_v4_worst_case_pool_layout,
)
from sglang.srt.mem_cache.cp_kv_layer_split.deepseek_v4_pool import (
    CpKvLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_TRANSFER_ATTENTION_STATE,
    DSV4_TRANSFER_C4_INDEXER_KV,
    DSV4_TRANSFER_C4_KV,
    DSV4_TRANSFER_C128_KV,
    DSV4_TRANSFER_INDEXER_STATE,
    DSV4_TRANSFER_SWA_KV,
    DeepSeekV4TokenToKVPool,
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
            envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_SWA_SHARDING.override(
                disable_swa
            )
        )
        stack.enter_context(
            envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_C4_SHARDING.override(disable_c4)
        )
        stack.enter_context(
            envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_C128_SHARDING.override(
                disable_c128
            )
        )
        stack.enter_context(
            envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_C4_INDEXER_SHARDING.override(
                disable_c4_indexer
            )
        )
        yield


class TestCpKvLayerSplitDeepSeekV4Layout(CustomTestCase):
    def test_layout_counts_owned_layers_per_family(self):
        ratios = [0, 4, 4, 128] + [0] * 56
        with _override_sharding_envs():
            layout = build_cp_kv_layer_split_deepseek_v4_pool_layout(
                0, 4, 60, 0, 60, ratios
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
            layout = build_cp_kv_layer_split_deepseek_v4_worst_case_pool_layout(
                4, 61, 0, 61, ratios
            )
            rank_layouts = [
                build_cp_kv_layer_split_deepseek_v4_pool_layout(
                    rank, 4, 61, 0, 61, ratios
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
            layout = build_cp_kv_layer_split_deepseek_v4_pool_layout(
                1, 2, 8, 0, 8, ratios
            )

        self.assertEqual(layout.c4_layer_num, 4)
        self.assertEqual(layout.c4_state_layer_num, 4)
        self.assertEqual(layout.c128_layer_num, 1)


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

    return SimpleNamespace(
        _swa_global_to_local={
            gid: i for i, gid in enumerate(sorted(swa_owned_global_ids))
        },
        swa_kv_pool=SimpleNamespace(kv_buffer=[None] * swa_buffer_count),
        layer_mapping=layer_mapping,
        compression_ratios=compression_ratios,
        _stage_start=stage_start,
        _stage_end=stage_end,
        _owns_c4_kv_layer_id=lambda gid, _owned=c4_owned: gid in _owned,
        _owns_c128_kv_layer_id=lambda gid, _owned=c128_owned: gid in _owned,
        _owns_indexer_kv_layer_id=lambda gid, _owned=c4_indexer_owned: gid in _owned,
        _owns_attention_state_layer_id=lambda gid: (
            gid in c4_attn_state_owned or gid in c128_attn_state_owned
        ),
        _owns_indexer_state_layer_id=lambda gid: gid in c4_indexer_state_owned,
    )


def _hicache_mapping(fake):
    return CpKvLayerSplitDeepSeekV4TokenToKVPool.get_hicache_host_layer_mapping(fake)


class TestCpKvLayerSplitDeepSeekV4HiCacheMapping(CustomTestCase):
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
        mapping = _hicache_mapping(fake)

        self.assertEqual(mapping["swa"], {0: 0, 1: 1})
        self.assertEqual(mapping["c4_kv"], {1: 0, 3: 1})
        self.assertEqual(mapping["c128_kv"], {})
        self.assertEqual(mapping["c4_indexer"], {1: 0, 3: 1})
        self.assertEqual(mapping["c4_state"], {1: 0, 3: 1})
        self.assertEqual(mapping["c128_state"], {})
        self.assertEqual(mapping["c4_indexer_state"], {1: 0, 3: 1})

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
        mapping = _hicache_mapping(fake)

        self.assertEqual(mapping["c4_kv"], {0: 0, 2: 1})
        self.assertEqual(mapping["c4_indexer"], {0: 0, 1: 1, 2: 2, 3: 3})

    def test_union_across_cp_ranks_covers_each_sharded_layer_once(self):
        ratios = [0, 4, 4, 128, 4, 128, 0, 4]
        rank0 = _make_fake_hicache_pool(
            compression_ratios=ratios,
            swa_owned_global_ids={0, 1, 2, 3},
            swa_buffer_count=2,
            c4_owned={1, 2},
            c128_owned={3},
            c4_indexer_owned={1, 2},
            c4_attn_state_owned={1, 2},
            c128_attn_state_owned={3},
            c4_indexer_state_owned={1, 2},
        )
        rank1 = _make_fake_hicache_pool(
            compression_ratios=ratios,
            swa_owned_global_ids={4, 5, 6, 7},
            swa_buffer_count=2,
            c4_owned={4, 7},
            c128_owned={5},
            c4_indexer_owned={4, 7},
            c4_attn_state_owned={4, 7},
            c128_attn_state_owned={5},
            c4_indexer_state_owned={4, 7},
        )

        m0 = _hicache_mapping(rank0)
        m1 = _hicache_mapping(rank1)

        self.assertEqual(set(m0["c4_kv"]) | set(m1["c4_kv"]), {1, 2, 4, 7})
        self.assertEqual(set(m0["c128_kv"]) | set(m1["c128_kv"]), {3, 5})
        self.assertFalse(set(m0["c4_kv"]) & set(m1["c4_kv"]))

        for mapping in (m0, m1):
            for family_mapping in mapping.values():
                values = sorted(family_mapping.values())
                self.assertEqual(values, list(range(len(values))))


def _bind_layer_split_methods(fake):
    method_names = (
        "owns_kv_layer_id",
        "_owns_c4_kv_layer_id",
        "_owns_c128_kv_layer_id",
        "_owns_indexer_kv_layer_id",
        "_owns_attention_state_layer_id",
        "_owns_indexer_state_layer_id",
        "_transfer_swa_layer_id",
        "_transfer_core_layer_id",
        "_transfer_indexer_layer_id",
    )
    for name in method_names:
        method = getattr(CpKvLayerSplitDeepSeekV4TokenToKVPool, name)
        setattr(fake, name, method.__get__(fake))
    return fake


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
    fake = SimpleNamespace(
        cp_rank=cp_rank,
        cp_size=cp_size,
        model_num_hidden_layers=len(ratios),
        _stage_start=0,
        _stage_end=len(ratios),
        compression_ratios=ratios,
        _shard_swa=shard_swa,
        _shard_c4=shard_c4,
        _shard_c128=shard_c128,
        _shard_c4_indexer=shard_c4_indexer,
        _swa_global_to_local={
            layer_id: local_id for local_id, layer_id in enumerate(owned_swa_layers)
        },
        swa_kv_pool=SimpleNamespace(kv_buffer=[None] * 4),
    )
    return _bind_layer_split_methods(fake)


def _kv_transfer_layout(fake):
    return CpKvLayerSplitDeepSeekV4TokenToKVPool.get_kv_transfer_layout(fake)


def _state_transfer_layout(fake):
    return CpKvLayerSplitDeepSeekV4TokenToKVPool.get_state_transfer_layout(fake)


class TestCpKvLayerSplitDeepSeekV4TransferLayout(CustomTestCase):
    def test_sharded_layout_describes_only_owned_buffers(self):
        fake = _make_fake_transfer_pool(cp_rank=1)

        self.assertEqual(
            _kv_transfer_layout(fake),
            [
                (DSV4_TRANSFER_C4_KV, 4),
                (DSV4_TRANSFER_C4_KV, 7),
                (DSV4_TRANSFER_C4_INDEXER_KV, 4),
                (DSV4_TRANSFER_C4_INDEXER_KV, 7),
                (DSV4_TRANSFER_C128_KV, 5),
            ],
        )
        self.assertEqual(
            _state_transfer_layout(fake),
            [
                (DSV4_TRANSFER_SWA_KV, 4),
                (DSV4_TRANSFER_SWA_KV, 5),
                (DSV4_TRANSFER_SWA_KV, 6),
                (DSV4_TRANSFER_SWA_KV, 7),
                (DSV4_TRANSFER_ATTENTION_STATE, 4),
                (DSV4_TRANSFER_ATTENTION_STATE, 5),
                (DSV4_TRANSFER_ATTENTION_STATE, 7),
                (DSV4_TRANSFER_INDEXER_STATE, 4),
                (DSV4_TRANSFER_INDEXER_STATE, 7),
            ],
        )

    def test_replicated_non_owner_layout_keeps_none_placeholders(self):
        fake = _make_fake_transfer_pool(cp_rank=1, shard_c4=False)

        self.assertEqual(
            _kv_transfer_layout(fake),
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
            _state_transfer_layout(fake),
            [
                (DSV4_TRANSFER_SWA_KV, 4),
                (DSV4_TRANSFER_SWA_KV, 5),
                (DSV4_TRANSFER_SWA_KV, 6),
                (DSV4_TRANSFER_SWA_KV, 7),
                None,
                None,
                None,
                (DSV4_TRANSFER_ATTENTION_STATE, 5),
                None,
                (DSV4_TRANSFER_INDEXER_STATE, 4),
                (DSV4_TRANSFER_INDEXER_STATE, 7),
            ],
        )

    def test_unified_kv_pool_reports_no_descriptor_matching_layout(self):
        fake = SimpleNamespace(_unified_kv=True)

        self.assertEqual(DeepSeekV4TokenToKVPool.get_kv_transfer_layout(fake), [])
        self.assertEqual(DeepSeekV4TokenToKVPool.get_state_transfer_layout(fake), [])


if __name__ == "__main__":
    unittest.main()
