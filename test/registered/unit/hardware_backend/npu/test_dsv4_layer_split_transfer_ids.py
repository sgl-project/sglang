"""Unit tests for DSV4 (NPU) layer-split PD transfer layer ids.

A layer-split prefill rank registers only its owned layers while decode
registers the full cache, so transfer entries must be paired by (buffer
section, global layer id) -- positional pairing would ship bytes into the
wrong decode layers. These tests pin the pool id sequences and the pairings
they produce.
"""

import unittest

from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_cache_layer_split import (
    LayerSplitDSV4NPUTokenToKVPool,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_layer_split_plan import (
    DSV4LayerShardPlan,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DSV4NPUTokenToKVPool,
    bucket_layer_ids,
)
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# 15 stage layers: c4 at 0,2,5,7,10,12; c128 at 4,9,14.
RATIOS = [4, 0, 4, 0, 128] * 3
C4_IDS = [0, 2, 5, 7, 10, 12]
C128_IDS = [4, 9, 14]


def _bare(pool_cls, **attrs):
    pool = object.__new__(pool_cls)
    for name, value in attrs.items():
        setattr(pool, name, value)
    return pool


def _full_pool():
    return _bare(
        DSV4NPUTokenToKVPool,
        compression_ratios=RATIOS,
        _stage_start=0,
        _stage_end=len(RATIOS),
    )


def _shard_pool(rank: int, shard_size: int = 2):
    return _bare(
        LayerSplitDSV4NPUTokenToKVPool,
        compression_ratios=RATIOS,
        _stage_start=0,
        _stage_end=len(RATIOS),
        _shard_plan=DSV4LayerShardPlan(
            rank=rank,
            shard_size=shard_size,
            num_layers=len(RATIOS),
            stage_start=0,
            ratios=RATIOS,
        ),
    )


class TestDSV4TransferLayerIds(CustomTestCase):
    def test_bucket_layer_ids(self):
        self.assertEqual(
            bucket_layer_ids(RATIOS, 0, len(RATIOS), 4), C4_IDS
        )
        self.assertEqual(
            bucket_layer_ids(RATIOS, 0, len(RATIOS), 128), C128_IDS
        )

    def test_full_pool_ids_cover_every_buf_entry(self):
        pool = _full_pool()
        # Main KV: three same-length sections (c4 KV, index K, index scale).
        self.assertEqual(pool.get_kv_layer_ids(), C4_IDS * 3)
        # SWA component: per-layer SWA KV, then c4 attention and indexer states.
        self.assertEqual(
            pool.get_state_layer_ids(), list(range(len(RATIOS))) + C4_IDS * 2
        )
        self.assertEqual(pool.get_c128_layer_ids(), C128_IDS)

    def test_shard_pool_ids_are_owned_only(self):
        # 15 layers over 2 ranks: rank0 owns [0,8), rank1 owns [8,15).
        rank0, rank1 = _shard_pool(0), _shard_pool(1)
        self.assertEqual(rank0.get_kv_layer_ids(), [0, 2, 5, 7] * 3)
        self.assertEqual(rank1.get_kv_layer_ids(), [10, 12] * 3)
        self.assertEqual(
            rank0.get_state_layer_ids(), list(range(8)) + [0, 2, 5, 7] * 2
        )
        self.assertEqual(
            rank1.get_state_layer_ids(), list(range(8, 15)) + [10, 12] * 2
        )
        self.assertEqual(rank0.get_c128_layer_ids(), [4])
        self.assertEqual(rank1.get_c128_layer_ids(), [9, 14])

    def test_ids_pair_owned_entries_to_matching_decode_layers(self):
        dst_ids = _full_pool().get_kv_layer_ids()
        src_ids = _shard_pool(1).get_kv_layer_ids()
        pairs = build_transfer_entry_pairs(src_ids, dst_ids, len(src_ids), len(dst_ids))

        self.assertEqual(len(pairs), len(src_ids))
        for src_idx, dst_idx in pairs:
            # Every transfer moves bytes between the SAME global layer, and
            # each source entry is used exactly once.
            self.assertEqual(src_ids[src_idx], dst_ids[dst_idx])
        self.assertEqual(sorted(src for src, _ in pairs), list(range(len(src_ids))))
        # Positional pairing would have been plain (i, i): the id sequences
        # must actually differ there for this test to guard anything.
        self.assertNotEqual(src_ids, dst_ids[: len(src_ids)])

    def test_state_component_ids_pair_across_shard(self):
        dst_ids = _full_pool().get_state_layer_ids()
        src_ids = _shard_pool(1).get_state_layer_ids()
        pairs = build_transfer_entry_pairs(src_ids, dst_ids, len(src_ids), len(dst_ids))

        self.assertEqual(len(pairs), len(src_ids))
        for src_idx, dst_idx in pairs:
            self.assertEqual(src_ids[src_idx], dst_ids[dst_idx])

    @unittest.skipUnless(is_npu(), "layer ids engage the transfer only on NPU")
    def test_build_kv_layer_ids_reports_npu_pool_ids(self):
        pool = _full_pool()
        self.assertEqual(
            build_kv_layer_ids(
                token_to_kv_pool=pool,
                draft_token_to_kv_pool=None,
                num_draft_entries=0,
                num_hidden_layers=len(RATIOS),
            ),
            C4_IDS * 3,
        )


if __name__ == "__main__":
    unittest.main()
