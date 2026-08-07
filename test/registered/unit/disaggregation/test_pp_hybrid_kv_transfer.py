"""Unit tests for full-attention KV transfer with prefill pp_size > 1 on
hybrid-linear models (HybridLinearKVPool)."""

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.prefill import _transfer_start_layer
from sglang.srt.disaggregation.utils import build_transfer_entry_pairs
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _full_attention_ids(*, num_layers: int, interval: int) -> list:
    return [i for i in range(num_layers) if i % interval == interval - 1]


def _hybrid_pool(*, start_layer: int) -> HybridLinearKVPool:
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.start_layer = start_layer
    return pool


class TestTransferStartLayer(CustomTestCase):
    """Bug regression: with prefill pp_size=2 on a 60-layer hybrid-linear model
    (full_attention_interval=4), stage 1's pool.start_layer is 30 — a global
    layer index counting linear layers. The decode peer's KV pointer list is
    dense over the 15 full-attention layers only, so slicing dst[30:38] yielded
    [] and an IndexError in mooncake send_kvcache_slice. The transfer offset
    must be the count of full-attention layers before the stage boundary."""

    def test_hybrid_stage1_translates_to_full_attention_offset(self):
        cfg = SimpleNamespace(
            full_attention_layer_ids=_full_attention_ids(num_layers=60, interval=4)
        )
        self.assertEqual(
            _transfer_start_layer(
                pool=_hybrid_pool(start_layer=30), hf_text_config=cfg
            ),
            7,
        )

    def test_hybrid_stage0_is_zero(self):
        cfg = SimpleNamespace(
            full_attention_layer_ids=_full_attention_ids(num_layers=60, interval=4)
        )
        self.assertEqual(
            _transfer_start_layer(pool=_hybrid_pool(start_layer=0), hf_text_config=cfg),
            0,
        )

    def test_non_hybrid_pool_keeps_global_start_layer(self):
        cfg = SimpleNamespace(full_attention_layer_ids=[])
        self.assertEqual(
            _transfer_start_layer(
                pool=SimpleNamespace(start_layer=30), hf_text_config=cfg
            ),
            30,
        )


class _RecordingKVManager:
    get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp

    def __init__(self, *, prefill_start_layer: int, pp_size: int):
        self.is_mla_backend = False
        self.is_hybrid_mla_backend = False
        self.enable_custom_mem_pool = False
        self.pp_size = pp_size
        self.kv_args = SimpleNamespace(prefill_start_layer=prefill_start_layer)
        self.blocks = []

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        self.blocks.extend(transfer_blocks)
        return 0


class TestHybridSendUsesLayerIdPairing(CustomTestCase):
    """Bug regression: a hybrid-linear (non-MLA-flagged) backend fell into the
    positional MHA slicing path of _send_kvcache_generic even when both peers
    published layer ids. For a stage with F full-attention layers against a
    decode peer with N (F < N, F not dividing N), the draft-KV modulo heuristic
    silently placed the V block at F * (N // F) instead of N — wrong layers
    transferred, no error. With layer ids published on both sides the pairing
    must be exact."""

    def _run_case(
        self, *, model_full_ids: list, stage_full_ids: list, start_offset: int
    ):
        num_stage = len(stage_full_ids)
        num_model = len(model_full_ids)
        src_ptrs = [1000 + i for i in range(2 * num_stage)]
        dst_ptrs = [2000 + i for i in range(2 * num_model)]
        item_lens = [10 + i for i in range(2 * num_stage)]
        manager = _RecordingKVManager(prefill_start_layer=start_offset, pp_size=2)
        rc = MooncakeKVManager._send_kvcache_generic(
            manager,
            mooncake_session_id="session",
            src_data_ptrs=src_ptrs,
            dst_data_ptrs=dst_ptrs,
            item_lens=item_lens,
            prefill_data_indices=np.array([0], dtype=np.int32),
            dst_data_indices=np.array([0], dtype=np.int32),
            executor=None,
            src_layer_ids=stage_full_ids * 2,
            dst_layer_ids=model_full_ids * 2,
        )
        self.assertEqual(rc, 0)
        expected = [
            (src_ptrs[i], dst_ptrs[start_offset + i], item_lens[i])
            for i in range(num_stage)
        ] + [
            (
                src_ptrs[num_stage + i],
                dst_ptrs[num_model + start_offset + i],
                item_lens[num_stage + i],
            )
            for i in range(num_stage)
        ]
        self.assertEqual(manager.blocks, expected)

    def test_stage1_f8_of_n15(self):
        ids = _full_attention_ids(num_layers=60, interval=4)
        self._run_case(model_full_ids=ids, stage_full_ids=ids[7:], start_offset=7)

    def test_stage0_f7_of_n15(self):
        ids = _full_attention_ids(num_layers=60, interval=4)
        self._run_case(model_full_ids=ids, stage_full_ids=ids[:7], start_offset=0)

    def test_f5_of_n12(self):
        ids = _full_attention_ids(num_layers=48, interval=4)
        self._run_case(model_full_ids=ids, stage_full_ids=ids[:5], start_offset=0)


class TestGetMhaKvPtrsWithPp(CustomTestCase):
    """Derived property: the modulo heuristic in get_mha_kv_ptrs_with_pp exists
    for the decode-has-draft-KV layout [K_main, V_main, draft_K, draft_V]. Pin
    that geometry (15 main + 1 draft layer) so a future rewrite of the
    heuristic (e.g. to fix the plain-MHA pp>1 F-not-dividing-N case) keeps the
    draft case intact."""

    def test_draft_kv_geometry_selects_main_v_block(self):
        manager = SimpleNamespace(kv_args=SimpleNamespace(prefill_start_layer=0))
        src_kv_ptrs = list(range(30))
        dst_kv_ptrs = list(range(100, 132))
        src_k, src_v, dst_k, dst_v, num_layers = (
            CommonKVManager.get_mha_kv_ptrs_with_pp(manager, src_kv_ptrs, dst_kv_ptrs)
        )
        self.assertEqual(src_k, src_kv_ptrs[:15])
        self.assertEqual(src_v, src_kv_ptrs[15:])
        self.assertEqual(dst_k, dst_kv_ptrs[:15])
        self.assertEqual(dst_v, dst_kv_ptrs[15:30])
        self.assertEqual(num_layers, 15)


class TestBuildTransferEntryPairsDuplicateIds(CustomTestCase):
    """Derived property: layer ids repeat across the K and V tensor groups, so
    pairing must consume dst occurrences in order (K with K, V with V) rather
    than by plain id lookup."""

    def test_k_then_v_occurrence_ordering(self):
        pairs = build_transfer_entry_pairs(
            src_layer_ids=[3, 7, 3, 7],
            dst_layer_ids=[3, 7, 11, 3, 7, 11],
            n_src=4,
            n_dst=6,
            allow_positional_fallback=False,
        )
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 3), (3, 4)])


if __name__ == "__main__":
    unittest.main()
