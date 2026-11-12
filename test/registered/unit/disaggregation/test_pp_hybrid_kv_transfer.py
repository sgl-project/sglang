"""Unit tests for full-attention KV transfer with prefill pp_size > 1 on
hybrid-linear models and NPU MLA pools."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.prefill import _transfer_start_layer
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.srt.runtime_context import get_memory, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _full_attention_ids(*, num_layers: int, interval: int) -> list:
    return [i for i in range(num_layers) if i % interval == interval - 1]


def _hybrid_pool(*, start_layer: int) -> HybridLinearKVPool:
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.start_layer = start_layer
    # The inner full-attention view its router reports as the wrapped view.
    pool.full_kv_pool = object()
    pool.full_attention_layer_id_mapping = {}
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


class _RecordingKVManager:
    get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp
    get_mla_kv_ptrs_with_pp = CommonKVManager.get_mla_kv_ptrs_with_pp

    def __init__(self, *, prefill_start_layer: int, pp_size: int):
        self.is_mla_backend = False
        self.is_hybrid_mla_backend = False
        self.enable_custom_mem_pool = False
        self.max_transfer_batch_indices = 0
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


class TestSingleRegionSWATransfer(CustomTestCase):
    def test_one_region_full_generates_transfer_block(self):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)
        manager = _RecordingKVManager(prefill_start_layer=0, pp_size=1)
        manager.kv_args.kv_data_ptrs = [1000]
        manager.kv_args.kv_item_lens = [64]
        manager.kv_args.kv_layer_ids = []
        manager._validate_envelope_kv_layout = (
            MooncakeKVManager._validate_envelope_kv_layout.__get__(manager)
        )
        manager._send_kvcache_generic = MooncakeKVManager._send_kvcache_generic.__get__(
            manager
        )
        with get_memory().override(enable_unified_memory=True):
            rc = MooncakeKVManager.send_kvcache(
                manager,
                mooncake_session_id="session",
                prefill_kv_indices=np.array([3, 4], dtype=np.int32),
                dst_kv_ptrs=[2000],
                dst_kv_indices=np.array([7, 8], dtype=np.int32),
                dst_kv_item_len=64,
                executor=None,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(manager.blocks, [(1192, 2448, 128)])


class _RecordingAscendManager:
    def __init__(self):
        self.is_hybrid_mla_backend = True
        self.pp_size = 2
        self.kv_args = SimpleNamespace(
            kv_data_ptrs=[101, 102, 103, 104],
            kv_item_lens=[11, 12, 13, 14],
            kv_layer_ids=[31, 35, 31, 35],
        )
        self.generic_call = None

    def _validate_envelope_kv_layout(self, *args):
        pass

    def _send_kvcache_generic(self, **kwargs):
        self.generic_call = kwargs
        return 7


class TestAscendHybridPpDispatch(CustomTestCase):
    def test_hybrid_pp_uses_layer_id_pairing_path(self):
        manager = _RecordingAscendManager()
        dst_layer_ids = [3, 7, 31, 35, 3, 7, 31, 35]
        rc = AscendKVManager.send_kvcache(
            manager,
            mooncake_session_id="session",
            prefill_kv_indices=np.array([1], dtype=np.int32),
            dst_kv_ptrs=list(range(8)),
            dst_kv_indices=np.array([2], dtype=np.int32),
            executor=None,
            dst_layer_ids=dst_layer_ids,
        )

        self.assertEqual(rc, 7)
        self.assertEqual(manager.generic_call["src_layer_ids"], [31, 35, 31, 35])
        self.assertEqual(manager.generic_call["dst_layer_ids"], dst_layer_ids)


class TestMambaSlotTransfer(CustomTestCase):
    def test_stage1_uses_paired_slot_sizes_and_offsets(self):
        manager = _RecordingKVManager(prefill_start_layer=1, pp_size=2)
        MooncakeKVManager._send_mamba_state(
            manager,
            req=SimpleNamespace(mooncake_session_id="session"),
            prefill_mamba_index=[2],
            src_state_data_ptrs=[1000],
            src_state_item_lens=[16],
            dst_state_data_ptrs=[2000, 3000],
            dst_mamba_index=[3],
            src_layer_ids=[7],
            dst_layer_ids=[3, 7],
            dst_state_item_lens=[8, 16],
        )
        self.assertEqual(manager.blocks, [(1032, 3048, 16)])

    def test_slot_size_mismatch_rejected_before_transfer(self):
        manager = _RecordingKVManager(prefill_start_layer=0, pp_size=1)
        with self.assertRaisesRegex(RuntimeError, "Mamba slot size mismatch"):
            MooncakeKVManager._send_mamba_state(
                manager,
                req=SimpleNamespace(mooncake_session_id="session"),
                prefill_mamba_index=[1],
                src_state_data_ptrs=[1000],
                src_state_item_lens=[16],
                dst_state_data_ptrs=[2000],
                dst_mamba_index=[1],
                dst_state_item_lens=[32],
            )
        self.assertEqual(manager.blocks, [])


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


def _plain_pool(*, layer_num: int) -> SimpleNamespace:
    """A one-sub-pool pool that is its own view."""
    pool = SimpleNamespace(layer_num=layer_num, start_layer=0)
    pool.layer_router = PlainPoolRouter(pool, IdSpaceKind.FULL)
    return pool


def _hybrid_pool_with_ids(*, layer_ids: list) -> HybridLinearKVPool:
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.full_kv_pool = object()
    pool.full_attention_layer_id_mapping = {
        layer_id: index for index, layer_id in enumerate(layer_ids)
    }
    pool.use_mla = False
    # No buffers: the layer ids are what these tests read.
    pool.get_contiguous_buf_infos = lambda: ([], [], [])
    return pool


def _static_storage(pool) -> StaticKVStorage:
    """The storage a static pool registers with PD peers through."""
    allocator = SimpleNamespace(storage=None, state=None, get_kvcache=lambda: pool)
    return StaticKVStorage(None, allocator)


def _npu_mla_pool(
    start_layer, layer_num, *, fp8=False, combined=False, indexer_layer_ids=()
):
    # CPU tensors exercise the real registration methods without NPU allocation.
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.start_layer, pool.layer_num = start_layer, layer_num
    pool.use_combined_kv_buffer = combined
    pool.dsa_kv_cache_store_fp8 = fp8
    pool.is_draft_worker, pool.dcp_size = False, 1
    pool.index_head_dim = 128 if indexer_layer_ids else None
    pool.indexer_layer_ids = tuple(indexer_layer_ids)

    def buffers(layer_ids, group):
        # Distinct contents expose pairing the right layer with the wrong group.
        return [torch.full((2, 2, 1, 4), group * 16 + lid) for lid in layer_ids]

    layers = range(start_layer, start_layer + layer_num)
    pool.k_buffer = buffers(layers, 0)
    pool.v_buffer = buffers(layers, 1)
    pool.kv_buffer = buffers(layers, 2)
    pool.index_k_buffer = buffers(indexer_layer_ids, 3)
    pool.index_k_scale_buffer = buffers(indexer_layer_ids, 4) if fp8 else None
    return pool


class TestNpuMlaLayerMetadata(CustomTestCase):
    def _ids(self, pool, draft=None):
        return build_kv_layer_ids(
            token_to_kv_pool=pool,
            storage=_static_storage(pool),
            draft_token_to_kv_pool=draft,
            num_draft_entries=(
                len(draft.get_contiguous_buf_infos()[0]) if draft is not None else 0
            ),
            num_hidden_layers=8,
        )

    def test_owner_ids_follow_registered_groups(self):
        layers, index_layers = [4, 5, 6, 7], [4, 6]
        cases = [
            ({}, ("k_buffer", "v_buffer"), layers * 2),
            (
                {"indexer_layer_ids": index_layers},
                ("k_buffer", "v_buffer", "index_k_buffer"),
                layers * 2 + index_layers,
            ),
            (
                {"fp8": True, "indexer_layer_ids": index_layers},
                ("k_buffer", "index_k_buffer", "index_k_scale_buffer"),
                layers + index_layers * 2,
            ),
            ({"combined": True}, ("kv_buffer",), layers),
        ]
        for options, groups, expected_ids in cases:
            with self.subTest(options=options):
                pool = _npu_mla_pool(4, 4, **options)
                buffers = [
                    buffer for group in groups for buffer in getattr(pool, group)
                ]
                ptrs, lens, strides = pool.get_contiguous_buf_infos()
                self.assertEqual(ptrs, [buffer.data_ptr() for buffer in buffers])
                self.assertEqual(lens, [buffer.nbytes for buffer in buffers])
                self.assertEqual(strides, [buffer[0].nbytes for buffer in buffers])
                self.assertEqual(self._ids(pool), expected_ids)
                self.assertEqual(len(expected_ids), len(ptrs))

    def test_pipeline_pairing_preserves_layer_and_tensor_group(self):
        layouts = [
            ((0, 8), (0, 8)),
            ((0, 4), (0, 8)),
            ((4, 4), (0, 8)),
            ((4, 4), (4, 4)),
        ]
        for fp8 in (False, True):
            for source, destination in layouts:
                with self.subTest(fp8=fp8, source=source, destination=destination):
                    pools = [
                        _npu_mla_pool(
                            start,
                            count,
                            fp8=fp8,
                            indexer_layer_ids=[
                                lid
                                for lid in range(start, start + count)
                                if lid % 3 == 0
                            ],
                        )
                        for start, count in (source, destination)
                    ]
                    src, dst = pools
                    src_buffers = [b for b, _ in src._get_disagg_buffer_entries()]
                    dst_buffers = [b for b, _ in dst._get_disagg_buffer_entries()]
                    pairs = build_transfer_entry_pairs(
                        self._ids(src),
                        self._ids(dst),
                        len(src_buffers),
                        len(dst_buffers),
                    )
                    self.assertEqual(len(pairs), len(src_buffers))
                    for i, j in pairs:
                        self.assertTrue(torch.equal(src_buffers[i], dst_buffers[j]))

    def test_dense_npu_draft_uses_positional_group_fallback(self):
        target = _npu_mla_pool(4, 4)
        for fp8, has_indexer, groups in (
            (False, False, 2),
            (False, True, 3),
            (True, True, 3),
        ):
            with self.subTest(fp8=fp8, has_indexer=has_indexer):
                draft = _npu_mla_pool(
                    0,
                    3,
                    fp8=fp8,
                    indexer_layer_ids=range(3) if has_indexer else (),
                )
                draft.is_draft_worker = True
                # Draft construction disables indexer elision and combined KV;
                # the existing fallback infers these complete tensor groups.
                with patch.object(
                    draft,
                    "get_kv_layer_ids",
                    side_effect=AssertionError("dense draft should use the fallback"),
                ):
                    self.assertEqual(
                        self._ids(target, draft),
                        [4, 5, 6, 7] * 2 + [8, 9, 10] * groups,
                    )


class TestBuildKvLayerIds(CustomTestCase):
    """Bug regression: enabling EAGLE appended draft KV buffers to kv_data_ptrs
    while kv_layer_ids described only the target's entries, so the ids were
    suppressed entirely and the transfer fell back to positional slicing. Under
    prefill pp_size > 1 that slices the wrong layers -- prefill pp=2 + EAGLE
    produced garbled decode output while pp=1 + EAGLE did not."""

    def _stage1_ids(self) -> list:
        full = _full_attention_ids(num_layers=60, interval=4)
        return [lid for lid in full if lid >= 30]

    def test_hybrid_draft_pool_is_remapped_out_of_the_target_range(self):
        """The EAGLE draft pool for a hybrid-linear model is itself a
        HybridLinearKVPool that numbers its single MTP layer from zero, so its
        raw ids collide with target layer 0 and must be remapped into the band."""
        target = _hybrid_pool_with_ids(layer_ids=self._stage1_ids())
        ids = build_kv_layer_ids(
            token_to_kv_pool=target,
            storage=_static_storage(target),
            draft_token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=[0]),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        stage1 = self._stage1_ids()
        self.assertEqual(ids, stage1 + stage1 + [60, 60])

    def test_ragged_draft_registration_is_rejected(self):
        target = _hybrid_pool_with_ids(layer_ids=self._stage1_ids())
        with self.assertRaises(RuntimeError):
            build_kv_layer_ids(
                token_to_kv_pool=target,
                storage=_static_storage(target),
                draft_token_to_kv_pool=_plain_pool(layer_num=2),
                num_draft_entries=3,
                num_hidden_layers=60,
            )


class TestDraftBandPairsAcrossPipelineStages(CustomTestCase):
    """Derived property: a pp=2 prefill stage and a pp=1 decode peer, both with
    an EAGLE draft pool, must pair on layer id -- the stage's 8 full-attention
    layers land on the decode peer's matching K and V entries, and the draft
    band lands on the decode peer's draft entries rather than on layer 0."""

    def test_stage1_pairs_onto_the_decode_layout(self):
        full = _full_attention_ids(num_layers=60, interval=4)
        stage1 = [lid for lid in full if lid >= 30]
        target = _hybrid_pool_with_ids(layer_ids=stage1)
        src = build_kv_layer_ids(
            token_to_kv_pool=target,
            storage=_static_storage(target),
            draft_token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=[0]),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        target = _hybrid_pool_with_ids(layer_ids=full)
        dst = build_kv_layer_ids(
            token_to_kv_pool=target,
            storage=_static_storage(target),
            draft_token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=[0]),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        pairs = build_transfer_entry_pairs(
            src, dst, len(src), len(dst), allow_positional_fallback=False
        )
        k_offset = len(full) - len(stage1)
        self.assertEqual(
            pairs,
            # K block, then V block, then the two draft entries at the tail.
            [(i, k_offset + i) for i in range(len(stage1))]
            + [(len(stage1) + i, len(full) + k_offset + i) for i in range(len(stage1))]
            + [
                (2 * len(stage1), 2 * len(full)),
                (2 * len(stage1) + 1, 2 * len(full) + 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
