"""Unit tests for PP-aware ordinary and hybrid-linear KV transfer entries."""

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.prefill import _transfer_start_layer
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
)
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import CPUFP8KVCacheMethod
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MHATokenToKVPoolMXFP8,
    PageMajorMHATokenToKVPool,
)
from sglang.srt.runtime_context import get_memory, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _mori_kv_manager_cls():
    # CPU runners lack the native mori package; stub only what mori/conn.py imports.
    try:
        importlib.import_module("mori.io")
    except (ImportError, OSError):
        for name in ("mori", "mori.cpp", "mori.io"):
            sys.modules.setdefault(name, MagicMock())
    from sglang.srt.disaggregation.mori.conn import MoriKVManager

    return MoriKVManager


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


def _hybrid_pool_with_ids(*, layer_ids: list) -> HybridLinearKVPool:
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.full_attention_layer_id_mapping = layer_ids
    pool.use_mla = False
    return pool


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
        ids = build_kv_layer_ids(
            token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=self._stage1_ids()),
            draft_token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=[0]),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        stage1 = self._stage1_ids()
        self.assertEqual(ids, stage1 + stage1 + [60, 60])

    def test_ragged_draft_registration_is_rejected(self):
        with self.assertRaises(RuntimeError):
            build_kv_layer_ids(
                token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=self._stage1_ids()),
                draft_token_to_kv_pool=SimpleNamespace(layer_num=2),
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
        src = build_kv_layer_ids(
            token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=stage1),
            draft_token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=[0]),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        dst = build_kv_layer_ids(
            token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=full),
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


class TestOrdinaryMhaEntryIds(CustomTestCase):
    """Regression: uneven ordinary-MHA stages used divisibility to locate V,
    corrupting target KV. Source draft entries also invalidate a half-split.
    Check the resulting tensor contents using the registered entry pairing.
    """

    @staticmethod
    def _pool(start, layers, quant_method=None):
        return MHATokenToKVPool(
            size=3,
            page_size=1,
            dtype=torch.float32,
            head_num=2,
            head_dim=4,
            layer_num=layers,
            device="cpu",
            enable_memory_saver=False,
            enable_alt_stream=False,
            start_layer=start,
            kv_cache_layout="nhd",
            quant_method=quant_method,
        )

    @staticmethod
    def _ids(target, draft=None):
        return build_kv_layer_ids(
            token_to_kv_pool=target,
            draft_token_to_kv_pool=draft,
            num_draft_entries=0 if draft is None else 2 * draft.layer_num,
            num_hidden_layers=15,
        )

    def test_uneven_stages_copy_target_and_separate_draft_entries(self):
        for source_draft, destination_draft in (
            (False, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                source_draft=source_draft, destination_draft=destination_draft
            ):
                target = self._pool(0, 15)
                draft = self._pool(0, 1) if destination_draft else None
                dst = target.k_buffer + target.v_buffer
                if draft is not None:
                    dst += draft.k_buffer + draft.v_buffer
                for tensor in dst:
                    tensor.fill_(-1)
                dst_ids = self._ids(target, draft)
                for start, layers in ((0, 7), (7, 8)):
                    source = self._pool(start, layers)
                    for bias, buffers in (
                        (10, source.k_buffer),
                        (100, source.v_buffer),
                    ):
                        for index, tensor in enumerate(buffers):
                            tensor.fill_(bias + start + index)
                    # PD-prefill draft buffers belong to the final stage.
                    src_draft = (
                        self._pool(0, 1) if source_draft and start == 7 else None
                    )
                    src = source.k_buffer + source.v_buffer
                    if src_draft is not None:
                        src_draft.k_buffer[0].fill_(201)
                        src_draft.v_buffer[0].fill_(202)
                        src += src_draft.k_buffer + src_draft.v_buffer
                    pairs = build_transfer_entry_pairs(
                        self._ids(source, src_draft), dst_ids, len(src), len(dst)
                    )
                    for i, j in pairs:
                        dst[j].copy_(src[i])
                for bias, buffers in ((10, target.k_buffer), (100, target.v_buffer)):
                    for layer, tensor in enumerate(buffers):
                        self.assertTrue(torch.all(tensor == bias + layer))
                if draft is not None:
                    self.assertTrue(
                        torch.all(draft.k_buffer[0] == (201 if source_draft else -1))
                    )
                    self.assertTrue(
                        torch.all(draft.v_buffer[0] == (202 if source_draft else -1))
                    )

    def test_nixl_prepares_matching_target_and_draft_regions(self):
        """Guard ordinary producer IDs reaching NIXL's prepared descriptors.

        Native handle creation is recorded, not executed on these CPU tensors.
        """
        source, destination = self._pool(7, 8), self._pool(0, 15)
        draft = self._pool(0, 1)
        src_ptrs, _, src_items = source.get_contiguous_buf_infos()
        dst_ptrs, _, dst_items = destination.get_contiguous_buf_infos()
        draft_ptrs, _, draft_items = draft.get_contiguous_buf_infos()
        manager = NixlKVManager.__new__(NixlKVManager)
        manager.pp_size = 2
        manager.attn_tp_size = 1
        manager.is_mla_backend = manager.is_hybrid_mla_backend = False
        manager.src_mem_kind = "VRAM"
        manager._num_slots_src = 4
        manager.prep_handles = {}
        manager._init_equal_tp_prep_handle = MagicMock()
        manager.kv_args = SimpleNamespace(
            kv_layer_ids=self._ids(source, draft),
            kv_data_ptrs=src_ptrs + draft_ptrs,
            kv_item_lens=src_items + draft_items,
            gpu_id=0,
        )
        peer = SimpleNamespace(
            dst_kv_layer_ids=self._ids(destination, draft),
            dst_kv_ptrs=dst_ptrs + draft_ptrs,
            dst_kv_item_lens=dst_items + draft_items,
            dst_kv_mem_kinds=["VRAM"] * 32,
            requires_dcp_relayout=False,
            decode_tp_size=1,
            dst_num_slots=4,
            agent_name="decode",
            gpu_id=0,
        )
        manager._prepare_payload_xfer(peer)
        calls = manager._init_equal_tp_prep_handle.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].args[0], "decode")
        expected = dst_ptrs[7:15] + dst_ptrs[22:30] + draft_ptrs
        self.assertEqual(calls[1].args[1], expected)
        self.assertEqual(calls[1].kwargs["kv_xfer_lens"], src_items + draft_items)
        self.assertEqual(
            calls[1].kwargs["kv_item_lens"],
            dst_items[7:15] + dst_items[22:30] + draft_items,
        )

    def test_nixl_hetero_tp_pairs_uneven_stage_by_layer_id(self):
        """Head-sliced NIXL descriptors must not use the half-split V offset."""
        source, destination = self._pool(7, 8), self._pool(0, 15)
        draft = self._pool(0, 1)
        src_ptrs, src_lens, src_items = source.get_contiguous_buf_infos()
        dst_ptrs, _, dst_items = destination.get_contiguous_buf_infos()
        draft_ptrs, _, _ = draft.get_contiguous_buf_infos()
        manager = NixlKVManager.__new__(NixlKVManager)
        manager.agent = MagicMock()
        manager.pp_size = 2
        manager.attn_tp_size = 1
        manager.prep_handle_slice_src = None
        manager.prep_handles_slice_dst = {}
        manager.kv_args = SimpleNamespace(
            gpu_id=0,
            engine_rank=0,
            page_size=1,
            prefill_start_layer=7,
            total_kv_head_num=2,
            kv_head_num=2,
            kv_layer_ids=self._ids(source),
            kv_data_ptrs=src_ptrs,
            kv_item_lens=src_items,
            kv_data_lens=src_lens,
        )
        peer = SimpleNamespace(
            agent_name="decode",
            decode_tp_size=2,
            decode_tp_rank=0,
            dst_kv_item_len=dst_items[0] // 2,
            dst_kv_ptrs=dst_ptrs + draft_ptrs,
            dst_kv_layer_ids=self._ids(destination, draft),
            dst_num_slots=3,
            gpu_id=0,
        )
        manager._init_hetero_tp_prep_handle("decode", peer)
        dst_call = next(
            call
            for call in manager.agent.prep_xfer_dlist.call_args_list
            if call.args[0] == "decode"
        )
        self.assertEqual(
            [int(run[0]) for run in dst_call.args[1]],
            dst_ptrs[7:15] + dst_ptrs[22:30],
        )

    def test_mori_pp1_version_skew_keeps_positional_pairing(self):
        """A PP=1 Mori peer without layer IDs keeps the matching-prefix pairing."""
        manager = object.__new__(_mori_kv_manager_cls())
        target, draft = self._pool(0, 15), self._pool(0, 1)
        dst = [f"dst{i}" for i in range(32)]
        manager.kv_mem_descs = [f"src{i}" for i in range(30)]
        for source_ids, destination_ids in (
            ([], self._ids(target, draft)),
            (self._ids(target), []),
        ):
            with self.subTest(source_has_ids=bool(source_ids)):
                manager.kv_args = SimpleNamespace(
                    kv_layer_ids=source_ids, prefill_start_layer=0
                )
                manager.pp_size = 1
                _, _, dst_k, dst_v, layers = manager._get_mha_mem_desc_slices(
                    dst, destination_ids
                )
                self.assertEqual((dst_k, dst_v, layers), (dst[:15], dst[15:30], 15))
                manager.pp_size = 2
                with self.assertRaisesRegex(RuntimeError, "both PD peers"):
                    manager._get_mha_mem_desc_slices(dst, destination_ids)

    def test_quantization_recipe_on_base_class_does_not_publish_ids(self):
        pool = self._pool(0, 2, quant_method=CPUFP8KVCacheMethod())
        self.assertTrue(pool.is_quantized_kv_cache)
        self.assertEqual(self._ids(pool), [])

    def test_mixed_pp_peers_require_coordinated_metadata_upgrade(self):
        ids = self._ids(self._pool(7, 8))
        for source_ids, destination_ids in ((ids, []), ([], ids)):
            with self.subTest(source_has_ids=bool(source_ids)):
                # Equal entry counts alone cannot establish stage identity.
                with self.assertRaisesRegex(RuntimeError, "both PD peers"):
                    build_transfer_entry_pairs(source_ids, destination_ids, 16, 16)
                # PP=1 retains the existing matching-prefix compatibility path.
                self.assertEqual(
                    build_transfer_entry_pairs(
                        source_ids,
                        destination_ids,
                        16,
                        16,
                        allow_positional_fallback=True,
                    ),
                    [(i, i) for i in range(16)],
                )

    def test_equal_entry_counts_do_not_hide_different_layer_spans(self):
        source, destination = self._pool(7, 8), self._pool(0, 8)
        with self.assertRaisesRegex(RuntimeError, "missing a transfer entry"):
            build_transfer_entry_pairs(
                self._ids(source), self._ids(destination), 16, 16
            )

    def test_special_layouts_do_not_inherit_plain_mha_ids(self):
        for cls in (
            MHATokenToKVPoolFP4,
            MHATokenToKVPoolMXFP8,
            PageMajorMHATokenToKVPool,
        ):
            with self.subTest(pool=cls.__name__):
                self.assertEqual(self._ids(cls.__new__(cls)), [])
        pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        pool.kv_cache_layout = "hnd"
        self.assertEqual(self._ids(pool), [])

    def test_registration_cardinality_must_match_the_described_entries(self):
        pool = self._pool(0, 2)
        pointers, lengths, item_lengths = pool.get_contiguous_buf_infos()
        for registration in (
            (pointers[:-1], lengths, item_lengths),
            (pointers, lengths[:-1], item_lengths),
            (pointers, lengths, item_lengths[:-1]),
        ):
            with self.subTest(lengths=tuple(map(len, registration))):
                with patch.object(
                    pool, "get_contiguous_buf_infos", return_value=registration
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Ordinary MHA registration"
                    ):
                        self._ids(pool)


if __name__ == "__main__":
    unittest.main()
