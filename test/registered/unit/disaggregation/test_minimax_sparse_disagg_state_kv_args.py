import concurrent.futures
import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_staging_slot_metadata,
    build_transfer_entry_pairs,
    setup_state_kv_args,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MiniMaxSparseKVPool
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _make_k_only_pool(start_layer: int = 0) -> MiniMaxSparseKVPool:
    """Mirror the released MiniMax-M3 config shape: all sparse layers K-only."""
    dense_layer_ids = [start_layer, start_layer + 1, start_layer + 2]
    sparse_layer_ids = [start_layer + 3 + i for i in range(4)]
    end_layer = sparse_layer_ids[-1] + 1
    return MiniMaxSparseKVPool(
        size=8,
        page_size=4,
        dtype=torch.float32,
        head_num=2,
        head_dim=8,
        idx_head_dim=16,
        dense_layer_ids=dense_layer_ids,
        sparse_layer_ids=sparse_layer_ids,
        disable_value_sparse_layer_ids=sparse_layer_ids,
        device="cpu",
        start_layer=start_layer,
        end_layer=end_layer,
    )


def _static_storage(pool, draft=None, total_kv_layers=None) -> StaticKVStorage:
    """The storage a static pool registers with PD peers through."""
    allocator = SimpleNamespace(storage=None, state=None, get_kvcache=lambda: pool)
    return StaticKVStorage(
        None, allocator, draft_token_to_kv_pool=draft, num_hidden_layers=total_kv_layers
    )


def _make_kv_pool(start_layer: int = 0) -> MiniMaxSparseKVPool:
    """Sparse layers with index value (index_kv_pool != None)."""
    dense_layer_ids = [start_layer, start_layer + 1]
    sparse_layer_ids = [start_layer + 2, start_layer + 3]
    end_layer = sparse_layer_ids[-1] + 1
    return MiniMaxSparseKVPool(
        size=8,
        page_size=4,
        dtype=torch.float32,
        head_num=2,
        head_dim=8,
        idx_head_dim=16,
        dense_layer_ids=dense_layer_ids,
        sparse_layer_ids=sparse_layer_ids,
        disable_value_sparse_layer_ids=[],
        device="cpu",
        start_layer=start_layer,
        end_layer=end_layer,
    )


class TestMiniMaxSparseDisaggStateKvArgs(unittest.TestCase):
    def test_index_kv_pool_raises(self):
        pool = _make_kv_pool()
        self.assertIsNotNone(pool.index_kv_pool)
        kv_args = KVArgs()
        with self.assertRaises(NotImplementedError):
            setup_state_kv_args(kv_args, _static_storage(pool))

    def test_decode_rejects_prefill_with_different_attention_tp(self):
        # Main page rows are head-major and the index state is per rank, so
        # neither can be resliced by head; reject before any transfer.
        kv_args = KVArgs()
        kv_args.page_size = 4
        setup_state_kv_args(kv_args, _static_storage(_make_k_only_pool()))
        manager = object.__new__(CommonKVManager)
        manager.kv_args = kv_args
        manager.attn_tp_size = 4
        manager.kv_cache_dtype_str = None
        manager.prefill_info_table = {}
        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=8,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=4,
            kv_cache_dtype=None,
            follow_bootstrap_room=True,
        )
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            with self.assertRaisesRegex(RuntimeError, "matching attention TP"):
                manager.try_ensure_parallel_info("prefill:8998")
        self.assertEqual(manager.prefill_info_table, {})


# MiniMax-M3.1: 60 layers, all sparse with a K-only index, and a 5-layer DSpark draft.
M31_LAYERS = 60
M31_DRAFT_LAYERS = 5
PP4_STAGES = ((0, 15), (15, 30), (30, 45), (45, 60))


def _m31_pool(
    start_layer: int, end_layer: int, *, nvfp4: bool, kv_heads: int = 4, size=256
):
    layers = list(range(M31_LAYERS))
    kwargs = dict(
        size=size,
        page_size=128,
        dtype=torch.bfloat16,
        head_num=kv_heads,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=layers,
        disable_value_sparse_layer_ids=layers,
        device="cpu",
        start_layer=start_layer,
        end_layer=end_layer,
    )
    return MiniMaxM3Nvfp4KVPool(**kwargs) if nvfp4 else MiniMaxSparseKVPool(**kwargs)


def _draft_pool(kv_heads: int, size: int) -> MHATokenToKVPool:
    # The DSpark draft's KV: bf16, token-major, the target's KV heads and head dim.
    return MHATokenToKVPool(
        size=size,
        page_size=128,
        dtype=torch.bfloat16,
        head_num=kv_heads,
        head_dim=128,
        layer_num=M31_DRAFT_LAYERS,
        device="cpu",
        enable_memory_saver=False,
    )


def _kv_ids(pool, draft=None):
    # The DSpark draft pool registers k0..k4 then v0..v4 after the target entries.
    return build_kv_layer_ids(
        token_to_kv_pool=pool,
        draft_token_to_kv_pool=draft,
        num_draft_entries=2 * M31_DRAFT_LAYERS if draft is not None else 0,
        num_hidden_layers=M31_LAYERS,
        storage=_static_storage(pool),
    )


def _state_kv_args(pool) -> KVArgs:
    kv_args = KVArgs()
    setup_state_kv_args(kv_args, _static_storage(pool))
    return kv_args


class _RecordingKVManager(MooncakeKVManager):
    """Records the transfer blocks instead of moving bytes."""

    def __init__(self, pp_size: int):  # no engine, bootstrap or threads
        self.is_mla_backend = False
        self.is_hybrid_mla_backend = False
        self.enable_custom_mem_pool = False
        self.max_transfer_batch_indices = 0
        self.state_strides_validated = set()
        self.pp_size = pp_size
        self.blocks = []

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        self.blocks.extend(transfer_blocks)
        return 0


class TestMiniMaxPPStagesPairByLayerId(unittest.TestCase):
    """PP4 prefill stages (layers from 0/15/30/45) against a decode peer holding all
    60 layers: KV entries and index-K state pair by global layer id, and the
    DSpark draft band (60-64) lands on the decode peer's draft entries."""

    def _check_kv_pairs(self, *, nvfp4: bool):
        decode = _m31_pool(0, M31_LAYERS, nvfp4=nvfp4, kv_heads=1)
        dst = _kv_ids(decode, _draft_pool(1, 256))
        groups = 1 if nvfp4 else 2  # one page row per layer, or K then V
        self.assertEqual(
            dst,
            list(range(M31_LAYERS)) * groups
            + list(range(M31_LAYERS, M31_LAYERS + M31_DRAFT_LAYERS)) * 2,
        )
        for start, end in PP4_STAGES:
            # Only the last stage hosts the DSpark draft.
            draft = end == M31_LAYERS
            with self.subTest(nvfp4=nvfp4, start=start):
                stage = _m31_pool(start, end, nvfp4=nvfp4)
                src = _kv_ids(stage, _draft_pool(4, 256) if draft else None)
                ptrs = stage.get_sparse_kv_buf_infos()[0]
                self.assertEqual(len(src), len(ptrs) + (10 if draft else 0))
                pairs = build_transfer_entry_pairs(src, dst, len(src), len(dst))
                n = end - start
                expected = [
                    (g * n + k, g * M31_LAYERS + start + k)
                    for g in range(groups)
                    for k in range(n)
                ]
                if draft:
                    band = groups * M31_LAYERS
                    expected += [
                        (groups * n + d, band + d) for d in range(2 * M31_DRAFT_LAYERS)
                    ]
                self.assertEqual(pairs, expected)

    def test_fp8_kv_entries_and_nvfp4_page_rows(self):
        for nvfp4 in (False, True):
            self._check_kv_pairs(nvfp4=nvfp4)

    def test_index_k_state_pairs_and_moves_to_the_stage_layers(self):
        for nvfp4 in (False, True):
            dst_args = _state_kv_args(_m31_pool(0, M31_LAYERS, nvfp4=nvfp4, kv_heads=1))
            self.assertEqual(dst_args.state_types, [StateType.MINIMAX_INDEX_K])
            self.assertEqual(dst_args.state_layer_ids[0], list(range(M31_LAYERS)))
            for start, end in PP4_STAGES:
                with self.subTest(nvfp4=nvfp4, start=start):
                    src_args = _state_kv_args(_m31_pool(start, end, nvfp4=nvfp4))
                    self.assertEqual(src_args.state_types, dst_args.state_types)
                    self.assertEqual(
                        src_args.state_layer_ids[0], list(range(start, end))
                    )
                    # TP-independent index rows: the same item length on both peers.
                    self.assertEqual(
                        src_args.state_item_lens[0],
                        dst_args.state_item_lens[0][start:end],
                    )
                    manager = _RecordingKVManager(pp_size=4)
                    req = SimpleNamespace(
                        mooncake_session_id="session",
                        room=7,
                        dst_state_indices=[[0]],
                    )
                    target = SimpleNamespace(
                        dst_state_data_ptrs=dst_args.state_data_ptrs,
                        dst_state_item_lens=dst_args.state_item_lens,
                        dst_state_dim_per_tensor=dst_args.state_dim_per_tensor,
                        dst_state_layer_ids=dst_args.state_layer_ids,
                        dst_attn_tp_size=1,
                        dst_tp_rank=0,
                    )
                    manager.kv_args = src_args
                    manager.attn_tp_size = 1
                    rc = manager.maybe_send_extra(req, [[1]], None, target)
                    self.assertEqual(rc, 0)
                    item = src_args.state_item_lens[0][0]
                    self.assertEqual(
                        manager.blocks,
                        [
                            (
                                src_args.state_data_ptrs[0][k] + item,
                                dst_args.state_data_ptrs[0][start + k],
                                item,
                            )
                            for k in range(end - start)
                        ],
                    )

    def test_components_follow_the_model_not_the_stage(self):
        """A stage holding only dense layers still registers an (empty) index-K
        component, and a stage holding only sparse layers an empty dense one, so
        both line up with a decode peer that has both."""

        def pool(start, end):
            return MiniMaxSparseKVPool(
                size=8,
                page_size=4,
                dtype=torch.float32,
                head_num=2,
                head_dim=8,
                idx_head_dim=16,
                dense_layer_ids=[0, 1, 2],
                sparse_layer_ids=[3, 4, 5, 6],
                disable_value_sparse_layer_ids=[3, 4, 5, 6],
                device="cpu",
                start_layer=start,
                end_layer=end,
            )

        full = _state_kv_args(pool(0, 7))
        expected_types = [StateType.MINIMAX_INDEX_K, StateType.MINIMAX_DENSE_KV]
        self.assertEqual(full.state_types, expected_types)
        self.assertEqual(full.state_layer_ids, [[3, 4, 5, 6], [0, 1, 2, 0, 1, 2]])
        for (start, end), index_ids, dense_ids in (
            ((0, 3), [], [0, 1, 2, 0, 1, 2]),
            ((3, 7), [3, 4, 5, 6], []),
            ((2, 5), [3, 4], [2, 2]),
        ):
            with self.subTest(start=start):
                stage = _state_kv_args(pool(start, end))
                self.assertEqual(stage.state_types, expected_types)
                self.assertEqual(stage.state_layer_ids, [index_ids, dense_ids])
                self.assertEqual(
                    [len(p) for p in stage.state_data_ptrs],
                    [len(index_ids), len(dense_ids)],
                )


class TestMiniMaxFp8Staging(unittest.TestCase):
    """SGLANG_DISAGG_STAGING_BUFFER on the token-major FP8 MiniMax KV: the
    sparse layers' K/V (and a DSpark draft of the same geometry) are staged, with
    slot ids that pair a PP stage against the full decode layout."""

    def _staging(self, pool, *, draft_dtype=None):
        draft = None
        if draft_dtype is not None:
            draft = MHATokenToKVPool(
                size=256,
                page_size=128,
                dtype=draft_dtype,
                head_num=pool.head_num,
                head_dim=128,
                layer_num=M31_DRAFT_LAYERS,
                device="cpu",
                enable_memory_saver=False,
            )
        return build_staging_slot_metadata(
            kv_layer_ids=_kv_ids(pool, draft=draft),
            num_draft_entries=2 * M31_DRAFT_LAYERS if draft is not None else 0,
            kv_pool=pool,
            draft_kv_pool=draft,
        )

    def _pool(self, start, end, kv_heads):
        layers = list(range(M31_LAYERS))
        return MiniMaxSparseKVPool(
            size=256,
            page_size=128,
            dtype=torch.float8_e4m3fn,
            head_num=kv_heads,
            head_dim=128,
            idx_head_dim=128,
            dense_layer_ids=[],
            sparse_layer_ids=layers,
            disable_value_sparse_layer_ids=layers,
            device="cpu",
            start_layer=start,
            end_layer=end,
        )

    def test_stage_slots_pair_with_the_full_decode(self):
        draft_ids = list(range(M31_LAYERS, M31_LAYERS + M31_DRAFT_LAYERS))
        decode = self._staging(
            self._pool(0, M31_LAYERS, 1), draft_dtype=torch.float8_e4m3fn
        )
        for start, end in PP4_STAGES:
            with self.subTest(start=start):
                draft = end == M31_LAYERS
                stage = self._pool(start, end, 4)
                k_buffers, v_buffers, slot_ids = self._staging(
                    stage, draft_dtype=torch.float8_e4m3fn if draft else None
                )
                local = list(range(start, end)) + (draft_ids if draft else [])
                # Gather order: every K buffer, then every V buffer.
                self.assertEqual(slot_ids, local + local)
                self.assertEqual(
                    k_buffers[0].data_ptr(), stage.get_key_buffer(start).data_ptr()
                )
                self.assertEqual(
                    v_buffers[end - start - 1].data_ptr(),
                    stage.get_value_buffer(end - 1).data_ptr(),
                )
                pairs = build_transfer_entry_pairs(
                    slot_ids, decode[2], len(slot_ids), len(decode[2])
                )
                self.assertEqual(
                    [slot_ids[i] for i, _ in pairs], [decode[2][j] for _, j in pairs]
                )

    def test_bf16_draft_is_rejected(self):
        # Slots are sized by the first buffer: a bf16 draft behind FP8 target K/V
        # would be gathered and scattered at half its size.
        with self.assertRaisesRegex(RuntimeError, "cannot stage this draft KV"):
            self._staging(self._pool(45, 60, 4), draft_dtype=torch.bfloat16)


class _MemcpyEngine:
    """Fake transfer engine: the peers share this process's memory."""

    def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
        for src, dst, length in zip(src_addrs, dst_addrs, lengths):
            ctypes.memmove(dst, src, length)
        return 0


def _fill_randn(tensor: torch.Tensor) -> None:
    tensor.copy_(torch.randn(tensor.shape))


def _m3_pool(*, start: int, end: int, kv_heads: int) -> MiniMaxSparseKVPool:
    """M3 shape with dense layers: dense 0-2, K-only sparse 3-6, token-major K/V."""
    return MiniMaxSparseKVPool(
        size=16,
        page_size=4,
        dtype=torch.float32,
        head_num=kv_heads,
        head_dim=8,
        idx_head_dim=16,
        dense_layer_ids=[0, 1, 2],
        sparse_layer_ids=[3, 4, 5, 6],
        disable_value_sparse_layer_ids=[3, 4, 5, 6],
        device="cpu",
        start_layer=start,
        end_layer=end,
    )


def _state_manager(pool, *, attn_tp_size: int, tp_rank: int, pp_size: int):
    manager = object.__new__(MooncakeKVManager)
    kv_args = _state_kv_args(pool)
    kv_args.engine_rank = tp_rank
    kv_args.page_size = pool.page_size
    kv_args.kv_head_num = pool.head_num
    kv_args.total_kv_head_num = 4
    manager.kv_args = kv_args
    manager.attn_tp_size = attn_tp_size
    manager.pp_size = pp_size
    manager.engine = _MemcpyEngine()
    manager.enable_deferred_decode_kv_release = False
    manager.enable_custom_mem_pool = False
    manager.max_transfer_batch_indices = 0
    manager.is_mla_backend = manager.is_hybrid_mla_backend = False
    manager.state_strides_validated = set()
    manager.request_status = {}
    return manager


class TestMiniMaxStateAcrossAttentionTP(unittest.TestCase):
    """MiniMax state between different attention TP sizes on mooncake: the
    one-head index K reaches every decode rank in full (written once), and the
    dense-layer K/V is head-sliced like token-major main KV."""

    SRC_PAGES = [3, 1]
    DST_PAGES = [2, 0]

    def _send(self, src, decode, *, dst_tp: int, dst_rank: int):
        dst_args = _state_kv_args(decode)
        req = SimpleNamespace(
            mooncake_session_id="session",
            room=7,
            dst_state_indices=[self.DST_PAGES, self.DST_PAGES],
        )
        target = SimpleNamespace(
            dst_state_data_ptrs=dst_args.state_data_ptrs,
            dst_state_item_lens=dst_args.state_item_lens,
            dst_state_dim_per_tensor=dst_args.state_dim_per_tensor,
            dst_state_layer_ids=dst_args.state_layer_ids,
            dst_attn_tp_size=dst_tp,
            dst_tp_rank=dst_rank,
        )
        executor = concurrent.futures.ThreadPoolExecutor(2)
        self.addCleanup(executor.shutdown)
        rc = src.maybe_send_extra(
            req, [self.SRC_PAGES, self.SRC_PAGES], executor, target
        )
        self.assertEqual(rc, 0)

    def _tokens(self, pages):
        return [p * 4 + t for p in pages for t in range(4)]

    def test_tp1_stage_to_tp2_decode(self):
        # PP stage [2, 5): dense layer 2, sparse 3-4.
        stage = _m3_pool(start=2, end=5, kv_heads=4)
        for layer in (2, 3, 4):
            _fill_randn(stage.get_key_buffer(layer))
            _fill_randn(stage.get_value_buffer(layer))
        for layer in (3, 4):
            _fill_randn(stage.get_index_k_buffer(layer))
        manager = _state_manager(stage, attn_tp_size=1, tp_rank=0, pp_size=3)
        src_tok, dst_tok = self._tokens(self.SRC_PAGES), self._tokens(self.DST_PAGES)
        for rank in range(2):
            with self.subTest(rank=rank):
                decode = _m3_pool(start=0, end=7, kv_heads=2)
                self._send(manager, decode, dst_tp=2, dst_rank=rank)
                heads = slice(2 * rank, 2 * rank + 2)
                for get in ("get_key_buffer", "get_value_buffer"):
                    src_buf = getattr(stage, get)(2)
                    dst_buf = getattr(decode, get)(2)
                    self.assertTrue(
                        torch.equal(dst_buf[dst_tok], src_buf[src_tok][:, heads])
                    )
                    # Other dense layers belong to other stages.
                    self.assertFalse(getattr(decode, get)(0).any())
                for layer in (3, 4):
                    self.assertTrue(
                        torch.equal(
                            decode.get_index_k_buffer(layer)[dst_tok],
                            stage.get_index_k_buffer(layer)[src_tok],
                        )
                    )
                self.assertFalse(decode.get_index_k_buffer(5).any())

    def test_tp2_to_tp1_elects_one_index_writer(self):
        source = _m3_pool(start=0, end=7, kv_heads=2)
        for layer in range(7):
            _fill_randn(source.get_key_buffer(layer))
        _fill_randn(source.get_index_k_buffer(3))
        decode = _m3_pool(start=0, end=7, kv_heads=4)
        manager = _state_manager(source, attn_tp_size=2, tp_rank=1, pp_size=1)
        self._send(manager, decode, dst_tp=1, dst_rank=0)
        dst_tok = self._tokens(self.DST_PAGES)
        src_tok = self._tokens(self.SRC_PAGES)
        # Rank 1 writes its two dense heads at head offset 2, and no index K.
        self.assertTrue(
            torch.equal(
                decode.get_key_buffer(0)[dst_tok][:, 2:],
                source.get_key_buffer(0)[src_tok],
            )
        )
        self.assertFalse(decode.get_key_buffer(0)[dst_tok][:, :2].any())
        self.assertFalse(decode.get_index_k_buffer(3).any())


class TestMiniMaxHandshakeAcrossAttentionTP(unittest.TestCase):
    """Only a backend that reslices MiniMax KV and state admits a prefill of
    another attention TP size; the others still fail at the handshake."""

    def _ensure(self, manager_cls):
        manager = object.__new__(manager_cls)
        manager.prefill_info_table = {}
        manager.kv_args = _state_kv_args(_m31_pool(0, M31_LAYERS, nvfp4=False))
        manager.kv_args.page_size = 128
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.attn_tp_size = 4
        manager.attn_cp_size = 1
        manager.kv_layout = None
        manager.dsv41_spec_layout = None
        manager.spec_kv_layout = None
        manager.is_mla_backend = manager.is_hybrid_mla_backend = False
        manager.dcp_size = 1
        manager._resolve_rank_mapping = lambda info: None
        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=4,
            page_size=128,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
        )
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            return CommonKVManager.try_ensure_parallel_info(manager, "prefill:1")

    def test_mooncake_admits_pp4_tp1_prefill_for_tp4_decode(self):
        self.assertTrue(self._ensure(MooncakeKVManager))

    def test_nixl_still_rejects(self):
        with self.assertRaisesRegex(RuntimeError, "matching attention TP sizes"):
            self._ensure(NixlKVManager)


if __name__ == "__main__":
    unittest.main()
