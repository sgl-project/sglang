import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
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


if __name__ == "__main__":
    unittest.main()
