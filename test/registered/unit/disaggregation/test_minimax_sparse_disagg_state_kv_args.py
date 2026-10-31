import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
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
    def test_pp_stage_publishes_global_layer_ids(self):
        # Layers 0-1 dense, 2-5 sparse; the second of two PP stages owns 3..5.
        def make_pool(start_layer, end_layer):
            return MiniMaxSparseKVPool(
                size=8,
                page_size=4,
                dtype=torch.float32,
                head_num=2,
                head_dim=8,
                idx_head_dim=16,
                dense_layer_ids=[0, 1],
                sparse_layer_ids=[2, 3, 4, 5],
                disable_value_sparse_layer_ids=[2, 3, 4, 5],
                device="cpu",
                start_layer=start_layer,
                end_layer=end_layer,
            )

        for (start, end), main_ids, index_ids, dense_ids in (
            ((0, 6), [2, 3, 4, 5] * 2, [2, 3, 4, 5], [0, 1] * 2),
            ((1, 3), [2] * 2, [2], [1] * 2),
            ((3, 6), [3, 4, 5] * 2, [3, 4, 5], None),
        ):
            with self.subTest(start=start, end=end):
                pool = make_pool(start, end)
                storage = _static_storage(pool)
                kv_args = KVArgs()
                setup_state_kv_args(kv_args, storage)
                self.assertEqual(
                    list(storage.regions(storage.owner).layer_ids), main_ids
                )
                self.assertEqual(
                    len(main_ids), len(storage.regions(storage.owner).ptrs)
                )
                expected_types = [StateType.MINIMAX_INDEX_K]
                expected_ids = [index_ids]
                if dense_ids is not None:
                    expected_types.append(StateType.MINIMAX_DENSE_KV)
                    expected_ids.append(dense_ids)
                self.assertEqual(kv_args.state_types, expected_types)
                self.assertEqual(kv_args.state_layer_ids, expected_ids)
                for ptrs, ids in zip(kv_args.state_data_ptrs, expected_ids):
                    self.assertEqual(len(ptrs), len(ids))

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


if __name__ == "__main__":
    unittest.main()
