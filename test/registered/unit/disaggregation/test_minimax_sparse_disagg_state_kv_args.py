import unittest

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
    def test_setup_state_kv_args_single_minimax_component(self):
        pool = _make_k_only_pool()
        kv_args = KVArgs()
        setup_state_kv_args(kv_args, pool)
        self.assertEqual(kv_args.state_types, [StateType.MINIMAX_INDEX_K])
        self.assertEqual(len(kv_args.state_data_ptrs), 1)
        self.assertEqual(len(kv_args.state_data_ptrs[0]), pool.index_k_pool.layer_num)
        self.assertEqual(len(kv_args.state_item_lens[0]), pool.index_k_pool.layer_num)

    def test_index_kv_pool_raises(self):
        pool = _make_kv_pool()
        self.assertIsNotNone(pool.index_kv_pool)
        kv_args = KVArgs()
        with self.assertRaises(NotImplementedError):
            setup_state_kv_args(kv_args, pool)


if __name__ == "__main__":
    unittest.main()
