import unittest

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.models import minimax_m3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
    def test_setup_state_kv_args_minimax_components(self):
        pool = _make_k_only_pool()
        kv_args = KVArgs()
        setup_state_kv_args(kv_args, pool)
        self.assertEqual(
            kv_args.state_types,
            [StateType.MINIMAX_INDEX_K, StateType.MINIMAX_DENSE_KV],
        )
        self.assertEqual(len(kv_args.state_data_ptrs), 2)
        self.assertEqual(len(kv_args.state_data_ptrs[0]), pool.index_k_pool.layer_num)
        self.assertEqual(len(kv_args.state_item_lens[0]), pool.index_k_pool.layer_num)
        self.assertEqual(len(kv_args.state_data_ptrs[1]), 6)
        self.assertEqual(len(kv_args.state_item_lens[1]), 6)

    def test_tc_piecewise_positions_match_local_pp_hidden_tokens(self):
        normalize = getattr(
            minimax_m3, "_normalize_qknorm_rope_positions_for_cuda_graph", None
        )
        self.assertIsNotNone(normalize)
        if normalize is None:
            return

        positions = torch.arange(3584)
        hidden_states = torch.zeros((3372, 3))
        normalized = normalize(positions, hidden_states)

        self.assertEqual(normalized.shape, (3372,))
        torch.testing.assert_close(normalized, positions[:3372])
        equal_positions = torch.arange(3372)
        self.assertIs(normalize(equal_positions, hidden_states), equal_positions)

    def test_index_kv_pool_raises(self):
        pool = _make_kv_pool()
        self.assertIsNotNone(pool.index_kv_pool)
        kv_args = KVArgs()
        with self.assertRaises(NotImplementedError):
            setup_state_kv_args(kv_args, pool)


if __name__ == "__main__":
    unittest.main()
