import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base.conn import KVArgs
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
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
            setup_state_kv_args(kv_args, pool)

    def test_decode_rejects_prefill_with_different_attention_tp(self):
        # Main page rows are head-major and the index state is per rank, so
        # neither can be resliced by head; reject before any transfer.
        kv_args = KVArgs()
        kv_args.page_size = 4
        setup_state_kv_args(kv_args, _make_k_only_pool())
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
