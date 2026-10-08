import unittest
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeCustomMemPoolBatch(CustomTestCase):
    def test_generic_intra_node_nvlink_combines_all_layers(self):
        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = True
        manager.custom_mem_pool_type = "INTRA_NODE_NVLINK"
        manager.max_transfer_batch_indices = 0
        manager.pp_size = 1
        manager._transfer_data = MagicMock(return_value=0)
        executor = MagicMock()

        layer_count = 43
        src_ptrs = [100_000 + i * 10_000 for i in range(layer_count)]
        dst_ptrs = [200_000 + i * 10_000 for i in range(layer_count)]
        item_lens = [100] * layer_count

        ret = manager._send_kvcache_generic(
            mooncake_session_id="session",
            src_data_ptrs=src_ptrs,
            dst_data_ptrs=dst_ptrs,
            item_lens=item_lens,
            prefill_data_indices=np.array([1, 2, 5], dtype=np.int32),
            dst_data_indices=np.array([11, 12, 15], dtype=np.int32),
            executor=executor,
        )

        self.assertEqual(ret, 0)
        executor.submit.assert_not_called()
        manager._transfer_data.assert_called_once()
        session_id, blocks = manager._transfer_data.call_args.args
        self.assertEqual(session_id, "session")
        self.assertEqual(len(blocks), layer_count * 2)
        self.assertEqual(
            blocks[:2],
            [
                (src_ptrs[0] + 100, dst_ptrs[0] + 1_100, 200),
                (src_ptrs[0] + 500, dst_ptrs[0] + 1_500, 100),
            ],
        )
        self.assertEqual(
            blocks[-2:],
            [
                (src_ptrs[-1] + 100, dst_ptrs[-1] + 1_100, 200),
                (src_ptrs[-1] + 500, dst_ptrs[-1] + 1_500, 100),
            ],
        )


if __name__ == "__main__":
    unittest.main()
