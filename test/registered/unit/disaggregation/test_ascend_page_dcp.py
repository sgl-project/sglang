import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAscendPageDcpSend(CustomTestCase):
    def test_rejects_incorrect_destination_geometry(self):
        mgr = object.__new__(AscendKVManager)
        mgr.kv_args = SimpleNamespace(
            kv_buf_groups=3,
            kv_data_ptrs=[1_000, 2_000, 3_000],
            kv_item_lens=[20, 8, 4],
            page_size=2,
            mla_compression_ratios=None,
        )
        mgr._dcp_remote_decode_layout = [False, True, True]
        mgr._transfer_data = Mock(return_value=0)

        with self.assertRaisesRegex(RuntimeError, "destination geometry"):
            mgr.send_kvcache_dcp(
                "decode",
                np.array([5], dtype=np.int32),
                [10_000, 20_000, 30_000],
                np.array([7], dtype=np.int32),
                dcp_token_item_lens=[10, 4, 2],
                dst_dcp_size=4,
                dst_dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=2,
                executor=None,
                dst_layer_ids=[],
                dst_kv_item_lens=[20, 8, 16],
            )


if __name__ == "__main__":
    unittest.main()
