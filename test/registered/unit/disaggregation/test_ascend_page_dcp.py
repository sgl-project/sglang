import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.ascend.conn import (
    AscendKVManager,
    _build_page_interleaved_dcp_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAscendPageDcpPlan(CustomTestCase):
    def test_assigns_whole_pages_to_dcp_ranks(self):
        src = np.array([5, 2, 11, 4], dtype=np.int32)
        dst = np.array([7], dtype=np.int32)

        for rank, expected_src in enumerate(src):
            local_src, local_dst, global_src, global_dst = (
                _build_page_interleaved_dcp_plan(
                    src,
                    dst,
                    page_size=2,
                    dcp_size=4,
                    dcp_rank=rank,
                    src_page_offset=0,
                    decode_prefix_len=0,
                    num_kv_tokens=8,
                )
            )
            np.testing.assert_array_equal(local_src, [expected_src])
            np.testing.assert_array_equal(local_dst, [7])
            np.testing.assert_array_equal(global_src, src)
            np.testing.assert_array_equal(global_dst, [28, 29, 30, 31])

    def test_chunk_offset_crosses_virtual_pages(self):
        args = {
            "src_page_indices": np.array([9, 3], dtype=np.int32),
            "dst_page_indices": np.array([4, 6], dtype=np.int32),
            "page_size": 2,
            "dcp_size": 2,
            "src_page_offset": 1,
            "decode_prefix_len": 4,
            "num_kv_tokens": 4,
        }
        rank0 = _build_page_interleaved_dcp_plan(dcp_rank=0, **args)
        rank1 = _build_page_interleaved_dcp_plan(dcp_rank=1, **args)

        np.testing.assert_array_equal(rank0[0], [3])
        np.testing.assert_array_equal(rank0[1], [6])
        np.testing.assert_array_equal(rank1[0], [9])
        np.testing.assert_array_equal(rank1[1], [4])
        np.testing.assert_array_equal(rank0[2], [9, 3])
        np.testing.assert_array_equal(rank0[3], [9, 12])

    def test_partial_page_and_validation(self):
        plan = _build_page_interleaved_dcp_plan(
            np.array([3], dtype=np.int32),
            np.array([8], dtype=np.int32),
            page_size=4,
            dcp_size=2,
            dcp_rank=0,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=1,
        )
        np.testing.assert_array_equal(plan[0], [3])
        with self.assertRaisesRegex(ValueError, "align"):
            _build_page_interleaved_dcp_plan(
                np.array([3], dtype=np.int32),
                np.array([8], dtype=np.int32),
                page_size=4,
                dcp_size=2,
                dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=4,
                num_kv_tokens=1,
            )

    def test_accepts_cp_filtered_page_subset(self):
        plan = _build_page_interleaved_dcp_plan(
            np.array([3], dtype=np.int32),
            np.array([4, 6], dtype=np.int32),
            page_size=2,
            dcp_size=2,
            dcp_rank=0,
            src_page_offset=2,
            decode_prefix_len=0,
            num_kv_tokens=4,
        )
        np.testing.assert_array_equal(plan[0], [3])
        np.testing.assert_array_equal(plan[1], [6])


class TestAscendPageDcpSend(CustomTestCase):
    def test_sends_local_and_global_entries_without_pack(self):
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

        result = mgr.send_kvcache_dcp(
            "decode",
            np.array([5, 6, 7, 8], dtype=np.int32),
            [10_000, 20_000, 30_000],
            np.array([7], dtype=np.int32),
            dcp_token_item_lens=[10, 4, 2],
            dst_dcp_size=4,
            dst_dcp_rank=2,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=8,
            executor=None,
            dst_layer_ids=[],
            pack_buffer=None,
            dst_kv_item_lens=[20, 32, 16],
        )

        self.assertEqual(result, 0)
        mgr._transfer_data.assert_called_once_with(
            "decode",
            [
                (1_140, 10_140, 20),
                (2_040, 20_224, 32),
                (3_020, 30_112, 16),
            ],
        )

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
