import unittest

import torch

from sglang.srt.speculative.dspark_components.kernels.compact_layout import (
    compact_row_index,
    compact_verify_ids,
)
from sglang.srt.speculative.dspark_components.kernels import qo_indptr as _qo_indptr_mod
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_OLD_QO_INDPTR_KERNEL_IMPL = _qo_indptr_mod._KERNEL_IMPL


def setUpModule():
    _qo_indptr_mod._KERNEL_IMPL = "torch"


def tearDownModule():
    _qo_indptr_mod._KERNEL_IMPL = _OLD_QO_INDPTR_KERNEL_IMPL


class TestDSparkCompactLayout(CustomTestCase):
    def test_row_index_marks_padded_tail_invalid(self):
        req, within, valid = compact_row_index(
            verify_lens=torch.tensor([2, 1], dtype=torch.int32),
            padded_total=5,
            device=torch.device("cpu"),
        )

        self.assertEqual(req.tolist(), [0, 0, 1, 2, 2])
        self.assertEqual(within.tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(valid.tolist(), [True, True, True, False, False])

    def test_verify_ids_padded_graph_keeps_anchor_draft_order_and_zero_padding(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([1, 3, 2], dtype=torch.int32),
            graph_num_tokens=8,
        )
        draft_block_ids = torch.tensor(
            [
                [100, 101, 102, 103],
                [200, 201, 202, 203],
                [300, 301, 302, 303],
            ],
            dtype=torch.int64,
        )
        draft_tokens = torch.tensor(
            [
                [11, 12, 13],
                [21, 22, 23],
                [31, 32, 33],
            ],
            dtype=torch.int64,
        )

        verify_ids = compact_verify_ids(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=torch.device("cpu"),
        )

        self.assertEqual(
            verify_ids.tolist(),
            [
                100,
                200,
                21,
                22,
                300,
                31,
                0,
                0,
            ],
        )

    def test_verify_ids_use_draft_tokens_for_non_anchor_slots(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([4], dtype=torch.int32),
            graph_num_tokens=4,
        )
        draft_block_ids = torch.tensor([[100, 999, 999, 999]], dtype=torch.int64)
        draft_tokens = torch.tensor([[11, 12, 13]], dtype=torch.int64)

        verify_ids = compact_verify_ids(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=torch.device("cpu"),
        )

        self.assertEqual(verify_ids.tolist(), [100, 11, 12, 13])


if __name__ == "__main__":
    unittest.main()
