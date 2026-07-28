import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.utils.common import (
    get_cuda_graph_batch_size_alignment,
    get_cuda_graph_max_batch_size,
    get_eager_max_batch_size,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDCPBatchCapacity(CustomTestCase):
    @patch("sglang.srt.utils.common.require_gathered_buffer", return_value=True)
    @patch("sglang.srt.utils.common.get_parallel")
    def test_cuda_graph_capacity_aligns_to_attention_tp(
        self, get_parallel, _require_gathered
    ):
        get_parallel.return_value = SimpleNamespace(attn_tp_size=4, attn_cp_size=1)
        args = SimpleNamespace(enable_two_batch_overlap=False)

        self.assertEqual(get_cuda_graph_batch_size_alignment(args), 4)
        self.assertEqual(get_cuda_graph_max_batch_size(args, 65), 68)

    @patch("sglang.srt.utils.common.require_gathered_buffer", return_value=False)
    @patch("sglang.srt.utils.common.get_parallel")
    def test_cuda_graph_capacity_combines_tbo_and_attention_cp(
        self, get_parallel, _require_gathered
    ):
        get_parallel.return_value = SimpleNamespace(attn_tp_size=1, attn_cp_size=4)
        args = SimpleNamespace(enable_two_batch_overlap=True)

        self.assertEqual(get_cuda_graph_batch_size_alignment(args), 8)
        self.assertEqual(get_cuda_graph_max_batch_size(args, 65), 72)

    @patch("sglang.srt.utils.common.require_mlp_sync", return_value=False)
    def test_eager_capacity_without_mlp_sync_is_unchanged(self, _require_mlp_sync):
        args = SimpleNamespace()
        self.assertEqual(get_eager_max_batch_size(args, 65), 65)


if __name__ == "__main__":
    unittest.main()
