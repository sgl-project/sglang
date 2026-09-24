"""Cross-node placement and scoped logits layout regressions."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.engine import _calculate_rank_ranges
from sglang.srt.layers.logits_processor import _reassemble_tp_lm_head_all_to_all_output
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestParallelConsumers(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _publish(self, **config):
        publish(
            ServerArgs(model_path="dummy", **config),
            role="scheduler",
            ranks=SpawnRanks(world_rank=0),
        )

    def test_rank_ranges_can_describe_another_node(self):
        self._publish(tp_size=4, pp_size=2, nnodes=4, node_rank=0)
        expected = [
            ([0], [0, 1]),
            ([0], [2, 3]),
            ([1], [0, 1]),
            ([1], [2, 3]),
        ]
        for node_rank, (pp, tp) in enumerate(expected):
            with self.subTest(node_rank=node_rank):
                pp_range, tp_range, pp_per_node, tp_per_node = _calculate_rank_ranges(
                    node_rank
                )
                self.assertEqual((list(pp_range), list(tp_range)), (pp, tp))
                self.assertEqual((pp_per_node, tp_per_node), (1, 2))

    def test_logits_reassembly_uses_the_active_tp_scope(self):
        self._publish(tp_size=4)
        source = torch.arange(24).reshape(8, 3)
        with get_parallel().override(tp_size=2, attn_tp_size=2, moe_tp_size=2):
            result = _reassemble_tp_lm_head_all_to_all_output(source)
            torch.testing.assert_close(result, torch.cat(source.chunk(2), dim=1))
        result = _reassemble_tp_lm_head_all_to_all_output(source)
        torch.testing.assert_close(result, torch.cat(source.chunk(4), dim=1))


if __name__ == "__main__":
    unittest.main()
