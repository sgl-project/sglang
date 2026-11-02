"""Tests for multimodal tensor transport topology detection."""

import unittest

from sglang.srt.multimodal.transport import determine_tensor_transport_mode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestTensorTransportMode(CustomTestCase):
    def test_transport_mode_uses_published_node_topology(self):
        cases = (
            (1, None, "cuda_ipc"),
            (1, "127.0.0.1:20000", "cuda_ipc"),
            (2, None, "default"),
            (2, "10.0.0.1:20000", "default"),
        )

        for nnodes, dist_init_addr, expected in cases:
            with self.subTest(nnodes=nnodes, dist_init_addr=dist_init_addr):
                with get_context().override_server_args(
                    nnodes=nnodes,
                    dist_init_addr=dist_init_addr,
                ):
                    self.assertEqual(determine_tensor_transport_mode(), expected)

    def test_multi_node_pipeline_keeps_first_stage_local(self):
        # Only the first pipeline stage reads tokenizer output; it stays on
        # node 0 as long as each TP group fits on one node.
        cases = (
            (dict(nnodes=8, pp_size=8), "cuda_ipc"),
            (dict(nnodes=2, pp_size=8), "cuda_ipc"),
            (dict(nnodes=4, pp_size=2), "default"),
            (dict(nnodes=2, pp_size=2, dp_size=2), "default"),
            (dict(nnodes=8, pp_size=8, use_ray=True), "default"),
        )

        for overrides, expected in cases:
            with self.subTest(**overrides):
                with get_context().override_server_args(**overrides):
                    self.assertEqual(determine_tensor_transport_mode(), expected)


if __name__ == "__main__":
    unittest.main()
