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


if __name__ == "__main__":
    unittest.main()
