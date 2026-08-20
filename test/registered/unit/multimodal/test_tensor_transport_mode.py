"""Tests for multimodal tensor transport topology detection."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.mm_utils import determine_tensor_transport_mode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTensorTransportMode(CustomTestCase):
    def test_single_node_rendezvous_address_keeps_local_transport(self):
        server_args = SimpleNamespace(
            nnodes=1,
            tp_size=2,
            dist_init_addr="127.0.0.1:20000",
        )

        self.assertEqual(determine_tensor_transport_mode(server_args), "cuda_ipc")

    def test_multi_node_uses_default_transport_without_address_heuristic(self):
        server_args = SimpleNamespace(
            nnodes=2,
            dist_init_addr=None,
        )

        self.assertEqual(determine_tensor_transport_mode(server_args), "default")


if __name__ == "__main__":
    unittest.main()
