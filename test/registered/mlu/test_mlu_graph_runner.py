import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_mlu  # noqa: F401

from sglang.srt.hardware_backend.mlu.graph_runner import (
    DecodeCudaGraphRunner,
    MLUGraphRunner,
)
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-mlu")


class TestMLUGraphRunner(CustomTestCase):
    def setUp(self):
        torch.mlu.set_device(0)
        self.device = torch.device("mlu", 0)

    def test_graph_runner_captures_and_replays_mlu_graph(self):
        with patch.object(DecodeCudaGraphRunner, "__init__", return_value=None):
            runner = MLUGraphRunner(SimpleNamespace())

        self.assertEqual(runner.attr_name, {})
        self.assertEqual(runner.attr_type, {})
        self.assertEqual(runner._cache_loc_dtype(), torch.int32)
        self.assertEqual(runner._position_dtype(), torch.int32)

        graph = runner._create_device_graph()
        self.assertIsInstance(graph, torch.mlu.MLUGraph)

        x = torch.ones(4, dtype=torch.float32, device=self.device)
        y = torch.empty_like(x)
        pool = torch.mlu.graph_pool_handle()
        stream = torch.mlu.Stream()

        def run_once():
            y.copy_(x + 1)
            return y

        out = runner._capture_graph(graph, pool, stream, run_once)
        x.fill_(3)
        graph.replay()
        torch.mlu.synchronize()

        self.assertIs(out, y)
        torch.testing.assert_close(out.cpu(), torch.full((4,), 4.0))


if __name__ == "__main__":
    unittest.main()
