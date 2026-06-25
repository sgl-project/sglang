import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_mlu  # noqa: F401

from sglang.srt.hardware_backend.mlu.graph_runner import (
    DecodeCudaGraphRunner,
    MLUCudaGraphBackend,
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

        x = torch.ones(4, dtype=torch.float32, device=self.device)
        y = torch.empty_like(x)
        stream = torch.mlu.Stream()
        tp_group = SimpleNamespace(barrier=lambda: None)
        runner.device_module = torch.mlu
        runner.model_runner = SimpleNamespace(tp_group=tp_group)
        backend = MLUCudaGraphBackend(runner)

        def run_once():
            y.copy_(x + 1)
            return y

        shape_key = "decode-bs-4"
        with backend.capture_session(stream):
            backend.capture_one(
                shape_key,
                run_once,
                capture_inputs={"x": x},
            )

        self.assertIsInstance(backend._graphs[shape_key], torch.mlu.MLUGraph)
        x.fill_(3)
        out = backend.replay(shape_key, static_forward_batch=None)
        torch.mlu.synchronize()

        self.assertIs(out, y)
        torch.testing.assert_close(out.cpu(), torch.full((4,), 4.0))


if __name__ == "__main__":
    unittest.main()
