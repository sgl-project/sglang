"""CPU-only tests for small PrefillCudaGraphRunner helpers."""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPrefillCudaGraphRunnerHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer")
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        cls.runner_cls = PrefillCudaGraphRunner

    def _runner_for_model(self, model):
        runner = object.__new__(self.runner_cls)
        runner.model_runner = SimpleNamespace(model=model)
        return runner

    def test_layer_model_positions_use_mrope_when_outer_model_enabled(self):
        positions = torch.arange(4)
        mrope_positions = torch.arange(12).reshape(3, 4)
        forward_batch = SimpleNamespace(
            positions=positions, mrope_positions=mrope_positions
        )
        runner = self._runner_for_model(SimpleNamespace(is_mrope_enabled=True))

        self.assertIs(runner._get_layer_model_positions(forward_batch), mrope_positions)

    def test_layer_model_positions_use_mrope_when_language_model_enabled(self):
        positions = torch.arange(4)
        mrope_positions = torch.arange(12).reshape(3, 4)
        forward_batch = SimpleNamespace(
            positions=positions, mrope_positions=mrope_positions
        )
        runner = self._runner_for_model(
            SimpleNamespace(
                is_mrope_enabled=False,
                language_model=SimpleNamespace(is_mrope_enabled=True),
            )
        )

        self.assertIs(runner._get_layer_model_positions(forward_batch), mrope_positions)

    def test_layer_model_positions_keep_positions_without_mrope_model(self):
        positions = torch.arange(4)
        mrope_positions = torch.arange(12).reshape(3, 4)
        forward_batch = SimpleNamespace(
            positions=positions, mrope_positions=mrope_positions
        )
        runner = self._runner_for_model(SimpleNamespace(is_mrope_enabled=False))

        self.assertIs(runner._get_layer_model_positions(forward_batch), positions)

    def test_layer_model_positions_keep_positions_without_mrope_tensor(self):
        positions = torch.arange(4)
        forward_batch = SimpleNamespace(positions=positions, mrope_positions=None)
        runner = self._runner_for_model(SimpleNamespace(is_mrope_enabled=True))

        self.assertIs(runner._get_layer_model_positions(forward_batch), positions)


if __name__ == "__main__":
    unittest.main()
