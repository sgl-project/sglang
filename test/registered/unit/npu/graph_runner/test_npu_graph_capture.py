"""Minimal real-device capture/replay coverage for ``torch.npu.NPUGraph``."""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

# Capturing an NPUGraph needs one Ascend card rather than a CPU mock runner.
register_npu_ci(est_time=1, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=1, suite="nightly-1-npu-a3", nightly=True)


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.is_available(), "requires an Ascend NPU"
)
class TestNPUGraphCapture(CustomTestCase):
    def test_replay_reads_updated_static_input(self):
        static_input = torch.tensor([1.0, 2.0], device="npu")
        torch.npu.synchronize()
        for _ in range(2):
            static_output = static_input * 2
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            static_output = static_input * 2

        static_input.copy_(torch.tensor([3.0, 4.0], device="npu"))
        graph.replay()
        torch.npu.synchronize()

        torch.testing.assert_close(
            static_output.cpu(), torch.tensor([6.0, 8.0]), rtol=0, atol=0
        )

    def test_engram_gate_capture_and_replay(self):
        from sglang.srt.layers.engram import engram_gate

        x = torch.tensor([[[1.0, 2.0], [-1.0, 2.0]]])
        kv = torch.tensor([[1.0, 1.0, 1.0, -1.0, 3.0, -2.0]])
        weight = torch.ones(2, 2)
        static_x, npu_kv, npu_weight = (t.to("npu") for t in (x, kv, weight))
        engram_gate(static_x, npu_kv, npu_weight, npu_weight, 1e-6, 1e-6)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = engram_gate(static_x, npu_kv, npu_weight, npu_weight, 1e-6, 1e-6)

        static_x.copy_((-x).to("npu"))
        graph.replay()
        torch.npu.synchronize()
        expected = engram_gate(-x, kv, weight, weight, 1e-6, 1e-6)
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
