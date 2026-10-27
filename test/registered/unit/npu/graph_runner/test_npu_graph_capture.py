"""Minimal real-device capture/replay coverage for ``torch.npu.NPUGraph``."""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.is_available(), "requires an Ascend NPU"
)
class TestNPUGraphCapture(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
