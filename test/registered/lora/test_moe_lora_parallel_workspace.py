"""CUDA fork/join guards for the MoE LoRA overlap workspace."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace, run_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestMoeLoraParallelWorkspace(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda", torch.cuda.current_device())

    def test_prepare_plan_owns_the_menu_entry_checks(self):
        runner = object.__new__(MoeLoraRunner)
        provider = SimpleNamespace(name="test")
        runner.providers = {"test": provider}
        runner._validate_plan_provider = mock.Mock()
        plan = mock.Mock()

        runner.prepare_plan(plan, provider_name="test")

        runner._validate_plan_provider.assert_called_once_with(plan, provider)

    def test_side_stream_is_capture_safe_and_fully_joined(self):
        workspace = MoeLoraWorkspace()

        source = torch.empty(64, dtype=torch.float32, device=self.device)
        side_output = torch.empty_like(source)
        compute_output = torch.empty_like(source)
        combined = torch.empty_like(source)

        def side() -> None:
            side_output.copy_(source)
            side_output.add_(1)

        def compute() -> torch.Tensor:
            compute_output.copy_(source)
            compute_output.mul_(2)
            return compute_output

        def launch_twice() -> torch.Tensor:
            # A shared runner workspace reuses region events sequentially
            # across layers. Exercise two same-name regions in one graph.
            for _ in range(2):
                result = run_parallel(
                    workspace,
                    name="shared_region",
                    device=self.device,
                    compute=compute,
                    side=side,
                )
            combined.copy_(result)
            combined.add_(side_output)
            return combined

        # Stand in for the CUDA graph warm-up forwards: one eager pass
        # creates the side stream and materializes both event handles.
        source.fill_(3)
        launch_twice()
        torch.cuda.synchronize(self.device)

        side_stream = workspace.side_stream(self.device)
        ready = workspace.event(self.device, "shared_region:ready")
        done = workspace.event(self.device, "shared_region:done")
        self.assertNotEqual(ready.cuda_event, 0)
        self.assertNotEqual(done.cuda_event, 0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_twice()

        for value in (3, 5, 7, 11):
            source.fill_(value)
            graph.replay()
            torch.cuda.synchronize(self.device)
            torch.testing.assert_close(
                combined,
                torch.full_like(combined, 3 * value + 1),
                rtol=0,
                atol=0,
            )

        self.assertIs(workspace.side_stream(self.device), side_stream)
        self.assertIs(workspace.event(self.device, "shared_region:ready"), ready)
        self.assertIs(workspace.event(self.device, "shared_region:done"), done)


if __name__ == "__main__":
    unittest.main()
