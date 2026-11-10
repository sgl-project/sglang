"""NPU scheduling delegates to sgl-kernel-npu; device coverage needs an NPU."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.speculative.dspark import dspark_schedule as schedule
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("cuda_alias", [False, True])
def test_npu_dispatch_uses_existing_kernel_with_or_without_cuda_alias(cuda_alias):
    confidence = Mock(device=SimpleNamespace(type="npu"))
    cfg = DSparkScheduleConfig(gamma=8)
    result = object()
    kernel = Mock(return_value=result)
    # Mock only the external kernel boundary, exercising execute() and triton().
    sample = ModuleType("sgl_kernel_npu.operators.sample")
    sample.schedule_verify_lens_topk_triton = kernel
    modules = {
        "sgl_kernel_npu": ModuleType("sgl_kernel_npu"),
        "sgl_kernel_npu.operators": ModuleType("sgl_kernel_npu.operators"),
        "sgl_kernel_npu.operators.sample": sample,
    }
    with (
        patch.dict(sys.modules, modules),
        patch.object(schedule, "_is_npu", True),
        patch.object(schedule, "inputs_on_cuda", return_value=cuda_alias) as predicate,
        patch.object(schedule, "schedule_verify_lens_topk_triton") as cuda_kernel,
        patch.object(schedule.ScheduleVerifyLensTopk, "torch") as reference,
    ):
        assert (
            schedule.ScheduleVerifyLensTopk.execute(
                confidence=confidence, budget=3, cfg=cfg
            )
            is result
        )
        kernel.assert_called_once_with(confidence=confidence, budget=3, cfg=cfg)
        predicate.assert_not_called()
        cuda_kernel.assert_not_called()
        reference.assert_not_called()


@pytest.mark.parametrize("cuda", [False, True])
def test_existing_cpu_and_cuda_dispatch_unchanged(cuda):
    confidence = torch.ones(2, 8)
    cfg = DSparkScheduleConfig(gamma=8)
    with (
        patch.object(schedule, "_is_npu", False),
        patch.object(schedule, "inputs_on_cuda", return_value=cuda),
        patch.object(schedule, "schedule_verify_lens_topk", return_value="cpu") as cpu,
        patch.object(
            schedule, "schedule_verify_lens_topk_triton", return_value="cuda"
        ) as cuda_kernel,
    ):
        assert schedule.ScheduleVerifyLensTopk.execute(
            confidence=confidence, budget=3, cfg=cfg
        ) == ("cuda" if cuda else "cpu")
        selected, unused = (cuda_kernel, cpu) if cuda else (cpu, cuda_kernel)
        selected.assert_called_once_with(confidence=confidence, budget=3, cfg=cfg)
        unused.assert_not_called()


_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="requires an Ascend NPU")
@pytest.mark.parametrize("budget", [2, 100])
def test_existing_npu_kernel_capture_replay_reads_updated_confidence(budget):
    cfg = DSparkScheduleConfig(gamma=3)
    static_input = torch.ones(2, 3, device="npu")
    for _ in range(2):
        schedule.ScheduleVerifyLensTopk.execute(
            confidence=static_input, budget=budget, cfg=cfg
        )
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        static_output = schedule.ScheduleVerifyLensTopk.execute(
            confidence=static_input, budget=budget, cfg=cfg
        )
    for values in (
        [[0.9, 0.8, 0.7], [0.2, 0.1, 0.1]],
        [[0.2, 0.1, 0.1], [0.9, 0.8, 0.7]],
        [[1.0] * 3] * 2,
        [[0.0] * 3] * 2,
    ):
        confidence = torch.tensor(values)
        static_input.copy_(confidence)
        graph.replay()
        torch.npu.synchronize()
        reference = schedule.schedule_verify_lens_topk(
            confidence=confidence, budget=budget, cfg=cfg
        )
        torch.testing.assert_close(static_output.cpu(), reference, rtol=0, atol=0)
