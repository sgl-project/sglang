"""Preserve upstream scheduling dispatch; real NPU execution needs a device."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.speculative.dspark import dspark_schedule as schedule
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_native_npu_dispatch_uses_torch_reference():
    # Exercise the real inputs_on_cuda predicate without an NPU runtime. A
    # native NPU tensor is a Tensor with is_cuda=False. CUDA-alias deployments
    # intentionally retain upstream's dispatch rather than adding an override.
    confidence = Mock(
        spec=torch.Tensor, device=SimpleNamespace(type="npu"), is_cuda=False
    )
    cfg = DSparkScheduleConfig(gamma=8)
    result = object()
    with (
        patch.object(schedule, "_is_npu", True),
        patch.object(schedule.ScheduleVerifyLensTopk, "triton") as kernel,
        patch.object(
            schedule.ScheduleVerifyLensTopk, "torch", return_value=result
        ) as reference,
    ):
        assert (
            schedule.ScheduleVerifyLensTopk.execute(
                confidence=confidence, budget=3, cfg=cfg
            )
            is result
        )
        reference.assert_called_once_with(confidence=confidence, budget=3, cfg=cfg)
        kernel.assert_not_called()


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
