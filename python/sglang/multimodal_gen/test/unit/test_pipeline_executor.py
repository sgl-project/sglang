import contextlib
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.executors import pipeline_executor
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase


class _RecordingExecutor(PipelineExecutor):
    def __init__(self):
        super().__init__(server_args=SimpleNamespace())
        self.single_inference_mode = None
        self.group_inference_mode = None
        self.single_grad_enabled = None
        self.group_grad_enabled = None

    def execute(self, stages, batch, server_args):
        self.single_inference_mode = torch.is_inference_mode_enabled()
        self.single_grad_enabled = torch.is_grad_enabled()
        return batch

    def execute_group(self, stages, batches, server_args):
        self.group_inference_mode = torch.is_inference_mode_enabled()
        self.group_grad_enabled = torch.is_grad_enabled()
        return batches


def _batch():
    return SimpleNamespace(profile=False, is_warmup=False)


class _NoGradPlatform:
    @classmethod
    @contextlib.contextmanager
    def inference_mode(cls):
        with torch.no_grad():
            yield


class _InferenceTensorPlatform:
    @classmethod
    def inference_mode(cls):
        return torch.inference_mode(mode=True)


def test_execute_with_profiling_uses_inference_tensor_platform(monkeypatch):
    monkeypatch.setattr(pipeline_executor, "current_platform", _InferenceTensorPlatform)
    executor = _RecordingExecutor()

    with torch.inference_mode(False):
        executor.execute_with_profiling([], _batch(), SimpleNamespace())

    assert executor.single_inference_mode is True
    assert executor.single_grad_enabled is False


def test_execute_with_profiling_uses_platform_inference_mode(monkeypatch):
    monkeypatch.setattr(pipeline_executor, "current_platform", _NoGradPlatform)
    executor = _RecordingExecutor()

    with torch.inference_mode(False):
        executor.execute_with_profiling([], _batch(), SimpleNamespace())

    assert executor.single_inference_mode is False
    assert executor.single_grad_enabled is False


def test_execute_group_with_profiling_uses_platform_inference_mode(monkeypatch):
    monkeypatch.setattr(pipeline_executor, "current_platform", _NoGradPlatform)
    executor = _RecordingExecutor()

    with torch.inference_mode(False):
        executor.execute_group_with_profiling(
            [], [_batch(), _batch()], SimpleNamespace()
        )

    assert executor.group_inference_mode is False
    assert executor.group_grad_enabled is False


def test_npu_platform_inference_mode_preserves_version_counters():
    with torch.inference_mode(False), NPUPlatformBase.inference_mode():
        tensor = torch.ones(1)

    assert tensor._version == 0
    assert torch.is_inference_mode_enabled() is False
