from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)


class _RecordingExecutor(PipelineExecutor):
    def __init__(self):
        super().__init__(server_args=SimpleNamespace())
        self.single_inference_mode = None
        self.group_inference_mode = None

    def execute(self, stages, batch, server_args):
        self.single_inference_mode = torch.is_inference_mode_enabled()
        return batch

    def execute_group(self, stages, batches, server_args):
        self.group_inference_mode = torch.is_inference_mode_enabled()
        return batches


def _batch():
    return SimpleNamespace(profile=False, is_warmup=False)


def test_execute_with_profiling_uses_inference_mode():
    executor = _RecordingExecutor()

    with torch.inference_mode(False):
        executor.execute_with_profiling([], _batch(), SimpleNamespace())

    assert executor.single_inference_mode is True


def test_execute_group_with_profiling_uses_inference_mode():
    executor = _RecordingExecutor()

    with torch.inference_mode(False):
        executor.execute_group_with_profiling([], [_batch(), _batch()], SimpleNamespace())

    assert executor.group_inference_mode is True
