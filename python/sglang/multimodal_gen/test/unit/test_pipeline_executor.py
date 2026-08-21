import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
)
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


class _TestServerArgs(SimpleNamespace):
    def should_cpu_offload_component(self, component_name):
        return self.component_modes.get(component_name) == COMPONENT_OFFLOAD


def _server_args(**overrides):
    values = {
        "use_fsdp_inference": False,
        "component_modes": {},
    }
    values.update(overrides)
    return _TestServerArgs(**values)


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


class _ComponentStage:
    def __init__(self, *component_names):
        self.component_names = component_names

    @staticmethod
    def _active_component_stage_name():
        return "FakeStage"

    def component_uses(self, server_args, stage_name=None):
        return [
            SimpleNamespace(component_name=component_name)
            for component_name in self.component_names
        ]


def test_execute_with_profiling_uses_inference_tensor_platform(monkeypatch):
    monkeypatch.setattr(pipeline_executor, "current_platform", _InferenceTensorPlatform)
    executor = _RecordingExecutor()

    with torch.inference_mode(False):
        executor.execute_with_profiling([], _batch(), _server_args())

    assert executor.single_inference_mode is True
    assert executor.single_grad_enabled is False


def test_execute_with_profiling_uses_platform_inference_mode(monkeypatch):
    monkeypatch.setattr(pipeline_executor, "current_platform", _NoGradPlatform)
    executor = _RecordingExecutor()

    with torch.inference_mode(False):
        executor.execute_with_profiling([], _batch(), _server_args())

    assert executor.single_inference_mode is False
    assert executor.single_grad_enabled is False


def test_execute_group_with_profiling_uses_platform_inference_mode(monkeypatch):
    monkeypatch.setattr(pipeline_executor, "current_platform", _NoGradPlatform)
    executor = _RecordingExecutor()

    with torch.inference_mode(False):
        executor.execute_group_with_profiling([], [_batch(), _batch()], _server_args())

    assert executor.group_inference_mode is False
    assert executor.group_grad_enabled is False


def test_group_payload_is_forwarded_to_component_residency_manager():
    executor = _RecordingExecutor()
    executor.component_residency_manager = Mock()
    batches = [_batch(), _batch()]
    server_args = _server_args()

    executor.begin_component_residency_request([], batches, server_args)

    executor.component_residency_manager.begin_request.assert_called_once_with(
        [], batches, server_args
    )


@pytest.mark.parametrize(
    ("server_args", "component_names"),
    [
        (_server_args(use_fsdp_inference=True), ("transformer",)),
        (
            _server_args(component_modes={"transformer": COMPONENT_OFFLOAD}),
            ("transformer",),
        ),
        (
            _server_args(component_modes={"text_encoder": COMPONENT_OFFLOAD}),
            ("text_encoder",),
        ),
        (
            _server_args(component_modes={"image_encoder": COMPONENT_OFFLOAD}),
            ("image_encoder",),
        ),
        (_server_args(component_modes={"vae": COMPONENT_OFFLOAD}), ("vae",)),
    ],
)
def test_stage_context_preserves_version_counters_when_needed(
    server_args, component_names
):
    stage = _ComponentStage(*component_names)

    with torch.inference_mode():
        with PipelineExecutor._stage_execution_context(stage, server_args):
            tensor = torch.ones(1)

            assert torch.is_inference_mode_enabled() is False
            assert torch.is_grad_enabled() is False
            assert tensor._version == 0


@pytest.mark.parametrize(
    ("server_args", "component_names"),
    [
        (
            _server_args(component_modes={"transformer": LAYERWISE_OFFLOAD}),
            ("transformer",),
        ),
        (
            _server_args(component_modes={"text_encoder": LAYERWISE_OFFLOAD}),
            ("text_encoder",),
        ),
    ],
)
def test_stage_context_allows_layerwise_inference_tensor_mode(
    server_args, component_names
):
    stage = _ComponentStage(*component_names)

    with torch.inference_mode():
        with PipelineExecutor._stage_execution_context(stage, server_args):
            assert torch.is_inference_mode_enabled() is True

    assert torch.is_inference_mode_enabled() is False


def test_npu_platform_inference_mode_preserves_version_counters():
    with torch.inference_mode(False), NPUPlatformBase.inference_mode():
        tensor = torch.ones(1)

    assert tensor._version == 0
    assert torch.is_inference_mode_enabled() is False
