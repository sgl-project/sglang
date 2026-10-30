import gzip
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.executors import pipeline_executor
from sglang.multimodal_gen.runtime.utils import profiler as profiler_module


@pytest.fixture(autouse=True)
def cpu_profiler(monkeypatch, tmp_path):
    # Exercise the real CPU profiler even on accelerator CI workers.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(profiler_module.current_platform, "is_npu", lambda: False)
    monkeypatch.setattr(
        profiler_module.envs, "SGLANG_DIFFUSION_TORCH_PROFILER_DIR", str(tmp_path)
    )
    monkeypatch.setattr(pipeline_executor, "get_world_rank", lambda: 0)
    for name in ("musa", "xpu"):
        device_module = getattr(torch, name, None)
        if device_module is not None:
            monkeypatch.setattr(device_module, "is_available", lambda: False)
    yield
    profiler = profiler_module.SGLDiffusionProfiler.get_instance()
    if profiler is not None:
        profiler.stop(export_trace=False)


def _payload(name):
    with torch.profiler.record_function(f"payload_{name}"):
        (torch.arange(16.0) + 1).sum()


def _recorded_payloads(folder):
    paths = list(folder.glob("*.trace.json.gz"))
    assert len(paths) == 1
    with gzip.open(paths[0], "rt") as stream:
        events = json.load(stream)["traceEvents"]
    return [
        event["name"].removeprefix("payload_")
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name", "").startswith("payload_")
    ]


class _ProfilerTestExecutor(pipeline_executor.PipelineExecutor):
    def __init__(self):
        self.component_residency_manager = Mock()

    def execute(self, stages, batch, server_args):
        raise NotImplementedError


def _request(requested, total, full_profile=False):
    return SimpleNamespace(
        profile=True,
        is_warmup=False,
        request_id="window",
        profile_all_stages=full_profile,
        num_profiled_timesteps=requested,
        num_inference_steps=total,
    )


def _run_stage(executor, role, callback):
    stage = SimpleNamespace(
        role_affinity=role,
        _active_component_stage_name=lambda: role.value,
        component_uses=lambda *_: (),
    )
    server_args = SimpleNamespace(use_fsdp_inference=False)
    executor.run_stage_with_context(stage, None, server_args, lambda *_: callback())
    executor._step_stage_profiler()


@pytest.mark.parametrize(
    "requested,total,expected",
    [
        (1, 8, ["1"]),
        (2, 8, ["1", "2"]),
        (3, 3, ["0", "1", "2"]),
        (5, 3, ["0", "1", "2"]),
        (-1, 3, ["0", "1", "2"]),
        (1, None, ["1"]),
    ],
)
def test_requested_steps_match_exported_trace(tmp_path, requested, total, expected):
    executor = _ProfilerTestExecutor()
    stopped_on_last_step = False
    with executor.profile_execution(_request(requested, total)):
        profiler = profiler_module.SGLDiffusionProfiler.get_instance()
        _run_stage(executor, RoleType.ENCODER, lambda: _payload("preprocess"))

        def denoise():
            nonlocal stopped_on_last_step
            for step in range(total or 8):
                _payload(step)
                profiler.step_denoising_step()
                if step == int(expected[-1]):
                    stopped_on_last_step = profiler.has_stopped

        _run_stage(executor, RoleType.DENOISER, denoise)
        _run_stage(executor, RoleType.DECODER, lambda: _payload("decode"))
    # Extra notifications after an early stop must not restart or rewrite it.
    profiler.step_denoising_step()
    profiler.stop()
    assert _recorded_payloads(tmp_path) == expected
    assert stopped_on_last_step
    assert profiler_module.SGLDiffusionProfiler.get_instance() is None


def test_early_request_end_exports_only_completed_active_steps(tmp_path):
    executor = _ProfilerTestExecutor()
    with executor.profile_execution(_request(3, 8)):
        profiler = profiler_module.SGLDiffusionProfiler.get_instance()
        _run_stage(executor, RoleType.ENCODER, lambda: _payload("preprocess"))

        def denoise():
            for step in range(2):
                _payload(step)
                profiler.step_denoising_step()
            assert not profiler.has_stopped

        _run_stage(executor, RoleType.DENOISER, denoise)
        _run_stage(executor, RoleType.DECODER, lambda: _payload("decode"))
    assert _recorded_payloads(tmp_path) == ["1"]


def test_full_stages_does_not_use_denoising_step_limit(tmp_path):
    executor = _ProfilerTestExecutor()
    with executor.profile_execution(_request(1, 2, full_profile=True)):
        profiler = profiler_module.SGLDiffusionProfiler.get_instance()
        _run_stage(executor, RoleType.ENCODER, lambda: _payload("encode"))

        def denoise():
            for step in range(2):
                _payload(f"denoise{step}")
                profiler.step_denoising_step()
                assert not profiler.has_stopped

        _run_stage(executor, RoleType.DENOISER, denoise)
        _run_stage(executor, RoleType.DECODER, lambda: _payload("decode"))
        assert not profiler.has_stopped
    profiler.step_stage()
    assert _recorded_payloads(tmp_path) == ["encode", "denoise0", "denoise1", "decode"]


def test_request_cancelled_before_denoising_never_starts_profiler(tmp_path):
    executor = _ProfilerTestExecutor()
    with pytest.raises(RuntimeError, match="cancelled"):
        with executor.profile_execution(_request(-1, 2)):
            profiler = profiler_module.SGLDiffusionProfiler.get_instance()
            _run_stage(executor, RoleType.ENCODER, lambda: _payload("preprocess"))
            raise RuntimeError("cancelled")
    profiler.stop()
    assert not list(tmp_path.glob("*.trace.json.gz"))
    assert profiler.has_stopped
    assert not profiler.has_started
    profiler.start()
    profiler.step_denoising_step()
    assert profiler_module.SGLDiffusionProfiler.get_instance() is None
