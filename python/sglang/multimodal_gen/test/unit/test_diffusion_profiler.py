import gzip
import json

import pytest
import torch

from sglang.multimodal_gen.runtime.utils import profiler as profiler_module


@pytest.fixture(autouse=True)
def cpu_profiler(monkeypatch):
    # Exercise the real CPU profiler even on accelerator CI workers.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(profiler_module.current_platform, "is_npu", lambda: False)
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
    profiler = profiler_module.SGLDiffusionProfiler(
        request_id="window",
        num_steps=requested,
        num_inference_steps=total,
        log_dir=str(tmp_path),
    )
    stopped_on_last_step = False
    for step in range(total or 8):
        _payload(step)
        profiler.step_denoising_step()
        if step == int(expected[-1]):
            stopped_on_last_step = profiler.has_stopped
    profiler.stop()
    # Extra notifications after an early stop must not restart or rewrite it.
    profiler.step_denoising_step()
    profiler.stop()
    assert _recorded_payloads(tmp_path) == expected
    assert stopped_on_last_step
    assert profiler_module.SGLDiffusionProfiler.get_instance() is None


def test_early_request_end_exports_only_completed_active_steps(tmp_path):
    profiler = profiler_module.SGLDiffusionProfiler(
        request_id="partial", num_steps=3, num_inference_steps=8, log_dir=str(tmp_path)
    )
    for step in range(2):
        _payload(step)
        profiler.step_denoising_step()
    assert not profiler.has_stopped
    profiler.stop()
    assert _recorded_payloads(tmp_path) == ["1"]


def test_full_stages_does_not_use_denoising_step_limit(tmp_path):
    profiler = profiler_module.SGLDiffusionProfiler(
        request_id="full", full_profile=True, num_steps=1, log_dir=str(tmp_path)
    )
    stages = ["encode", "denoise0", "denoise1", "decode"]
    for stage in stages:
        _payload(stage)
        profiler.step_denoising_step()
        profiler.step_stage()
        assert not profiler.has_stopped
    profiler.stop()
    profiler.step_stage()
    assert _recorded_payloads(tmp_path) == stages
