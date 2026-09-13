import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.runtime import launch_server as launcher
from sglang.multimodal_gen.test.server import test_server_utils as utils
from sglang.multimodal_gen.test.server.test_server_utils import PerformanceValidator
from sglang.multimodal_gen.test.server.testcase_configs import (
    PerformanceSummary,
    ScenarioConfig,
    ToleranceConfig,
)


@pytest.fixture
def validator(monkeypatch):
    monkeypatch.setenv("SGLANG_GEN_BASELINE", "0")
    scenario = ScenarioConfig.from_dict(
        {
            "stages_ms": {},
            "denoise_step_ms": {},
            "expected_e2e_ms": 1000,
            "expected_avg_denoise_ms": 0,
            "expected_median_denoise_ms": 0,
            "expected_load_ms": 4000,
        }
    )
    return PerformanceValidator(scenario, ToleranceConfig(0.25, 0, 0, 0, 0), [])


def test_slow_loading_fails_even_when_inference_passes(validator):
    summary = PerformanceSummary(1000, 0, 0, {}, [], {}, {}, load_time_ms=6000)
    validator.validate_e2e(summary)
    with pytest.raises(AssertionError, match="Load-inclusive E2E"):
        validator.validate_load_inclusive_e2e(summary)


def test_fast_loading_cannot_hide_inference_regression(validator):
    summary = PerformanceSummary(2000, 0, 0, {}, [], {}, {}, load_time_ms=1000)
    validator.validate_load_inclusive_e2e(summary)
    with pytest.raises(AssertionError, match="E2E Latency"):
        validator.validate_e2e(summary)


@pytest.mark.parametrize("duration", [None, 0, -1, float("nan"), float("inf")])
def test_missing_or_invalid_load_duration_fails(validator, duration):
    summary = PerformanceSummary(1000, 0, 0, {}, [], {}, {}, load_time_ms=duration)
    with pytest.raises(AssertionError, match="Load duration missing or invalid"):
        validator.validate_load_inclusive_e2e(summary)


@pytest.mark.parametrize("duration", [None, 0, -1, float("nan"), float("inf")])
def test_missing_or_invalid_load_baseline_fails(validator, duration):
    validator.scenario.expected_load_ms = duration
    summary = PerformanceSummary(1000, 0, 0, {}, [], {}, {}, load_time_ms=4000)
    with pytest.raises(AssertionError, match="Load baseline missing or invalid"):
        validator.validate_load_inclusive_e2e(summary)


def test_repeated_requests_each_include_one_load(validator):
    for inference_ms in (1000, 900, 950):
        summary = PerformanceSummary(
            inference_ms, 0, 0, {}, [], {}, {}, load_time_ms=4000
        )
        validator.validate_e2e(summary)
        validator.validate_load_inclusive_e2e(summary)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("warmup_seconds", [0, 120])
def test_server_load_clock_excludes_warmup(
    monkeypatch, tmp_path, workers, warmup_seconds, validator
):
    clock = [1_000_000_000]
    output = io.StringIO()
    monkeypatch.setattr(utils.time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(utils.current_platform, "is_hip", lambda: False)
    monkeypatch.setattr(utils.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        utils, "prepare_perf_log", lambda: (tmp_path, tmp_path / "perf.jsonl")
    )
    monkeypatch.setattr(launcher, "configure_logger", Mock())
    monkeypatch.setattr(launcher, "logger", Mock())
    launcher.logger.info.side_effect = lambda message, *args: output.write(
        (message % args) + "\n"
    )

    def ready():
        clock[0] += 1_000_000_000
        return {"status": "ready"}

    monkeypatch.setattr(
        launcher.mp, "Pipe", lambda **kwargs: (Mock(recv=ready), Mock())
    )
    monkeypatch.setattr(launcher.mp, "Process", Mock())
    monkeypatch.setattr(launcher, "shutdown_scheduler_processes", Mock())

    def warmup(args):
        clock[0] += warmup_seconds * 1_000_000_000

    monkeypatch.setattr(launcher, "launch_http_server_only", warmup)
    args = SimpleNamespace(
        num_gpus=workers,
        nnodes=1,
        node_rank=0,
        master_port=1234,
        webui=False,
        pipeline_config=SimpleNamespace(supports_action_endpoint=lambda: False),
    )

    def spawn(*unused_args, **unused_kwargs):
        launcher.launch_server(args)
        return SimpleNamespace(pid=1234, stdout=io.StringIO(output.getvalue()))

    monkeypatch.setattr(utils.subprocess, "Popen", spawn)
    manager = utils.ServerManager("test", 1234)
    monkeypatch.setattr(manager, "_wait_for_ready", Mock())
    context = manager.start()
    context._log_thread.join(timeout=5)
    assert not context._log_thread.is_alive()
    assert context.load_time_ms == workers * 1000
    summary = PerformanceSummary(
        1000, 0, 0, {}, [], {}, {}, load_time_ms=context.load_time_ms
    )
    validator.validate_e2e(summary)
    validator.validate_load_inclusive_e2e(summary)
