# SPDX-License-Identifier: Apache-2.0
"""Prometheus lifecycle, replica ownership and disabled-path regressions."""

import os
import subprocess
import sys
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prometheus_client import CollectorRegistry

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    RequestTracker,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.scheduler import (
    Scheduler,
    _SequentiallyReturnedOutputs,
)
from sglang.multimodal_gen.runtime.observability import metrics as metrics_module
from sglang.multimodal_gen.runtime.observability.metrics import DiffusionMetrics
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils import perf_logger


@pytest.fixture
def metrics():
    registry = CollectorRegistry()
    collector = DiffusionMetrics(role="monolithic", replica="0", registry=registry)
    return collector, registry


@pytest.mark.parametrize("enabled", [False, True])
def test_role_checks_http_port_only_when_exporting_metrics(enabled, monkeypatch):
    args = ServerArgs.__new__(ServerArgs)
    args.disagg_role = RoleType.DENOISER
    args.enable_metrics = enabled
    args.strict_ports = True
    require_port = Mock()
    monkeypatch.setattr(args, "_require_port", require_port)
    args._adjust_network_ports()
    http_checks = [
        call.args for call in require_port.call_args_list if call.args[1] == "HTTP"
    ]
    assert http_checks == ([(args.port, "HTTP")] if enabled else [])


def sample(registry, name, **labels):
    return registry.get_sample_value(
        "sglang:diffusion_" + name, {"role": "monolithic", "replica": "0", **labels}
    )


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_scheduler_counts_original_requests_and_cleans_up(
    metrics, sequential, failure, enabled
):
    collector, registry = metrics
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.metrics = collector if enabled else None
    scheduler._disagg_role = RoleType.MONOLITHIC
    scheduler._disagg_metrics = None
    scheduler.receiver = None
    scheduler.context = Mock()

    class NoScanQueue(deque):
        def __iter__(self):
            raise AssertionError("metrics must not scan the waiting queue")

    scheduler.waiting_queue = NoScanQueue()
    scheduler._running = True
    scheduler._consecutive_error_count = 0
    scheduler._max_consecutive_errors = 1
    scheduler._log_warmup_result = Mock()
    scheduler._log_batch_metrics_summary = Mock()
    scheduler._cleanup_disagg = Mock()
    scheduler.return_result = Mock()
    scheduler.process_received_reqs_with_req_based_warmup = lambda reqs: reqs
    reqs = [Req(sampling_params=SamplingParams(prompt="test")) for _ in range(2)]
    # one multi-output request is still one original scheduler request
    group = [Req(sampling_params=SamplingParams(prompt="group")) for _ in range(3)]
    reqs.append(group)
    scheduler.recv_reqs = lambda: [(None, req) for req in reqs]
    scheduler.get_next_batch_to_run = lambda: [(None, req) for req in reqs]

    def dispatch(items):
        if enabled:
            assert sample(registry, "num_running_reqs") == 3
            assert sample(registry, "num_queue_reqs") == 0
        scheduler._running = False
        if failure and not sequential:
            raise RuntimeError("forward failed")

        def outputs():
            for index in range(len(items)):
                if failure and index == 1:
                    raise RuntimeError("forward failed")
                yield OutputBatch()

        return (
            _SequentiallyReturnedOutputs(outputs()) if sequential else list(outputs())
        )

    scheduler._dispatch_items = dispatch
    scheduler.event_loop()
    if not enabled:
        assert not collector._requests
        assert (
            sample(registry, "requests_total", status="success", is_warmup="false")
            is None
        )
        return
    errors = 2 if sequential else 3
    assert sample(registry, "requests_total", status="success", is_warmup="false") == (
        (1 if sequential else None) if failure else 3
    )
    if failure:
        assert (
            sample(registry, "requests_total", status="error", is_warmup="false")
            == errors
        )
    assert sample(registry, "num_running_reqs") == 0
    assert sample(registry, "num_queue_reqs") == 0
    assert not collector._requests


@pytest.mark.parametrize(
    "terminal", [RequestState.DONE, RequestState.FAILED, RequestState.TIMED_OUT]
)
def test_disagg_lifecycle_counts_once_including_retries(metrics, terminal):
    collector, registry = metrics
    tracker = RequestTracker(collector)
    tracker.submit("request", is_warmup=True)
    for state in (
        RequestState.ENCODER_RUNNING,
        RequestState.ENCODER_DONE,
        RequestState.DENOISING_RUNNING,
        RequestState.DENOISING_WAITING,
        RequestState.DENOISING_RUNNING,
        RequestState.DENOISING_DONE,
        RequestState.DECODER_RUNNING,
        terminal,
    ):
        tracker.transition("request", state)
    tracker.remove("request")
    assert (
        sample(
            registry,
            "requests_total",
            status="success" if terminal == RequestState.DONE else "error",
            is_warmup="true",
        )
        == 1
    )
    assert sample(registry, "queue_time_seconds_count", is_warmup="true") == 1
    assert sample(registry, "num_running_reqs") == 0
    tracker.submit("cancelled")
    tracker.remove("cancelled")
    assert sample(registry, "num_queue_reqs") == 0
    assert sample(registry, "requests_total", status="error", is_warmup="false") == 1


def test_only_replica_leaders_construct_metrics(monkeypatch):
    construct = Mock()
    monkeypatch.setattr(metrics_module, "DiffusionMetrics", construct)
    monkeypatch.setattr(metrics_module, "_metrics", None)
    args = SimpleNamespace(
        num_gpus=4,
        dp_size=2,
        enable_metrics=False,
        disagg_role=RoleType.MONOLITHIC,
        scheduler_endpoint_for=lambda replica: f"tcp://localhost:{5555 + replica}",
    )
    for rank in range(4):
        assert metrics_module.init_metrics(args, rank) is None
    construct.assert_not_called()
    args.enable_metrics = True
    for rank in range(4):
        metrics_module.init_metrics(args, rank)
    assert construct.call_count == 2
    assert [call.kwargs["replica"] for call in construct.call_args_list] == [
        "tcp://localhost:5555",
        "tcp://localhost:5556",
    ]


def test_disabled_metrics_skip_status_collection_and_timing(monkeypatch):
    worker = GPUWorker.__new__(GPUWorker)
    worker.metrics = None
    worker.pipeline = Mock()
    worker._update_lora_metrics()
    worker.pipeline.get_lora_status.assert_not_called()
    monkeypatch.setattr(metrics_module, "_metrics", None)
    monkeypatch.setenv("SGLANG_DIFFUSION_STAGE_LOGGING", "0")
    timer = Mock(side_effect=AssertionError("disabled metrics must not time stages"))
    monkeypatch.setattr(perf_logger.time, "perf_counter", timer)
    with perf_logger.StageProfiler("test", Mock(), None):
        pass
    timer.assert_not_called()


def test_stage_metrics_do_not_synchronize_and_bound_step_labels(metrics, monkeypatch):
    collector, registry = metrics
    monkeypatch.setattr(metrics_module, "_metrics", collector)
    monkeypatch.delenv("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", raising=False)
    device = Mock()
    monkeypatch.setattr(perf_logger.torch, "get_device_module", lambda: device)
    for step in range(3):
        with perf_logger.StageProfiler(f"denoising_step_{step}", Mock(), None):
            pass
    device.synchronize.assert_not_called()
    assert (
        sample(registry, "stage_host_latency_seconds_count", stage="DenoisingStep") == 3
    )


def test_lora_deduplicates_adapters_and_resets_inactive_modules(metrics):
    collector, registry = metrics
    active = {
        "transformer": [{"nicknames": ["a", "b"]}],
        "transformer_2": [{"nicknames": ["a"]}],
    }
    collector.update_lora({"loaded_adapters": ["a", "b"], "active": active})
    assert sample(registry, "lora_active_adapters") == 2
    collector.update_lora({"loaded_adapters": ["a", "b"], "active": {}})
    assert sample(registry, "lora_active_modules") == 0
    assert sample(registry, "lora_module_active", module="transformer") == 0


def test_multiprocess_scrape_keeps_role_and_replica_gauges_separate(tmp_path):
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    worker = """
import sys
from sglang.multimodal_gen.runtime.observability.metrics import DiffusionMetrics
m = DiffusionMetrics(role=sys.argv[1], replica=sys.argv[2])
for i in range(int(sys.argv[3])):
    m.enqueue(i, is_warmup=False)
m.dispatch(0)
m.finish(0, error=False)
"""
    for role, replica, count in [
        ("monolithic", "0", 2),
        ("monolithic", "1", 3),
        ("decoder", "0", 1),
    ]:
        subprocess.run(
            [sys.executable, "-c", worker, role, replica, str(count)],
            env=env,
            check=True,
            timeout=90,
        )
    scrape = """
import asyncio
from types import SimpleNamespace
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest, multiprocess
from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
r = CollectorRegistry()
multiprocess.MultiProcessCollector(r)
assert r.get_sample_value('sglang:diffusion_num_queue_reqs', {'role':'monolithic','replica':'0'}) == 1
assert r.get_sample_value('sglang:diffusion_num_queue_reqs', {'role':'monolithic','replica':'1'}) == 2
assert r.get_sample_value('sglang:diffusion_num_queue_reqs', {'role':'decoder','replica':'0'}) == 0
assert b'sglang:diffusion_requests_total' in generate_latest(r)
args = SimpleNamespace(enable_metrics=True, pipeline_config=SimpleNamespace(
    supports_action_endpoint=lambda: False, supports_openpi_endpoint=lambda: False))
app = create_app(args)
app.state.server_warmup_done = asyncio.Event()
response = TestClient(app).get('/metrics')
assert response.status_code == 200
assert 'sglang:diffusion_requests_total' in response.text
args.enable_metrics = False
assert TestClient(create_app(args)).get('/metrics').status_code == 404
"""
    subprocess.run([sys.executable, "-c", scrape], env=env, check=True, timeout=90)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
