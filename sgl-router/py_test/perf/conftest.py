from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import pytest
import requests
from py_test.fixtures.ports import find_free_port
from py_test.perf.cluster import (
    PDCluster,
    detect_ib_device,
    ensure_environment_ready,
    launch_disaggregated_cluster,
)
from py_test.shared.genai_bench import BenchmarkStats

logger = logging.getLogger(__name__)

_PD_PERF_MARKER = "perf"
_PYTEST_HTML_ACTIVE = False


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--pd-perf-model-path",
        action="store",
        default=None,
        help="Override the model path used for PD perf benchmarks.",
    )


@dataclass
class RouterHandle:
    policy: str
    url: str
    metrics_port: int
    process: subprocess.Popen

    def stop(self) -> None:
        _terminate(self.process)


@dataclass
class PolicyPerfRecord:
    policy: str
    ttft_mean: float
    e2e_latency_mean: float
    input_throughput_mean: float
    output_throughput_mean: float
    experiment_dir: str


def build_perf_record(
    policy: str, stats: Iterable[BenchmarkStats], experiment_dir: str
) -> PolicyPerfRecord:
    stats_list = list(stats)
    if not stats_list:
        raise AssertionError("No benchmark stats were collected")
    ttft_mean = sum(s.ttft_mean for s in stats_list) / len(stats_list)
    e2e_mean = sum(s.e2e_latency_mean for s in stats_list) / len(stats_list)
    input_mean = sum(s.input_throughput_mean for s in stats_list) / len(stats_list)
    output_mean = sum(s.output_throughput_mean for s in stats_list) / len(stats_list)
    return PolicyPerfRecord(
        policy=policy,
        ttft_mean=ttft_mean,
        e2e_latency_mean=e2e_mean,
        input_throughput_mean=input_mean,
        output_throughput_mean=output_mean,
        experiment_dir=experiment_dir,
    )


def pytest_configure(config) -> None:
    global _PYTEST_HTML_ACTIVE
    config.addinivalue_line(
        "markers", f"{_PD_PERF_MARKER}: mark as PD performance benchmark"
    )
    _PYTEST_HTML_ACTIVE = config.pluginmanager.hasplugin("html")


@pytest.fixture(scope="session")
def perf_model_path(request) -> str:
    option = request.config.getoption("--pd-perf-model-path")
    if option:
        return option
    return os.environ.get(
        "PD_PERF_MODEL_PATH", "/raid/models/meta-llama/Llama-3.1-8B-Instruct"
    )


@pytest.fixture(scope="session")
def perf_environment(perf_model_path: str) -> dict:
    ib_device = detect_ib_device()
    if ib_device is None:
        pytest.skip("No active InfiniBand device detected; skipping PD perf suite")

    try:
        ensure_environment_ready(perf_model_path, require_ib=False)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    return {"ib_device": ib_device}


@pytest.fixture(scope="session")
def pd_perf_cluster(
    perf_model_path: str, perf_environment: dict
) -> Iterable[PDCluster]:
    cluster = launch_disaggregated_cluster(
        model_path=perf_model_path,
        ib_device=perf_environment["ib_device"],
    )
    try:
        yield cluster
    finally:
        cluster.stop()


@pytest.fixture
def pd_router_factory(pd_perf_cluster: PDCluster) -> Callable[[str], RouterHandle]:
    def _launch(policy: str) -> RouterHandle:
        host = "127.0.0.9"
        port = find_free_port()
        metrics_port = find_free_port()
        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--policy",
            policy,
            "--host",
            host,
            "--port",
            str(port),
            "--prometheus-host",
            "127.0.0.1",
            "--prometheus-port",
            str(metrics_port),
            "--log-level",
            "warn",
        ]
        for worker in pd_perf_cluster.prefills:
            cmd += ["--prefill", worker.url, str(worker.bootstrap_port or "none")]
        for worker in pd_perf_cluster.decodes:
            cmd += ["--decode", worker.url]

        env = os.environ.copy()
        env.setdefault("RUST_BACKTRACE", "1")

        logger.info("Starting PD router for policy=%s on %s:%d", policy, host, port)
        proc = subprocess.Popen(cmd, env=env)
        router_url = f"http://{host}:{port}"
        try:
            _wait_router_health(router_url, timeout=120.0)
        except Exception:
            _terminate(proc)
            raise
        return RouterHandle(
            policy=policy, url=router_url, metrics_port=metrics_port, process=proc
        )

    return _launch


@pytest.fixture
def router_smoke_checker(perf_model_path: str) -> Callable[[RouterHandle], None]:
    def _check(router: RouterHandle) -> None:
        _run_router_smoke(router, perf_model_path)

    return _check


def _wait_router_health(base_url: str, timeout: float) -> None:
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                resp = session.get(f"{base_url}/health", timeout=5)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    raise TimeoutError(f"Router at {base_url} failed health check within {timeout}s")


def _terminate(proc: subprocess.Popen, timeout: float = 60.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    deadline = time.time() + timeout
    while proc.poll() is None and time.time() < deadline:
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()


def _run_router_smoke(router: RouterHandle, model_path: str) -> None:
    headers = {"Authorization": "Bearer test-token"}
    chat_payload = {
        "model": model_path,
        "messages": [
            {"role": "user", "content": "Write a short haiku about GPUs"},
        ],
        "stream": False,
        "max_tokens": 64,
    }
    resp = requests.post(
        f"{router.url}/v1/chat/completions",
        headers=headers,
        json=chat_payload,
        timeout=180,
    )
    resp.raise_for_status()
    if "choices" not in resp.json():
        raise AssertionError("Chat completion response missing choices field")

    stream_payload = {
        "model": model_path,
        "messages": [
            {"role": "user", "content": "Count from 1 to 3"},
        ],
        "stream": True,
        "max_tokens": 16,
    }
    with requests.post(
        f"{router.url}/v1/chat/completions",
        headers=headers,
        json=stream_payload,
        timeout=180,
        stream=True,
    ) as stream_resp:
        stream_resp.raise_for_status()
        saw_data = False
        for line in stream_resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                saw_data = True
                break
        if not saw_data:
            raise AssertionError("Streaming response did not emit any data chunks")


def _format_perf_summary(records: Iterable[PolicyPerfRecord]) -> str:
    lines = [
        "Policy | TTFT (s) | E2E (s) | Input TP (tok/s) | Output TP (tok/s)",
        "------ | -------- | ------- | ---------------- | -----------------",
    ]
    for record in records:
        lines.append(
            f"{record.policy} | {record.ttft_mean:.2f} | {record.e2e_latency_mean:.2f} | "
            f"{record.input_throughput_mean:.0f} | {record.output_throughput_mean:.0f}"
        )
    return "\n".join(lines)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when != "call":
        return
    records = getattr(item, "_pd_perf_records", None)
    if not records:
        return
    summary = _format_perf_summary(records)
    report.sections.append(("pd-perf", summary))
    if _PYTEST_HTML_ACTIVE:
        setattr(report, "pd_perf_summary", summary)
