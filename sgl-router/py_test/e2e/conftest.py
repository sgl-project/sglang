import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional
from urllib.parse import urlparse

import pytest
import requests

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
)

logger = logging.getLogger(__name__)


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _parse_url(base_url: str) -> tuple[str, str]:
    """Parse a base URL and return (host, port) as strings.

    This is more robust than simple string splitting and supports different schemes
    and URL shapes like trailing paths.
    """
    parsed = urlparse(base_url)
    return parsed.hostname or "127.0.0.1", (
        str(parsed.port) if parsed.port is not None else ""
    )


def _wait_router_health(base_url: str, timeout: float) -> None:
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                r = session.get(f"{base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    raise TimeoutError("Router failed to become healthy in time")


def _popen_launch_router(
    model: str,
    base_url: str,
    dp_size: int,
    timeout: float,
    policy: str = "cache_aware",
) -> subprocess.Popen:
    host, port = _parse_url(base_url)

    prom_port = _find_available_port()

    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--dp",
        str(dp_size),
        "--router-policy",
        policy,
        "--allow-auto-truncate",
        "--router-prometheus-port",
        str(prom_port),
        "--router-prometheus-host",
        "127.0.0.1",
    ]

    proc = subprocess.Popen(cmd)
    _wait_router_health(base_url, timeout)
    return proc


def _popen_launch_worker(
    model: str,
    base_url: str,
    *,
    dp_size: int | None = None,
    api_key: str | None = None,
    base_gpu_id: int | None = 0,
) -> subprocess.Popen:
    host, port = _parse_url(base_url)

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--base-gpu-id",
        str(base_gpu_id or 0),
    ]
    if dp_size is not None:
        cmd += ["--dp-size", str(dp_size)]
    if api_key is not None:
        cmd += ["--api-key", api_key]
    return subprocess.Popen(cmd)


def _popen_launch_router_only(
    base_url: str,
    policy: str = "round_robin",
    timeout: float = 120.0,
    *,
    dp_aware: bool = False,
    api_key: str | None = None,
) -> subprocess.Popen:
    host, port = _parse_url(base_url)

    prom_port = _find_available_port()
    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        host,
        "--port",
        port,
        "--policy",
        policy,
    ]
    if dp_aware:
        cmd += ["--dp-aware"]
    if api_key is not None:
        cmd += ["--api-key", api_key]
    cmd += [
        "--prometheus-port",
        str(prom_port),
        "--prometheus-host",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(cmd)
    _wait_router_health(base_url, timeout)
    return proc


def _terminate(proc: subprocess.Popen, timeout: float = 120) -> None:
    if proc is None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)


def _which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception as e:
        logger.warning("shutil.which(%r) failed: %s", cmd, e)
        return None


def _graceful_stop_popen(p: subprocess.Popen) -> None:
    if p is None:
        return
    try:
        if p.poll() is None:
            p.terminate()
            for _ in range(5):
                if p.poll() is not None:
                    break
                time.sleep(1)
            if p.poll() is None:
                p.kill()
    except Exception as e:
        logger.warning("Exception during graceful stop of popen: %s", e)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _graceful_stop_pid(pid: int) -> None:
    try:
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            for _ in range(5):
                if not _pid_alive(pid):
                    break
                time.sleep(1)
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass


def _graceful_stop_any(obj) -> None:
    try:
        if isinstance(obj, subprocess.Popen):
            _graceful_stop_popen(obj)
            return
        if isinstance(obj, int):
            _graceful_stop_pid(obj)
            return
        proc_obj = getattr(obj, "proc", None)
        if isinstance(proc_obj, subprocess.Popen):
            _graceful_stop_popen(proc_obj)
    except Exception:
        pass


@pytest.fixture(scope="session")
def genai_bench_runner() -> Callable[..., None]:
    """Provide a callable to run genai-bench and validate metrics.

    Usage in tests:
      def test(..., genai_bench_runner):
          genai_bench_runner(router_url=..., model_path=..., experiment_folder=...)
    """

    def _run(
        *,
        router_url: str,
        model_path: str,
        experiment_folder: str,
        timeout_sec: int | None = None,
        thresholds: dict | None = None,
        extra_env: dict | None = None,
        num_concurrency: int = 32,
        traffic_scenario: str = "D(4000,100)",
        max_requests_per_run: int | None = None,
        clean_experiment: bool = True,
        kill_procs: list | None = None,
        drain_delay_sec: int = 6,
    ) -> None:
        cli = _which("genai-bench")
        if not cli:
            pytest.fail(
                "genai-bench CLI not found; please install it to run benchmarks"
            )

        # Clean previous experiment folder under current working directory
        if clean_experiment:
            exp_dir = Path.cwd() / experiment_folder
            if exp_dir.exists():
                shutil.rmtree(exp_dir, ignore_errors=True)

        # Default requests per run if not provided
        mrr = (
            max_requests_per_run
            if max_requests_per_run is not None
            else num_concurrency * 3
        )

        cmd = [
            cli,
            "benchmark",
            "--api-backend",
            "openai",
            "--api-base",
            router_url,
            "--api-key",
            "dummy-token",
            "--api-model-name",
            model_path,
            "--model-tokenizer",
            model_path,
            "--task",
            "text-to-text",
            "--num-concurrency",
            str(num_concurrency),
            "--traffic-scenario",
            traffic_scenario,
            "--max-requests-per-run",
            str(mrr),
            "--max-time-per-run",
            "2",
            "--experiment-folder-name",
            experiment_folder,
            "--experiment-base-dir",
            str(Path.cwd()),
        ]

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        to = timeout_sec or int(os.environ.get("GENAI_BENCH_TEST_TIMEOUT", "120"))
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout = stderr = ""
        rc = None
        try:
            try:
                stdout, stderr = proc.communicate(timeout=to)
            except subprocess.TimeoutExpired:
                # Simple: kill the CLI process if it doesn't exit in time
                try:
                    proc.kill()
                except Exception:
                    pass
                stdout, stderr = proc.communicate()
            rc = proc.returncode

            # Prefer exact path under cwd; fallback to rglob search
            base = Path.cwd()
            direct = base / experiment_folder
            candidates = [direct] if direct.is_dir() else []
            if not candidates:
                for p in base.rglob(experiment_folder):
                    if p.is_dir() and p.name == experiment_folder:
                        candidates = [p]
                        break
            if not candidates:
                raise AssertionError(
                    "Benchmark failed: experiment folder not found: "
                    f"{experiment_folder}\nExit code: {rc}\nSTDOUT (tail):\n{stdout[-1000:]}\nSTDERR (tail):\n{stderr[-1000:]}"
                )
            actual_folder = candidates[0]

            json_files = [
                p
                for p in actual_folder.rglob("*.json")
                if "experiment_metadata" not in p.name
            ]
            if not json_files:
                raise AssertionError(
                    "Benchmark failed: no JSON results found\n"
                    f"Exit code: {rc}\nSTDOUT (tail):\n{stdout[-1000:]}\nSTDERR (tail):\n{stderr[-1000:]}"
                )

            th = thresholds  # None means "log only", no validation

            for jf in json_files:
                with jf.open("r") as f:
                    data = json.load(f)
                stats = data.get("aggregated_metrics", {}).get("stats", {})
            ttft_mean = float(stats.get("ttft", {}).get("mean", float("inf")))
            e2e_latency_mean = float(
                stats.get("e2e_latency", {}).get("mean", float("inf"))
            )
            input_tp_mean = float(stats.get("input_throughput", {}).get("mean", 0.0))
            output_tp_mean = float(stats.get("output_throughput", {}).get("mean", 0.0))

            logger.info(
                "genai-bench[%s] %s ttft_mean=%.3fs e2e_latency_mean=%.3fs input_tp_mean=%.1f tok/s output_tp_mean=%.1f tok/s",
                experiment_folder,
                jf.name,
                ttft_mean,
                e2e_latency_mean,
                input_tp_mean,
                output_tp_mean,
            )

            if th is not None:
                assert (
                    ttft_mean <= th["ttft_mean_max"]
                ), f"TTFT validation failed: {ttft_mean} > {th['ttft_mean_max']} (file={jf.name})"
                assert (
                    e2e_latency_mean <= th["e2e_latency_mean_max"]
                ), f"E2E latency validation failed: {e2e_latency_mean} > {th['e2e_latency_mean_max']} (file={jf.name})"
                assert (
                    input_tp_mean >= th["input_throughput_mean_min"]
                ), f"Input throughput validation failed: {input_tp_mean} < {th['input_throughput_mean_min']} (file={jf.name})"
                assert (
                    output_tp_mean >= th["output_throughput_mean_min"]
                ), f"Output throughput validation failed: {output_tp_mean} < {th['output_throughput_mean_min']} (file={jf.name})"

        finally:
            # Always attempt to stop workers to avoid resource leakage
            if kill_procs:
                # Give router/workers a small grace period to finish any last drains
                if drain_delay_sec > 0:
                    try:
                        time.sleep(drain_delay_sec)
                    except Exception:
                        pass
                for p in kill_procs:
                    _graceful_stop_any(p)
                try:
                    time.sleep(2)
                except Exception:
                    pass

    return _run


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: mark as end-to-end test")


@pytest.fixture(scope="session")
def e2e_model() -> str:
    # Always use the default test model
    return DEFAULT_MODEL_NAME_FOR_TEST


@pytest.fixture
def e2e_router(e2e_model: str):
    # Keep this available but tests below use router-only to avoid GPU contention
    base_url = DEFAULT_URL_FOR_TEST
    proc = _popen_launch_router(
        e2e_model, base_url, dp_size=2, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    )
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)


@pytest.fixture
def e2e_router_only_rr():
    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _popen_launch_router_only(base_url, policy="round_robin")
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)


@pytest.fixture(scope="session")
def e2e_primary_worker(e2e_model: str):
    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _popen_launch_worker(e2e_model, base_url)
    # Router health gate will handle worker readiness
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)


@pytest.fixture
def e2e_router_only_rr_dp_aware_api():
    """Router-only with dp-aware enabled and an API key."""
    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    api_key = "secret"
    proc = _popen_launch_router_only(
        base_url, policy="round_robin", timeout=180.0, dp_aware=True, api_key=api_key
    )
    try:
        yield SimpleNamespace(proc=proc, url=base_url, api_key=api_key)
    finally:
        _terminate(proc)


@pytest.fixture
def e2e_worker_dp2_api(e2e_model: str, e2e_router_only_rr_dp_aware_api):
    """Worker with dp-size=2 and the same API key as the dp-aware router."""
    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    api_key = e2e_router_only_rr_dp_aware_api.api_key
    proc = _popen_launch_worker(e2e_model, base_url, dp_size=2, api_key=api_key)
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)


@pytest.fixture(scope="session")
def e2e_two_workers_dp2(e2e_model: str):
    """Launch two workers, each with dp_size=2, mapped to GPUs [0,1] and [2,3]."""
    workers = []
    try:
        # Worker A on GPUs 0-1
        port_a = _find_available_port()
        url_a = f"http://127.0.0.1:{port_a}"
        proc_a = _popen_launch_worker(e2e_model, url_a, dp_size=2, base_gpu_id=0)
        workers.append(SimpleNamespace(proc=proc_a, url=url_a))

        # Worker B on GPUs 2-3
        port_b = _find_available_port()
        url_b = f"http://127.0.0.1:{port_b}"
        proc_b = _popen_launch_worker(e2e_model, url_b, dp_size=2, base_gpu_id=2)
        workers.append(SimpleNamespace(proc=proc_b, url=url_b))

        yield workers
    finally:
        for w in workers:
            _terminate(w.proc)
