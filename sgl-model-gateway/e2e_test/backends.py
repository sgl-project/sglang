"""Backend configurations for E2E tests.

This module defines the available backends for E2E testing:
- grpc: Local gRPC workers with SGLang router
- http: Local HTTP workers with SGLang router
- openai: OpenAI API backend
- xai: xAI API backend

Each backend configuration specifies:
- model: Model path or name
- launcher: Function to launch the backend
- launcher_kwargs: Arguments for the launcher
- needs_workers: Whether local GPU workers are needed
- api_key_env: Environment variable for API key (if needed)
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import requests

if TYPE_CHECKING:
    import openai

from infra.model_specs import _resolve_model_path

logger = logging.getLogger(__name__)


# Default ports for each backend type (can be overridden)
DEFAULT_PORTS = {
    "grpc": 30030,
    "grpc_harmony": 30031,
    "http": 30020,
    "openai": 30010,
    "xai": 30011,
    "oracle_store": 30040,
}

# Prometheus port offset from main port
PROMETHEUS_PORT_OFFSET = 1000


def get_open_port() -> int:
    """Get an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def kill_process_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    """Kill a process and all its children."""
    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        parent.send_signal(sig)
    except ImportError:
        # Fallback if psutil not available
        os.kill(pid, sig)
    except Exception as e:
        logger.warning("Failed to kill process tree for PID %d: %s", pid, e)


def wait_for_health(
    url: str,
    timeout: float = 60,
    api_key: str | None = None,
) -> None:
    """Wait for a server's /health endpoint to return 200."""
    start = time.time()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", headers=headers, timeout=5)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)

    raise TimeoutError(f"Server at {url} did not become healthy within {timeout}s")


def wait_for_workers_ready(
    router_url: str,
    expected_workers: int,
    timeout: float = 300,
    api_key: str | None = None,
) -> None:
    """Wait for router to have all workers connected."""
    start = time.time()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{router_url}/workers", headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("total", 0) >= expected_workers:
                    logger.info(
                        "All %d workers connected after %.1fs",
                        expected_workers,
                        time.time() - start,
                    )
                    return
        except requests.RequestException:
            pass
        time.sleep(2)

    raise TimeoutError(
        f"Router at {router_url} did not get {expected_workers} workers within {timeout}s"
    )


@dataclass
class ClusterInfo:
    """Information about a running cluster."""

    base_url: str
    router_process: subprocess.Popen
    worker_processes: list[subprocess.Popen]
    model: str
    backend: str

    def shutdown(self) -> None:
        """Shutdown the cluster."""
        # Kill router first
        if self.router_process.poll() is None:
            kill_process_tree(self.router_process.pid)

        # Kill workers
        for proc in self.worker_processes:
            if proc.poll() is None:
                kill_process_tree(proc.pid)


def launch_grpc_cluster(
    model: str,
    base_url: str | None = None,
    *,
    num_workers: int = 1,
    tp_size: int = 1,
    policy: str = "round_robin",
    api_key: str | None = None,
    worker_args: list[str] | None = None,
    router_args: list[str] | None = None,
    timeout: float = 300,
    show_output: bool | None = None,
) -> ClusterInfo:
    """Launch gRPC workers and router.

    Args:
        model: Model path
        base_url: Base URL for router (auto-assigns port if None)
        num_workers: Number of workers to launch
        tp_size: Tensor parallelism size
        policy: Routing policy
        api_key: Optional API key for router auth
        worker_args: Additional worker arguments
        router_args: Additional router arguments
        timeout: Startup timeout in seconds
        show_output: Show subprocess output (default: SHOW_ROUTER_LOGS env var)

    Returns:
        ClusterInfo with running processes
    """
    if show_output is None:
        show_output = os.environ.get("SHOW_ROUTER_LOGS", "0") == "1"

    # Determine router port
    if base_url:
        router_port = int(base_url.split(":")[-1])
    else:
        router_port = get_open_port()
        base_url = f"http://127.0.0.1:{router_port}"

    logger.info("Launching gRPC cluster: %d workers, tp=%d", num_workers, tp_size)

    # Launch workers
    workers = []
    worker_urls = []

    for i in range(num_workers):
        worker_port = get_open_port()
        worker_url = f"grpc://127.0.0.1:{worker_port}"
        worker_urls.append(worker_url)

        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--grpc-mode",
            "--mem-fraction-static",
            "0.8",
            "--log-level",
            "warning",
        ]

        if tp_size > 1:
            cmd.extend(["--tp-size", str(tp_size)])

        if worker_args:
            cmd.extend(worker_args)

        logger.info("Starting worker %d on port %d", i + 1, worker_port)

        proc = subprocess.Popen(
            cmd,
            stdout=None if show_output else subprocess.PIPE,
            stderr=None if show_output else subprocess.PIPE,
            start_new_session=True,
        )
        workers.append(proc)

    # Wait for workers to initialize
    logger.info("Waiting for workers to initialize (20s)...")
    time.sleep(20)

    # Verify workers are alive
    for i, worker in enumerate(workers):
        if worker.poll() is not None:
            # Cleanup
            for w in workers:
                try:
                    kill_process_tree(w.pid)
                except Exception:
                    pass
            raise RuntimeError(f"Worker {i + 1} died during startup")

    # Launch router
    router_cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--prometheus-port",
        str(router_port + PROMETHEUS_PORT_OFFSET),
        "--policy",
        policy,
        "--model-path",
        model,
        "--log-level",
        "warn",
        "--worker-urls",
        *worker_urls,
    ]

    if api_key:
        router_cmd.extend(["--api-key", api_key])

    if router_args:
        router_cmd.extend(router_args)

    logger.info("Starting router on port %d", router_port)

    router_proc = subprocess.Popen(
        router_cmd,
        stdout=None if show_output else subprocess.PIPE,
        stderr=None if show_output else subprocess.PIPE,
        start_new_session=True,
    )

    # Wait for router to be ready with all workers
    try:
        wait_for_workers_ready(base_url, num_workers, timeout=timeout, api_key=api_key)
    except TimeoutError:
        # Cleanup on failure
        kill_process_tree(router_proc.pid)
        for w in workers:
            kill_process_tree(w.pid)
        raise

    logger.info("gRPC cluster ready at %s with %d workers", base_url, num_workers)

    return ClusterInfo(
        base_url=base_url,
        router_process=router_proc,
        worker_processes=workers,
        model=model,
        backend="grpc",
    )


def launch_openai_router(
    backend: str,  # "openai" or "xai"
    base_url: str | None = None,
    *,
    history_backend: str = "memory",
    router_args: list[str] | None = None,
    timeout: float = 60,
    show_output: bool | None = None,
) -> ClusterInfo:
    """Launch router with OpenAI/xAI backend.

    Args:
        backend: "openai" or "xai"
        base_url: Base URL for router (auto-assigns port if None)
        history_backend: "memory" or "oracle"
        router_args: Additional router arguments
        timeout: Startup timeout in seconds
        show_output: Show subprocess output

    Returns:
        ClusterInfo with running router
    """
    if show_output is None:
        show_output = os.environ.get("SHOW_ROUTER_LOGS", "0") == "1"

    # Determine port
    if base_url:
        router_port = int(base_url.split(":")[-1])
    else:
        router_port = get_open_port()
        base_url = f"http://127.0.0.1:{router_port}"

    # Get API key
    if backend == "openai":
        worker_url = "https://api.openai.com"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
    elif backend == "xai":
        worker_url = "https://api.x.ai"
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable required")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    logger.info("Launching %s router on port %d", backend, router_port)

    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--prometheus-port",
        str(router_port + PROMETHEUS_PORT_OFFSET),
        "--backend",
        "openai",
        "--worker-urls",
        worker_url,
        "--history-backend",
        history_backend,
        "--log-level",
        "warn",
    ]

    if router_args:
        cmd.extend(router_args)

    env = os.environ.copy()
    if backend == "openai":
        env["OPENAI_API_KEY"] = api_key
    else:
        env["XAI_API_KEY"] = api_key

    router_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=None if show_output else subprocess.PIPE,
        stderr=None if show_output else subprocess.PIPE,
        start_new_session=True,
    )

    try:
        wait_for_health(base_url, timeout=timeout)
    except TimeoutError:
        kill_process_tree(router_proc.pid)
        raise

    logger.info("%s router ready at %s", backend, base_url)

    return ClusterInfo(
        base_url=base_url,
        router_process=router_proc,
        worker_processes=[],
        model="",  # Cloud API - model specified per request
        backend=backend,
    )


# Backend configuration registry
BACKENDS: dict[str, dict[str, Any]] = {
    "grpc": {
        "description": "Local gRPC workers with SGLang router",
        "model": _resolve_model_path("meta-llama/Llama-3.1-8B-Instruct"),
        "launcher": launch_grpc_cluster,
        "launcher_kwargs": {
            "num_workers": 1,
            "tp_size": 1,
            "policy": "round_robin",
        },
        "needs_workers": True,
        "api_key_env": None,
    },
    "grpc_harmony": {
        "description": "Local gRPC workers with Harmony model",
        "model": _resolve_model_path("openai/gpt-oss-20b"),
        "launcher": launch_grpc_cluster,
        "launcher_kwargs": {
            "num_workers": 1,
            "tp_size": 2,
            "policy": "round_robin",
            "worker_args": ["--reasoning-parser=gpt-oss"],
            "router_args": ["--history-backend", "memory"],
        },
        "needs_workers": True,
        "api_key_env": None,
    },
    "openai": {
        "description": "OpenAI API backend",
        "model": "gpt-4o-mini",
        "launcher": launch_openai_router,
        "launcher_kwargs": {
            "backend": "openai",
            "history_backend": "memory",
        },
        "needs_workers": False,
        "api_key_env": "OPENAI_API_KEY",
    },
    "xai": {
        "description": "xAI API backend",
        "model": "grok-2-latest",
        "launcher": launch_openai_router,
        "launcher_kwargs": {
            "backend": "xai",
            "history_backend": "memory",
        },
        "needs_workers": False,
        "api_key_env": "XAI_API_KEY",
    },
    "oracle_store": {
        "description": "OpenAI API with Oracle history backend",
        "model": "gpt-4o-mini",
        "launcher": launch_openai_router,
        "launcher_kwargs": {
            "backend": "openai",
            "history_backend": "oracle",
        },
        "needs_workers": False,
        "api_key_env": "OPENAI_API_KEY",
    },
}


def get_backend_config(backend: str) -> dict[str, Any]:
    """Get configuration for a backend."""
    if backend not in BACKENDS:
        raise KeyError(
            f"Unknown backend: {backend}. Available: {list(BACKENDS.keys())}"
        )
    return BACKENDS[backend]


def launch_backend(backend: str, **kwargs: Any) -> ClusterInfo:
    """Launch a backend cluster.

    Args:
        backend: Backend name from BACKENDS
        **kwargs: Override launcher kwargs

    Returns:
        ClusterInfo with running cluster
    """
    cfg = get_backend_config(backend)

    # Merge kwargs with defaults
    launcher_kwargs = {**cfg["launcher_kwargs"], **kwargs}

    # Add model for grpc backends
    if cfg["needs_workers"]:
        return cfg["launcher"](cfg["model"], **launcher_kwargs)
    else:
        return cfg["launcher"](**launcher_kwargs)
