"""Cloud backend configurations for E2E tests.

This module handles cloud API backends (OpenAI, xAI) that don't need local GPU workers.
For local backends (gRPC, HTTP), use ModelPool from infra/ to launch workers,
then launch the router separately pointing to those workers.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from infra import get_open_port, kill_process_tree, wait_for_health

logger = logging.getLogger(__name__)


@dataclass
class RouterInstance:
    """A running router instance (for cloud backends)."""

    base_url: str
    router_process: subprocess.Popen
    backend: str

    def shutdown(self) -> None:
        """Shutdown the router."""
        if self.router_process.poll() is None:
            kill_process_tree(self.router_process.pid)


def launch_cloud_router(
    backend: str,  # "openai" or "xai"
    *,
    history_backend: str = "memory",
    router_args: list[str] | None = None,
    timeout: float = 60,
    show_output: bool | None = None,
) -> RouterInstance:
    """Launch router with cloud API backend (OpenAI/xAI).

    Args:
        backend: "openai" or "xai"
        history_backend: "memory" or "oracle"
        router_args: Additional router arguments
        timeout: Startup timeout in seconds
        show_output: Show subprocess output

    Returns:
        RouterInstance with running router
    """
    if show_output is None:
        show_output = os.environ.get("SHOW_ROUTER_LOGS", "0") == "1"

    router_port = get_open_port()
    prometheus_port = get_open_port()
    base_url = f"http://127.0.0.1:{router_port}"

    # Get API key and worker URL
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
        raise ValueError(f"Unsupported cloud backend: {backend}")

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
        str(prometheus_port),
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

    return RouterInstance(
        base_url=base_url,
        router_process=router_proc,
        backend=backend,
    )


# Cloud backend configurations
CLOUD_BACKENDS: dict[str, dict[str, Any]] = {
    "openai": {
        "description": "OpenAI API backend",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "history_backend": "memory",
    },
    "xai": {
        "description": "xAI API backend",
        "model": "grok-2-latest",
        "api_key_env": "XAI_API_KEY",
        "history_backend": "memory",
    },
    "oracle_store": {
        "description": "OpenAI API with Oracle history backend",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "history_backend": "oracle",
    },
}


def get_cloud_backend_config(backend: str) -> dict[str, Any]:
    """Get configuration for a cloud backend."""
    if backend not in CLOUD_BACKENDS:
        raise KeyError(
            f"Unknown cloud backend: {backend}. Available: {list(CLOUD_BACKENDS.keys())}"
        )
    return CLOUD_BACKENDS[backend]


def launch_cloud_backend(backend: str, **kwargs: Any) -> RouterInstance:
    """Launch a cloud backend router.

    Args:
        backend: Backend name from CLOUD_BACKENDS
        **kwargs: Override launcher kwargs

    Returns:
        RouterInstance with running router
    """
    cfg = get_cloud_backend_config(backend)

    # Determine actual backend type (openai or xai)
    if backend == "oracle_store":
        actual_backend = "openai"
    else:
        actual_backend = backend

    history_backend = kwargs.pop("history_backend", cfg["history_backend"])

    return launch_cloud_router(
        actual_backend,
        history_backend=history_backend,
        **kwargs,
    )
