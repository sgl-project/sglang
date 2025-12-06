"""
pytest configuration for e2e_response_api tests.

This configures pytest to not collect base test classes that are meant to be inherited.
"""

import os

import openai
import pytest  # noqa: F401
from router_fixtures import (
    popen_launch_openai_xai_router,
    popen_launch_workers_and_router,
)
from util import kill_process_tree

# ------------------------------
# Backend Configuration Map
# ------------------------------
BACKENDS = {
    "openai": {
        "model": "gpt-5-nano",
        "base_url_port": "http://127.0.0.1:30010",
        "launcher": popen_launch_openai_xai_router,
        "launcher_kwargs": {
            "backend": "openai",
            "history_backend": "memory",
        },
        "api_key_env": "OPENAI_API_KEY",
        "needs_workers": False,
    },
    "xai": {
        "model": "grok-4-fast",
        "base_url_port": "http://127.0.0.1:30023",
        "launcher": popen_launch_openai_xai_router,
        "launcher_kwargs": {
            "backend": "xai",
            "history_backend": "memory",
        },
        "api_key_env": "XAI_API_KEY",
        "needs_workers": False,
    },
    "grpc": {
        "model": "/home/ubuntu/models/Qwen/Qwen2.5-14B-Instruct",
        "base_url_port": "http://127.0.0.1:30030",
        "launcher": popen_launch_workers_and_router,
        "launcher_kwargs": {
            "timeout": 90,
            "num_workers": 1,
            "tp_size": 2,
            "policy": "round_robin",
            "worker_args": ["--context-length=1000"],
            "router_args": [
                "--history-backend",
                "memory",
                "--tool-call-parser",
                "qwen",
            ],
        },
        "api_key_env": None,  # grpc does not use API keys
        "needs_workers": True,
    },
    "grpc_harmony": {
        "model": "/home/ubuntu/models/openai/gpt-oss-20b",
        "base_url_port": "http://127.0.0.1:30030",
        "launcher": popen_launch_workers_and_router,
        "launcher_kwargs": {
            "timeout": 90,
            "num_workers": 1,
            "tp_size": 2,
            "policy": "round_robin",
            "worker_args": ["--reasoning-parser=gpt-oss"],
            "router_args": ["--history-backend", "memory"],
        },
        "api_key_env": None,
        "needs_workers": True,
    },
    "oracle_store": {
        "model": "gpt-5-nano",
        "base_url_port": "http://127.0.0.1:30040",
        "launcher": popen_launch_openai_xai_router,
        "launcher_kwargs": {
            "backend": "openai",
            "history_backend": "oracle",
        },
        "api_key_env": "OPENAI_API_KEY",
        "needs_workers": False,
    },
}


@pytest.fixture(scope="class")
def setup_backend(request):
    backend = request.param
    if backend not in BACKENDS:
        raise RuntimeError(f"Unknown backend {backend}")

    cfg = BACKENDS[backend]

    # Launch cluster
    cluster = (
        cfg["launcher"](
            cfg["model"],
            cfg["base_url_port"],
            **cfg["launcher_kwargs"],
        )
        if cfg["launcher"] is popen_launch_workers_and_router
        else cfg["launcher"](
            backend=cfg["launcher_kwargs"]["backend"],
            base_url=cfg["base_url_port"],
            history_backend=cfg["launcher_kwargs"]["history_backend"],
        )
    )

    # Build client
    api_key = os.environ.get(cfg["api_key_env"]) if cfg["api_key_env"] else None
    client = openai.Client(
        api_key=api_key,
        base_url=cluster["base_url"] + "/v1",
    )

    # Yield data to test
    try:
        yield backend, cfg["model"], client
    finally:
        # Always kill router
        kill_process_tree(cluster["router"].pid)

        # If workers exist, kill them as well
        if cfg["needs_workers"]:
            for w in cluster.get("workers", []):
                kill_process_tree(w.pid)
