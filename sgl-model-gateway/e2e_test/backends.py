"""Cloud runtime configurations for E2E tests.

This module handles cloud API runtimes (OpenAI, xAI) that don't need local GPU workers.
For local runtimes (gRPC, HTTP), use ModelPool from infra/ to launch workers,
then launch the gateway separately pointing to those workers.

Cloud runtimes vs History backends:
- Cloud runtimes: Where models run (openai, xai)
- History backends: Gateway plugin for conversation storage (memory, oracle)
These are orthogonal - any cloud runtime can use any history backend.
"""

from __future__ import annotations

import logging
from typing import Any

from infra import Gateway

logger = logging.getLogger(__name__)


# Cloud runtime configurations (where models run)
CLOUD_RUNTIMES: dict[str, dict[str, Any]] = {
    "openai": {
        "description": "OpenAI API",
        "model": "gpt-5-nano",
        "api_key_env": "OPENAI_API_KEY",
    },
    "xai": {
        "description": "xAI API",
        "model": "grok-2-latest",
        "api_key_env": "XAI_API_KEY",
    },
}

# Backward compatibility alias
CLOUD_BACKENDS = CLOUD_RUNTIMES


def get_cloud_runtime_config(runtime: str) -> dict[str, Any]:
    """Get configuration for a cloud runtime."""
    if runtime not in CLOUD_RUNTIMES:
        raise KeyError(
            f"Unknown cloud runtime: {runtime}. Available: {list(CLOUD_RUNTIMES.keys())}"
        )
    return CLOUD_RUNTIMES[runtime]


def launch_cloud_gateway(
    runtime: str,  # "openai" or "xai"
    *,
    history_backend: str = "memory",
    extra_args: list[str] | None = None,
    timeout: float = 60,
    show_output: bool | None = None,
) -> Gateway:
    """Launch gateway with cloud API runtime.

    Args:
        runtime: Cloud runtime ("openai" or "xai")
        history_backend: History storage backend ("memory" or "oracle")
        extra_args: Additional router arguments
        timeout: Startup timeout in seconds
        show_output: Show subprocess output

    Returns:
        Gateway instance with running router
    """
    if runtime not in CLOUD_RUNTIMES:
        raise ValueError(f"Unknown cloud runtime: {runtime}")

    gateway = Gateway()
    gateway.start(
        cloud_backend=runtime,
        history_backend=history_backend,
        timeout=timeout,
        show_output=show_output,
        extra_args=extra_args,
    )
    return gateway


# Backward compatibility aliases
get_cloud_backend_config = get_cloud_runtime_config
launch_cloud_backend = launch_cloud_gateway
launch_cloud_router = launch_cloud_gateway
