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
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "xai": {
        "description": "xAI API",
        "model": "grok-2-latest",
        "api_key_env": "XAI_API_KEY",
    },
}

# Keep CLOUD_BACKENDS as alias for backward compatibility during migration
# TODO: Remove after e2e_response_api migration
CLOUD_BACKENDS: dict[str, dict[str, Any]] = {
    **CLOUD_RUNTIMES,
    # Legacy entry for tests that parameterize on history backend
    "oracle_store": {
        "description": "OpenAI API with Oracle history backend",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "history_backend": "oracle",
        "_runtime": "openai",  # Actual runtime to use
    },
}


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


# Backward compatibility aliases - TODO: Remove after migration
def get_cloud_backend_config(backend: str) -> dict[str, Any]:
    """Deprecated: Use get_cloud_runtime_config instead."""
    if backend not in CLOUD_BACKENDS:
        raise KeyError(
            f"Unknown cloud backend: {backend}. Available: {list(CLOUD_BACKENDS.keys())}"
        )
    return CLOUD_BACKENDS[backend]


def launch_cloud_backend(backend: str, **kwargs: Any) -> Gateway:
    """Deprecated: Use launch_cloud_gateway instead."""
    cfg = get_cloud_backend_config(backend)

    # Handle legacy oracle_store entry
    runtime = cfg.get("_runtime", backend)
    history_backend = kwargs.pop(
        "history_backend", cfg.get("history_backend", "memory")
    )

    return launch_cloud_gateway(
        runtime,
        history_backend=history_backend,
        extra_args=kwargs.pop("router_args", None),
        **kwargs,
    )


# Keep old function name as alias
launch_cloud_router = launch_cloud_gateway
