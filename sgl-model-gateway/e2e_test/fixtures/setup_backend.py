"""Backend setup fixtures for E2E tests.

This module provides fixtures for launching gateways/routers for different backends.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from infra import ModelPool

from .markers import get_marker_kwargs, get_marker_value

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def setup_backend(request: pytest.FixtureRequest, model_pool: "ModelPool"):
    """Class-scoped fixture that launches a router for each test class.

    Routers are cheap to start (~1-2s) compared to workers (~30-60s), so we
    launch a fresh router per test class for isolation while reusing the
    expensive workers from model_pool.

    Backend types:
    - "http", "grpc": Gets existing worker from model_pool, launches router
    - "pd": Launches prefill/decode workers via model_pool, launches PD router
    - "openai", "xai", etc.: Launches cloud router (no local workers)

    Configuration via markers:
    - @pytest.mark.model("model-id"): Override default model
    - @pytest.mark.workers(count=1): Number of regular workers behind router
    - @pytest.mark.workers(prefill=1, decode=1): PD worker configuration
    - @pytest.mark.gateway(policy="round_robin", timeout=60): Gateway configuration

    Returns:
        Tuple of (backend_name, model_path, openai_client, gateway)

    Usage:
        @pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
        class TestBasic:
            def test_chat(self, setup_backend):
                backend, model, client, gateway = setup_backend
    """
    import openai
    from infra import (
        DEFAULT_MODEL,
        DEFAULT_ROUTER_TIMEOUT,
        ENV_MODEL,
        ENV_SKIP_BACKEND_SETUP,
        LOCAL_MODES,
        ConnectionMode,
        Gateway,
        WorkerIdentity,
        WorkerType,
    )

    backend_name = request.param

    # Skip if requested
    if os.environ.get(ENV_SKIP_BACKEND_SETUP, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"{ENV_SKIP_BACKEND_SETUP} is set")

    # Get model from marker or env var or default
    model_id = get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    # Get worker configuration from marker
    workers_config = get_marker_kwargs(
        request, "workers", defaults={"count": 1, "prefill": None, "decode": None}
    )

    # Get gateway configuration from marker
    gateway_config = get_marker_kwargs(
        request,
        "gateway",
        defaults={
            "policy": "round_robin",
            "timeout": DEFAULT_ROUTER_TIMEOUT,
            "extra_args": None,
        },
    )

    # PD disaggregation backend
    if backend_name == "pd":
        yield from _setup_pd_backend(
            request, model_pool, model_id, workers_config, gateway_config
        )
        return

    # Check if this is a local backend (grpc, http)
    try:
        connection_mode = ConnectionMode(backend_name)
        is_local = connection_mode in LOCAL_MODES
    except ValueError:
        is_local = False
        connection_mode = None

    # Local backends: use worker from pool + launch gateway
    if is_local:
        yield from _setup_local_backend(
            request,
            model_pool,
            backend_name,
            model_id,
            connection_mode,
            workers_config,
            gateway_config,
        )
        return

    # Cloud backends: launch cloud router
    yield from _setup_cloud_backend(backend_name)


def _setup_pd_backend(
    request: pytest.FixtureRequest,
    model_pool: "ModelPool",
    model_id: str,
    workers_config: dict,
    gateway_config: dict,
):
    """Setup PD disaggregation backend."""
    import openai
    from infra import ConnectionMode, Gateway, WorkerIdentity, WorkerType

    # Check PD requirements
    try:
        import sgl_kernel  # noqa: F401
    except ImportError:
        pytest.skip("sgl_kernel not available, required for PD disaggregation")

    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get PD configuration from workers marker
    num_prefill = workers_config.get("prefill") or 1
    num_decode = workers_config.get("decode") or 1

    # Check GPU requirements
    required_gpus = num_prefill + num_decode
    gpu_count = torch.cuda.device_count()
    if gpu_count < required_gpus:
        pytest.skip(
            f"PD tests require {required_gpus} GPUs "
            f"({num_prefill} prefill + {num_decode} decode), found {gpu_count}"
        )

    # Try to use pre-launched PD workers, or launch additional ones if needed
    existing_prefills = model_pool.get_workers_by_type(model_id, WorkerType.PREFILL)
    existing_decodes = model_pool.get_workers_by_type(model_id, WorkerType.DECODE)

    # Calculate how many more we need
    missing_prefill = max(0, num_prefill - len(existing_prefills))
    missing_decode = max(0, num_decode - len(existing_decodes))

    if missing_prefill == 0 and missing_decode == 0:
        prefills = existing_prefills[:num_prefill]
        decodes = existing_decodes[:num_decode]
        logger.info(
            "Using pre-launched PD workers: %d prefill, %d decode",
            len(prefills),
            len(decodes),
        )
    else:
        # Build WorkerIdentity list for missing workers
        workers_to_launch: list[WorkerIdentity] = []
        for i in range(missing_prefill):
            workers_to_launch.append(
                WorkerIdentity(
                    model_id,
                    ConnectionMode.HTTP,
                    WorkerType.PREFILL,
                    len(existing_prefills) + i,
                )
            )
        for i in range(missing_decode):
            workers_to_launch.append(
                WorkerIdentity(
                    model_id,
                    ConnectionMode.HTTP,
                    WorkerType.DECODE,
                    len(existing_decodes) + i,
                )
            )

        logger.info(
            "Have %d/%d prefill, %d/%d decode. Launching %d more workers",
            len(existing_prefills),
            num_prefill,
            len(existing_decodes),
            num_decode,
            len(workers_to_launch),
        )
        new_instances = model_pool.launch_workers(
            workers_to_launch, startup_timeout=300
        )

        new_prefills = [w for w in new_instances if w.worker_type == WorkerType.PREFILL]
        new_decodes = [w for w in new_instances if w.worker_type == WorkerType.DECODE]
        prefills = existing_prefills + new_prefills
        decodes = existing_decodes + new_decodes

    # Acquire references to prevent eviction during test
    all_workers = prefills + decodes
    for worker in all_workers:
        worker.acquire()

    model_path = prefills[0].model_path if prefills else None

    # Launch PD gateway
    gateway = Gateway()
    gateway.start(
        prefill_workers=prefills,
        decode_workers=decodes,
        policy=gateway_config["policy"],
        timeout=gateway_config["timeout"],
        extra_args=gateway_config["extra_args"],
    )

    client = openai.OpenAI(
        base_url=f"{gateway.base_url}/v1",
        api_key="not-used",
    )

    logger.info(
        "Setup PD backend: model=%s, %d prefill + %d decode workers, "
        "gateway=%s, policy=%s",
        model_id,
        len(prefills),
        len(decodes),
        gateway.base_url,
        gateway_config["policy"],
    )

    try:
        yield "pd", model_path, client, gateway
    finally:
        logger.info("Tearing down PD gateway")
        gateway.shutdown()
        # Release references to allow eviction
        for worker in all_workers:
            worker.release()


def _setup_local_backend(
    request: pytest.FixtureRequest,
    model_pool: "ModelPool",
    backend_name: str,
    model_id: str,
    connection_mode,
    workers_config: dict,
    gateway_config: dict,
):
    """Setup local backend (grpc, http)."""
    import openai
    from infra import Gateway, WorkerIdentity, WorkerType

    num_workers = workers_config.get("count") or 1
    instances: list = []  # Track instances for reference counting

    try:
        if num_workers > 1:
            existing = model_pool.get_workers_by_type(model_id, WorkerType.REGULAR)
            existing_for_mode = [w for w in existing if w.mode == connection_mode]

            if len(existing_for_mode) >= num_workers:
                instances = existing_for_mode[:num_workers]
            else:
                missing = num_workers - len(existing_for_mode)
                workers_to_launch = [
                    WorkerIdentity(
                        model_id,
                        connection_mode,
                        WorkerType.REGULAR,
                        len(existing_for_mode) + i,
                    )
                    for i in range(missing)
                ]
                new_instances = model_pool.launch_workers(
                    workers_to_launch, startup_timeout=300
                )
                instances = existing_for_mode + new_instances

            if not instances:
                pytest.fail(f"Failed to get {num_workers} workers for {model_id}")
            worker_urls = [inst.worker_url for inst in instances]
            model_path = instances[0].model_path
        else:
            instance = model_pool.get(model_id, connection_mode)
            instances = [instance]
            worker_urls = [instance.worker_url]
            model_path = instance.model_path

        # Acquire references to prevent eviction during test
        for inst in instances:
            inst.acquire()
    except RuntimeError as e:
        pytest.fail(str(e))

    # Launch gateway
    gateway = Gateway()
    gateway.start(
        worker_urls=worker_urls,
        model_path=model_path,
        policy=gateway_config["policy"],
        timeout=gateway_config["timeout"],
        extra_args=gateway_config["extra_args"],
    )

    client = openai.OpenAI(
        base_url=f"{gateway.base_url}/v1",
        api_key="not-used",
    )

    logger.info(
        "Setup %s backend: model=%s, workers=%d, gateway=%s, policy=%s",
        backend_name,
        model_id,
        num_workers,
        gateway.base_url,
        gateway_config["policy"],
    )

    try:
        yield backend_name, model_path, client, gateway
    finally:
        logger.info("Tearing down gateway for %s backend", backend_name)
        gateway.shutdown()
        # Release references to allow eviction
        for inst in instances:
            inst.release()


def _setup_cloud_backend(backend_name: str):
    """Setup cloud backend (openai, xai, etc.)."""
    import openai
    from backends import CLOUD_BACKENDS, launch_cloud_backend

    if backend_name not in CLOUD_BACKENDS:
        pytest.fail(f"Unknown backend: {backend_name}")

    cfg = CLOUD_BACKENDS[backend_name]
    api_key_env = cfg.get("api_key_env")

    if api_key_env and not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set, skipping {backend_name} tests")

    logger.info("Launching cloud backend: %s", backend_name)
    router = launch_cloud_backend(backend_name)

    api_key = os.environ.get(api_key_env) if api_key_env else "not-used"
    client = openai.OpenAI(
        base_url=f"{router.base_url}/v1",
        api_key=api_key,
    )

    try:
        yield backend_name, cfg["model"], client, router
    finally:
        logger.info("Tearing down cloud backend: %s", backend_name)
        router.shutdown()


@pytest.fixture
def backend_router(request: pytest.FixtureRequest, model_pool: "ModelPool"):
    """Function-scoped fixture for launching a fresh router per test.

    This launches a new Gateway for each test, pointing to workers from the pool.
    Use for tests that need isolated router state.

    Usage:
        @pytest.mark.parametrize("backend_router", ["grpc", "http"], indirect=True)
        def test_router_state(backend_router):
            gateway = backend_router
    """
    from infra import DEFAULT_MODEL, ENV_MODEL, ConnectionMode, Gateway

    backend_name = request.param
    model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    connection_mode = ConnectionMode(backend_name)

    try:
        instance = model_pool.get(model_id, connection_mode)
    except KeyError:
        pytest.skip(f"Model {model_id}:{backend_name} not available in pool")
    except RuntimeError as e:
        pytest.fail(str(e))

    # Acquire reference to prevent eviction during test
    instance.acquire()

    gateway = Gateway()
    gateway.start(
        worker_urls=[instance.worker_url],
        model_path=instance.model_path,
    )

    try:
        yield gateway
    finally:
        gateway.shutdown()
        # Release reference to allow eviction
        instance.release()
