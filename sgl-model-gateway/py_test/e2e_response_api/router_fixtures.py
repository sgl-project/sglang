"""
Fixtures for launching OpenAI/XAI router for response API e2e testing.

This module provides fixtures for launching SGLang router with OpenAI or XAI backends:
    1. Launch router with --backend openai pointing to OpenAI or XAI API
    2. Configure history backend (memory or oracle)

This supports testing the Response API against real cloud providers.
"""

import logging
import os
import socket
import subprocess
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def wait_for_workers_ready(
    router_url: str,
    expected_workers: int,
    timeout: int = 300,
    api_key: Optional[str] = None,
) -> None:
    """
    Wait for router to have all workers connected.

    Polls the /workers endpoint until the 'total' field matches expected_workers.

    Example response from /workers endpoint:
    {"workers":[],"total":0,"stats":{"prefill_count":0,"decode_count":0,"regular_count":0}}

    Args:
        router_url: Base URL of router (e.g., "http://127.0.0.1:30000")
        expected_workers: Number of workers expected to be connected
        timeout: Max seconds to wait
        api_key: Optional API key for authentication
    """
    start_time = time.time()
    last_error = None
    attempt = 0

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with requests.Session() as session:
        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = int(time.time() - start_time)

            # Log progress every 10 seconds
            if elapsed > 0 and elapsed % 10 == 0 and attempt % 10 == 0:
                logger.info(
                    f"  Still waiting for workers... ({elapsed}/{timeout}s elapsed)"
                )

            try:
                response = session.get(
                    f"{router_url}/workers", headers=headers, timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    total_workers = data.get("total", 0)

                    if total_workers == expected_workers:
                        logger.info(
                            f"  All {expected_workers} workers connected after {elapsed}s"
                        )
                        return
                    else:
                        last_error = f"Workers: {total_workers}/{expected_workers}"
                else:
                    last_error = f"HTTP {response.status_code}"
            except requests.ConnectionError:
                last_error = "Connection refused (router not ready yet)"
            except requests.Timeout:
                last_error = "Timeout"
            except requests.RequestException as e:
                last_error = str(e)
            except (ValueError, KeyError) as e:
                last_error = f"Invalid response: {e}"

            time.sleep(1)

    raise TimeoutError(
        f"Router at {router_url} did not get {expected_workers} workers within {timeout}s.\n"
        f"Last status: {last_error}\n"
        f"Hint: Run with SHOW_ROUTER_LOGS=1 to see startup logs"
    )


def find_free_port() -> int:
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_router_ready(
    router_url: str,
    timeout: int = 60,
    api_key: Optional[str] = None,
) -> None:
    """
    Wait for router to be ready.

    Polls the /health endpoint until it returns 200.

    Args:
        router_url: Base URL of router (e.g., "http://127.0.0.1:30000")
        timeout: Max seconds to wait
        api_key: Optional API key for authentication
    """
    start_time = time.time()
    last_error = None
    attempt = 0

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with requests.Session() as session:
        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = int(time.time() - start_time)

            # Log progress every 10 seconds
            if elapsed > 0 and elapsed % 10 == 0 and attempt % 10 == 0:
                logger.info(
                    f"  Still waiting for router... ({elapsed}/{timeout}s elapsed)"
                )

            try:
                response = session.get(
                    f"{router_url}/health", headers=headers, timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"  Router ready after {elapsed}s")
                    return
                else:
                    last_error = f"HTTP {response.status_code}"
            except requests.ConnectionError:
                last_error = "Connection refused (router not ready yet)"
            except requests.Timeout:
                last_error = "Timeout"
            except requests.RequestException as e:
                last_error = str(e)

            time.sleep(1)

    raise TimeoutError(
        f"Router at {router_url} did not become ready within {timeout}s.\n"
        f"Last status: {last_error}\n"
        f"Hint: Run with SHOW_ROUTER_LOGS=1 to see startup logs"
    )


def popen_launch_openai_xai_router(
    backend: str,  # "openai" or "xai"
    base_url: str,
    timeout: int = 60,
    history_backend: str = "memory",
    api_key: Optional[str] = None,
    router_args: Optional[list] = None,
    stdout=None,
    stderr=None,
    prometheus_port: Optional[int] = None,
) -> dict:
    """
    Launch SGLang router with OpenAI or XAI backend.

    This approach:
    1. Starts router with --backend openai
    2. Points to OpenAI or XAI API via --worker-urls
    3. Configures history backend (memory or oracle)
    4. Waits for router health check to pass

    Args:
        backend: "openai" or "xai"
        base_url: Base URL for router (e.g., "http://127.0.0.1:30000")
        timeout: Timeout for router startup (default: 60s)
        history_backend: "memory" or "oracle" (default: memory)
        api_key: Optional API key for router authentication
        router_args: Additional arguments for router
        stdout: Optional file handle for router stdout
        stderr: Optional file handle for router stderr

    Returns:
        dict with:
            - router: router process object
            - base_url: router URL (HTTP endpoint)

    Example:
        >>> cluster = popen_launch_openai_xai_router(
        ...     "openai", "http://127.0.0.1:30000"
        ... )
        >>> # Use cluster['base_url'] for HTTP requests
        >>> # Cleanup:
        >>> kill_process_tree(cluster['router'].pid)
    """
    show_output = os.environ.get("SHOW_ROUTER_LOGS", "0") == "1"

    # Parse router port from base_url
    if ":" in base_url.split("//")[-1]:
        router_port = int(base_url.split(":")[-1])
    else:
        router_port = find_free_port()

    logger.info(f"\n{'='*70}")
    logger.info(f"Launching {backend.upper()} router")
    logger.info(f"{'='*70}")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  Router port: {router_port}")
    logger.info(f"  History backend: {history_backend}")

    # Determine worker URL based on backend
    if backend == "openai":
        worker_url = "https://api.openai.com"
        # Get API key from environment
        backend_api_key = os.environ.get("OPENAI_API_KEY")
        if not backend_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set for OpenAI backend"
            )
    elif backend == "xai":
        worker_url = "https://api.x.ai"
        # Get API key from environment
        backend_api_key = os.environ.get("XAI_API_KEY")
        if not backend_api_key:
            raise ValueError(
                "XAI_API_KEY environment variable must be set for XAI backend"
            )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    logger.info(f"  Worker URL: {worker_url}")

    # Build router command
    router_cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--backend",
        "openai",
        "--worker-urls",
        worker_url,
        "--history-backend",
        history_backend,
        "--log-level",
        "warn",
    ]

    # Note: Not adding --api-key to router command for local testing
    # The router will not require authentication

    # Add Prometheus port to avoid conflicts (use unique port or disable)
    if prometheus_port is None:
        # Auto-assign a unique prometheus port based on router port
        prometheus_port = router_port + 1000
    router_cmd.extend(["--prometheus-port", str(prometheus_port)])

    # Add router-specific args
    if router_args:
        router_cmd.extend(router_args)

    if show_output:
        logger.info(f"  Command: {' '.join(router_cmd)}")

    # Set up environment with backend API key
    env = os.environ.copy()
    if backend == "openai":
        env["OPENAI_API_KEY"] = backend_api_key
    else:
        env["XAI_API_KEY"] = backend_api_key

    # Launch router
    if show_output:
        router_proc = subprocess.Popen(
            router_cmd,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    else:
        router_proc = subprocess.Popen(
            router_cmd,
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.PIPE,
            env=env,
        )

    print(f"  PID: {router_proc.pid}")

    # Wait for router to be ready
    router_url = f"http://127.0.0.1:{router_port}"
    print(f"\nWaiting for router to start at {router_url}...")

    try:
        wait_for_router_ready(router_url, timeout=timeout, api_key=None)
        logger.info(f"✓ Router ready at {router_url}")
    except TimeoutError:
        logger.error(f"✗ Router failed to start")
        # Cleanup: kill router
        try:
            router_proc.kill()
        except:
            pass
        raise

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ {backend.upper()} router ready!")
    logger.info(f"  Router: {router_url}")
    logger.info(f"{'='*70}\n")

    return {
        "router": router_proc,
        "base_url": router_url,
    }


def popen_launch_workers_and_router(
    model: str,
    base_url: str,
    timeout: int = 300,
    num_workers: int = 2,
    policy: str = "round_robin",
    api_key: Optional[str] = None,
    worker_args: Optional[list] = None,
    router_args: Optional[list] = None,
    tp_size: int = 1,
    env: Optional[dict] = None,
    stdout=None,
    stderr=None,
) -> dict:
    """
    Launch SGLang workers and gRPC router separately.

    This approach:
    1. Starts N SGLang workers with --grpc-mode flag
    2. Waits for workers to initialize (process startup)
    3. Starts a gRPC router pointing to those workers
    4. Waits for router health check to pass (router validates worker connectivity)

    This matches production deployment patterns better than the integrated approach.

    Args:
        model: Model path (e.g., /home/ubuntu/models/llama-3.1-8b-instruct)
        base_url: Base URL for router (e.g., "http://127.0.0.1:8080")
        timeout: Timeout for server startup (default: 300s)
        num_workers: Number of workers to launch
        policy: Routing policy (round_robin, random, power_of_two, cache_aware)
        api_key: Optional API key for router
        worker_args: Additional arguments for workers (e.g., ["--context-len", "8192"])
        router_args: Additional arguments for router (e.g., ["--max-total-token", "1536"])
        tp_size: Tensor parallelism size for workers (default: 1)
        env: Optional environment variables for workers (e.g., {"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "256"})
        stdout: Optional file handle for worker stdout (default: subprocess.PIPE)
        stderr: Optional file handle for worker stderr (default: subprocess.PIPE)

    Returns:
        dict with:
            - workers: list of worker process objects
            - worker_urls: list of gRPC worker URLs
            - router: router process object
            - base_url: router URL (HTTP endpoint)

    Example:
        >>> cluster = popen_launch_workers_and_router(model, base_url, num_workers=2)
        >>> # Use cluster['base_url'] for HTTP requests
        >>> # Cleanup:
        >>> for worker in cluster['workers']:
        >>>     kill_process_tree(worker.pid)
        >>> kill_process_tree(cluster['router'].pid)
    """
    show_output = os.environ.get("SHOW_ROUTER_LOGS", "0") == "1"

    # Parse router port from base_url
    if ":" in base_url.split("//")[-1]:
        router_port = int(base_url.split(":")[-1])
    else:
        router_port = find_free_port()

    logger.info(f"\n{'='*70}")
    logger.info(f"Launching gRPC cluster (separate workers + router)")
    logger.info(f"{'='*70}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Router port: {router_port}")
    logger.info(f"  Workers: {num_workers}")
    logger.info(f"  TP size: {tp_size}")
    logger.info(f"  Policy: {policy}")

    # Step 1: Launch workers with gRPC enabled
    workers = []
    worker_urls = []

    for i in range(num_workers):
        worker_port = find_free_port()
        worker_url = f"grpc://127.0.0.1:{worker_port}"
        worker_urls.append(worker_url)

        logger.info(f"\n[Worker {i+1}/{num_workers}]")
        logger.info(f"  Port: {worker_port}")
        logger.info(f"  URL: {worker_url}")

        # Build worker command
        worker_cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--grpc-mode",  # Enable gRPC for this worker
            "--mem-fraction-static",
            "0.8",
        ]

        # Add TP size
        if tp_size > 1:
            worker_cmd.extend(["--tp-size", str(tp_size)])

        # Add worker-specific args
        if worker_args:
            worker_cmd.extend(worker_args)

        # Launch worker with optional environment variables
        if show_output:
            worker_proc = subprocess.Popen(
                worker_cmd,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
        else:
            worker_proc = subprocess.Popen(
                worker_cmd,
                stdout=stdout if stdout is not None else subprocess.PIPE,
                stderr=stderr if stderr is not None else subprocess.PIPE,
                env=env,
            )

        workers.append(worker_proc)
        logger.info(f"  PID: {worker_proc.pid}")

    # Give workers a moment to start binding to ports
    # The router will check worker health when it starts
    logger.info(f"\nWaiting for {num_workers} workers to initialize (20s)...")
    time.sleep(20)

    # Quick check: make sure worker processes are still alive
    for i, worker in enumerate(workers):
        if worker.poll() is not None:
            logger.error(
                f"  ✗ Worker {i+1} died during startup (exit code: {worker.poll()})"
            )
            # Cleanup: kill all workers
            for w in workers:
                try:
                    w.kill()
                except:
                    pass
            raise RuntimeError(f"Worker {i+1} failed to start")

    logger.info(
        f"✓ All {num_workers} workers started (router will verify connectivity)"
    )

    # Step 2: Launch router pointing to workers
    logger.info(f"\n[Router]")
    logger.info(f"  Port: {router_port}")
    logger.info(f"  Worker URLs: {', '.join(worker_urls)}")

    # Build router command
    router_cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--prometheus-port",
        "9321",
        "--policy",
        policy,
        "--model-path",
        model,
        "--log-level",
        "warn",
    ]

    # Add worker URLs
    router_cmd.append("--worker-urls")
    router_cmd.extend(worker_urls)

    # Add API key
    if api_key:
        router_cmd.extend(["--api-key", api_key])

    # Add router-specific args
    if router_args:
        router_cmd.extend(router_args)

    if show_output:
        logger.info(f"  Command: {' '.join(router_cmd)}")

    # Launch router
    if show_output:
        router_proc = subprocess.Popen(router_cmd)
    else:
        router_proc = subprocess.Popen(
            router_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    logger.info(f"  PID: {router_proc.pid}")

    # Wait for router to be ready
    router_url = f"http://127.0.0.1:{router_port}"
    logger.info(f"\nWaiting for router to start at {router_url}...")

    try:
        wait_for_workers_ready(
            router_url, expected_workers=num_workers, timeout=180, api_key=api_key
        )
        logger.info(f"✓ Router ready at {router_url}")
    except TimeoutError:
        logger.error(f"✗ Router failed to start")
        # Cleanup: kill router and all workers
        try:
            router_proc.kill()
        except:
            pass
        for worker in workers:
            try:
                worker.kill()
            except:
                pass
        raise

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ gRPC cluster ready!")
    logger.info(f"  Router: {router_url}")
    logger.info(f"  Workers: {len(workers)}")
    logger.info(f"{'='*70}\n")

    return {
        "workers": workers,
        "worker_urls": worker_urls,
        "router": router_proc,
        "base_url": router_url,
    }
