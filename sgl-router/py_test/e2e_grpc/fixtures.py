"""
Fixtures for launching gRPC router + workers for e2e testing.

This module provides a drop-in replacement for popen_launch_server that launches
a gRPC router with integrated workers using a single command:

    python3 -m sglang_router.launch_server --grpc-mode ...

Tests from test/srt/openai_server can be reused by simply swapping:
    popen_launch_server() → popen_launch_grpc_router()
"""

import socket
import subprocess
import time
from typing import Optional

import requests


def find_free_port() -> int:
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(url: str, timeout: int = 300) -> None:
    """Wait for a service to become healthy."""
    start_time = time.time()
    last_error = None
    attempt = 0

    with requests.Session() as session:
        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = int(time.time() - start_time)

            # Print progress every 10 seconds
            if elapsed > 0 and elapsed % 10 == 0 and attempt % 10 == 0:
                print(f"  Still waiting... ({elapsed}/{timeout}s elapsed)")

            try:
                response = session.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"  Health check succeeded after {elapsed}s")
                    return
                last_error = f"HTTP {response.status_code}"
            except requests.ConnectionError as e:
                last_error = f"Connection refused (server not ready yet)"
            except requests.Timeout as e:
                last_error = f"Timeout"
            except requests.RequestException as e:
                last_error = str(e)

            time.sleep(1)

    raise TimeoutError(
        f"Service at {url} did not become healthy within {timeout}s.\n"
        f"Last error: {last_error}\n"
        f"Hint: Run with SHOW_ROUTER_LOGS=1 to see startup logs"
    )


def popen_launch_worker(
    model: str,
    port: Optional[int] = None,
    base_gpu_id: int = 0,
    tp_size: int = 1,
    api_key: Optional[str] = None,
    other_args: Optional[list] = None,
) -> subprocess.Popen:
    """
    Launch a single SGLang worker with gRPC support.

    Example command this generates:
        python3 -m sglang.launch_server \
          --model /home/ubuntu/models/llama-3.1-8b-instruct \
          --port 8080 \
          --tp-size 2 \
          --grpc-mode

    Args:
        model: Model path (e.g., /home/ubuntu/models/llama-3.1-8b-instruct)
        port: Port to bind (auto-assign if None)
        base_gpu_id: GPU ID to use
        tp_size: Tensor parallelism size (default: 1)
        api_key: Optional API key
        other_args: Additional arguments to pass to launch_server

    Returns:
        subprocess.Popen object for the worker
    """
    port = port or find_free_port()
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--base-gpu-id",
        str(base_gpu_id),
        "--grpc-mode",  # Enable gRPC mode
    ]

    if tp_size > 1:
        cmd.extend(["--tp-size", str(tp_size)])

    if api_key:
        cmd.extend(["--api-key", api_key])

    if other_args:
        cmd.extend(other_args)

    proc = subprocess.Popen(cmd)
    url = f"http://127.0.0.1:{port}"

    # Wait for worker to be ready
    wait_for_health(url, timeout=300)

    # Store URL on process object for later reference
    proc.url = url  # type: ignore
    proc.port = port  # type: ignore

    return proc


def popen_launch_grpc_router(
    model: str,
    base_url: str,
    timeout: int = 300,
    num_workers: int = 2,
    policy: str = "round_robin",
    api_key: Optional[str] = None,
    other_args: Optional[list] = None,
    tp_size: int = 1,
) -> dict:
    """
    Launch gRPC router with integrated workers using a single command.

    This is a drop-in replacement for popen_launch_server() that uses:
        python3 -m sglang_router.launch_server --grpc-mode ...

    This single command launches both workers AND router together!

    Example command generated:
        python3 -m sglang_router.launch_server \\
          --host 0.0.0.0 \\
          --port 8080 \\
          --model /home/ubuntu/models/llama-3.1-8b-instruct \\
          --tp-size 2 \\
          --dp-size 2 \\
          --grpc-mode \\
          --router-tool-call-parser llama \\
          --router-model-path /home/ubuntu/models/llama-3.1-8b-instruct \\
          --attention-backend fa3 \\
          --router-policy round_robin

    Args:
        model: Model path (e.g., /home/ubuntu/models/llama-3.1-8b-instruct)
        base_url: Base URL for router (e.g., "http://127.0.0.1:8080")
        timeout: Timeout for server startup (default: 300s)
        num_workers: Number of workers (maps to DP size)
        policy: Routing policy (round_robin, random, power_of_two, cache_aware)
        api_key: Optional API key
        other_args: Additional arguments (e.g., ["--attention-backend", "fa3"])
        tp_size: Tensor parallelism size (default: 1)

    Returns:
        dict with:
            - process: the single process running router + workers
            - base_url: router URL (same as input)

    Example:
        >>> # Original test code:
        >>> process = popen_launch_server(model, base_url)
        >>>
        >>> # New test code:
        >>> cluster = popen_launch_grpc_router(model, base_url, num_workers=2)
        >>> # Cleanup: kill_process_tree(cluster['process'].pid)
    """
    import os
    show_output = os.environ.get("SHOW_ROUTER_LOGS", "0") == "1"

    # Parse port from base_url
    if ":" in base_url.split("//")[-1]:
        port = int(base_url.split(":")[-1])
    else:
        port = find_free_port()

    # Find a free prometheus port
    prom_port = find_free_port()

    print(f"Launching gRPC router with integrated workers...")
    print(f"  Model: {model}")
    print(f"  Port: {port}")
    print(f"  Workers (DP): {num_workers}")
    print(f"  TP size: {tp_size}")
    print(f"  Policy: {policy}")

    # Build command
    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model,
        "--dp-size",
        str(num_workers),
        "--grpc-mode",
        "--router-tool-call-parser",
        "llama",
        "--router-model-path",
        model,
        "--attention-backend",
        "fa3",
        "--router-policy",
        policy,
        "--router-prometheus-port",
        str(prom_port),
        "--router-health-check-timeout-secs",
        str(timeout),
        "--router-health-check-interval-secs",
        "60",
    ]

    # Add TP size
    if tp_size > 1:
        cmd.extend(["--tp-size", str(tp_size)])

    # Add API key
    if api_key:
        cmd.extend(["--api-key", api_key])

    # Add other args
    if other_args:
        cmd.extend(other_args)

    print(f"  Command: {' '.join(cmd)}")

    # Launch process
    if show_output:
        # Show all router output
        proc = subprocess.Popen(cmd)
    else:
        # Suppress output (clean test runs)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    # Build URL
    router_url = f"http://127.0.0.1:{port}"

    # Wait for router to be ready
    print(f"Waiting for router to start at {router_url}...")
    print(f"Process PID: {proc.pid}")

    try:
        wait_for_health(router_url, timeout=timeout)
        print(f"✓ gRPC router ready at {router_url}")
    except TimeoutError as e:
        # Check if process is still alive
        poll = proc.poll()
        if poll is not None:
            print(f"\n✗ ERROR: Router process exited with code {poll}")
            print("Process died during startup. Run with SHOW_ROUTER_LOGS=1 to see why.")
            # Try to get any output
            if proc.stdout:
                try:
                    stdout, stderr = proc.communicate(timeout=1)
                    if stdout:
                        print(f"\nStdout:\n{stdout.decode()}")
                    if stderr:
                        print(f"\nStderr:\n{stderr.decode()}")
                except:
                    pass
        raise

    return {
        "process": proc,
        "base_url": router_url,
    }


def popen_launch_grpc_router_pd(
    model: str,
    base_url: str,
    timeout: int = 300,
    num_prefill_workers: int = 1,
    num_decode_workers: int = 1,
    prefill_policy: str = "round_robin",
    decode_policy: str = "round_robin",
    api_key: Optional[str] = None,
    other_args: Optional[list] = None,
) -> dict:
    """
    Launch gRPC router in Prefill-Decode disaggregation mode.

    Args:
        model: Model path or name
        base_url: Base URL for router
        timeout: Timeout for server startup
        num_prefill_workers: Number of prefill workers
        num_decode_workers: Number of decode workers
        prefill_policy: Policy for prefill workers
        decode_policy: Policy for decode workers
        api_key: Optional API key
        other_args: Additional arguments to pass to workers

    Returns:
        dict with:
            - router: router process
            - prefill_workers: list of prefill worker processes
            - decode_workers: list of decode worker processes
            - base_url: router URL
    """
    import subprocess

    # Parse port from base_url
    if ":" in base_url.split("//")[-1]:
        router_port = int(base_url.split(":")[-1])
    else:
        router_port = find_free_port()

    prefill_workers = []
    decode_workers = []
    prefill_specs = []
    decode_urls = []

    print(f"Launching PD cluster: {num_prefill_workers} prefill + {num_decode_workers} decode workers...")

    # Launch prefill workers
    for i in range(num_prefill_workers):
        bootstrap_port = find_free_port()
        worker_port = find_free_port()
        worker_url = f"http://127.0.0.1:{worker_port}"

        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--disaggregation-mode",
            "prefill",
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--disaggregation-bootstrap-port",
            str(bootstrap_port),
            "--base-gpu-id",
            str(i),
        ]
        if api_key:
            cmd.extend(["--api-key", api_key])
        if other_args:
            cmd.extend(other_args)

        proc = subprocess.Popen(cmd)
        wait_for_health(worker_url, timeout=300)
        prefill_workers.append(proc)
        prefill_specs.append((f"grpc://127.0.0.1:{worker_port}", bootstrap_port))
        print(f"  Prefill worker {i+1} ready at {worker_url}")

    # Launch decode workers
    for i in range(num_decode_workers):
        worker_port = find_free_port()
        worker_url = f"http://127.0.0.1:{worker_port}"

        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--disaggregation-mode",
            "decode",
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--base-gpu-id",
            str(num_prefill_workers + i),
        ]
        if api_key:
            cmd.extend(["--api-key", api_key])
        if other_args:
            cmd.extend(other_args)

        proc = subprocess.Popen(cmd)
        wait_for_health(worker_url, timeout=300)
        decode_workers.append(proc)
        decode_urls.append(f"grpc://127.0.0.1:{worker_port}")
        print(f"  Decode worker {i+1} ready at {worker_url}")

    # Launch PD router
    print(f"Launching gRPC PD router at {base_url}...")
    router_cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--pd-disaggregation",
        "--model-path",
        model,
        "--prefill-policy",
        prefill_policy,
        "--decode-policy",
        decode_policy,
    ]

    # Add prefill workers
    for url, bport in prefill_specs:
        router_cmd.extend(["--prefill", url, str(bport)])

    # Add decode workers
    for url in decode_urls:
        router_cmd.extend(["--decode", url])

    # Add Prometheus port
    prom_port = find_free_port()
    router_cmd.extend(["--prometheus-port", str(prom_port)])

    if api_key:
        router_cmd.extend(["--api-key", api_key])

    router_proc = subprocess.Popen(router_cmd)
    wait_for_health(base_url, timeout=timeout)
    print(f"gRPC PD router ready at {base_url}")

    return {
        "router": router_proc,
        "prefill_workers": prefill_workers,
        "decode_workers": decode_workers,
        "base_url": base_url,
    }
