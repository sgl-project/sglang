"""
Unified Benchmark Driver for Multi-Node SGLang Benchmarking

This script launches an SGLang server with either MP or Ray actor backend,
waits for it to be ready, runs sglang.bench_serving, and reports results.

Usage:
    # Ray actor backend with TP=8
    python benchmark_driver.py --backend ray-actor --model Qwen/Qwen2-7B --tp 8

    # Ray actor backend with TP=4, PP=2
    python benchmark_driver.py --backend ray-actor --model Qwen/Qwen2-7B --tp 4 --pp-size 2

    # MP backend with TP=8
    python benchmark_driver.py --backend mp --model Qwen/Qwen2-7B --tp 8 --nnodes 2

Submit via Anyscale:
    anyscale job submit -f examples/anyscale/job_benchmark_ray_tp8.yaml
"""

import argparse
import json
import subprocess
import sys
import time

import requests


def fix_packages():
    """Fix package compatibility on head node.

    Must be called BEFORE importing sglang to avoid import errors.
    """
    print("Fixing package compatibility on head node...")

    commands = [
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
         "numpy>=1.26.0,<2.0"],
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
         "sgl-kernel", "--force-reinstall"],
    ]

    for cmd in commands:
        print(f"  Running: {' '.join(cmd[:6])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Warning: Command failed with code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")

    print("Package fix complete on head node.")


def fix_packages_on_workers():
    """Fix packages on all worker nodes via Ray remote tasks."""
    print("\nFixing packages on worker nodes...")
    subprocess.run([sys.executable, "examples/anyscale/fix_worker_packages.py"])
    print("Worker nodes ready.")


def wait_for_server(url: str, timeout: int = 600) -> bool:
    """Wait for the server health check to pass."""
    start_time = time.time()
    health_url = f"{url}/health"

    print(f"Waiting for server at {health_url} to be ready...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"Server at {url} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(5)
        elapsed = int(time.time() - start_time)
        print(f"Still waiting... ({elapsed}s elapsed)")

    print(f"Timeout waiting for server at {url}")
    return False


def run_benchmark(url: str, args) -> dict:
    """Run sglang.bench_serving and return results."""
    output_file = f"/tmp/benchmark_{args.backend}_{args.tp}tp_{args.pp_size}pp.jsonl"

    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", url,
        "--model", args.model,
        "--dataset-name", args.dataset,
        "--num-prompts", str(args.num_prompts),
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--random-range-ratio", str(args.range_ratio),
        "--output-file", output_file,
    ]

    # Add request rate if not infinite
    if args.request_rate != float("inf"):
        cmd.extend(["--request-rate", str(args.request_rate)])

    print(f"\n{'='*60}")
    print("Running Benchmark")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Benchmark failed with return code {result.returncode}")
        return None

    # Parse results from output file (JSONL format, get last line)
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1])
    except Exception as e:
        print(f"Failed to parse results: {e}")

    return None


def print_results_summary(results: dict, args):
    """Print a summary of benchmark results."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Backend: {args.backend}")
    print(f"  Model: {args.model}")
    print(f"  TP: {args.tp}, PP: {args.pp_size}")
    print(f"  World Size: {args.tp * args.pp_size}")
    print()
    print(f"Workload:")
    print(f"  Num Prompts: {args.num_prompts}")
    print(f"  Input Length: {args.input_len}")
    print(f"  Output Length: {args.output_len}")
    print()

    if results:
        print(f"Results:")
        print(f"  Completed Requests: {results.get('completed', 'N/A')}")
        print(f"  Duration: {results.get('duration', 'N/A'):.2f}s")
        print(f"  Output Throughput: {results.get('output_throughput', 'N/A'):.2f} tokens/sec")
        print(f"  Request Throughput: {results.get('request_throughput', 'N/A'):.2f} req/sec")
        print()
        print(f"Latency Metrics:")
        print(f"  Mean TTFT: {results.get('mean_ttft_ms', 'N/A'):.2f} ms")
        print(f"  Mean TPOT: {results.get('mean_tpot_ms', 'N/A'):.2f} ms")
        print(f"  Mean E2E Latency: {results.get('mean_e2e_latency_ms', 'N/A'):.2f} ms")
        print(f"  P99 E2E Latency: {results.get('p99_e2e_latency_ms', 'N/A'):.2f} ms")
    else:
        print("Results: FAILED - No results available")

    print(f"{'='*60}")


def launch_ray_actor_backend(args):
    """Launch server using HttpServerActor on a GPU worker node.

    This creates a Ray actor that runs on a GPU worker node, avoiding the need
    to import sglang on the head node. The actor starts an HTTP server that
    can be accessed from any node in the cluster.

    Returns (actor, url) where actor is the Ray actor handle.
    """
    import ray

    if not ray.is_initialized():
        ray.init()

    print(f"\n{'='*60}")
    print("Launching SGLang with Ray Actor Backend (HTTP Server)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"TP: {args.tp}, PP: {args.pp_size}")

    # Import the launch function (safe - doesn't import sglang internals)
    from sglang.srt.entrypoints.http_server_actor import create_and_launch_http_server_actor

    # Create actor on rank 0's GPU worker node with topology discovery
    actor = create_and_launch_http_server_actor(
        model_path=args.model,
        tp_size=args.tp,
        pp_size=args.pp_size,
        port=args.port,
        host="0.0.0.0",
        use_ray=True,  # Enable Ray actor backend for schedulers
    )

    # Get URL and node info in a single RPC call (avoids multiple round-trips)
    status = ray.get(actor.get_status.remote())
    url = status["url"]
    node_ip = status["node_ip"]
    print(f"Server actor created on worker node: {node_ip}")
    print(f"Server URL: {url}")

    # Wait for server process to start (with progress output)
    print("\nWaiting for server to initialize...")
    start_time = time.time()
    while time.time() - start_time < 60:
        status = ray.get(actor.get_status.remote())
        if status["alive"]:
            print(f"  Server process is running ({int(time.time() - start_time)}s)")
            break
        time.sleep(2)
        print(f"  Still starting... ({int(time.time() - start_time)}s)")

    return actor, url


def launch_mp_backend(args):
    """Launch server using MP backend.

    This starts the driver.py script which handles multi-node coordination.
    Returns (process, url) where process is the subprocess running the server.
    """
    print(f"\n{'='*60}")
    print("Launching SGLang with MP Backend")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"TP: {args.tp}, PP: {args.pp_size}, Nodes: {args.nnodes}")

    # Import driver module to access its functions
    cmd = [
        sys.executable, "examples/anyscale/driver.py",
        "--model-path", args.model,
        "--tp", str(args.tp),
        "--pp-size", str(args.pp_size),
        "--nnodes", str(args.nnodes),
        "--port", str(args.port),
    ]

    # Start driver in background - it will keep running until we stop it
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output and look for the master server URL from driver.py
    # driver.py outputs "Master server URL: http://{ip}:{port}" when ready
    print("Starting MP backend (waiting for server URL)...")
    url = None
    start_time = time.time()
    timeout = 600  # 10 minutes timeout for server startup

    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if line:
            print(f"  [driver] {line.rstrip()}")
            # Look for the master server URL in driver.py output
            if "Master server URL:" in line:
                # Extract URL from line like "Master server URL: http://10.0.1.2:30000"
                url = line.split("Master server URL:")[-1].strip()
                print(f"Found master server URL: {url}")
                break
        if process.poll() is not None:
            print("Driver process exited unexpectedly!")
            return None, None

    if url is None:
        print("Failed to get master server URL from driver.py output")
        return None, None

    print(f"MP backend started, server URL: {url}")
    return process, url


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark driver for multi-node SGLang"
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ray-actor", "mp"],
        required=True,
        help="Backend to use: ray-actor (Engine with use_ray=True) or mp (launch_server processes)",
    )

    # Model and parallelism
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-7B",
        help="Model path (HuggingFace model ID or local path)",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--nnodes", type=int, default=2, help="Number of nodes (MP backend only)")
    parser.add_argument("--port", type=int, default=30000, help="Server port")

    # Benchmark parameters
    parser.add_argument("--dataset", type=str, default="random", help="Dataset name")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts")
    parser.add_argument("--input-len", type=int, default=512, help="Random input length")
    parser.add_argument("--output-len", type=int, default=256, help="Random output length")
    parser.add_argument("--range-ratio", type=float, default=0.5, help="Random range ratio")
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (inf for saturated throughput)",
    )

    # Control flags
    parser.add_argument(
        "--skip-package-fix",
        action="store_true",
        help="Skip package compatibility fixes",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Keep server running after benchmark (for debugging)",
    )

    args = parser.parse_args()

    # Step 1: Fix packages
    if not args.skip_package_fix:
        fix_packages()
        fix_packages_on_workers()

    # Step 2: Launch server based on backend
    process = None
    url = None

    try:
        if args.backend == "ray-actor":
            process, url = launch_ray_actor_backend(args)
            if process is None:
                print("Failed to launch Ray actor backend")
                sys.exit(1)
        else:  # mp
            process, url = launch_mp_backend(args)
            if process is None:
                print("Failed to launch MP backend")
                sys.exit(1)

        # Step 3: Wait for server to be ready
        if not wait_for_server(url, timeout=600):
            print("Server failed to start within timeout")
            sys.exit(1)

        # Step 4: Run benchmark
        results = run_benchmark(url, args)

        # Step 5: Print results summary
        print_results_summary(results, args)

        # Step 6: Keep running if requested
        if args.keep_server:
            print("\nServer is running. Press Ctrl+C to shutdown...")
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                print("\nShutting down...")

    finally:
        # Cleanup
        if process is not None:
            print("Terminating server...")
            if hasattr(process, "shutdown"):
                # Ray actor - use shutdown method
                try:
                    import ray

                    ray.get(process.shutdown.remote(), timeout=30)
                except Exception as e:
                    print(f"Warning during shutdown: {e}")
            else:
                # Subprocess - terminate directly
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("Force killing server process...")
                    process.kill()
                    process.wait()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
