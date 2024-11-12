import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List

import requests
from sglang_router import PolicyType, Router

# Global processes list for cleanup
_processes: List[subprocess.Popen] = []


def cleanup_processes(signum=None, frame=None):
    """Cleanup function to kill all worker processes."""
    print("\nCleaning up processes...")
    for process in _processes:
        try:
            # Kill the entire process group
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
            process.wait()
        except:
            pass
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, cleanup_processes)
signal.signal(signal.SIGTERM, cleanup_processes)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch SGLang Router Server")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host address to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=30000, help="Base port number for workers"
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=2,
        help="Number of worker processes (degree of parallelism)",
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--local-tokenizer-path",
        type=str,
        required=True,
        help="Path to the local tokenizer",
    )
    return parser.parse_args()


def launch_workers(args) -> tuple[List[subprocess.Popen], List[str]]:
    """Launch all worker processes concurrently using subprocess."""
    processes = []
    worker_urls = []

    # Launch each worker process
    for i in range(args.dp):
        port = args.port + i
        url = f"http://{args.host}:{port}"
        worker_urls.append(url)
        # TODO: replace this with launch_server, and move this file to sglang/ because it depends on sglang
        # We don't
        command = f"export CUDA_VISIBLE_DEVICES={i}; python -m sglang.launch_server --model-path {args.model_path} --host {args.host} --port {port}"
        print(command)
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
        _processes.append(process)  # Add to global list for cleanup

    return processes, worker_urls


def wait_for_healthy_workers(worker_urls: List[str], timeout: int = 300) -> bool:
    """Block until all workers are healthy or timeout is reached."""
    start_time = time.time()
    healthy_workers: Dict[str, bool] = {url: False for url in worker_urls}

    while time.time() - start_time < timeout:
        print("checking healthiness...")
        all_healthy = True

        for url in worker_urls:
            if not healthy_workers[url]:  # Only check workers that aren't healthy yet
                try:
                    response = requests.get(f"{url}/health")
                    if response.status_code == 200:
                        print(f"Worker at {url} is healthy")
                        healthy_workers[url] = True
                    else:
                        all_healthy = False
                except requests.RequestException:
                    all_healthy = False

        if all_healthy:
            print("All workers are healthy!")
            return True

        time.sleep(5)

    # If we get here, we've timed out
    unhealthy_workers = [url for url, healthy in healthy_workers.items() if not healthy]
    print(f"Timeout waiting for workers: {unhealthy_workers}")
    return False


def main():
    """Main function to launch the router and workers."""
    args = parse_args()
    processes = None

    try:
        # Launch all workers concurrently
        processes, worker_urls = launch_workers(args)

        # Block until all workers are healthy
        if not wait_for_healthy_workers(worker_urls):
            raise RuntimeError("Failed to start all workers")

        # Initialize and start the router
        router = Router(
            worker_urls=worker_urls,
            policy=PolicyType.ApproxTree,
            tokenizer_path=args.local_tokenizer_path,
        )

        print("Starting router...")
        router.start()

        # Keep the main process running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup: Kill all worker processes
        if processes:
            for process in processes:
                process.kill()


if __name__ == "__main__":
    main()
