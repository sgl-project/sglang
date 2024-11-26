import argparse
import copy
import logging
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from typing import List

import requests
from sglang_router.launch_router import RouterArgs, launch_router

from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_port_available
from sglang.utils import get_exception_traceback


def setup_logger():
    logger = logging.getLogger("router")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[Router (Python)] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Create new process group
def run_server(server_args, dp_rank):
    os.setpgrp()  # Create new process group

    # Set DP_RANK environment variable
    os.environ["DP_RANK"] = str(dp_rank)

    launch_server(server_args)


def launch_server_process(
    server_args: ServerArgs, worker_port: int, dp_id: int
) -> mp.Process:
    """Launch a single server process with the given args and port."""
    server_args = copy.deepcopy(server_args)
    server_args.port = worker_port
    server_args.base_gpu_id = dp_id * server_args.tp_size
    server_args.dp_size = 1

    proc = mp.Process(target=run_server, args=(server_args, dp_id))
    proc.start()
    return proc


def cleanup_processes(processes: List[mp.Process]):
    logger = logging.getLogger("router")
    logger.info("Cleaning up processes...")
    for proc in processes:
        if proc.is_alive():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.join(timeout=3)
                if proc.is_alive():
                    logger.warning(
                        f"Process {proc.pid} did not terminate gracefully, force killing..."
                    )
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def setup_signal_handlers(cleanup_func):
    """Setup handlers for various termination signals."""

    def signal_handler(signum, frame):
        cleanup_func()
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, signal_handler)


def wait_for_server_health(host: str, port: int, timeout: int = 300) -> bool:
    """Wait for server to be healthy by checking /health endpoint."""
    start_time = time.time()
    url = f"http://{host}:{port}/health"

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def find_available_ports(base_port: int, count: int) -> List[int]:
    """Find consecutive available ports starting from base_port."""
    available_ports = []
    current_port = base_port

    while len(available_ports) < count:
        if is_port_available(current_port):
            available_ports.append(current_port)
        current_port += random.randint(100, 1000)

    return available_ports


def main():
    logger = setup_logger()

    # CUDA runtime isn't fork-safe, which can lead to subtle bugs or crashes
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Launch SGLang router and server processes"
    )

    ServerArgs.add_cli_args(parser)
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    parser.add_argument(
        "--router-dp-worker-base-port",
        type=int,
        default=31000,
        help="Base port number for data parallel workers",
    )

    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

    # Find available ports for workers
    worker_ports = find_available_ports(
        args.router_dp_worker_base_port, server_args.dp_size
    )

    # Start server processes
    server_processes = []

    try:
        for i, worker_port in enumerate(worker_ports):
            logger.info(f"Launching DP server process {i} on port {worker_port}")
            proc = launch_server_process(server_args, worker_port, i)
            server_processes.append(proc)

        # Setup cleanup handler
        setup_signal_handlers(lambda: cleanup_processes(server_processes))

        # Wait for all servers to be healthy
        all_healthy = True

        for port in worker_ports:
            if not wait_for_server_health(server_args.host, port):
                logger.error(f"Server on port {port} failed to become healthy")
                all_healthy = False
                break

        if not all_healthy:
            logger.error("Not all servers are healthy. Shutting down...")
            cleanup_processes(server_processes)
            sys.exit(1)

        logger.info("All servers are healthy. Starting router...")

        # Update router args with worker URLs
        router_args.worker_urls = [
            f"http://{server_args.host}:{port}" for port in worker_ports
        ]

        # Start the router
        router = launch_router(router_args)

        if router is None:
            logger.error("Failed to start router. Shutting down...")
            cleanup_processes(server_processes)
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.error(get_exception_traceback())
    finally:
        logger.info("Cleaning up processes...")
        cleanup_processes(server_processes)


if __name__ == "__main__":
    main()
