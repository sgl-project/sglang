import argparse
import asyncio
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
from setproctitle import setproctitle
from sglang_router.launch_router import RouterArgs, launch_router
from sglang_router.multi_model import MultiModelConfigError, load_multi_model_config
from sglang_router.resource_aware_router import (
    RuntimeEndpoint,
    run_resource_aware_router,
)

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import is_port_available


def setup_logger():
    logger = logging.getLogger("router")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[Router (Python)] %(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


# Create new process group
def run_server(server_args, dp_rank, worker_env: dict[str, str] | None = None):
    """
    Note:

    1. Without os.setpgrp(), all processes share the same PGID. When you press Ctrl+C, the terminal sends SIGINT to all processes in the group simultaneously.
    This can cause leaf processes to terminate first, which messes up the cleaning order and produces orphaned processes.

    Terminal (PGID=100)
    └── Main Python Process (PGID=100)
        └── Server Process 1 (PGID=100)
            └── Scheduler 1
            └── Detokenizer 1
        └── Server Process 2 (PGID=100)
            └── Scheduler 2
            └── Detokenizer 2

    2. With os.setpgrp(), the main Python process and its children are in a separate group. Now:

    Terminal (PGID=100)
    └── Main Python Process (PGID=200)
        └── Server Process 1 (PGID=300)
            └── Scheduler 1
            └── Detokenizer 1
        └── Server Process 2 (PGID=400)
            └── Scheduler 2
            └── Detokenizer 2
    """
    # create new process group
    os.setpgrp()

    # Set device visibility before importing a server entrypoint. This lets the
    # supervisor give every model replica an independent, possibly non-
    # contiguous physical GPU slice while SGLang still sees local device IDs.
    if worker_env:
        os.environ.update(worker_env)

    setproctitle("sglang::server")
    # Set SGLANG_DP_RANK environment variable
    os.environ["SGLANG_DP_RANK"] = str(dp_rank)

    # Launch server in appropriate mode (HTTP or gRPC)
    if server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    else:
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


def launch_server_process(
    server_args: ServerArgs,
    worker_port: int,
    dp_id: int,
    *,
    base_gpu_id: int | None = None,
    worker_env: dict[str, str] | None = None,
) -> mp.Process:
    """Launch a single server process with the given args and port."""
    changes = {
        "port": worker_port,
        "base_gpu_id": (
            dp_id * server_args.tp_size if base_gpu_id is None else base_gpu_id
        ),
        "dp_size": 1,
    }
    # Three channels, newest first. A wheel that has the read-only record but not
    # `replace_resolved` still has `_late_resolution`, and plain assignment there
    # raises; only a wheel with neither accepts `setattr`.
    replace_resolved = getattr(server_args, "replace_resolved", None)
    if replace_resolved is not None:
        worker_args = replace_resolved("sglang_router.launch_server_process", **changes)
    else:
        worker_args = copy.deepcopy(server_args)
        late = getattr(worker_args, "_late_resolution", None)
        if late is not None:
            late("sglang_router.launch_server_process", **changes)
        else:
            for field, value in changes.items():
                setattr(worker_args, field, value)
    server_args = worker_args

    proc = mp.Process(
        target=run_server,
        args=(server_args, dp_id, dict(worker_env) if worker_env else None),
    )
    proc.start()
    return proc


def wait_for_server_health(host: str, port: int, timeout: int = 300) -> bool:
    """Wait for server to be healthy by checking /health endpoint."""
    start_time = time.perf_counter()
    url = f"http://{_connectable_host(host)}:{port}/health"

    while time.perf_counter() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def _connectable_host(host: str) -> str:
    """Convert all-interface bind addresses to a host usable by local clients."""
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def run_router(router_args: RouterArgs):
    """Run the router in its own process group for deterministic cleanup."""
    os.setpgrp()
    launch_router(router_args)


def wait_for_worker_registration(
    router_host: str,
    router_port: int,
    worker_id: str,
    model_id: str,
    timeout: int,
) -> bool:
    """Wait until IGW's asynchronous worker job is visible in its registry."""
    start_time = time.perf_counter()
    url = f"http://{_connectable_host(router_host)}:{router_port}/workers"
    while time.perf_counter() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                workers = response.json().get("workers", [])
                if any(
                    worker.get("id") == worker_id
                    and worker.get("model_id") == model_id
                    for worker in workers
                ):
                    return True
        except (requests.exceptions.RequestException, ValueError):
            pass
        time.sleep(1)
    return False


def register_igw_worker(
    router_host: str,
    router_port: int,
    worker_url: str,
    model_id: str,
    timeout: int,
) -> str:
    """Register a healthy local runtime with IGW under its configured model ID."""
    url = f"http://{_connectable_host(router_host)}:{router_port}/workers"
    try:
        response = requests.post(
            url,
            json={"url": worker_url, "model_id": model_id},
            timeout=10,
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"failed to register {model_id!r} worker {worker_url} with IGW: {exc}"
        ) from exc

    if response.status_code not in (200, 201, 202):
        raise RuntimeError(
            f"IGW rejected {model_id!r} worker {worker_url}: "
            f"HTTP {response.status_code}: {response.text}"
        )

    try:
        worker_id = response.json().get("worker_id")
    except ValueError as exc:
        raise RuntimeError(
            f"IGW returned a non-JSON registration response for {worker_url}"
        ) from exc
    if not worker_id:
        raise RuntimeError(f"IGW did not return a worker ID for {worker_url}")
    if not wait_for_worker_registration(
        router_host, router_port, worker_id, model_id, timeout
    ):
        raise RuntimeError(
            f"IGW did not register {model_id!r} worker {worker_url} within {timeout}s"
        )
    return worker_id


def find_available_ports(base_port: int, count: int) -> List[int]:
    """Find consecutive available ports starting from base_port."""
    available_ports = []
    current_port = base_port

    while len(available_ports) < count:
        if is_port_available(current_port):
            available_ports.append(current_port)
        current_port += random.randint(100, 1000)

    return available_ports


def cleanup_processes(processes: List[mp.Process]):
    for process in processes:
        if process.pid is None:
            continue
        logger.info(f"Terminating process group {process.pid}")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process group may already be terminated
            pass

    # Wait for processes to terminate
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            logger.warning(
                f"Process {process.pid} did not terminate gracefully, forcing kill"
            )
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    logger.info("All process groups terminated")


def launch_multi_model_server(config_path: str):
    """Launch a static set of isolated model replicas behind one IGW endpoint.

    Each ``gpu_group`` in the JSON configuration starts one SGLang server with
    ``dp_size=1``. Repeating groups for a model is standard process-level DP;
    IGW owns the model-aware request routing. This intentionally does not mix
    models in SRT's native DP controller or in a CUDA/NCCL process group.
    """
    try:
        multi_model_config = load_multi_model_config(config_path)
    except MultiModelConfigError as exc:
        raise RuntimeError(f"invalid multi-model config: {exc}") from exc

    router_kwargs = dict(multi_model_config.router_args)
    router_kwargs["enable_igw"] = True
    router_kwargs["worker_urls"] = []
    try:
        router_args = RouterArgs(**router_kwargs)
    except TypeError as exc:
        raise RuntimeError(f"invalid config.router arguments: {exc}") from exc

    worker_args_by_replica = []
    for replica in multi_model_config.replicas:
        worker_kwargs = dict(replica.server_args)
        worker_kwargs.update(
            {
                "base_gpu_id": 0,
                "dp_size": 1,
                "host": multi_model_config.worker_host,
                "model_path": replica.model_path,
                "served_model_name": replica.model_id,
            }
        )
        try:
            worker_args_by_replica.append((replica, ServerArgs(**worker_kwargs)))
        except TypeError as exc:
            raise RuntimeError(
                f"invalid server_args for model {replica.model_id!r}: {exc}"
            ) from exc

    router_process = mp.Process(target=run_router, args=(router_args,))
    router_process.start()
    managed_processes = [router_process]
    server_processes: list[mp.Process] = []
    registered_workers: list[tuple[str, str]] = []

    def cleanup_and_exit(_sig, _frame):
        cleanup_processes(server_processes)
        cleanup_processes(managed_processes)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    signal.signal(signal.SIGQUIT, cleanup_and_exit)

    try:
        if not wait_for_server_health(
            router_args.host, router_args.port, multi_model_config.startup_timeout_secs
        ):
            raise RuntimeError(
                f"IGW did not become healthy on {router_args.host}:{router_args.port}"
            )

        worker_ports = find_available_ports(
            multi_model_config.worker_base_port, len(multi_model_config.replicas)
        )
        launches = []
        for replica_index, ((replica, worker_args), worker_port) in enumerate(
            zip(worker_args_by_replica, worker_ports, strict=True)
        ):
            worker_url = (
                f"http://{_connectable_host(multi_model_config.worker_host)}:{worker_port}"
            )
            logger.info(
                "Launching model %s replica %s on %s with CUDA_VISIBLE_DEVICES=%s",
                replica.model_id,
                replica_index,
                worker_url,
                ",".join(map(str, replica.gpu_ids)),
            )
            process = launch_server_process(
                worker_args,
                worker_port,
                # Each gpu_group is an independent Runtime, not a rank inside
                # one native SGLang DP controller. Its local DP rank must
                # therefore always be zero; otherwise non-first Runtimes can
                # serve requests while publishing an empty /v1/loads response.
                0,
                base_gpu_id=0,
                worker_env={"CUDA_VISIBLE_DEVICES": ",".join(map(str, replica.gpu_ids))},
            )
            server_processes.append(process)
            launches.append((replica, worker_port, worker_url))

        for replica, worker_port, worker_url in launches:
            if not wait_for_server_health(
                multi_model_config.worker_host,
                worker_port,
                multi_model_config.startup_timeout_secs,
            ):
                raise RuntimeError(
                    f"{replica.model_id!r} worker did not become healthy on {worker_url}"
                )
            worker_id = register_igw_worker(
                router_args.host,
                router_args.port,
                worker_url,
                replica.model_id,
                multi_model_config.startup_timeout_secs,
            )
            registered_workers.append((replica.model_id, worker_id))
            logger.info(
                "Registered model %s worker %s with IGW", replica.model_id, worker_id
            )

        if multi_model_config.model_resolver is not None:
            resolver_config = multi_model_config.model_resolver
            endpoints = tuple(
                RuntimeEndpoint(model_id=replica.model_id, url=worker_url)
                for replica, _worker_port, worker_url in launches
            )
            backend_url = (
                f"http://{_connectable_host(router_args.host)}:{router_args.port}"
            )
            resolver_process = mp.Process(
                target=run_resource_aware_router,
                args=(resolver_config, endpoints, backend_url),
            )
            resolver_process.start()
            managed_processes.append(resolver_process)
            if not wait_for_server_health(
                resolver_config.host,
                resolver_config.port,
                multi_model_config.startup_timeout_secs,
            ):
                raise RuntimeError(
                    "Resource-aware Router did not become healthy on "
                    f"{resolver_config.host}:{resolver_config.port}"
                )
            logger.info(
                "Resource-aware Router is ready on http://%s:%s; backend Router is %s",
                resolver_config.host,
                resolver_config.port,
                backend_url,
            )

        logger.info(
            "Multi-model gateway is ready with %d worker replicas across %d models",
            len(registered_workers),
            len({model_id for model_id, _ in registered_workers}),
        )
        router_process.join()
    finally:
        cleanup_processes(server_processes)
        cleanup_processes(managed_processes)


def main():
    # CUDA runtime isn't fork-safe, which can lead to subtle bugs or crashes.
    # Set this before either legacy or multi-model launch paths create children.
    mp.set_start_method("spawn")

    if "--multi-model-config" in sys.argv:
        multi_model_parser = argparse.ArgumentParser(
            description="Launch isolated SGLang model replicas behind one IGW endpoint"
        )
        multi_model_parser.add_argument(
            "--multi-model-config",
            required=True,
            help="Path to the JSON configuration for the static multi-model supervisor.",
        )
        launch_multi_model_server(
            multi_model_parser.parse_args().multi_model_config
        )
        return

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
    # No extra retry/CB flags here; RouterArgs.add_cli_args already defines them with router- prefix

    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    # Older released wheels resolve in the constructor and have no gate.
    if hasattr(server_args, "resolve_once"):
        server_args.resolve_once()
    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

    # Find available ports for workers. The count is the operator's requested
    # replica count, which is the raw field on purpose: `--dwdp-size` makes
    # resolution declare a `dp_size` that describes one multi-rank server's
    # internal topology, and spawning that many single-rank children would ask
    # for dp_size^2 GPUs.
    worker_ports = find_available_ports(
        args.router_dp_worker_base_port, server_args.dp_size
    )

    # Start server processes
    server_processes = []

    for i, worker_port in enumerate(worker_ports):
        logger.info(f"Launching DP server process {i} on port {worker_port}")
        proc = launch_server_process(server_args, worker_port, i)
        server_processes.append(proc)

    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_processes(server_processes))
    signal.signal(
        signal.SIGTERM, lambda sig, frame: cleanup_processes(server_processes)
    )
    signal.signal(
        signal.SIGQUIT, lambda sig, frame: cleanup_processes(server_processes)
    )

    # Update router args with worker URLs
    # Use grpc:// protocol if server is in gRPC mode, otherwise http://
    protocol = "grpc" if server_args.grpc_mode else "http"
    router_args.worker_urls = [
        f"{protocol}://{server_args.host}:{port}" for port in worker_ports
    ]

    # Start the router
    try:
        launch_router(router_args)
    except Exception as e:
        logger.error(f"Failed to start router: {e}")
        cleanup_processes(server_processes)
        sys.exit(1)


if __name__ == "__main__":
    main()
